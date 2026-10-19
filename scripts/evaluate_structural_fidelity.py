from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robot_portrait.apdrawinggan_preprocess import (  # noqa: E402
    FaceNotDetectedError,
    YuNetConfig,
    detect_yunet_five_points,
)


REGIONS = (
    "eyes",
    "eyebrows",
    "nose",
    "mouth",
    "contour",
    "hair",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Automatically evaluate regional structural fidelity between "
            "each method's actual photo input and generated portrait. "
            "This is a proxy metric, not a subjective artistic-quality score."
        )
    )
    parser.add_argument(
        "--benchmark-csv",
        type=Path,
        default=Path("outputs/evaluation_benchmark/benchmark_rows.csv"),
        help="CSV produced by scripts/benchmark_end_to_end.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/structural_fidelity"),
        help="Output directory for CSV/JSON results.",
    )
    parser.add_argument(
        "--yunet-model",
        type=Path,
        default=Path(
            "third_party/models/face_detection_yunet_2023mar.onnx"
        ),
        help="YuNet ONNX model path.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.75,
        help=(
            "YuNet threshold used only for evaluation-region localization. "
            "This does not change the generation benchmark."
        ),
    )
    parser.add_argument(
        "--edge-tolerance",
        type=int,
        default=2,
        help="Pixel tolerance used for edge matching.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Optional directory for ROI/edge debug images.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path)


def read_image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    buffer = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to decode image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise OSError(f"Unable to encode image: {path}")
    encoded.tofile(path)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def paired_success_rows(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    completed_by_sample: dict[str, set[str]] = {}
    for row in rows:
        if row.get("status") != "completed":
            continue
        completed_by_sample.setdefault(
            row["sample_id"],
            set(),
        ).add(row["method"])

    paired_ids = {
        sample_id
        for sample_id, methods in completed_by_sample.items()
        if {"baseline", "apdrawinggan"} <= methods
    }

    selected = [
        row
        for row in rows
        if (
            row.get("status") == "completed"
            and row.get("sample_id") in paired_ids
            and row.get("method") in {"baseline", "apdrawinggan"}
        )
    ]
    selected.sort(
        key=lambda row: (
            row["sample_id"],
            0 if row["method"] == "baseline" else 1,
        )
    )
    return selected


def actual_source_and_output(
    row: dict[str, str],
) -> tuple[Path, Path]:
    run_dir = Path(row["output_dir"])
    method = row["method"]

    if method == "baseline":
        return (
            run_dir / "prepared.png",
            run_dir / "baseline.png",
        )

    aligned = (
        run_dir
        / "apdrawinggan_preprocess"
        / "aligned.png"
    )
    source = aligned if aligned.is_file() else run_dir / "prepared.png"
    return (
        source,
        run_dir / "apdrawinggan.png",
    )


def detect_face(
    source: np.ndarray,
    model: Path,
    score_threshold: float,
):
    return detect_yunet_five_points(
        source,
        YuNetConfig(
            model_path=model,
            score_threshold=score_threshold,
            max_input_side=640,
        ),
    )


def clip_box(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    left = max(0, min(width - 1, int(round(x0))))
    top = max(0, min(height - 1, int(round(y0))))
    right = max(left + 1, min(width, int(round(x1))))
    bottom = max(top + 1, min(height, int(round(y1))))
    return left, top, right, bottom


def rect_mask(
    shape: tuple[int, int],
    boxes: list[tuple[int, int, int, int]],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for left, top, right, bottom in boxes:
        mask[top:bottom, left:right] = 255
    return mask


def region_masks(
    shape: tuple[int, int],
    face,
) -> dict[str, np.ndarray]:
    height, width = shape
    left, top, right, bottom = face.face_box
    fw = float(right - left)
    fh = float(bottom - top)

    points = np.asarray(face.points, dtype=np.float64)
    eye_a, eye_b, nose, mouth_a, mouth_b = points

    eye_span = max(
        1.0,
        float(np.linalg.norm(eye_a - eye_b)),
    )
    mouth_span = max(
        1.0,
        float(np.linalg.norm(mouth_a - mouth_b)),
    )

    eye_boxes = []
    for eye in (eye_a, eye_b):
        eye_boxes.append(
            clip_box(
                eye[0] - 0.34 * eye_span,
                eye[1] - 0.24 * fh,
                eye[0] + 0.34 * eye_span,
                eye[1] + 0.20 * fh,
                width,
                height,
            )
        )

    eyebrow_boxes = []
    for eye in (eye_a, eye_b):
        eyebrow_boxes.append(
            clip_box(
                eye[0] - 0.38 * eye_span,
                eye[1] - 0.34 * fh,
                eye[0] + 0.38 * eye_span,
                eye[1] - 0.06 * fh,
                width,
                height,
            )
        )

    nose_box = clip_box(
        nose[0] - 0.22 * fw,
        min(eye_a[1], eye_b[1]) + 0.04 * fh,
        nose[0] + 0.22 * fw,
        max(mouth_a[1], mouth_b[1]) - 0.05 * fh,
        width,
        height,
    )

    mouth_cx = float((mouth_a[0] + mouth_b[0]) / 2.0)
    mouth_cy = float((mouth_a[1] + mouth_b[1]) / 2.0)
    mouth_box = clip_box(
        mouth_cx - 0.85 * mouth_span,
        mouth_cy - 0.16 * fh,
        mouth_cx + 0.85 * mouth_span,
        mouth_cy + 0.20 * fh,
        width,
        height,
    )

    # Face contour ring: outer face box minus an inset interior.
    contour = np.zeros((height, width), dtype=np.uint8)
    contour[
        max(0, top):min(height, bottom),
        max(0, left):min(width, right),
    ] = 255
    inset_x = int(round(0.16 * fw))
    inset_y = int(round(0.12 * fh))
    il = min(width, max(0, left + inset_x))
    ir = min(width, max(il, right - inset_x))
    it = min(height, max(0, top + inset_y))
    ib = min(height, max(it, bottom - inset_y))
    if ir > il and ib > it:
        contour[it:ib, il:ir] = 0

    eye_y = float((eye_a[1] + eye_b[1]) / 2.0)
    hair_box = clip_box(
        left - 0.08 * fw,
        top - 0.08 * fh,
        right + 0.08 * fw,
        eye_y - 0.08 * fh,
        width,
        height,
    )

    return {
        "eyes": rect_mask((height, width), eye_boxes),
        "eyebrows": rect_mask((height, width), eyebrow_boxes),
        "nose": rect_mask((height, width), [nose_box]),
        "mouth": rect_mask((height, width), [mouth_box]),
        "contour": contour,
        "hair": rect_mask((height, width), [hair_box]),
    }


def source_edges(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(
        enhanced,
        (3, 3),
        0,
    )
    edges = cv2.Canny(
        blurred,
        50,
        150,
        L2gradient=True,
    )
    return edges > 0


def generated_lines(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # The project's generated portraits are white-background line drawings.
    return gray < 230


def regional_edge_f1(
    reference: np.ndarray,
    predicted: np.ndarray,
    region_mask: np.ndarray,
    tolerance: int,
) -> dict[str, float | int | None]:
    region = region_mask > 0
    ref = reference & region
    pred = predicted & region

    ref_count = int(ref.sum())
    pred_count = int(pred.sum())

    if ref_count == 0 or pred_count == 0:
        return {
            "precision": None,
            "recall": None,
            "f1": None,
            "reference_pixels": ref_count,
            "predicted_pixels": pred_count,
            "density_ratio": (
                None
                if ref_count == 0
                else pred_count / ref_count
            ),
        }

    radius = max(0, int(tolerance))
    if radius == 0:
        ref_d = ref
        pred_d = pred
    else:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * radius + 1, 2 * radius + 1),
        )
        ref_d = cv2.dilate(
            ref.astype(np.uint8),
            kernel,
        ) > 0
        pred_d = cv2.dilate(
            pred.astype(np.uint8),
            kernel,
        ) > 0

    precision = float((pred & ref_d).sum()) / pred_count
    recall = float((ref & pred_d).sum()) / ref_count
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "reference_pixels": ref_count,
        "predicted_pixels": pred_count,
        "density_ratio": pred_count / ref_count,
    }


def evaluate_one(
    row: dict[str, str],
    model: Path,
    score_threshold: float,
    tolerance: int,
    debug_dir: Path | None,
) -> dict[str, Any]:
    source_path, generated_path = actual_source_and_output(row)
    source = read_image(source_path)
    generated = read_image(generated_path)

    if generated.shape[:2] != source.shape[:2]:
        generated = cv2.resize(
            generated,
            (source.shape[1], source.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    face = detect_face(
        source,
        model,
        score_threshold,
    )
    masks = region_masks(
        source.shape[:2],
        face,
    )
    ref = source_edges(source)
    pred = generated_lines(generated)

    output: dict[str, Any] = {
        "sample_id": row["sample_id"],
        "category": row.get("category", ""),
        "method": row["method"],
        "source_path": str(source_path),
        "generated_path": str(generated_path),
        "evaluation_yunet_score": float(face.score),
    }

    f1_values = []
    for region in REGIONS:
        metrics = regional_edge_f1(
            ref,
            pred,
            masks[region],
            tolerance,
        )
        for key, value in metrics.items():
            output[f"{region}_{key}"] = value
        if metrics["f1"] is not None:
            f1_values.append(float(metrics["f1"]))

    output["regional_f1_mean"] = (
        statistics.fmean(f1_values)
        if f1_values
        else None
    )

    if debug_dir is not None:
        canvas = generated.copy()
        colors = {
            "eyes": (255, 0, 0),
            "eyebrows": (0, 128, 255),
            "nose": (0, 255, 255),
            "mouth": (0, 0, 255),
            "contour": (0, 255, 0),
            "hair": (255, 0, 255),
        }
        for region, mask in masks.items():
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(
                canvas,
                contours,
                -1,
                colors[region],
                1,
            )
        write_image(
            debug_dir
            / f"{row['sample_id']}_{row['method']}_regions.png",
            canvas,
        )

    return output


def numeric(
    rows: list[dict[str, Any]],
    field: str,
) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(field)
        if value is None or value == "":
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            pass
    return values


def stats(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": (
            statistics.stdev(values)
            if len(values) >= 2
            else 0.0
        ),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def summarize(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    fields = [
        *(f"{region}_f1" for region in REGIONS),
        "regional_f1_mean",
    ]

    per_method: dict[str, Any] = {}
    for method in ("baseline", "apdrawinggan"):
        method_rows = [
            row
            for row in results
            if row["method"] == method
        ]
        per_method[method] = {
            field: stats(numeric(method_rows, field))
            for field in fields
        }

    paired_delta: dict[str, Any] = {}
    sample_map: dict[str, dict[str, dict[str, Any]]] = {}
    for row in results:
        sample_map.setdefault(
            row["sample_id"],
            {},
        )[row["method"]] = row

    for field in fields:
        deltas = []
        for sample_id, methods in sample_map.items():
            if (
                "baseline" not in methods
                or "apdrawinggan" not in methods
            ):
                continue
            a = methods["baseline"].get(field)
            b = methods["apdrawinggan"].get(field)
            if a is None or b is None:
                continue
            deltas.append(float(b) - float(a))
        paired_delta[field] = stats(deltas)

    by_category: dict[str, Any] = {}
    categories = sorted(
        {row.get("category", "") for row in results}
    )
    for category in categories:
        by_category[category] = {}
        for method in ("baseline", "apdrawinggan"):
            subset = [
                row
                for row in results
                if (
                    row.get("category") == category
                    and row["method"] == method
                )
            ]
            by_category[category][method] = {
                field: stats(numeric(subset, field))
                for field in fields
            }

    return {
        "schema_version": 1,
        "interpretation": (
            "Regional edge F1 is an automatic structural-fidelity proxy "
            "between the method-specific photo input and generated line art. "
            "It is not a subjective artistic-style score."
        ),
        "paired_samples": len(
            {
                row["sample_id"]
                for row in results
            }
        ),
        "rows": len(results),
        "per_method": per_method,
        "paired_apdrawinggan_minus_baseline": paired_delta,
        "by_category": by_category,
    }


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    fields = [
        "sample_id",
        "category",
        "method",
        "source_path",
        "generated_path",
        "evaluation_yunet_score",
    ]
    for region in REGIONS:
        fields.extend(
            [
                f"{region}_precision",
                f"{region}_recall",
                f"{region}_f1",
                f"{region}_reference_pixels",
                f"{region}_predicted_pixels",
                f"{region}_density_ratio",
            ]
        )
    fields.append("regional_f1_mean")

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()

    csv_path = resolve(args.benchmark_csv).resolve()
    output_dir = resolve(args.output_dir).resolve()
    model = resolve(args.yunet_model).resolve()
    debug_dir = (
        resolve(args.debug_dir).resolve()
        if args.debug_dir is not None
        else None
    )

    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    if not model.is_file():
        raise FileNotFoundError(model)
    if args.edge_tolerance < 0:
        raise ValueError("--edge-tolerance must be >= 0")

    rows = paired_success_rows(
        load_rows(csv_path)
    )

    if len(rows) != 36:
        print(
            f"Warning: expected 36 paired-success rows, found {len(rows)}.",
            file=sys.stderr,
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    if debug_dir is not None:
        debug_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    print("RobotPortraitArt Automatic Structural Fidelity V1")
    print(f"Rows:             {len(rows)}")
    print(f"YuNet threshold:  {args.score_threshold}")
    print(f"Edge tolerance:   {args.edge_tolerance}px")
    print(f"Output:           {output_dir}")

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        label = f"{row['sample_id']} / {row['method']}"
        try:
            result = evaluate_one(
                row,
                model,
                args.score_threshold,
                args.edge_tolerance,
                debug_dir,
            )
            results.append(result)
            print(
                f"[{index:02d}/{len(rows):02d}] "
                f"{label:28s} "
                f"meanF1={result['regional_f1_mean']:.4f}"
            )
        except Exception as exc:
            failures.append(
                {
                    "sample_id": row.get("sample_id"),
                    "category": row.get("category"),
                    "method": row.get("method"),
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            print(
                f"[{index:02d}/{len(rows):02d}] "
                f"{label:28s} FAILED: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    write_csv(
        output_dir / "structural_fidelity_rows.csv",
        results,
    )

    summary = summarize(results)
    (
        output_dir / "structural_fidelity_summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        output_dir / "structural_fidelity_failures.json"
    ).write_text(
        json.dumps(
            failures,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("")
    print("Finished.")
    print(f"  Evaluated: {len(results)}")
    print(f"  Failed:    {len(failures)}")
    print(
        "  CSV:       "
        + str(output_dir / "structural_fidelity_rows.csv")
    )
    print(
        "  Summary:   "
        + str(output_dir / "structural_fidelity_summary.json")
    )

    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
