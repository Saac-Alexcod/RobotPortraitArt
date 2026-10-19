from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

CATEGORY_ORDER = (
    "easy",
    "small_face",
    "dark",
    "soft",
    "rotated",
)

CATEGORY_RULES = {
    "easy": {
        "face_score_min": 0.90,
        "brightness_min": 80.0,
        "brightness_max": 190.0,
        "face_area_min": 0.08,
        "face_area_max": 0.35,
        "abs_roll_max": 12.0,
        "sharpness_min": 50.0,
    },
    "small_face": {
        "face_score_min": 0.75,
        "brightness_min": 60.0,
        "brightness_max": 210.0,
        "face_area_min": 0.02,
        "face_area_max": 0.06,
        "abs_roll_max": 18.0,
        "sharpness_min": 20.0,
    },
    "dark": {
        "face_score_min": 0.75,
        "brightness_min": 30.0,
        "brightness_max": 60.0,
        "face_area_min": 0.05,
        "face_area_max": 0.35,
        "abs_roll_max": 18.0,
        "sharpness_min": 20.0,
    },
    "soft": {
        "face_score_min": 0.75,
        "brightness_min": 60.0,
        "brightness_max": 210.0,
        "face_area_min": 0.05,
        "face_area_max": 0.35,
        "abs_roll_max": 18.0,
        "sharpness_min": 10.0,
        "sharpness_max": 40.0,
    },
    "rotated": {
        "face_score_min": 0.75,
        "brightness_min": 50.0,
        "brightness_max": 210.0,
        "face_area_min": 0.05,
        "face_area_max": 0.40,
        "abs_roll_min": 20.0,
        "abs_roll_max": 45.0,
        "sharpness_min": 20.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare controlled RobotPortraitArt Benchmark Faces V1.3 "
            "from CelebA plus local/core and stress-test images."
        ),
    )
    parser.add_argument(
        "--archive",
        type=Path,
        required=True,
        help="First CelebA multi-volume archive file, ending in .001.",
    )
    parser.add_argument(
        "--partition-file",
        type=Path,
        required=True,
        help="Official CelebA list_eval_partition.txt.",
    )
    parser.add_argument(
        "--seven-zip",
        type=Path,
        default=None,
        help="Optional explicit path to 7z.exe.",
    )
    parser.add_argument(
        "--yunet-model",
        type=Path,
        default=Path(
            "third_party/models/"
            "face_detection_yunet_2023mar.onnx"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/benchmark_faces_v1_3"),
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=1200,
    )
    parser.add_argument(
        "--per-category",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--extra-dir",
        type=Path,
        default=None,
        help="Local test-image directory.",
    )
    parser.add_argument(
        "--extra-count",
        type=int,
        default=5,
        help="Maximum number of local images to inspect.",
    )
    parser.add_argument(
        "--local-core-count",
        type=int,
        default=4,
        help="Number of YuNet-detected local images placed in core.",
    )
    parser.add_argument(
        "--demo-image",
        type=Path,
        default=Path("data/input/demo_face.png"),
        help=(
            "Optional project smoke-test image. Its YuNet status is recorded "
            "but it is not included in the benchmark core."
        ),
    )
    parser.add_argument(
        "--keep-candidates",
        action="store_true",
    )
    parser.add_argument(
        "--force",
        action="store_true",
    )
    return parser.parse_args()


def find_seven_zip(explicit: Path | None) -> Path:
    if explicit is not None:
        if explicit.is_file():
            return explicit
        raise FileNotFoundError(f"7z executable not found: {explicit}")

    found = shutil.which("7z")
    if found:
        return Path(found)

    for candidate in (
        Path(r"D:\Software\7-Zip\7z.exe"),
        Path(r"C:\Program Files\7-Zip\7z.exe"),
        Path(r"C:\Program Files (x86)\7-Zip\7z.exe"),
    ):
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Unable to locate 7z.exe. Add it to PATH or pass --seven-zip."
    )


def seven_zip_version(executable: Path) -> str:
    result = subprocess.run(
        [str(executable)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    text = (result.stdout or result.stderr).strip()
    return text.splitlines()[0] if text else "unknown"


def validate_archive_volumes(first_volume: Path) -> list[Path]:
    if not first_volume.is_file():
        raise FileNotFoundError(f"Archive not found: {first_volume}")

    if not first_volume.name.endswith(".001"):
        raise ValueError("--archive must point to the .001 volume.")

    prefix = first_volume.name[:-3]
    volumes = sorted(
        first_volume.parent.glob(prefix + "[0-9][0-9][0-9]")
    )
    expected = [
        f"{prefix}{index:03d}"
        for index in range(1, len(volumes) + 1)
    ]
    actual = [path.name for path in volumes]

    if not volumes or actual != expected:
        raise RuntimeError(
            "Archive volumes are missing or non-contiguous.\n"
            f"Expected: {expected}\nActual:   {actual}"
        )
    return volumes


def read_test_partition(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Partition file not found: {path}")

    names: list[str] = []
    with path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
    ) as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) == 2 and parts[1] == "2":
                names.append(parts[0])

    if not names:
        raise RuntimeError("No CelebA test-partition rows found.")
    return names


def extract_candidates(
    seven_zip: Path,
    archive: Path,
    filenames: list[str],
    work_dir: Path,
) -> Path:
    extract_root = work_dir / "extracted"
    extract_root.mkdir(parents=True, exist_ok=True)

    list_file = work_dir / "candidate_files.txt"
    list_file.write_text(
        "\n".join(f"img_celeba\\{name}" for name in filenames) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    command = [
        str(seven_zip),
        "x",
        str(archive),
        f"@{list_file}",
        f"-o{extract_root}",
        "-y",
        "-bd",
        "-scsUTF-8",
    ]

    print("")
    print("Extracting deterministic CelebA candidate set...")
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"7-Zip extraction failed with exit code {result.returncode}."
        )

    image_dir = extract_root / "img_celeba"
    if not image_dir.is_dir():
        raise RuntimeError(
            f"Expected extracted directory was not created: {image_dir}"
        )

    extracted_names = {
        path.name
        for path in image_dir.iterdir()
        if path.is_file()
    }
    missing = sorted(set(filenames) - extracted_names)
    if missing:
        raise RuntimeError(
            f"{len(missing)} candidate files were not extracted. "
            f"Examples: {', '.join(missing[:10])}"
        )

    return image_dir


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to decode image: {path}")
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"Unable to encode PNG: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def create_detector(model_path: Path) -> cv2.FaceDetectorYN:
    if not model_path.is_file():
        raise FileNotFoundError(f"YuNet model not found: {model_path}")

    return cv2.FaceDetectorYN.create(
        str(model_path),
        "",
        (320, 320),
        0.35,
        0.3,
        5000,
    )


def analyze_image(
    path: Path,
    detector: cv2.FaceDetectorYN,
) -> dict[str, float | int] | None:
    try:
        image = read_image(path)
    except ValueError:
        return None

    height, width = image.shape[:2]
    if width < 32 or height < 32:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    sharpness = float(
        cv2.Laplacian(gray, cv2.CV_64F).var()
    )

    detector.setInputSize((width, height))
    _, faces = detector.detect(image)
    if faces is None or len(faces) == 0:
        return None

    face = max(faces, key=lambda row: float(row[-1]))
    face_w = float(face[2])
    face_h = float(face[3])

    right_eye_x = float(face[4])
    right_eye_y = float(face[5])
    left_eye_x = float(face[6])
    left_eye_y = float(face[7])

    roll_deg = math.degrees(
        math.atan2(
            left_eye_y - right_eye_y,
            left_eye_x - right_eye_x,
        )
    )
    face_area_ratio = (
        max(face_w, 0.0)
        * max(face_h, 0.0)
        / float(width * height)
    )

    return {
        "width": width,
        "height": height,
        "brightness": brightness,
        "sharpness": sharpness,
        "face_score": float(face[-1]),
        "face_area_ratio": face_area_ratio,
        "roll_deg": roll_deg,
    }


def qualifies(category: str, item: dict) -> bool:
    rules = CATEGORY_RULES[category]
    abs_roll = abs(float(item["roll_deg"]))

    checks = [
        float(item["face_score"]) >= rules["face_score_min"],
        float(item["brightness"]) >= rules["brightness_min"],
        float(item["brightness"]) <= rules["brightness_max"],
        float(item["face_area_ratio"]) >= rules["face_area_min"],
        float(item["face_area_ratio"]) <= rules["face_area_max"],
        abs_roll <= rules["abs_roll_max"],
        float(item["sharpness"]) >= rules["sharpness_min"],
    ]

    if "abs_roll_min" in rules:
        checks.append(abs_roll >= rules["abs_roll_min"])
    if "sharpness_max" in rules:
        checks.append(
            float(item["sharpness"]) <= rules["sharpness_max"]
        )

    return all(checks)


def rank_category(
    category: str,
    candidates: Iterable[dict],
) -> list[dict]:
    eligible = [
        item
        for item in candidates
        if qualifies(category, item)
    ]

    if category == "easy":
        return sorted(
            eligible,
            key=lambda item: (
                -item["face_score"],
                abs(item["brightness"] - 135.0),
                abs(item["face_area_ratio"] - 0.20),
                abs(item["roll_deg"]),
                -item["sharpness"],
            ),
        )

    if category == "small_face":
        return sorted(
            eligible,
            key=lambda item: (
                item["face_area_ratio"],
                abs(item["brightness"] - 135.0),
                -item["face_score"],
            ),
        )

    if category == "dark":
        return sorted(
            eligible,
            key=lambda item: (
                item["brightness"],
                abs(item["roll_deg"]),
                -item["face_score"],
            ),
        )

    if category == "soft":
        return sorted(
            eligible,
            key=lambda item: (
                item["sharpness"],
                abs(item["brightness"] - 135.0),
                -item["face_score"],
            ),
        )

    if category == "rotated":
        return sorted(
            eligible,
            key=lambda item: (
                -abs(item["roll_deg"]),
                -item["face_score"],
            ),
        )

    raise ValueError(f"Unknown category: {category}")


def select_public(
    candidates: list[dict],
    per_category: int,
) -> tuple[list[tuple[str, dict]], dict[str, int]]:
    eligible_counts = {
        category: len(rank_category(category, candidates))
        for category in CATEGORY_ORDER
    }

    selected: list[tuple[str, dict]] = []
    used: set[str] = set()

    for category in CATEGORY_ORDER:
        count = 0
        for item in rank_category(category, candidates):
            if item["filename"] in used:
                continue
            selected.append((category, item))
            used.add(item["filename"])
            count += 1
            if count >= per_category:
                break

        if count < per_category:
            raise RuntimeError(
                f"Category {category!r} has only {count} unique samples "
                f"after cross-category deduplication. "
                f"Eligible before deduplication: "
                f"{eligible_counts[category]}."
            )

    return selected, eligible_counts


def select_reserve_public(
    candidates: list[dict],
    selected: list[tuple[str, dict]],
) -> dict:
    used = {
        item["filename"]
        for _, item in selected
    }

    for item in rank_category("easy", candidates):
        if item["filename"] not in used:
            return item

    raise RuntimeError(
        "Unable to find an additional unique reserve-standard "
        "CelebA sample from the controlled easy pool."
    )


def local_images(directory: Path | None) -> list[Path]:
    if directory is None or not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )


def format_metric(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.6f}"


def build_row(
    *,
    sample_id: str,
    benchmark_group: str,
    source_kind: str,
    source_filename: str,
    category: str,
    output_filename: str,
    output_path: Path,
    metrics: dict[str, float | int] | None,
) -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "benchmark_group": benchmark_group,
        "source_kind": source_kind,
        "source_filename": source_filename,
        "category": category,
        "output_filename": output_filename,
        "sha256": sha256_file(output_path),
        "yunet_detected": "true" if metrics else "false",
        "width": format_metric(metrics["width"] if metrics else None),
        "height": format_metric(metrics["height"] if metrics else None),
        "brightness": format_metric(
            metrics["brightness"] if metrics else None
        ),
        "sharpness": format_metric(
            metrics["sharpness"] if metrics else None
        ),
        "face_score": format_metric(
            metrics["face_score"] if metrics else None
        ),
        "face_area_ratio": format_metric(
            metrics["face_area_ratio"] if metrics else None
        ),
        "roll_deg": format_metric(
            metrics["roll_deg"] if metrics else None
        ),
    }


def make_contact_sheet(
    image_paths: list[Path],
    labels: list[str],
    output_path: Path,
    columns: int = 5,
) -> None:
    if not image_paths:
        return

    tile_w = 240
    image_h = 240
    tile_h = 280
    rows = math.ceil(len(image_paths) / columns)

    sheet = np.full(
        (rows * tile_h, columns * tile_w, 3),
        245,
        dtype=np.uint8,
    )

    for index, (image_path, label) in enumerate(
        zip(image_paths, labels, strict=True)
    ):
        image = read_image(image_path)
        height, width = image.shape[:2]

        scale = min(tile_w / width, image_h / height)
        resized = cv2.resize(
            image,
            (
                max(1, int(width * scale)),
                max(1, int(height * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )

        canvas = np.full(
            (image_h, tile_w, 3),
            255,
            dtype=np.uint8,
        )
        y0 = (image_h - resized.shape[0]) // 2
        x0 = (tile_w - resized.shape[1]) // 2
        canvas[
            y0:y0 + resized.shape[0],
            x0:x0 + resized.shape[1],
        ] = resized

        row = index // columns
        column = index % columns
        top = row * tile_h
        left = column * tile_w

        sheet[
            top:top + image_h,
            left:left + tile_w,
        ] = canvas

        cv2.putText(
            sheet,
            label[:34],
            (left + 8, top + image_h + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )

    write_png(output_path, sheet)


def main() -> None:
    args = parse_args()

    if args.candidate_count <= 0:
        raise ValueError("--candidate-count must be positive.")
    if args.per_category <= 0:
        raise ValueError("--per-category must be positive.")
    if args.local_core_count < 0:
        raise ValueError("--local-core-count cannot be negative.")

    seven_zip = find_seven_zip(args.seven_zip)
    volumes = validate_archive_volumes(args.archive)
    test_names = read_test_partition(args.partition_file)

    output_dir: Path = args.output_dir
    if output_dir.exists():
        if not args.force:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}\n"
                "Use --force to intentionally rebuild it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    work_dir = output_dir.parent / ".benchmark_faces_v1_3_work"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    candidate_names = rng.sample(
        test_names,
        k=min(args.candidate_count, len(test_names)),
    )

    print("RobotPortraitArt Benchmark Faces V1.3.1")
    print(f"7-Zip: {seven_zip}")
    print(f"7-Zip version: {seven_zip_version(seven_zip)}")
    print(
        f"Archive volumes: {len(volumes)} "
        f"({volumes[0].name} ... {volumes[-1].name})"
    )
    print(f"CelebA test entries: {len(test_names)}")
    print(f"Candidate sample: {len(candidate_names)}")
    print(f"Seed: {args.seed}")

    try:
        candidate_dir = extract_candidates(
            seven_zip,
            args.archive,
            candidate_names,
            work_dir,
        )

        detector = create_detector(args.yunet_model)

        candidates: list[dict] = []
        detection_failures: list[str] = []

        print("")
        print("Running YuNet on CelebA candidates...")
        for index, filename in enumerate(candidate_names, start=1):
            path = candidate_dir / filename
            metrics = analyze_image(path, detector)

            if metrics is None:
                detection_failures.append(filename)
            else:
                candidates.append(
                    {
                        "filename": filename,
                        "path": path,
                        **metrics,
                    }
                )

            if index % 100 == 0 or index == len(candidate_names):
                print(
                    f"  {index}/{len(candidate_names)} scanned, "
                    f"{len(candidates)} detected"
                )

        selected, eligible_counts = select_public(
            candidates,
            args.per_category,
        )
        reserve_public = select_reserve_public(
            candidates,
            selected,
        )

        print("")
        print("Controlled-category eligible counts:")
        for category in CATEGORY_ORDER:
            print(
                f"  {category:10s}: "
                f"{eligible_counts[category]}"
            )

        rows: list[dict[str, str]] = []
        core_paths: list[Path] = []
        core_labels: list[str] = []
        stress_paths: list[Path] = []
        stress_labels: list[str] = []

        for index, (category, item) in enumerate(selected, start=1):
            sample_id = f"public_{index:02d}"
            output_name = f"{sample_id}.png"
            output_path = output_dir / output_name
            write_png(output_path, read_image(item["path"]))

            core_paths.append(output_path)
            core_labels.append(f"{sample_id} | {category}")
            rows.append(
                build_row(
                    sample_id=sample_id,
                    benchmark_group="core",
                    source_kind="celeba_test",
                    source_filename=item["filename"],
                    category=category,
                    output_filename=output_name,
                    output_path=output_path,
                    metrics=item,
                )
            )

        reserve_id = "public_16"
        reserve_output_name = f"{reserve_id}.png"
        reserve_output_path = output_dir / reserve_output_name
        write_png(
            reserve_output_path,
            read_image(reserve_public["path"]),
        )
        core_paths.append(reserve_output_path)
        core_labels.append(
            f"{reserve_id} | reserve_standard"
        )
        rows.append(
            build_row(
                sample_id=reserve_id,
                benchmark_group="core",
                source_kind="celeba_test",
                source_filename=reserve_public["filename"],
                category="reserve_standard",
                output_filename=reserve_output_name,
                output_path=reserve_output_path,
                metrics=reserve_public,
            )
        )

        local_candidates: list[
            tuple[Path, dict[str, float | int] | None]
        ] = []
        for path in local_images(args.extra_dir)[:args.extra_count]:
            local_candidates.append(
                (path, analyze_image(path, detector))
            )

        detected_local = [
            item
            for item in local_candidates
            if item[1] is not None
        ]
        failed_local = [
            item
            for item in local_candidates
            if item[1] is None
        ]

        if len(detected_local) < args.local_core_count:
            raise RuntimeError(
                "Not enough YuNet-detected local images for core: "
                f"need {args.local_core_count}, "
                f"found {len(detected_local)}."
            )

        core_local = detected_local[:args.local_core_count]
        unused_detected_local = detected_local[args.local_core_count:]

        for index, (source_path, metrics) in enumerate(
            core_local,
            start=1,
        ):
            sample_id = f"local_{index:02d}"
            output_name = f"{sample_id}.png"
            output_path = output_dir / output_name
            write_png(output_path, read_image(source_path))

            core_paths.append(output_path)
            core_labels.append(f"{sample_id} | local")
            rows.append(
                build_row(
                    sample_id=sample_id,
                    benchmark_group="core",
                    source_kind="local_extra",
                    source_filename=source_path.name,
                    category="local",
                    output_filename=output_name,
                    output_path=output_path,
                    metrics=metrics,
                )
            )

        demo_status = {
            "path": str(args.demo_image),
            "exists": args.demo_image.is_file(),
            "yunet_detected": False,
        }
        if args.demo_image.is_file():
            demo_metrics = analyze_image(
                args.demo_image,
                detector,
            )
            demo_status["yunet_detected"] = (
                demo_metrics is not None
            )

        stress_index = 0
        for source_path, metrics in failed_local:
            stress_index += 1
            sample_id = f"stress_{stress_index:02d}"
            output_name = f"{sample_id}.png"
            output_path = output_dir / output_name
            write_png(output_path, read_image(source_path))

            stress_paths.append(output_path)
            stress_labels.append(f"{sample_id} | OOD")
            rows.append(
                build_row(
                    sample_id=sample_id,
                    benchmark_group="stress",
                    source_kind="local_extra",
                    source_filename=source_path.name,
                    category="ood_yunet_failure",
                    output_filename=output_name,
                    output_path=output_path,
                    metrics=metrics,
                )
            )

        expected_core = (
            len(CATEGORY_ORDER) * args.per_category
            + 1
            + args.local_core_count
        )
        if len(core_paths) != expected_core:
            raise RuntimeError(
                f"Core benchmark count mismatch: "
                f"expected {expected_core}, got {len(core_paths)}."
            )

        fields = [
            "sample_id",
            "benchmark_group",
            "source_kind",
            "source_filename",
            "category",
            "output_filename",
            "sha256",
            "yunet_detected",
            "width",
            "height",
            "brightness",
            "sharpness",
            "face_score",
            "face_area_ratio",
            "roll_deg",
        ]

        manifest_path = output_dir / "benchmark_manifest.csv"
        with manifest_path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fields,
            )
            writer.writeheader()
            writer.writerows(rows)

        make_contact_sheet(
            core_paths,
            core_labels,
            output_dir / "benchmark_core_contact_sheet.png",
        )

        if stress_paths:
            make_contact_sheet(
                stress_paths,
                stress_labels,
                output_dir / "benchmark_stress_contact_sheet.png",
                columns=min(5, len(stress_paths)),
            )

        diagnostics = {
            "category_rules": CATEGORY_RULES,
            "eligible_counts": eligible_counts,
            "reserve_public": {
                "filename": reserve_public["filename"],
                "face_score": reserve_public["face_score"],
                "face_area_ratio": reserve_public["face_area_ratio"],
                "brightness": reserve_public["brightness"],
                "sharpness": reserve_public["sharpness"],
                "roll_deg": reserve_public["roll_deg"],
            },
            "demo_smoke_test": demo_status,
            "yunet_detection_failures": detection_failures,
            "unused_detected_local_images": [
                path.name
                for path, _ in unused_detected_local
            ],
        }
        (
            output_dir / "selection_diagnostics.json"
        ).write_text(
            json.dumps(
                diagnostics,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        summary = {
            "schema_version": 2,
            "benchmark_name": "RobotPortraitArt Benchmark Faces V1",
            "selection_script_version": "1.3.1",
            "celeba_partition": "test",
            "seed": args.seed,
            "candidate_count_requested": args.candidate_count,
            "candidate_count_extracted": len(candidate_names),
            "yunet_detected_candidates": len(candidates),
            "yunet_detection_failures": len(detection_failures),
            "public_controlled_count": len(selected),
            "public_reserve_count": 1,
            "public_count": len(selected) + 1,
            "local_core_count": len(core_local),
            "project_demo_in_core": False,
            "project_demo_yunet_detected": demo_status["yunet_detected"],
            "core_count": len(core_paths),
            "stress_count": len(stress_paths),
            "core_expected_count": expected_core,
            "public_categories": {
                category: args.per_category
                for category in CATEGORY_ORDER
            },
            "archive_first_volume": str(args.archive),
            "archive_volume_count": len(volumes),
            "partition_file": str(args.partition_file),
            "seven_zip": str(seven_zip),
            "seven_zip_version": seven_zip_version(seven_zip),
            "yunet_model": str(args.yunet_model),
            "demo_image": str(args.demo_image),
        }
        (
            output_dir / "benchmark_summary.json"
        ).write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print("")
        print("Benchmark V1.3.1 candidate prepared successfully.")
        print(f"  CelebA controlled:  {len(selected)}")
        print("  CelebA reserve:     1")
        print(f"  Local-photo core:   {len(core_local)}")
        print(f"  Core total:         {len(core_paths)}")
        print(f"  Stress total:       {len(stress_paths)}")
        print(f"  Output:             {output_dir}")
        print(
            "  Core contact sheet: "
            f"{output_dir / 'benchmark_core_contact_sheet.png'}"
        )
        if stress_paths:
            print(
                "  Stress contact sheet: "
                f"{output_dir / 'benchmark_stress_contact_sheet.png'}"
            )

    finally:
        if args.keep_candidates:
            print(f"Keeping candidate directory: {work_dir}")
        elif work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
