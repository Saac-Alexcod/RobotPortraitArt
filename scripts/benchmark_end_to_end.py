from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "1.0"

ROOT = Path(__file__).resolve().parents[1]
RUN_PIPELINE = ROOT / "scripts" / "run_pipeline.py"

METHODS = ("baseline", "apdrawinggan")

CSV_FIELDS = [
    "sample_id",
    "benchmark_group",
    "category",
    "source_filename",
    "input_sha256",
    "method",
    "status",
    "resumed",
    "returncode",
    "error_type",
    "error_message",
    "output_dir",
    "manifest_path",
    "generic_preprocess_mode",
    "generic_face_detected",
    "warning_count",
    "apd_preprocess_mode",
    "apd_detection_score",
    "apd_alignment_scale",
    "apd_alignment_rotation_deg",
    "apd_alignment_rms_error",
    "apd_mask_foreground_ratio",
    "stroke_count",
    "raw_stroke_count",
    "closed_stroke_count",
    "pen_down_length_px",
    "pen_up_length_px",
    "connected_component_count",
    "odd_vertex_count",
    "theoretical_min_trail_count",
    "edge_reuse_count",
    "two_opt_improvement_px",
    "two_opt_improvement_ratio",
    "two_opt_move_count",
    "two_opt_pass_count",
    "svg_bytes",
    "preprocess_ms",
    "generate_ms",
    "vectorize_ms",
    "rasterize_ms",
    "total_ms",
    "wall_time_ms",
]

SUMMARY_METRICS = [
    "stroke_count",
    "raw_stroke_count",
    "closed_stroke_count",
    "pen_down_length_px",
    "pen_up_length_px",
    "connected_component_count",
    "odd_vertex_count",
    "theoretical_min_trail_count",
    "edge_reuse_count",
    "two_opt_improvement_px",
    "two_opt_improvement_ratio",
    "svg_bytes",
    "preprocess_ms",
    "generate_ms",
    "vectorize_ms",
    "rasterize_ms",
    "total_ms",
    "wall_time_ms",
    "apd_detection_score",
    "apd_alignment_rms_error",
    "apd_mask_foreground_ratio",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the locked RobotPortraitArt face benchmark end-to-end "
            "through Baseline and/or APDrawingGAN and summarize robot "
            "drawing metrics from pipeline manifests."
        ),
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("outputs/benchmark_faces_v1_3"),
        help="Directory containing benchmark_manifest.csv and benchmark PNGs.",
    )
    parser.add_argument(
        "--benchmark-manifest",
        type=Path,
        default=None,
        help=(
            "Optional explicit benchmark manifest. "
            "Defaults to <benchmark-dir>/benchmark_manifest.csv."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluation_benchmark"),
        help="Benchmark run/output directory.",
    )
    parser.add_argument(
        "--group",
        choices=("core", "stress", "all"),
        default="core",
        help="Benchmark group to run. Formal report benchmark should use core.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
        help="Generation methods to evaluate.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/centerline.yaml"),
        help="Pipeline YAML config. The formal benchmark uses centerline mode.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Pipeline preprocessing size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Pipeline seed.",
    )
    parser.add_argument(
        "--apd-python",
        default=os.environ.get(
            "ROBOT_PORTRAIT_APD_PYTHON",
            "python3",
        ),
        help=(
            "WSL APDrawingGAN Python. Defaults to "
            "ROBOT_PORTRAIT_APD_PYTHON or python3."
        ),
    )
    parser.add_argument(
        "--apd-timeout",
        type=float,
        default=180.0,
        help="APDrawingGAN backend inference timeout in seconds.",
    )
    parser.add_argument(
        "--subprocess-timeout",
        type=float,
        default=360.0,
        help="Hard timeout for one complete run_pipeline.py invocation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N selected samples (useful for smoke tests).",
    )
    parser.add_argument(
        "--sample-id",
        action="append",
        default=[],
        help=(
            "Run only a specific sample ID. Repeat this option for multiple IDs."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun selected sample/method pairs even if a completed manifest exists.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the one-time run_pipeline.py --list-methods availability check.",
    )
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="Do not verify benchmark image SHA-256 hashes before running.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed sample/method run.",
    )
    return parser.parse_args()


def resolve_from_root(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def nested_get(
    payload: dict[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
        if value is None:
            return default
    return value


def parse_json_from_output(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object found in command output.")
    return json.loads(text[start : end + 1])


def git_metadata() -> dict[str, Any]:
    def run_git(*args: str) -> str | None:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    commit = run_git("rev-parse", "HEAD")
    status = run_git("status", "--short")
    branch = run_git("branch", "--show-current")
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status) if status is not None else None,
        "status_short": status,
    }


def benchmark_rows(
    manifest_path: Path,
    benchmark_dir: Path,
    group: str,
    sample_ids: set[str],
    limit: int | None,
    verify_hashes: bool,
) -> list[dict[str, str]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Benchmark manifest not found: {manifest_path}"
        )

    with manifest_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    selected = []
    for row in rows:
        row_group = row.get("benchmark_group", "")
        if group != "all" and row_group != group:
            continue

        sample_id = row.get("sample_id", "")
        if sample_ids and sample_id not in sample_ids:
            continue

        filename = row.get("output_filename", "")
        if not filename:
            raise ValueError(
                f"Benchmark row {sample_id!r} has no output_filename."
            )

        input_path = benchmark_dir / filename
        if not input_path.is_file():
            raise FileNotFoundError(
                f"Benchmark image not found: {input_path}"
            )

        expected_hash = row.get("sha256", "").strip().upper()
        if verify_hashes and expected_hash:
            actual_hash = sha256_file(input_path)
            if actual_hash != expected_hash:
                raise RuntimeError(
                    "Locked benchmark hash mismatch for "
                    f"{sample_id}: expected {expected_hash}, "
                    f"got {actual_hash}."
                )

        row["_input_path"] = str(input_path)
        selected.append(row)

    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be a positive integer.")
        selected = selected[:limit]

    if not selected:
        raise RuntimeError(
            "No benchmark rows matched the requested group/sample filters."
        )

    return selected


def preflight(
    methods: list[str],
    apd_python: str,
    apd_timeout: float,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(RUN_PIPELINE),
        "--list-methods",
        "--apd-python",
        apd_python,
        "--apd-timeout",
        str(apd_timeout),
    ]

    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=max(120.0, apd_timeout + 30.0),
    )

    combined = "\n".join(
        part
        for part in (result.stdout, result.stderr)
        if part
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Generator preflight command failed.\n" + combined[-4000:]
        )

    payload = parse_json_from_output(result.stdout)
    for method in methods:
        status = payload.get(method)
        if not isinstance(status, dict):
            raise RuntimeError(
                f"Preflight did not report requested method {method!r}."
            )
        if not status.get("available"):
            raise RuntimeError(
                f"Method {method!r} is unavailable: "
                f"{status.get('reason')}"
            )

    return payload


def manifest_matches_resume(
    manifest: dict[str, Any],
    method: str,
    config_path: Path,
    size: int,
    seed: int,
) -> bool:
    if manifest.get("status") != "completed":
        return False
    if manifest.get("method") != method:
        return False

    config = manifest.get("config", {})
    if config.get("preprocess_size") != size:
        return False
    if config.get("seed") != seed:
        return False

    source_path = config.get("source_path")
    if source_path:
        try:
            if Path(source_path).resolve() != config_path.resolve():
                return False
        except OSError:
            return False

    vector_mode = nested_get(manifest, "config", "vector", "mode")
    if vector_mode != "centerline":
        return False

    return True


def empty_result_row(
    benchmark_row: dict[str, str],
    method: str,
    run_dir: Path,
) -> dict[str, Any]:
    return {
        "sample_id": benchmark_row.get("sample_id", ""),
        "benchmark_group": benchmark_row.get("benchmark_group", ""),
        "category": benchmark_row.get("category", ""),
        "source_filename": benchmark_row.get("source_filename", ""),
        "input_sha256": benchmark_row.get("sha256", ""),
        "method": method,
        "status": "failed",
        "resumed": False,
        "returncode": "",
        "error_type": "",
        "error_message": "",
        "output_dir": str(run_dir),
        "manifest_path": str(run_dir / "manifest.json"),
        "generic_preprocess_mode": "",
        "generic_face_detected": "",
        "warning_count": "",
        "apd_preprocess_mode": "",
        "apd_detection_score": "",
        "apd_alignment_scale": "",
        "apd_alignment_rotation_deg": "",
        "apd_alignment_rms_error": "",
        "apd_mask_foreground_ratio": "",
        "stroke_count": "",
        "raw_stroke_count": "",
        "closed_stroke_count": "",
        "pen_down_length_px": "",
        "pen_up_length_px": "",
        "connected_component_count": "",
        "odd_vertex_count": "",
        "theoretical_min_trail_count": "",
        "edge_reuse_count": "",
        "two_opt_improvement_px": "",
        "two_opt_improvement_ratio": "",
        "two_opt_move_count": "",
        "two_opt_pass_count": "",
        "svg_bytes": "",
        "preprocess_ms": "",
        "generate_ms": "",
        "vectorize_ms": "",
        "rasterize_ms": "",
        "total_ms": "",
        "wall_time_ms": "",
    }


def result_row_from_manifest(
    benchmark_row: dict[str, str],
    method: str,
    run_dir: Path,
    manifest: dict[str, Any],
    *,
    resumed: bool,
    wall_time_ms: float | None,
    returncode: int | None,
) -> dict[str, Any]:
    row = empty_result_row(
        benchmark_row,
        method,
        run_dir,
    )

    preprocessing = manifest.get("preprocessing", {})
    generation = manifest.get("generation", {})
    vector = manifest.get("vectorization", {})
    timings = manifest.get("timings_ms", {})
    warnings = manifest.get("warnings", [])

    apd = nested_get(
        generation,
        "metadata",
        "apdrawinggan_preprocess",
        default={},
    )
    if not isinstance(apd, dict):
        apd = {}

    face_box = preprocessing.get("face_box")

    row.update(
        {
            "status": manifest.get("status", "completed"),
            "resumed": resumed,
            "returncode": "" if returncode is None else returncode,
            "generic_preprocess_mode": preprocessing.get("mode", ""),
            "generic_face_detected": bool(face_box),
            "warning_count": len(warnings) if isinstance(warnings, list) else "",
            "apd_preprocess_mode": apd.get("mode", ""),
            "apd_detection_score": nested_get(
                apd,
                "detection",
                "score",
                default="",
            ),
            "apd_alignment_scale": nested_get(
                apd,
                "alignment",
                "transform",
                "scale",
                default="",
            ),
            "apd_alignment_rotation_deg": nested_get(
                apd,
                "alignment",
                "transform",
                "rotation_degrees",
                default="",
            ),
            "apd_alignment_rms_error": nested_get(
                apd,
                "alignment",
                "transform",
                "rms_error",
                default="",
            ),
            "apd_mask_foreground_ratio": nested_get(
                apd,
                "mask",
                "foreground_ratio",
                default="",
            ),
            "stroke_count": vector.get("stroke_count", ""),
            "raw_stroke_count": vector.get("raw_stroke_count", ""),
            "closed_stroke_count": vector.get("closed_stroke_count", ""),
            "pen_down_length_px": vector.get("pen_down_length_px", ""),
            "pen_up_length_px": vector.get("pen_up_length_px", ""),
            "connected_component_count": vector.get(
                "connected_component_count",
                "",
            ),
            "odd_vertex_count": vector.get("odd_vertex_count", ""),
            "theoretical_min_trail_count": vector.get(
                "theoretical_min_trail_count",
                "",
            ),
            "edge_reuse_count": vector.get("edge_reuse_count", ""),
            "two_opt_improvement_px": vector.get(
                "two_opt_improvement_px",
                "",
            ),
            "two_opt_improvement_ratio": vector.get(
                "two_opt_improvement_ratio",
                "",
            ),
            "two_opt_move_count": vector.get("two_opt_move_count", ""),
            "two_opt_pass_count": vector.get("two_opt_pass_count", ""),
            "svg_bytes": vector.get("bytes", ""),
            "preprocess_ms": timings.get("preprocess", ""),
            "generate_ms": timings.get("generate", ""),
            "vectorize_ms": timings.get("vectorize", ""),
            "rasterize_ms": timings.get("rasterize", ""),
            "total_ms": timings.get("total", ""),
            "wall_time_ms": (
                ""
                if wall_time_ms is None
                else round(float(wall_time_ms), 3)
            ),
        }
    )
    return row


def console_log(
    path: Path,
    command: list[str],
    stdout: str,
    stderr: str,
    returncode: int | str,
) -> None:
    command_text = subprocess.list2cmdline(command)
    path.write_text(
        "\n".join(
            [
                f"command: {command_text}",
                f"returncode: {returncode}",
                "",
                "===== STDOUT =====",
                stdout,
                "",
                "===== STDERR =====",
                stderr,
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_one(
    benchmark_row: dict[str, str],
    method: str,
    args: argparse.Namespace,
    config_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    sample_id = benchmark_row["sample_id"]
    input_path = Path(benchmark_row["_input_path"])
    run_dir = output_root / "runs" / sample_id / method
    manifest_path = run_dir / "manifest.json"

    if not args.force and manifest_path.is_file():
        try:
            manifest = load_json(manifest_path)
        except (OSError, json.JSONDecodeError):
            manifest = {}

        if manifest_matches_resume(
            manifest,
            method,
            config_path,
            args.size,
            args.seed,
        ):
            print(
                f"[resume] {sample_id:12s} {method:12s} "
                "completed manifest found"
            )
            return result_row_from_manifest(
                benchmark_row,
                method,
                run_dir,
                manifest,
                resumed=True,
                wall_time_ms=None,
                returncode=None,
            )

    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(RUN_PIPELINE),
        "--input",
        str(input_path),
        "--output-dir",
        str(run_dir),
        "--config",
        str(config_path),
        "--method",
        method,
        "--size",
        str(args.size),
        "--seed",
        str(args.seed),
    ]

    if method == "apdrawinggan":
        command.extend(
            [
                "--apd-python",
                args.apd_python,
                "--apd-timeout",
                str(args.apd_timeout),
            ]
        )

    print(
        f"[run]    {sample_id:12s} {method:12s} "
        f"-> {run_dir.relative_to(ROOT) if run_dir.is_relative_to(ROOT) else run_dir}"
    )

    started = time.perf_counter()

    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=args.subprocess_timeout,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        console_log(
            run_dir / "console.log",
            command,
            result.stdout,
            result.stderr,
            result.returncode,
        )

        if result.returncode != 0:
            row = empty_result_row(
                benchmark_row,
                method,
                run_dir,
            )
            message = (result.stderr or result.stdout).strip()
            row.update(
                {
                    "status": "failed",
                    "returncode": result.returncode,
                    "error_type": "ProcessError",
                    "error_message": message[-2000:],
                    "wall_time_ms": round(elapsed_ms, 3),
                }
            )
            return row

        if not manifest_path.is_file():
            row = empty_result_row(
                benchmark_row,
                method,
                run_dir,
            )
            row.update(
                {
                    "status": "failed",
                    "returncode": result.returncode,
                    "error_type": "MissingManifest",
                    "error_message": (
                        "run_pipeline.py returned success but manifest.json "
                        "was not produced."
                    ),
                    "wall_time_ms": round(elapsed_ms, 3),
                }
            )
            return row

        manifest = load_json(manifest_path)
        return result_row_from_manifest(
            benchmark_row,
            method,
            run_dir,
            manifest,
            resumed=False,
            wall_time_ms=elapsed_ms,
            returncode=result.returncode,
        )

    except subprocess.TimeoutExpired as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        stdout = (
            exc.stdout.decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr = (
            exc.stderr.decode("utf-8", errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        console_log(
            run_dir / "console.log",
            command,
            stdout,
            stderr,
            "TIMEOUT",
        )

        row = empty_result_row(
            benchmark_row,
            method,
            run_dir,
        )
        row.update(
            {
                "status": "timeout",
                "error_type": "TimeoutExpired",
                "error_message": (
                    f"Run exceeded {args.subprocess_timeout:.1f}s hard timeout."
                ),
                "wall_time_ms": round(elapsed_ms, 3),
            }
        )
        return row


def numeric_values(
    rows: list[dict[str, Any]],
    field: str,
) -> list[float]:
    values: list[float] = []
    for row in rows:
        raw = row.get(field, "")
        if raw in ("", None):
            continue
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            continue
    return values


def metric_stats(
    rows: list[dict[str, Any]],
    field: str,
) -> dict[str, Any] | None:
    values = numeric_values(rows, field)
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


def summarize_subset(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    completed = [
        row
        for row in rows
        if row.get("status") == "completed"
    ]
    failed = [
        row
        for row in rows
        if row.get("status") != "completed"
    ]
    attempted = len(rows)

    return {
        "attempted": attempted,
        "completed": len(completed),
        "failed": len(failed),
        "success_rate": (
            len(completed) / attempted
            if attempted
            else None
        ),
        "metrics": {
            field: stats
            for field in SUMMARY_METRICS
            if (
                stats := metric_stats(
                    completed,
                    field,
                )
            ) is not None
        },
    }


def build_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    per_method: dict[str, Any] = {}
    by_category: dict[str, Any] = {}

    for method in METHODS:
        method_rows = [
            row
            for row in rows
            if row.get("method") == method
        ]
        if not method_rows:
            continue

        per_method[method] = summarize_subset(method_rows)

        categories = sorted(
            {
                str(row.get("category", ""))
                for row in method_rows
            }
        )
        by_category[method] = {
            category: summarize_subset(
                [
                    row
                    for row in method_rows
                    if row.get("category") == category
                ]
            )
            for category in categories
        }

    return {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "rows": len(rows),
        "per_method": per_method,
        "by_category": by_category,
    }


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CSV_FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_progress(
    output_root: Path,
    rows: list[dict[str, Any]],
) -> None:
    write_csv(
        output_root / "benchmark_rows.csv",
        rows,
    )

    summary = build_summary(rows)
    (
        output_root / "benchmark_summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    failures = [
        {
            "sample_id": row.get("sample_id"),
            "benchmark_group": row.get("benchmark_group"),
            "category": row.get("category"),
            "method": row.get("method"),
            "status": row.get("status"),
            "returncode": row.get("returncode"),
            "error_type": row.get("error_type"),
            "error_message": row.get("error_message"),
            "output_dir": row.get("output_dir"),
        }
        for row in rows
        if row.get("status") != "completed"
    ]
    (
        output_root / "failures.json"
    ).write_text(
        json.dumps(
            failures,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()

    if not RUN_PIPELINE.is_file():
        raise FileNotFoundError(
            f"run_pipeline.py not found: {RUN_PIPELINE}"
        )

    benchmark_dir = resolve_from_root(
        args.benchmark_dir
    ).resolve()
    manifest_path = (
        resolve_from_root(args.benchmark_manifest).resolve()
        if args.benchmark_manifest is not None
        else benchmark_dir / "benchmark_manifest.csv"
    )
    config_path = resolve_from_root(args.config).resolve()
    output_root = resolve_from_root(args.output_dir).resolve()

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Pipeline config not found: {config_path}"
        )

    output_root.mkdir(parents=True, exist_ok=True)

    selected = benchmark_rows(
        manifest_path,
        benchmark_dir,
        args.group,
        set(args.sample_id),
        args.limit,
        verify_hashes=not args.skip_hash_check,
    )

    print("RobotPortraitArt End-to-End Evaluation Benchmark")
    print(f"Script version:   {SCRIPT_VERSION}")
    print(f"Benchmark group:  {args.group}")
    print(f"Samples:          {len(selected)}")
    print(f"Methods:          {', '.join(args.methods)}")
    print(f"Config:           {config_path}")
    print(f"Size / seed:      {args.size} / {args.seed}")
    print(f"Output:           {output_root}")

    preflight_payload: dict[str, Any] | None = None
    if not args.skip_preflight:
        print("")
        print("Running generator availability preflight...")
        preflight_payload = preflight(
            args.methods,
            args.apd_python,
            args.apd_timeout,
        )
        print("Preflight: PASS")

    metadata = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "started_at_utc": utc_now(),
        "repository_root": str(ROOT),
        "benchmark_dir": str(benchmark_dir),
        "benchmark_manifest": str(manifest_path),
        "benchmark_manifest_sha256": sha256_file(manifest_path),
        "benchmark_group": args.group,
        "selected_sample_ids": [
            row["sample_id"]
            for row in selected
        ],
        "methods": args.methods,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "size": args.size,
        "seed": args.seed,
        "apd_python": args.apd_python,
        "apd_timeout_seconds": args.apd_timeout,
        "subprocess_timeout_seconds": args.subprocess_timeout,
        "hash_check_enabled": not args.skip_hash_check,
        "force": args.force,
        "git": git_metadata(),
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "preflight": preflight_payload,
    }
    metadata_path = output_root / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    result_rows: list[dict[str, Any]] = []
    total_runs = len(selected) * len(args.methods)
    run_index = 0

    try:
        for benchmark_row in selected:
            for method in args.methods:
                run_index += 1
                print("")
                print(
                    f"=== Run {run_index}/{total_runs}: "
                    f"{benchmark_row['sample_id']} / {method} ==="
                )

                row = run_one(
                    benchmark_row,
                    method,
                    args,
                    config_path,
                    output_root,
                )
                result_rows.append(row)
                write_progress(
                    output_root,
                    result_rows,
                )

                if row["status"] == "completed":
                    print(
                        "[ok]     "
                        f"stroke={row['stroke_count']} "
                        f"pen_up={row['pen_up_length_px']} "
                        f"total_ms={row['total_ms']}"
                    )
                else:
                    print(
                        "[failed] "
                        f"{row['error_type']}: "
                        f"{row['error_message']}"
                    )
                    if args.fail_fast:
                        raise RuntimeError(
                            "Stopping because --fail-fast was requested."
                        )

    except KeyboardInterrupt:
        print("")
        print("Interrupted by user. Partial CSV/JSON outputs were preserved.")
        write_progress(
            output_root,
            result_rows,
        )
        return 130

    metadata["completed_at_utc"] = utc_now()
    metadata["result_rows"] = len(result_rows)
    metadata["completed_runs"] = sum(
        1
        for row in result_rows
        if row.get("status") == "completed"
    )
    metadata["failed_runs"] = sum(
        1
        for row in result_rows
        if row.get("status") != "completed"
    )
    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_progress(
        output_root,
        result_rows,
    )

    print("")
    print("Benchmark finished.")
    print(
        f"  Completed: {metadata['completed_runs']}/{len(result_rows)}"
    )
    print(
        f"  Failed:    {metadata['failed_runs']}/{len(result_rows)}"
    )
    print(
        f"  Rows CSV:  {output_root / 'benchmark_rows.csv'}"
    )
    print(
        f"  Summary:   {output_root / 'benchmark_summary.json'}"
    )
    print(
        f"  Failures:  {output_root / 'failures.json'}"
    )

    return 0 if metadata["failed_runs"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
