"""Deterministically compare vectorization presets on the SAME raster.

Usage (Windows PowerShell, run from project root):
  python scripts/compare_vector_presets.py --input outputs/web_jobs/<job-id>/apdrawinggan.png

No GAN inference is used. Metrics and side-by-side render files go into
outputs/vector_comparison/<stem>/; changing the input image changes the test.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robot_portrait.io import read_image
from robot_portrait.vectorize import raster_to_svg, svg_to_raster

PARAMETERS = (
    "min_area", "epsilon_ratio", "stroke_width", "mode",
    "centerline_threshold", "centerline_spur_length_px",
    "centerline_min_stroke_length_px", "centerline_simplify_epsilon_px",
    "centerline_max_thinning_iterations",
    "centerline_junction_cluster_radius_px",
    "centerline_merge_angle_deg", "centerline_merge_lookahead_px",
    "centerline_ordering_2opt_max_passes",
    "centerline_ordering_2opt_min_improvement_px",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="Existing GAN portrait raster PNG, not the original face photo")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    picture = read_image(args.input)
    out = args.output_dir or ROOT / "outputs" / "vector_comparison" / args.input.stem
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for name, config_name in (
        ("stable", "centerline.yaml"),
        ("clean_experimental", "centerline_clean_experimental.yaml"),
    ):
        config = yaml.safe_load((ROOT / "configs" / config_name).read_text(encoding="utf-8"))
        vector = config["vector"]
        params = {key: vector[key] for key in PARAMETERS if key in vector}
        svg = out / f"{name}.svg"
        metadata = raster_to_svg(picture, svg, **params)
        svg_to_raster(svg, out / f"{name}.png")
        row = {"preset": name, **metadata}
        rows.append(row)
    (out / "metrics.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    cols = ["preset", "stroke_count", "pen_down_length_px", "pen_up_length_px", "two_opt_improvement_ratio"]
    with (out / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved paired SVG/PNG/metrics to {out}")
    print("The experimental preset is NOT validated as higher quality; judge identity preservation visually.")


if __name__ == "__main__":
    main()
