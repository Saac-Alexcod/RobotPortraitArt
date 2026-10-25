"""Run the course comparison on real images; export measurements and human-review sheets."""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/examples")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/course_evaluation")
    parser.add_argument("--methods", nargs="+", choices=("baseline", "apdrawinggan", "cartoon"),
                        default=["baseline", "apdrawinggan", "cartoon"])
    parser.add_argument("--distro", default=os.environ.get("ROBOT_PORTRAIT_APD_DISTRO", ""))
    parser.add_argument("--python", default=os.environ.get("ROBOT_PORTRAIT_APD_PYTHON", ""))
    args = parser.parse_args()
    from start_web import resolve_distro, inspect_apd
    if any(method != "baseline" for method in args.methods):
        distro = resolve_distro(args.distro)
        report = inspect_apd(distro, args.python, False)
        if not report["ready"]:
            raise RuntimeError(report)
        os.environ["ROBOT_PORTRAIT_APD_DISTRO"] = distro
        os.environ["ROBOT_PORTRAIT_APD_PYTHON"] = report["python"]
    from robot_portrait.web.runtime import build_pipeline
    inputs = sorted(path for path in args.input_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not inputs:
        parser.error("No input images found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, cards, reviews = [], [], []
    for source in inputs:
        for method in args.methods:
            name = f"{source.stem}_{method}"
            row = {"sample": source.name, "method": method}
            print(f"Running {name}", flush=True)
            try:
                pipeline = build_pipeline(vector_mode="centerline", size=256 if method == "cartoon" else 512)
                result = pipeline.run(source, args.output_dir / name, method=method)
                for key in ("source_to_portrait", "source_to_vector", "vector_retention"):
                    for metric, value in result.evaluation[key].items():
                        row[f"{key}_{metric}"] = value
                row.update(status="completed", error="", seconds=result.timings_ms["total"] / 1000,
                           strokes=result.vectorization.get("stroke_count"))
                labels = [("原图裁剪", result.prepared_path.name), ("风格图", result.raster_path.name),
                          ("矢量绘制", result.vector_render_path.name)]
                # APD's metrics compare its aligned original, not the shared Haar crop.
                aligned = result.generation["metadata"].get("apdrawinggan_preprocess", {}).get("files", {}).get("aligned_image")
                if aligned:
                    import shutil
                    shutil.copy2(aligned, result.output_dir / "evaluation_reference.png")
                    labels[0] = ("对齐原图", "evaluation_reference.png")
                images = ''.join(f'<figure><img src="{html.escape(name)}/{html.escape(file)}"><figcaption>{label}</figcaption></figure>' for label, file in labels)
                cards.append(f'<section><h2>{html.escape(name)}</h2><div class="images">{images}</div><pre>{html.escape(json.dumps(result.evaluation, ensure_ascii=False, indent=2))}</pre><a href="{html.escape(name)}/{method}.svg">SVG</a></section>')
                reviews.append({"sample": source.name, "method": method, "reviewer": "",
                                "identity_1_to_5": "", "style_quality_1_to_5": "",
                                "vector_likeness_1_to_5": "", "artifact_free_1_to_5": "", "comment": ""})
            except Exception as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                print(row["error"], flush=True)
                cards.append(f'<section><h2>{html.escape(name)}</h2><p>FAILED: {html.escape(row["error"])}</p></section>')
            rows.append(row)
            (args.output_dir / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    for filename, records, keys in [("scores.csv", rows, fields), ("human_review.csv", reviews, list(reviews[0]) if reviews else [])]:
        with (args.output_dir / filename).open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=keys)
            writer.writeheader()
            writer.writerows(records)
    document = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>肖像风格化与矢量化评估</title><style>body{font-family:system-ui;max-width:1100px;margin:40px auto;background:#f4f6fa}section{background:white;padding:24px;margin:20px 0}.images{display:flex}figure{margin:8px;flex:1}img{width:100%}pre{white-space:pre-wrap}</style><h1>肖像风格化与矢量化 · 实测对照</h1><p>结构指标不是艺术质量评分。统一评估分辨率 256、轮廓容差 2 px；空参考轮廓记为缺失，失败样例保留。生成原生尺寸：卡通 256，其余 512。未训练新模型，使用公开预训练权重。样本仅用于演示，不代表总体性能。</p>' + ''.join(cards) + '</html>'
    (args.output_dir / "report.html").write_text(document, encoding="utf-8")
    print(f"Report: {args.output_dir / 'report.html'}", flush=True)
    return int(any(row["status"] == "failed" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
