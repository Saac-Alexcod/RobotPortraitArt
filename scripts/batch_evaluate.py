"""对同一测试集的多个方法批量计算指标并输出 CSV、summary.json。"""

from pathlib import Path
import argparse
import csv
import json
import sys
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robot_portrait.metrics import ssim_score, edge_f1, optional_lpips

FIELDS = ["sample_id", "method", "ssim", "edge_f1", "lpips", "svg_paths", "svg_bytes", "eyes_score", "eyebrows_score", "nose_score", "mouth_score", "contour_score", "hair_score", "notes"]


def _images(directory: Path) -> dict[str, Path]:
    return {p.stem: p for p in directory.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}}


def evaluate(reference_dir: Path, methods: list[str], output_csv: Path, summary_json: Path, svg_dir: Path | None = None) -> dict:
    if not reference_dir.exists():
        raise FileNotFoundError(f"找不到参考图目录：{reference_dir}")
    references = _images(reference_dir)
    rows = []
    for spec in methods:
        if "=" not in spec:
            raise ValueError(f"方法参数应为 方法名=目录：{spec}")
        method, directory = spec.split("=", 1)
        prediction_dir = Path(directory)
        if not prediction_dir.exists():
            print(f"警告：方法目录不存在，将为所有样本写入缺失行：{prediction_dir}", file=sys.stderr)
        predictions = _images(prediction_dir) if prediction_dir.exists() else {}
        for sample_id, ref_path in references.items():
            pred_path = predictions.get(sample_id)
            if not pred_path:
                rows.append({"sample_id": sample_id, "method": method, "ssim": "", "edge_f1": "", "lpips": "", "svg_paths": "", "svg_bytes": "", "eyes_score": "", "eyebrows_score": "", "nose_score": "", "mouth_score": "", "contour_score": "", "hair_score": "", "notes": "缺少预测图，尚未运行"})
                continue
            pred, ref = cv2.imread(str(pred_path)), cv2.imread(str(ref_path))
            if pred is None or ref is None:
                raise ValueError(f"无法读取评价图像：pred={pred_path}, ref={ref_path}")
            lpips_value = optional_lpips(str(pred_path), str(ref_path))
            row = {"sample_id": sample_id, "method": method, "ssim": ssim_score(pred, ref), "edge_f1": edge_f1(pred, ref), "lpips": lpips_value if lpips_value is not None else "", "svg_paths": "", "svg_bytes": "", "eyes_score": "", "eyebrows_score": "", "nose_score": "", "mouth_score": "", "contour_score": "", "hair_score": "", "notes": "人工局部评分待填写"}
            # --svg-dir may be a shared flat directory or a method-subdirectory root.
            method_svg_dir = (svg_dir / method) if svg_dir and (svg_dir / method).exists() else svg_dir
            candidate = (method_svg_dir / f"{sample_id}.svg") if method_svg_dir else prediction_dir / f"{sample_id}.svg"
            if candidate.exists():
                text = candidate.read_text(encoding="utf-8")
                row["svg_paths"] = text.count("<path ")
                row["svg_bytes"] = candidate.stat().st_size
            rows.append(row)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {"seed": 42, "reference_count": len(references), "methods": [m.split("=", 1)[0] for m in methods], "rows": len(rows), "completed_rows": sum(bool(row["ssim"] != "") for row in rows), "note": "数值来自实际运行；空白项表示资源尚未准备或人工评分尚未填写。"}
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="批量计算 SSIM、Edge-F1、LPIPS 和 SVG 复杂度。")
    ap.add_argument("--reference-dir", required=True, help="参考图目录")
    ap.add_argument("--method", action="append", required=True, help="重复传入：方法名=预测图目录")
    ap.add_argument("--output-csv", default="outputs/metrics.csv")
    ap.add_argument("--summary-json", default="outputs/summary.json")
    ap.add_argument("--svg-dir", default="", help="SVG 目录；支持扁平目录或按方法名分子目录")
    ap.add_argument("--seed", type=int, default=42, help="记录用随机种子；评价过程本身不含随机采样")
    args = ap.parse_args()
    summary = evaluate(Path(args.reference_dir), args.method, Path(args.output_csv), Path(args.summary_json), Path(args.svg_dir) if args.svg_dir else None)
    summary["seed"] = args.seed
    Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
