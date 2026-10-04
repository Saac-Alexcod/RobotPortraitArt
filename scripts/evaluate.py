from pathlib import Path
import argparse, sys, json, cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robot_portrait.metrics import ssim_score, edge_f1, optional_lpips


def main():
    ap = argparse.ArgumentParser(description="计算肖像结果的 SSIM、Edge-F1 和可选 LPIPS。")
    ap.add_argument("--pred", required=True, help="预测图像")
    ap.add_argument("--ref", required=True, help="参考图像")
    ap.add_argument("--json_out", default="", help="可选 JSON 输出路径")
    args = ap.parse_args()

    pred_path, ref_path = Path(args.pred), Path(args.ref)
    missing = [str(path) for path in (pred_path, ref_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "找不到评价图像：" + ", ".join(missing) +
            "。艺术质量评价需要真实参考线稿，请将它放入 data/reference_style/ 并保持同名 stem。"
            "如果尚无参考线稿，可先评价 SVG 保真度："
            "python scripts/evaluate.py --pred outputs/face_line_raster.png "
            "--ref outputs/face_line.png --json_out outputs/face_vector_fidelity_metrics.json"
        )
    pred, ref = cv2.imread(str(pred_path)), cv2.imread(str(ref_path))
    if pred is None or ref is None:
        raise ValueError("预测图像或参考图像格式无法读取，请确认文件是有效的 JPG/PNG 图像。")

    result = {
        "SSIM_higher_is_better": ssim_score(pred, ref),
        "EdgeF1_higher_is_better": edge_f1(pred, ref),
        "LPIPS_lower_is_better": optional_lpips(args.pred, args.ref),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
