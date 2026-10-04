from pathlib import Path
import argparse, json, random, sys, yaml, cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robot_portrait.lineart import portrait_lineart
from robot_portrait.vectorize import raster_to_svg


def main():
    ap = argparse.ArgumentParser(description="运行 OpenCV 人像线稿基线，并可选输出 SVG。")
    ap.add_argument("--input", required=True, help="输入照片路径")
    ap.add_argument("--output", required=True, help="线稿 PNG 输出路径")
    ap.add_argument("--svg", help="SVG 输出路径")
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"), help="配置文件路径")
    ap.add_argument("--seed", type=int, default=42, help="固定随机种子（基线本身是确定性的）")
    ap.add_argument("--metadata", default="", help="配置记录 JSON；默认写到 PNG 同名 .json")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件：{config_path}")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if "baseline" not in cfg or "vector" not in cfg:
        raise ValueError("配置文件必须包含 baseline 和 vector 两个节")
    img = cv2.imread(args.input)
    if img is None:
        raise FileNotFoundError(f"找不到输入图像：{args.input}")

    out = portrait_lineart(img, **cfg["baseline"])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(args.output, out):
        raise OSError(f"无法写入线稿输出：{args.output}")
    print(f"[完成] 栅格图：{args.output}")

    if args.svg:
        info = raster_to_svg(out, args.svg, **cfg["vector"])
        print(f"[完成] SVG：{args.svg} | {info}")

    metadata = Path(args.metadata) if args.metadata else Path(args.output).with_suffix(".json")
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(json.dumps({
        "seed": args.seed,
        "input": str(Path(args.input)),
        "output": str(Path(args.output)),
        "svg": str(Path(args.svg)) if args.svg else None,
        "config": str(config_path),
        "baseline": cfg["baseline"],
        "vector": cfg["vector"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[完成] 配置记录：{metadata}")


if __name__ == "__main__":
    main()
