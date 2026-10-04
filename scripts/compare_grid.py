"""生成原图、模型结果和 SVG 回栅格结果的对比联系图。"""

from pathlib import Path
import argparse
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robot_portrait.vectorize import svg_to_raster


def _read(path: str | None, label: str) -> np.ndarray | None:
    if not path:
        return None
    image = cv2.imread(path)
    if image is None:
        print(f"警告：跳过不存在或无法读取的{label}：{path}", file=sys.stderr)
    return image


def make_grid(items: list[tuple[str, np.ndarray]], output: str | Path, size: int = 320) -> None:
    """将已有图像统一缩放并横向拼接。"""
    if not items:
        raise ValueError("没有可用于生成对比图的图像")
    tiles = []
    for label, image in items:
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        canvas = np.full((size + 42, size, 3), 255, np.uint8)
        canvas[42:] = image
        cv2.putText(canvas, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 1, cv2.LINE_AA)
        tiles.append(canvas)
    grid = np.concatenate(tiles, axis=1)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), grid)


def main() -> None:
    ap = argparse.ArgumentParser(description="生成肖像方法横向对比图，缺失结果会自动跳过。")
    ap.add_argument("--original", required=True, help="原图")
    ap.add_argument("--baseline", help="OpenCV 基线结果")
    ap.add_argument("--gan", help="APDrawingGAN 或其他模型结果")
    ap.add_argument("--svg", help="SVG 文件；会先回栅格化")
    ap.add_argument("--output", required=True, help="对比图输出路径")
    ap.add_argument("--require-svg", action="store_true", help="指定 SVG 后若回栅格失败则以错误退出")
    args = ap.parse_args()
    items = []
    for label, path in (("Original", args.original), ("Baseline", args.baseline), ("APDrawingGAN", args.gan)):
        image = _read(path, label)
        if image is not None:
            items.append((label, image))
    if args.svg and Path(args.svg).exists():
        png_path = Path(args.output).with_name(Path(args.output).stem + "_svg_render.png")
        try:
            raster_info = svg_to_raster(args.svg, png_path)
            if raster_info.get("backend") == "opencv-fallback":
                print("提示：系统原生 Cairo 不可用，已使用 OpenCV 回栅格本项目生成的 SVG。", file=sys.stderr)
            image = _read(str(png_path), "SVG 回栅格结果")
            if image is not None:
                items.append(("Vector Render", image))
        except RuntimeError as exc:
            if args.require_svg:
                raise
            print(f"警告：{exc}", file=sys.stderr)
    make_grid(items, args.output)
    print(f"[完成] 对比图：{args.output} | 列：{', '.join(label for label, _ in items)}")


if __name__ == "__main__":
    main()
