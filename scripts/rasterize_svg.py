"""将 SVG 线稿回栅格化为 PNG，供 SSIM/Edge-F1 等指标使用。"""

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robot_portrait.vectorize import svg_to_raster


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 cairosvg 将 SVG 回栅格化为 PNG。")
    parser.add_argument("--input", required=True, help="输入 SVG")
    parser.add_argument("--output", required=True, help="输出 PNG")
    parser.add_argument("--width", type=int, default=0, help="可选输出宽度")
    parser.add_argument("--verbose", action="store_true", help="显示回栅格后端的完整失败原因")
    args = parser.parse_args()
    if args.width < 0:
        raise ValueError("--width 必须为正整数或省略")
    try:
        info = svg_to_raster(args.input, args.output, args.width or None)
    except RuntimeError as exc:
        parser.error(str(exc))
    if info.get("backend") == "opencv-fallback":
        print("提示：系统原生 Cairo 不可用，已使用 OpenCV 回栅格本项目生成的 SVG。", file=sys.stderr)
    if not args.verbose:
        info.pop("fallback_reason", None)
    print(info)


if __name__ == "__main__":
    main()
