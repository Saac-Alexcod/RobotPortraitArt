"""将数据目录中的图片生成联系图，便于检查数据集是否正确放置。"""

from pathlib import Path
import argparse
from PIL import Image, ImageDraw, ImageFont, ImageOps

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_contact_sheet(input_dir: str | Path, output: str | Path, max_images: int = 100, cols: int = 5, thumb: int = 180) -> dict:
    """扫描目录并输出带文件名标签的图片联系图。"""
    input_dir = Path(input_dir)
    if not input_dir.exists():
        candidates = []
        search_root = input_dir.parent
        if search_root.exists():
            candidates = [str(path) for path in search_root.rglob("img_align_celeba") if path.is_dir()][:5]
        hint = ""
        if candidates:
            hint = "\n检测到可能的实际目录：" + "；".join(candidates)
        raise FileNotFoundError(
            f"数据目录不存在：{input_dir}\n"
            "请先下载并解压数据集；当前项目通常使用 data/CelebA/Img/img_align_celeba/."
            + hint
        )
    if not input_dir.is_dir():
        raise NotADirectoryError(f"数据路径不是目录：{input_dir}")
    paths = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in EXTENSIONS)[:max_images]
    if not paths:
        raise FileNotFoundError(f"目录中没有可显示的图片：{input_dir}")
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + 28)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(paths):
        with Image.open(path) as source:
            image = ImageOps.fit(source.convert("RGB"), (thumb, thumb))
        x, y = (index % cols) * thumb, (index // cols) * (thumb + 28)
        sheet.paste(image, (x, y))
        draw.text((x + 4, y + thumb + 5), path.name[:26], fill="black")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return {"images": len(paths), "output": str(output), "rows": rows, "cols": cols}


def main() -> None:
    ap = argparse.ArgumentParser(description="生成 CelebA、输入照片或参考风格目录的图片联系图。")
    ap.add_argument("--input-dir", required=True, help="待检查的数据目录")
    ap.add_argument("--output", required=True, help="联系图输出 PNG")
    ap.add_argument("--max-images", type=int, default=100)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--thumb", type=int, default=180)
    args = ap.parse_args()
    try:
        result = build_contact_sheet(args.input_dir, args.output, args.max_images, args.cols, args.thumb)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"警告：{exc}")
        print("当前未生成联系图；请准备数据后重新运行。")
        return
    print(result)


if __name__ == "__main__":
    main()
