"""人脸图像预处理：EXIF 旋转、检测、居中方形裁剪和统一尺寸。"""

from pathlib import Path
import argparse
import json
import sys

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robot_portrait.io import write_image


def _load_haar_detector(
    cascade_path: str | Path | None = None,
) -> cv2.CascadeClassifier | None:
    """
    从内存加载 Haar Cascade。

    Windows 下 OpenCV 的 FileStorage 对包含非 ASCII 字符的路径
    可能处理失败，因此由 Python 负责读取 XML，再通过 OpenCV
    FileStorage 的 MEMORY 模式加载。
    """
    if not hasattr(cv2, "CascadeClassifier") or not hasattr(cv2, "data"):
        return None

    path = (
        Path(cascade_path)
        if cascade_path
        else Path(cv2.data.haarcascades)
        / "haarcascade_frontalface_default.xml"
    )

    try:
        cascade_text = path.read_text(encoding="utf-8")

        storage = cv2.FileStorage(
            cascade_text,
            cv2.FileStorage_READ | cv2.FileStorage_MEMORY,
        )

    except (OSError, UnicodeError, cv2.error):
        return None

    try:
        if not storage.isOpened():
            return None

        node = storage.getFirstTopLevelNode()

        if node.empty():
            return None

        detector = cv2.CascadeClassifier()

        if not detector.read(node) or detector.empty():
            return None

        return detector

    except cv2.error:
        return None

    finally:
        storage.release()


def prepare_image(
    input_path: str | Path,
    output_path: str | Path,
    size: int = 512,
) -> dict:
    """预处理一张照片；检测不到人脸时使用中心裁剪并输出中文警告。"""

    if int(size) <= 0:
        raise ValueError("size 必须是正整数")

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        candidates = []

        if input_path.parent.exists():
            candidates = [
                path.name
                for path in sorted(input_path.parent.iterdir())
                if path.is_file()
                and path.suffix.lower()
                in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".webp",
                }
            ][:10]

        hint = ""

        if candidates:
            hint = "\n该目录中可用的图片有：" + ", ".join(candidates)

        raise FileNotFoundError(
            f"找不到输入照片：{input_path}\n"
            "请把自己的照片放入 data/input/，"
            "或将 --input 改为已有图片路径。"
            + hint
        )

    try:
        with Image.open(input_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")

    except Exception as exc:
        raise ValueError(
            f"无法读取输入照片：{input_path}，"
            "请确认它是有效的 JPG/PNG 图像。"
        ) from exc

    rgb = np.asarray(image)

    bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR,
    )

    gray = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2GRAY,
    )

    faces = ()

    detector = _load_haar_detector()

    if detector is not None:
        try:
            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40),
            )

        except cv2.error:
            print(
                "警告：Haar 人脸检测失败，"
                "改用中心方形裁剪。",
                file=sys.stderr,
            )

    else:
        print(
            "警告：无法加载 Haar 人脸检测器，"
            "改用中心方形裁剪。",
            file=sys.stderr,
        )

    if len(faces):
        x, y, w, h = max(
            faces,
            key=lambda item: item[2] * item[3],
        )

        margin = int(
            max(w, h) * 0.35
        )

        cx = x + w // 2
        cy = y + h // 2

        side = min(
            bgr.shape[:2]
        )

        side = min(
            side,
            max(w, h) + 2 * margin,
        )

        left = cx - side // 2
        top = cy - side // 2

        right = left + side
        bottom = top + side

        mode = "face"

    else:
        print(
            "警告：未检测到人脸，"
            "改用中心方形裁剪。",
            file=sys.stderr,
        )

        side = min(
            bgr.shape[:2]
        )

        left = (
            bgr.shape[1] - side
        ) // 2

        top = (
            bgr.shape[0] - side
        ) // 2

        right = left + side
        bottom = top + side

        mode = "center"

    # 将方框平移回图像内部，保证裁剪中心仍然对准人脸。
    side = min(
        right - left,
        bottom - top,
    )

    left = min(
        max(0, left),
        bgr.shape[1] - side,
    )

    top = min(
        max(0, top),
        bgr.shape[0] - side,
    )

    crop = bgr[
        top : top + side,
        left : left + side,
    ]

    side = min(
        crop.shape[:2]
    )

    crop = crop[
        :side,
        :side,
    ]

    crop = cv2.resize(
        crop,
        (size, size),
        interpolation=cv2.INTER_AREA,
    )

    write_image(
        output_path,
        crop,
    )

    return {
        "input": str(input_path),
        "output": str(output_path),
        "size": size,
        "mode": mode,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="人脸 EXIF 旋转、检测、居中裁剪和尺寸统一工具。"
    )

    ap.add_argument(
        "--input",
        required=True,
        help="输入照片路径",
    )

    ap.add_argument(
        "--output",
        required=True,
        help="预处理照片路径",
    )

    ap.add_argument(
        "--size",
        type=int,
        default=512,
        help="输出边长，默认 512",
    )

    ap.add_argument(
        "--metadata",
        default="",
        help="可选 JSON 记录；默认写到输出同名 .json",
    )

    args = ap.parse_args()

    result = prepare_image(
        args.input,
        args.output,
        args.size,
    )

    metadata = (
        Path(args.metadata)
        if args.metadata
        else Path(args.output).with_suffix(".json")
    )

    metadata.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(result)
    print(f"配置记录：{metadata}")


if __name__ == "__main__":
    main()