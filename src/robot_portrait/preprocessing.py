from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image, ImageOps


CropMode = Literal["face", "center"]
Box = tuple[int, int, int, int]


@dataclass(slots=True)
class PreprocessResult:
    """人像预处理阶段的结构化结果。

    坐标使用 NumPy 切片约定：
    right 和 bottom 为不包含端点的坐标。

    尺寸统一表示为：
    (width, height)
    """

    image: np.ndarray = field(repr=False)

    mode: CropMode

    original_size: tuple[int, int]
    output_size: tuple[int, int]

    crop_box: Box
    face_box: Box | None

    detector: str

    warnings: tuple[str, ...] = ()

    def to_metadata(
        self,
        *,
        input_path: str | Path | None = None,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """生成可以直接序列化为 JSON 的预处理信息。"""

        def box_to_dict(
            box: Box | None,
        ) -> dict[str, int] | None:
            if box is None:
                return None

            left, top, right, bottom = box

            return {
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
            }

        width, height = self.output_size

        original_width, original_height = (
            self.original_size
        )

        return {
            "input": (
                str(Path(input_path))
                if input_path is not None
                else None
            ),
            "output": (
                str(Path(output_path))
                if output_path is not None
                else None
            ),

            # 保留旧版 metadata 中的 size 字段，
            # 避免破坏已有脚本和实验记录。
            "size": (
                width
                if width == height
                else [width, height]
            ),

            "mode": self.mode,

            "detector": self.detector,

            "original_size": {
                "width": original_width,
                "height": original_height,
            },

            "output_size": {
                "width": width,
                "height": height,
            },

            "crop_box": box_to_dict(
                self.crop_box
            ),

            "face_box": box_to_dict(
                self.face_box
            ),

            "warnings": list(
                self.warnings
            ),
        }


def load_haar_detector(
    cascade_path: str | Path | None = None,
) -> cv2.CascadeClassifier | None:
    """加载 Haar 正脸检测器。

    文件由 Python 读取，再通过 OpenCV MEMORY 模式解析，
    避免 Windows 非 ASCII 路径导致 FileStorage 失败。
    """

    if (
        not hasattr(cv2, "CascadeClassifier")
        or not hasattr(cv2, "data")
    ):
        return None

    path = (
        Path(cascade_path)
        if cascade_path is not None
        else (
            Path(cv2.data.haarcascades)
            / "haarcascade_frontalface_default.xml"
        )
    )

    try:
        cascade_text = path.read_text(
            encoding="utf-8"
        )

        storage = cv2.FileStorage(
            cascade_text,
            cv2.FileStorage_READ
            | cv2.FileStorage_MEMORY,
        )

    except (
        OSError,
        UnicodeError,
        cv2.error,
    ):
        return None

    try:
        if not storage.isOpened():
            return None

        node = (
            storage
            .getFirstTopLevelNode()
        )

        if node.empty():
            return None

        detector = (
            cv2.CascadeClassifier()
        )

        if (
            not detector.read(node)
            or detector.empty()
        ):
            return None

        return detector

    except cv2.error:
        return None

    finally:
        storage.release()


def load_portrait_image(
    input_path: str | Path,
) -> np.ndarray:
    """读取照片、处理 EXIF 方向，并转换为 BGR。"""

    path = Path(input_path)

    if not path.exists():
        candidates: list[str] = []

        if path.parent.exists():
            candidates = [
                candidate.name
                for candidate
                in sorted(
                    path.parent.iterdir()
                )
                if (
                    candidate.is_file()
                    and candidate.suffix.lower()
                    in {
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".bmp",
                        ".webp",
                        ".tif",
                        ".tiff",
                    }
                )
            ][:10]

        hint = ""

        if candidates:
            hint = (
                "\n该目录中可用的图片有："
                + ", ".join(candidates)
            )

        raise FileNotFoundError(
            f"找不到输入照片：{path}\n"
            "请把自己的照片放入 data/input/，"
            "或将 --input 改为已有图片路径。"
            + hint
        )

    if not path.is_file():
        raise ValueError(
            f"输入照片路径不是文件：{path}"
        )

    try:
        with Image.open(path) as source:
            image = (
                ImageOps
                .exif_transpose(source)
                .convert("RGB")
            )

    except Exception as exc:
        raise ValueError(
            f"无法读取输入照片：{path}，"
            "请确认它是有效的 JPG/PNG 图像。"
        ) from exc

    rgb = np.asarray(image)

    return cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR,
    )


def _largest_face(
    gray: np.ndarray,
    detector: Any,
) -> tuple[Box | None, str | None]:
    """找到检测结果中面积最大的人脸。"""

    if detector is None:
        return (
            None,
            "无法加载 Haar 人脸检测器，"
            "改用中心方形裁剪。",
        )

    try:
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )

    except cv2.error:
        return (
            None,
            "Haar 人脸检测失败，"
            "改用中心方形裁剪。",
        )

    valid_faces = []

    for face in faces:
        x, y, width, height = (
            int(value)
            for value in face
        )

        if width > 0 and height > 0:
            valid_faces.append(
                (
                    x,
                    y,
                    width,
                    height,
                )
            )

    if not valid_faces:
        return (
            None,
            "未检测到人脸，"
            "改用中心方形裁剪。",
        )

    x, y, width, height = max(
        valid_faces,
        key=lambda item:
            item[2] * item[3],
    )

    return (
        (
            x,
            y,
            x + width,
            y + height,
        ),
        None,
    )


def _face_crop_box(
    image_width: int,
    image_height: int,
    face_box: Box,
    margin_ratio: float,
) -> Box:
    """根据人脸框生成带边距的方形裁剪框。"""

    x1, y1, x2, y2 = face_box

    face_width = x2 - x1
    face_height = y2 - y1

    margin = int(
        max(
            face_width,
            face_height,
        )
        * margin_ratio
    )

    side = min(
        image_width,
        image_height,
        max(
            face_width,
            face_height,
        )
        + 2 * margin,
    )

    side = max(
        1,
        int(side),
    )

    center_x = (
        x1
        + face_width // 2
    )

    center_y = (
        y1
        + face_height // 2
    )

    left = (
        center_x
        - side // 2
    )

    top = (
        center_y
        - side // 2
    )

    # 将裁剪框整体移动回图像内部，
    # 而不是直接裁掉越界部分。
    left = min(
        max(0, left),
        image_width - side,
    )

    top = min(
        max(0, top),
        image_height - side,
    )

    return (
        left,
        top,
        left + side,
        top + side,
    )


def _center_crop_box(
    image_width: int,
    image_height: int,
) -> Box:
    """计算中心方形裁剪区域。"""

    side = min(
        image_width,
        image_height,
    )

    left = (
        image_width - side
    ) // 2

    top = (
        image_height - side
    ) // 2

    return (
        left,
        top,
        left + side,
        top + side,
    )


def preprocess_bgr(
    image: np.ndarray,
    *,
    size: int = 512,
    detector: Any = None,
    detector_name: str = "custom",
    margin_ratio: float = 0.35,
) -> PreprocessResult:
    """预处理一张已经解码为 BGR 的图像。

    这个接口主要供 Pipeline、API 和测试直接使用。

    detector=None 时不会自动加载 Haar，
    而是明确使用中心裁剪回退。

    需要默认 Haar 检测时使用 preprocess_portrait()。
    """

    if (
        not isinstance(
            image,
            np.ndarray,
        )
        or image.size == 0
    ):
        raise ValueError(
            "image 必须是非空 numpy 数组"
        )

    if (
        image.ndim != 3
        or image.shape[2] != 3
    ):
        raise ValueError(
            "image 必须是 HxWx3 的 BGR 图像"
        )

    if int(size) <= 0:
        raise ValueError(
            "size 必须是正整数"
        )

    if margin_ratio < 0:
        raise ValueError(
            "margin_ratio 不能为负数"
        )

    size = int(size)

    image_height, image_width = (
        image.shape[:2]
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    face_box, warning = (
        _largest_face(
            gray,
            detector,
        )
    )

    warning_messages: list[str] = []

    if warning:
        warning_messages.append(
            warning
        )

    if face_box is not None:
        crop_box = _face_crop_box(
            image_width,
            image_height,
            face_box,
            margin_ratio,
        )

        mode: CropMode = "face"

    else:
        crop_box = _center_crop_box(
            image_width,
            image_height,
        )

        mode = "center"

    left, top, right, bottom = (
        crop_box
    )

    crop = image[
        top:bottom,
        left:right,
    ]

    if crop.size == 0:
        raise RuntimeError(
            "预处理产生了空裁剪区域："
            f"{crop_box}"
        )

    # 保持现有项目行为不变。
    resized = cv2.resize(
        crop,
        (size, size),
        interpolation=cv2.INTER_AREA,
    )

    return PreprocessResult(
        image=resized,

        mode=mode,

        original_size=(
            image_width,
            image_height,
        ),

        output_size=(
            size,
            size,
        ),

        crop_box=crop_box,

        face_box=face_box,

        detector=detector_name,

        warnings=tuple(
            warning_messages
        ),
    )


def preprocess_portrait(
    input_path: str | Path,
    *,
    size: int = 512,
    cascade_path: str | Path | None = None,
) -> PreprocessResult:
    """使用项目默认 Haar 检测器预处理一张肖像照片。"""

    image = load_portrait_image(
        input_path
    )

    detector = load_haar_detector(
        cascade_path
    )

    detector_name = (
        "haar_frontalface_default"
        if detector is not None
        else "unavailable"
    )

    return preprocess_bgr(
        image,
        size=size,
        detector=detector,
        detector_name=detector_name,
    )