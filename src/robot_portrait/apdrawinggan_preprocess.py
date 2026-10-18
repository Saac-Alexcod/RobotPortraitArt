from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np


FaceBox = tuple[int, int, int, int]
Point = tuple[float, float]

YUNET_LANDMARK_NAMES = (
    "right_eye",
    "left_eye",
    "nose_tip",
    "right_mouth_corner",
    "left_mouth_corner",
)

_GIT_LFS_POINTER_PREFIX = (
    b"version https://git-lfs.github.com/spec/v1"
)


class FaceNotDetectedError(RuntimeError):
    """Raised when YuNet does not find a usable face."""


class YuNetModelError(RuntimeError):
    """Raised when the YuNet model cannot be loaded."""


@dataclass(frozen=True, slots=True)
class YuNetConfig:
    """Configuration for OpenCV YuNet five-point face detection."""

    model_path: Path
    score_threshold: float = 0.9
    nms_threshold: float = 0.3
    top_k: int = 5000
    max_input_side: int = 640

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_path",
            Path(self.model_path),
        )

        if not 0.0 < float(self.score_threshold) <= 1.0:
            raise ValueError(
                "score_threshold 必须位于 (0, 1] 区间"
            )

        if not 0.0 <= float(self.nms_threshold) <= 1.0:
            raise ValueError(
                "nms_threshold 必须位于 [0, 1] 区间"
            )

        if int(self.top_k) <= 0:
            raise ValueError(
                "top_k 必须是正整数"
            )

        if int(self.max_input_side) <= 0:
            raise ValueError(
                "max_input_side 必须是正整数"
            )


@dataclass(frozen=True, slots=True)
class FivePointFace:
    """A single face and its five APDrawingGAN-compatible landmarks.

    Point order follows OpenCV YuNet and the MTCNN/APDrawingGAN convention:

    1. right eye
    2. left eye
    3. nose tip
    4. right mouth corner
    5. left mouth corner

    "Right" and "left" refer to the subject, not image coordinates.
    """

    points: tuple[Point, Point, Point, Point, Point]
    face_box: FaceBox
    score: float
    original_size: tuple[int, int]
    detector: str = "opencv-yunet"

    def __post_init__(self) -> None:
        array = np.asarray(
            self.points,
            dtype=np.float64,
        )

        if (
            array.shape != (5, 2)
            or not np.isfinite(array).all()
        ):
            raise ValueError(
                "points 必须恰好包含 5 个有限的 (x, y) 点"
            )

        left, top, right, bottom = self.face_box
        if (
            right <= left
            or bottom <= top
        ):
            raise ValueError(
                "face_box 必须具有正面积"
            )

        width, height = self.original_size
        if width <= 0 or height <= 0:
            raise ValueError(
                "original_size 必须为正尺寸"
            )

        if not np.isfinite(float(self.score)):
            raise ValueError(
                "score 必须为有限数值"
            )

    def rounded_points(
        self,
    ) -> tuple[
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
    ]:
        rounded = np.rint(
            np.asarray(
                self.points,
                dtype=np.float64,
            )
        ).astype(np.int64)

        return tuple(
            (int(x), int(y))
            for x, y in rounded
        )  # type: ignore[return-value]

    def to_metadata(
        self,
    ) -> dict[str, object]:
        return {
            "detector": self.detector,
            "score": float(self.score),
            "original_size": {
                "width": int(
                    self.original_size[0]
                ),
                "height": int(
                    self.original_size[1]
                ),
            },
            "face_box": {
                "left": int(self.face_box[0]),
                "top": int(self.face_box[1]),
                "right": int(self.face_box[2]),
                "bottom": int(self.face_box[3]),
            },
            "landmarks": {
                name: {
                    "x": float(point[0]),
                    "y": float(point[1]),
                }
                for name, point in zip(
                    YUNET_LANDMARK_NAMES,
                    self.points,
                    strict=True,
                )
            },
        }


DetectorFactory = Callable[
    [np.ndarray, tuple[int, int], YuNetConfig],
    Any,
]


def _load_model_buffer(
    model_path: Path,
) -> np.ndarray:
    if not model_path.is_file():
        raise FileNotFoundError(
            "找不到 YuNet ONNX 模型："
            f"{model_path}"
        )

    try:
        data = model_path.read_bytes()
    except OSError as exc:
        raise YuNetModelError(
            "无法读取 YuNet ONNX 模型："
            f"{model_path}"
        ) from exc

    if not data:
        raise YuNetModelError(
            "YuNet ONNX 模型为空："
            f"{model_path}"
        )

    if data.startswith(
        _GIT_LFS_POINTER_PREFIX
    ):
        raise YuNetModelError(
            "YuNet 模型当前只是 Git LFS pointer，"
            "请先拉取真实 ONNX 文件。"
        )

    return np.frombuffer(
        data,
        dtype=np.uint8,
    )


def _default_detector_factory(
    model_buffer: np.ndarray,
    input_size: tuple[int, int],
    config: YuNetConfig,
) -> Any:
    empty_config = np.empty(
        (0,),
        dtype=np.uint8,
    )

    try:
        # Use the in-memory overload instead of a model path.
        # This is robust when the project/model path contains
        # Chinese or other non-ASCII characters on Windows.
        return cv2.FaceDetectorYN.create(
            "onnx",
            model_buffer,
            empty_config,
            input_size,
            float(config.score_threshold),
            float(config.nms_threshold),
            int(config.top_k),
        )
    except cv2.error as exc:
        raise YuNetModelError(
            "OpenCV 无法解析 YuNet ONNX 模型："
            f"{config.model_path}"
        ) from exc


def _validate_image(
    image: np.ndarray,
) -> None:
    if (
        not isinstance(image, np.ndarray)
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

    if image.dtype != np.uint8:
        raise ValueError(
            "image 必须使用 uint8 像素类型"
        )


def _scaled_detection_image(
    image: np.ndarray,
    max_input_side: int,
) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    longest = max(width, height)

    if longest <= max_input_side:
        return image, 1.0

    scale = (
        float(max_input_side)
        / float(longest)
    )

    scaled_width = max(
        1,
        int(round(width * scale)),
    )
    scaled_height = max(
        1,
        int(round(height * scale)),
    )

    resized = cv2.resize(
        image,
        (
            scaled_width,
            scaled_height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    return resized, scale


def _select_primary_face(
    faces: np.ndarray,
) -> np.ndarray:
    array = np.asarray(
        faces,
        dtype=np.float64,
    )

    if array.size == 0:
        raise FaceNotDetectedError(
            "YuNet 未检测到人脸"
        )

    if array.ndim == 1:
        array = array.reshape(1, -1)

    if (
        array.ndim != 2
        or array.shape[1] < 15
    ):
        raise RuntimeError(
            "YuNet 返回了无法识别的检测结果形状："
            f"{array.shape}"
        )

    valid_rows: list[np.ndarray] = []

    for row in array:
        relevant = row[:15]

        if not np.isfinite(
            relevant
        ).all():
            continue

        width = float(row[2])
        height = float(row[3])

        if (
            width <= 0
            or height <= 0
        ):
            continue

        valid_rows.append(row)

    if not valid_rows:
        raise FaceNotDetectedError(
            "YuNet 未返回有效的人脸框"
        )

    # Portrait workflow: prefer the visually dominant face.
    # Confidence is used only as a tie-breaker.
    return max(
        valid_rows,
        key=lambda row: (
            float(row[2])
            * float(row[3]),
            float(row[14]),
        ),
    )


def detect_yunet_five_points(
    image: np.ndarray,
    config: YuNetConfig,
    *,
    detector_factory: DetectorFactory | None = None,
) -> FivePointFace:
    """Detect the dominant face and five landmarks with OpenCV YuNet."""

    _validate_image(image)

    original_height, original_width = (
        image.shape[:2]
    )

    detection_image, scale = (
        _scaled_detection_image(
            image,
            int(config.max_input_side),
        )
    )

    detection_height, detection_width = (
        detection_image.shape[:2]
    )

    model_buffer = _load_model_buffer(
        config.model_path
    )

    factory = (
        detector_factory
        or _default_detector_factory
    )

    detector = factory(
        model_buffer,
        (
            detection_width,
            detection_height,
        ),
        config,
    )

    try:
        _, faces = detector.detect(
            detection_image
        )
    except cv2.error as exc:
        raise RuntimeError(
            "YuNet 人脸检测执行失败"
        ) from exc

    if faces is None:
        raise FaceNotDetectedError(
            "YuNet 未检测到人脸"
        )

    row = _select_primary_face(
        faces
    )

    inverse_scale = 1.0 / scale

    x = float(row[0]) * inverse_scale
    y = float(row[1]) * inverse_scale
    width = (
        float(row[2])
        * inverse_scale
    )
    height = (
        float(row[3])
        * inverse_scale
    )

    left = max(
        0,
        int(np.floor(x)),
    )
    top = max(
        0,
        int(np.floor(y)),
    )
    right = min(
        original_width,
        int(np.ceil(x + width)),
    )
    bottom = min(
        original_height,
        int(np.ceil(y + height)),
    )

    if (
        right <= left
        or bottom <= top
    ):
        raise FaceNotDetectedError(
            "YuNet 检测到的人脸框落在图像范围之外"
        )

    landmark_values = (
        row[4:14]
        .reshape(5, 2)
        * inverse_scale
    )

    points = tuple(
        (
            float(point[0]),
            float(point[1]),
        )
        for point in landmark_values
    )

    return FivePointFace(
        points=points,  # type: ignore[arg-type]
        face_box=(
            left,
            top,
            right,
            bottom,
        ),
        score=float(row[14]),
        original_size=(
            original_width,
            original_height,
        ),
    )


def _coerce_points(
    value: (
        FivePointFace
        | Sequence[
            Sequence[float]
        ]
        | np.ndarray
    ),
) -> np.ndarray:
    points = (
        np.asarray(
            value.points,
            dtype=np.float64,
        )
        if isinstance(
            value,
            FivePointFace,
        )
        else np.asarray(
            value,
            dtype=np.float64,
        )
    )

    if (
        points.shape != (5, 2)
        or not np.isfinite(
            points
        ).all()
    ):
        raise ValueError(
            "landmark 必须恰好包含 "
            "5 个有限的 (x, y) 点"
        )

    return points


def write_apdrawinggan_landmarks(
    path: str | Path,
    value: (
        FivePointFace
        | Sequence[
            Sequence[float]
        ]
        | np.ndarray
    ),
) -> Path:
    """Write APDrawingGAN's five-line integer landmark format."""

    output_path = Path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rounded = np.rint(
        _coerce_points(value)
    ).astype(np.int64)

    output_path.write_text(
        "".join(
            f"{int(x)} {int(y)}\n"
            for x, y in rounded
        ),
        encoding="utf-8",
    )

    return output_path


def draw_five_point_preview(
    image: np.ndarray,
    detection: FivePointFace,
) -> np.ndarray:
    """Draw an English-labelled preview for manual verification."""

    _validate_image(image)

    preview = image.copy()

    left, top, right, bottom = (
        detection.face_box
    )

    cv2.rectangle(
        preview,
        (left, top),
        (
            max(left, right - 1),
            max(top, bottom - 1),
        ),
        (0, 255, 0),
        2,
    )

    labels = (
        "R-eye",
        "L-eye",
        "Nose",
        "R-mouth",
        "L-mouth",
    )

    for label, point in zip(
        labels,
        detection.rounded_points(),
        strict=True,
    ):
        x, y = point

        cv2.circle(
            preview,
            (x, y),
            4,
            (0, 0, 255),
            -1,
        )

        cv2.putText(
            preview,
            label,
            (
                x + 5,
                y - 5,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return preview
APDRAWINGGAN_ALIGNMENT_SIZE = 512

_APDRAWINGGAN_TEMPLATE_BASE = np.array(
    [
        [180.0, 230.0],
        [300.0, 230.0],
        [240.0, 301.0],
        [186.0, 365.6],
        [294.0, 365.6],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True, slots=True)
class SimilarityTransform:
    """Least-squares 2-D similarity transform from source to target."""

    matrix: np.ndarray
    scale: float
    rotation_degrees: float
    reflected: bool
    rms_error: float

    def __post_init__(self) -> None:
        matrix = np.asarray(
            self.matrix,
            dtype=np.float64,
        )

        if (
            matrix.shape != (2, 3)
            or not np.isfinite(matrix).all()
        ):
            raise ValueError(
                "matrix 必须是有限的 2x3 仿射矩阵"
            )

        if (
            not np.isfinite(float(self.scale))
            or float(self.scale) <= 0.0
        ):
            raise ValueError(
                "scale 必须是有限正数"
            )

        if not np.isfinite(
            float(self.rotation_degrees)
        ):
            raise ValueError(
                "rotation_degrees 必须是有限数值"
            )

        if (
            not np.isfinite(float(self.rms_error))
            or float(self.rms_error) < 0.0
        ):
            raise ValueError(
                "rms_error 必须是有限非负数"
            )

        object.__setattr__(
            self,
            "matrix",
            matrix.copy(),
        )

    def transform_points(
        self,
        points: (
            Sequence[Sequence[float]]
            | np.ndarray
        ),
    ) -> np.ndarray:
        array = np.asarray(
            points,
            dtype=np.float64,
        )

        if (
            array.ndim != 2
            or array.shape[1] != 2
            or not np.isfinite(array).all()
        ):
            raise ValueError(
                "points 必须是有限的 Nx2 坐标数组"
            )

        homogeneous = np.concatenate(
            [
                array,
                np.ones(
                    (array.shape[0], 1),
                    dtype=np.float64,
                ),
            ],
            axis=1,
        )

        return (
            homogeneous
            @ self.matrix.T
        )

    def to_metadata(
        self,
    ) -> dict[str, object]:
        return {
            "matrix": (
                self.matrix
                .astype(float)
                .tolist()
            ),
            "scale": float(
                self.scale
            ),
            "rotation_degrees": float(
                self.rotation_degrees
            ),
            "reflected": bool(
                self.reflected
            ),
            "rms_error": float(
                self.rms_error
            ),
        }


@dataclass(frozen=True, slots=True)
class AlignedFace:
    """Image and five landmarks after APDrawingGAN alignment."""

    image: np.ndarray
    points: tuple[
        Point,
        Point,
        Point,
        Point,
        Point,
    ]
    transform: SimilarityTransform
    output_size: int

    def __post_init__(self) -> None:
        image = np.asarray(
            self.image
        )

        if (
            image.shape
            != (
                self.output_size,
                self.output_size,
                3,
            )
            or image.dtype != np.uint8
        ):
            raise ValueError(
                "aligned image 必须是 "
                "output_size x output_size x 3 的 uint8 图像"
            )

        points = np.asarray(
            self.points,
            dtype=np.float64,
        )

        if (
            points.shape != (5, 2)
            or not np.isfinite(points).all()
        ):
            raise ValueError(
                "aligned points 必须包含 5 个有限坐标"
            )

    def rounded_points(
        self,
    ) -> tuple[
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
    ]:
        rounded = np.rint(
            np.asarray(
                self.points,
                dtype=np.float64,
            )
        ).astype(np.int64)

        return tuple(
            (int(x), int(y))
            for x, y in rounded
        )  # type: ignore[return-value]

    def to_metadata(
        self,
    ) -> dict[str, object]:
        return {
            "output_size": int(
                self.output_size
            ),
            "landmarks": [
                {
                    "name": name,
                    "x": float(point[0]),
                    "y": float(point[1]),
                }
                for name, point in zip(
                    YUNET_LANDMARK_NAMES,
                    self.points,
                    strict=True,
                )
            ],
            "transform": (
                self.transform
                .to_metadata()
            ),
        }


def apdrawinggan_alignment_template(
    output_size: int = APDRAWINGGAN_ALIGNMENT_SIZE,
) -> np.ndarray:
    """Return the exact five-point template used by APDrawingGAN.

    This reproduces ``face_align_512.m``:
        (coord5point - 240) / 560 * output_size + output_size / 2
    """

    output_size = int(
        output_size
    )

    if output_size <= 0:
        raise ValueError(
            "output_size 必须是正整数"
        )

    return (
        (
            _APDRAWINGGAN_TEMPLATE_BASE
            - 240.0
        )
        / 560.0
        * float(output_size)
        + float(output_size) / 2.0
    )


def read_apdrawinggan_landmarks(
    path: str | Path,
) -> np.ndarray:
    """Read APDrawingGAN's five-line landmark text format."""

    landmark_path = Path(path)

    if not landmark_path.is_file():
        raise FileNotFoundError(
            "找不到 landmark 文件："
            f"{landmark_path}"
        )

    rows: list[
        list[float]
    ] = []

    try:
        for line in (
            landmark_path
            .read_text(
                encoding="utf-8"
            )
            .splitlines()
        ):
            stripped = (
                line.strip()
            )

            if not stripped:
                continue

            parts = (
                stripped.split()
            )

            if len(parts) != 2:
                raise ValueError

            rows.append(
                [
                    float(parts[0]),
                    float(parts[1]),
                ]
            )
    except (
        OSError,
        UnicodeError,
        ValueError,
    ) as exc:
        raise ValueError(
            "landmark 文件格式无效，"
            "应为 5 行 x y 坐标"
        ) from exc

    return _coerce_points(
        np.asarray(
            rows,
            dtype=np.float64,
        )
    )


def _fit_similarity_candidate(
    source: np.ndarray,
    target: np.ndarray,
    *,
    reflected: bool,
) -> SimilarityTransform:
    source_mean = (
        source.mean(
            axis=0
        )
    )
    target_mean = (
        target.mean(
            axis=0
        )
    )

    source_centered = (
        source
        - source_mean
    )
    target_centered = (
        target
        - target_mean
    )

    covariance = (
        target_centered.T
        @ source_centered
    )

    left, singular_values, right_t = (
        np.linalg.svd(
            covariance
        )
    )

    orthogonal = (
        left
        @ right_t
    )

    current_det = float(
        np.linalg.det(
            orthogonal
        )
    )

    desired_det = (
        -1.0
        if reflected
        else 1.0
    )

    correction = np.eye(
        2,
        dtype=np.float64,
    )
    correction[
        -1,
        -1,
    ] = (
        desired_det
        / current_det
    )

    rotation = (
        left
        @ correction
        @ right_t
    )

    denominator = float(
        np.sum(
            source_centered
            * source_centered
        )
    )

    numerator = float(
        np.sum(
            singular_values
            * np.diag(
                correction
            )
        )
    )

    scale = (
        numerator
        / denominator
    )

    if (
        not np.isfinite(scale)
        or scale
        <= np.finfo(
            np.float64
        ).eps
    ):
        raise ValueError(
            "无法从 landmark 拟合有效的相似变换"
        )

    translation = (
        target_mean
        - scale
        * (
            rotation
            @ source_mean
        )
    )

    matrix = np.column_stack(
        (
            scale
            * rotation,
            translation,
        )
    )

    homogeneous = np.concatenate(
        [
            source,
            np.ones(
                (
                    source.shape[0],
                    1,
                ),
                dtype=np.float64,
            ),
        ],
        axis=1,
    )

    predicted = (
        homogeneous
        @ matrix.T
    )

    residual = (
        predicted
        - target
    )

    rms_error = float(
        np.sqrt(
            np.mean(
                np.sum(
                    residual
                    * residual,
                    axis=1,
                )
            )
        )
    )

    rotation_degrees = float(
        np.degrees(
            np.arctan2(
                rotation[1, 0],
                rotation[0, 0],
            )
        )
    )

    return SimilarityTransform(
        matrix=matrix,
        scale=scale,
        rotation_degrees=(
            rotation_degrees
        ),
        reflected=reflected,
        rms_error=rms_error,
    )


def fit_similarity_transform(
    source_points: (
        FivePointFace
        | Sequence[
            Sequence[float]
        ]
        | np.ndarray
    ),
    target_points: (
        Sequence[
            Sequence[float]
        ]
        | np.ndarray
    ),
    *,
    allow_reflection: bool = True,
) -> SimilarityTransform:
    """Fit APDrawingGAN/MATLAB-style least-squares similarity transform.

    MATLAB's legacy ``cp2tform(..., 'similarity')`` allows an optional
    reflection.  To reproduce that behavior, this implementation evaluates
    both the non-reflective and reflective solutions and keeps the one with
    the lower landmark RMS error.
    """

    source = _coerce_points(
        source_points
    )
    target = _coerce_points(
        target_points
    )

    source_centered = (
        source
        - source.mean(
            axis=0
        )
    )
    target_centered = (
        target
        - target.mean(
            axis=0
        )
    )

    if (
        np.linalg.matrix_rank(
            source_centered
        )
        < 2
        or np.linalg.matrix_rank(
            target_centered
        )
        < 2
    ):
        raise ValueError(
            "landmark 点集退化，"
            "无法拟合二维相似变换"
        )

    candidates = [
        _fit_similarity_candidate(
            source,
            target,
            reflected=False,
        )
    ]

    if allow_reflection:
        try:
            candidates.append(
                _fit_similarity_candidate(
                    source,
                    target,
                    reflected=True,
                )
            )
        except ValueError:
            pass

    return min(
        candidates,
        key=lambda candidate: (
            candidate.rms_error,
            candidate.reflected,
        ),
    )


def align_apdrawinggan_face(
    image: np.ndarray,
    landmarks: (
        FivePointFace
        | Sequence[
            Sequence[float]
        ]
        | np.ndarray
    ),
    *,
    output_size: int = APDRAWINGGAN_ALIGNMENT_SIZE,
    allow_reflection: bool = True,
) -> AlignedFace:
    """Align an image to the five-point geometry used by APDrawingGAN."""

    _validate_image(
        image
    )

    output_size = int(
        output_size
    )

    if output_size <= 0:
        raise ValueError(
            "output_size 必须是正整数"
        )

    source = _coerce_points(
        landmarks
    )

    target = (
        apdrawinggan_alignment_template(
            output_size
        )
    )

    transform = (
        fit_similarity_transform(
            source,
            target,
            allow_reflection=(
                allow_reflection
            ),
        )
    )

    aligned = cv2.warpAffine(
        image,
        transform.matrix,
        (
            output_size,
            output_size,
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=(
            cv2.BORDER_CONSTANT
        ),
        borderValue=(
            255,
            255,
            255,
        ),
    )

    transformed = (
        transform
        .transform_points(
            source
        )
    )

    points = tuple(
        (
            float(point[0]),
            float(point[1]),
        )
        for point in transformed
    )

    return AlignedFace(
        image=aligned,
        points=points,  # type: ignore[arg-type]
        transform=transform,
        output_size=output_size,
    )
