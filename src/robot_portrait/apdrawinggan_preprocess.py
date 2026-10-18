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

PPHUMANSEG_INPUT_SIZE = (192, 192)

_HUMANSEG_LFS_POINTER_PREFIX = (
    b"version https://git-lfs.github.com/spec/v1"
)


class HumanSegModelError(RuntimeError):
    """Raised when the PPHumanSeg ONNX model cannot be loaded."""


@dataclass(frozen=True, slots=True)
class HumanSegConfig:
    """Configuration for OpenCV Zoo PPHumanSeg inference."""

    model_path: Path
    backend_id: int = cv2.dnn.DNN_BACKEND_OPENCV
    target_id: int = cv2.dnn.DNN_TARGET_CPU
    foreground_class: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_path",
            Path(self.model_path),
        )

        if int(self.foreground_class) < 0:
            raise ValueError(
                "foreground_class 必须是非负整数"
            )


@dataclass(frozen=True, slots=True)
class HumanMaskResult:
    """Binary person foreground mask compatible with APDrawingGAN."""

    mask: np.ndarray
    foreground_ratio: float
    model: str = "opencv-zoo-pphumanseg-2023mar"

    def __post_init__(self) -> None:
        mask = np.asarray(
            self.mask
        )

        if (
            mask.ndim != 2
            or mask.dtype != np.uint8
            or mask.size == 0
        ):
            raise ValueError(
                "mask 必须是非空 uint8 单通道图像"
            )

        values = set(
            np.unique(mask).tolist()
        )

        if not values.issubset(
            {0, 255}
        ):
            raise ValueError(
                "mask 必须是 0/255 二值图"
            )

        ratio = float(
            self.foreground_ratio
        )

        if (
            not np.isfinite(ratio)
            or ratio < 0.0
            or ratio > 1.0
        ):
            raise ValueError(
                "foreground_ratio 必须位于 [0, 1]"
            )

    def to_metadata(
        self,
    ) -> dict[str, object]:
        height, width = (
            self.mask.shape
        )

        return {
            "model": self.model,
            "width": int(width),
            "height": int(height),
            "foreground_ratio": float(
                self.foreground_ratio
            ),
            "foreground_value": 255,
            "background_value": 0,
        }


HumanSegNetFactory = Callable[
    [np.ndarray, HumanSegConfig],
    Any,
]


def _load_humanseg_model_buffer(
    model_path: Path,
) -> np.ndarray:
    if not model_path.is_file():
        raise FileNotFoundError(
            "找不到 PPHumanSeg ONNX 模型："
            f"{model_path}"
        )

    try:
        data = model_path.read_bytes()
    except OSError as exc:
        raise HumanSegModelError(
            "无法读取 PPHumanSeg ONNX 模型："
            f"{model_path}"
        ) from exc

    if not data:
        raise HumanSegModelError(
            "PPHumanSeg ONNX 模型为空："
            f"{model_path}"
        )

    if data.startswith(
        _HUMANSEG_LFS_POINTER_PREFIX
    ):
        raise HumanSegModelError(
            "PPHumanSeg 模型当前只是 Git LFS pointer，"
            "请先拉取真实 ONNX 文件。"
        )

    return np.frombuffer(
        data,
        dtype=np.uint8,
    )


def _default_humanseg_net_factory(
    model_buffer: np.ndarray,
    config: HumanSegConfig,
) -> Any:
    try:
        # In-memory loading avoids Windows Unicode-path problems.
        net = cv2.dnn.readNetFromONNX(
            model_buffer
        )

        net.setPreferableBackend(
            int(
                config.backend_id
            )
        )

        net.setPreferableTarget(
            int(
                config.target_id
            )
        )

        return net
    except cv2.error as exc:
        raise HumanSegModelError(
            "OpenCV 无法解析 PPHumanSeg ONNX 模型："
            f"{config.model_path}"
        ) from exc


def _humanseg_blob(
    image: np.ndarray,
) -> np.ndarray:
    """Reproduce OpenCV Zoo PPHumanSeg preprocessing."""

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    resized = cv2.resize(
        rgb,
        PPHUMANSEG_INPUT_SIZE,
        interpolation=cv2.INTER_LINEAR,
    )

    normalized = (
        resized.astype(
            np.float32,
            copy=False,
        )
        / 255.0
    )

    normalized -= 0.5
    normalized /= 0.5

    return cv2.dnn.blobFromImage(
        normalized
    )


def _decode_humanseg_output(
    output: np.ndarray,
    *,
    output_size: tuple[int, int],
    foreground_class: int,
) -> np.ndarray:
    """Reproduce OpenCV Zoo PPHumanSeg postprocessing exactly.

    The reference implementation first resizes the per-class logits with
    bilinear interpolation and only then applies argmax.  Resizing an already
    discretized class map would move boundaries differently.
    """

    logits = np.asarray(
        output
    )

    if (
        logits.ndim != 4
        or logits.shape[0] != 1
        or logits.shape[1] < 2
        or logits.shape[2] <= 0
        or logits.shape[3] <= 0
        or not np.isfinite(
            logits
        ).all()
    ):
        raise RuntimeError(
            "PPHumanSeg 返回了无法识别的输出形状："
            f"{logits.shape}"
        )

    if (
        foreground_class
        >= logits.shape[1]
    ):
        raise RuntimeError(
            "foreground_class 超出模型输出类别数："
            f"{foreground_class} >= {logits.shape[1]}"
        )

    width, height = (
        output_size
    )

    logits_hwc = (
        logits[0]
        .transpose(
            1,
            2,
            0,
        )
    )

    if (
        logits_hwc.shape[1] != width
        or logits_hwc.shape[0] != height
    ):
        logits_hwc = cv2.resize(
            logits_hwc,
            (
                width,
                height,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

    if logits_hwc.ndim != 3:
        raise RuntimeError(
            "PPHumanSeg resize 后的 logits 形状无效："
            f"{logits_hwc.shape}"
        )

    classes = np.argmax(
        logits_hwc,
        axis=2,
    ).astype(
        np.uint8
    )

    foreground = (
        classes
        == int(
            foreground_class
        )
    ).astype(
        np.uint8
    )

    return (
        foreground
        * 255
    ).astype(
        np.uint8,
        copy=False,
    )


def generate_pphumanseg_mask(
    image: np.ndarray,
    config: HumanSegConfig,
    *,
    net_factory: HumanSegNetFactory | None = None,
) -> HumanMaskResult:
    """Generate APDrawingGAN's non-background mask with PPHumanSeg.

    APDrawingGAN calls this a "background mask", but its dataset code treats
    pixels >= 0.5 as ``mask2`` (non-background/portrait).  Therefore this
    function intentionally writes 255 for person/foreground and 0 for
    background.
    """

    _validate_image(
        image
    )

    model_buffer = (
        _load_humanseg_model_buffer(
            config.model_path
        )
    )

    factory = (
        net_factory
        or _default_humanseg_net_factory
    )

    net = factory(
        model_buffer,
        config,
    )

    blob = _humanseg_blob(
        image
    )

    try:
        net.setInput(
            blob
        )
        output = net.forward()
    except cv2.error as exc:
        raise RuntimeError(
            "PPHumanSeg 推理执行失败"
        ) from exc

    height, width = (
        image.shape[:2]
    )

    mask = (
        _decode_humanseg_output(
            output,
            output_size=(
                width,
                height,
            ),
            foreground_class=(
                int(
                    config.foreground_class
                )
            ),
        )
    )

    foreground_ratio = float(
        np.count_nonzero(
            mask
        )
        / mask.size
    )

    return HumanMaskResult(
        mask=mask,
        foreground_ratio=(
            foreground_ratio
        ),
    )


def draw_human_mask_preview(
    image: np.ndarray,
    result: HumanMaskResult,
) -> np.ndarray:
    """Visualize foreground while dimming background."""

    _validate_image(
        image
    )

    if (
        result.mask.shape
        != image.shape[:2]
    ):
        raise ValueError(
            "mask 尺寸必须与 image 一致"
        )

    preview = image.copy()
    foreground = (
        result.mask
        >= 128
    )

    dimmed = (
        preview.astype(
            np.float32
        )
        * 0.30
        + 255.0
        * 0.70
    )

    preview[
        ~foreground
    ] = np.clip(
        dimmed[
            ~foreground
        ],
        0,
        255,
    ).astype(
        np.uint8
    )

    return preview


def compare_binary_masks(
    predicted: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float]:
    """Compare two 0/255-style foreground masks."""

    predicted_array = np.asarray(
        predicted
    )
    reference_array = np.asarray(
        reference
    )

    if (
        predicted_array.ndim != 2
        or reference_array.ndim != 2
        or predicted_array.shape
        != reference_array.shape
        or predicted_array.size == 0
    ):
        raise ValueError(
            "两个 mask 必须是同尺寸的非空二维数组"
        )

    pred = (
        predicted_array
        >= 128
    )

    ref = (
        reference_array
        >= 128
    )

    intersection = int(
        np.count_nonzero(
            pred & ref
        )
    )

    union = int(
        np.count_nonzero(
            pred | ref
        )
    )

    pred_count = int(
        np.count_nonzero(
            pred
        )
    )

    ref_count = int(
        np.count_nonzero(
            ref
        )
    )

    total = int(
        pred.size
    )

    matches = int(
        np.count_nonzero(
            pred == ref
        )
    )

    iou = (
        1.0
        if union == 0
        else intersection
        / union
    )

    denominator = (
        pred_count
        + ref_count
    )

    dice = (
        1.0
        if denominator == 0
        else (
            2.0
            * intersection
            / denominator
        )
    )

    return {
        "iou": float(iou),
        "dice": float(dice),
        "pixel_accuracy": float(
            matches
            / total
        ),
        "predicted_foreground_ratio": float(
            pred_count
            / total
        ),
        "reference_foreground_ratio": float(
            ref_count
            / total
        ),
    }
