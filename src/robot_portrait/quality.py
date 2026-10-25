"""Reproducible structural measurements, not an automatic artistic grade."""
from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import structural_similarity
from skimage.morphology import skeletonize


def gray(image: np.ndarray) -> np.ndarray:
    value = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.resize(value, (256, 256), interpolation=cv2.INTER_AREA)


def ink(image: np.ndarray, *, line_art: bool) -> np.ndarray:
    value = gray(image)
    if line_art:
        return skeletonize(value < 230)
    return cv2.Canny(cv2.GaussianBlur(value, (3, 3), 0), 50, 150, L2gradient=True) > 0


def tolerant_f1(reference: np.ndarray, predicted: np.ndarray, radius: int = 2) -> float | None:
    # Empty drawings must never earn a perfect similarity score.
    if not reference.any():
        return None
    if not predicted.any():
        return 0.0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    ref_d = cv2.dilate(reference.astype(np.uint8), kernel) > 0
    pred_d = cv2.dilate(predicted.astype(np.uint8), kernel) > 0
    precision = float((predicted & ref_d).sum() / predicted.sum())
    recall = float((reference & pred_d).sum() / reference.sum())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate_portrait(reference: np.ndarray, portrait: np.ndarray, vector: np.ndarray,
                      vector_input: np.ndarray, *, cartoon: bool, reference_name: str) -> dict:
    source_edges = ink(reference, line_art=False)
    portrait_edges = ink(portrait, line_art=not cartoon)
    vector_edges = ink(vector, line_art=True)
    input_edges = ink(vector_input, line_art=True)
    def pair(a, b, ea, eb):
        return {
            "ssim": float(structural_similarity(gray(a), gray(b), data_range=255)),
            "edge_f1": tolerant_f1(ea, eb),
        }
    return {
        "protocol": "portrait-structure-v1", "comparison_size": 256,
        "edge_tolerance_px": 2, "reference": reference_name,
        "source_to_portrait": pair(reference, portrait, source_edges, portrait_edges),
        "source_to_vector": pair(reference, vector, source_edges, vector_edges),
        "vector_retention": pair(vector_input, vector, input_edges, vector_edges),
        "human_review": {"identity": None, "style_quality": None, "vector_likeness": None, "artifacts": None},
        "note": "SSIM 和容差 Edge-F1 仅衡量结构保真度，不等同于人脸身份、逼真度或课程成绩；艺术效果需人工盲评。卡通矢量图为单笔轮廓，不保留颜色。",
    }
