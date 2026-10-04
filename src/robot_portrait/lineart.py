from __future__ import annotations
import cv2
import numpy as np


def resize_keep_ratio(img: np.ndarray, max_side: int = 1024) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    if scale == 1.0:
        return img
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def portrait_lineart(
    bgr: np.ndarray,
    max_side: int = 1024,
    bilateral_d: int = 9,
    bilateral_sigma_color: int = 75,
    bilateral_sigma_space: int = 75,
    adaptive_block_size: int = 11,
    adaptive_c: int = 2,
    morph_kernel: int = 2,
) -> np.ndarray:
    """确定性的栅格基线：将照片转换为白底黑线的人像线稿。"""
    bgr = resize_keep_ratio(bgr, max_side)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    smooth = cv2.bilateralFilter(
        gray, bilateral_d, bilateral_sigma_color, bilateral_sigma_space
    )

    block = max(3, int(adaptive_block_size))
    if block % 2 == 0:
        block += 1

    line = cv2.adaptiveThreshold(
        smooth, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block, adaptive_c
    )

    # 在保留人脸主要笔画的同时去除细小的孤立噪声。
    if morph_kernel > 0:
        k = np.ones((morph_kernel, morph_kernel), np.uint8)
        line = cv2.morphologyEx(line, cv2.MORPH_OPEN, k)

    return line
