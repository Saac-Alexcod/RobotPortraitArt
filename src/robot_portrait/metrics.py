from __future__ import annotations
import cv2
import numpy as np
from skimage.metrics import structural_similarity
import warnings


def _gray_resize(img: np.ndarray, size: int = 512) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def ssim_score(pred: np.ndarray, ref: np.ndarray, size: int = 512) -> float:
    a, b = _gray_resize(pred, size), _gray_resize(ref, size)
    return float(structural_similarity(a, b, data_range=255))


def edge_f1(pred: np.ndarray, ref: np.ndarray, size: int = 512) -> float:
    a, b = _gray_resize(pred, size), _gray_resize(ref, size)
    ea = cv2.Canny(a, 80, 160) > 0
    eb = cv2.Canny(b, 80, 160) > 0
    tp = np.logical_and(ea, eb).sum()
    fp = np.logical_and(ea, ~eb).sum()
    fn = np.logical_and(~ea, eb).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return float(2 * precision * recall / (precision + recall + 1e-8))


def optional_lpips(pred_path: str, ref_path: str):
    """如果已安装 torch/lpips，则返回 LPIPS；否则返回 None。"""
    try:
        import torch, lpips
        from PIL import Image
        import torchvision.transforms as T
        loss_fn = lpips.LPIPS(net="alex")
        tfm = T.Compose([T.Resize((256,256)), T.ToTensor(), T.Normalize([.5]*3,[.5]*3)])
        a = tfm(Image.open(pred_path).convert("RGB")).unsqueeze(0)
        b = tfm(Image.open(ref_path).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            return float(loss_fn(a, b).item())
    except Exception as exc:
        warnings.warn(f"LPIPS 不可用，本次仅计算 SSIM/Edge-F1：{type(exc).__name__}: {exc}")
        return None
