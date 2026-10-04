from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET


def raster_to_svg(
    line_img: np.ndarray,
    output_svg: str | Path,
    min_area: float = 12.0,
    epsilon_ratio: float = 0.0025,
    stroke_width: float = 1.2,
) -> dict:
    """追踪黑色线条区域，并将简化后的轮廓路径写入 SVG。"""
    if line_img.ndim == 3:
        gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = line_img

    # 将黑色笔画转换为白色前景，便于提取轮廓
    fg = 255 - gray
    _, fg = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(fg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    h, w = gray.shape[:2]
    paths = []
    kept = 0
    for cnt in contours:
        area = abs(cv2.contourArea(cnt))
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        eps = max(0.5, epsilon_ratio * peri)
        approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
        if len(approx) < 2:
            continue
        d = [f"M {approx[0,0]} {approx[0,1]}"]
        for x, y in approx[1:]:
            d.append(f"L {x} {y}")
        d.append("Z")
        paths.append(
            f'<path d="{" ".join(d)}" fill="none" stroke="black" '
            f'stroke-width="{stroke_width}" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        kept += 1

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}">\n'
        f'<rect width="100%" height="100%" fill="white"/>\n'
        + "\n".join(paths)
        + "\n</svg>\n"
    )
    output_svg = Path(output_svg)
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    output_svg.write_text(svg, encoding="utf-8")
    return {"width": w, "height": h, "path_count": kept, "bytes": output_svg.stat().st_size}


def _opencv_svg_to_raster(svg_path: Path, output_png: Path, output_width: int | None) -> dict:
    """Rasterize the M/L/Z-only SVG subset emitted by :func:`raster_to_svg`."""
    root = ET.parse(svg_path).getroot()
    viewbox = root.attrib.get("viewBox", "").replace(",", " ").split()
    if len(viewbox) == 4:
        _, _, source_width, source_height = map(float, viewbox)
    else:
        source_width = float(root.attrib["width"])
        source_height = float(root.attrib["height"])
    target_width = int(output_width or round(source_width))
    if target_width <= 0 or source_width <= 0 or source_height <= 0:
        raise ValueError("SVG 尺寸无效")
    target_height = max(1, round(source_height * target_width / source_width))
    scale_x, scale_y = target_width / source_width, target_height / source_height
    canvas = np.full((target_height, target_width, 3), 255, dtype=np.uint8)
    path_count = 0
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] != "path":
            continue
        data = element.attrib.get("d", "")
        commands = re.sub(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", "", data)
        if re.sub(r"[\s,MLZmlz]", "", commands):
            raise ValueError("OpenCV fallback 仅支持本项目生成的 M/L/Z SVG 路径")
        numbers = [float(value) for value in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", data)]
        if len(numbers) < 4 or len(numbers) % 2:
            continue
        points = np.asarray([
            [round(numbers[index] * scale_x), round(numbers[index + 1] * scale_y)]
            for index in range(0, len(numbers), 2)
        ], dtype=np.int32).reshape((-1, 1, 2))
        stroke_width = float(element.attrib.get("stroke-width", "1"))
        thickness = max(1, round(stroke_width * (scale_x + scale_y) / 2))
        cv2.polylines(canvas, [points], True, (0, 0, 0), thickness, cv2.LINE_AA)
        path_count += 1
    if path_count == 0:
        raise ValueError("SVG 中没有可回栅格化的 path")
    if not cv2.imwrite(str(output_png), canvas):
        raise OSError(f"无法写入回栅格图像：{output_png}")
    return {"png": str(output_png), "bytes": output_png.stat().st_size, "backend": "opencv-fallback", "paths": path_count}


def svg_to_raster(svg_path: str | Path, output_png: str | Path, output_width: int | None = None) -> dict:
    """优先使用 CairoSVG；缺少原生 Cairo 时回退到项目 SVG 专用 OpenCV 后端。"""

    svg_path = Path(svg_path)
    output_png = Path(output_png)
    if not svg_path.exists():
        raise FileNotFoundError(f"找不到 SVG 文件：{svg_path}")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    cairo_error = None
    try:
        import cairosvg
        kwargs = {"output_width": int(output_width)} if output_width else {}
        cairosvg.svg2png(url=str(svg_path), write_to=str(output_png), **kwargs)
        return {"png": str(output_png), "bytes": output_png.stat().st_size, "backend": "cairosvg"}
    except Exception as exc:
        cairo_error = exc

    try:
        info = _opencv_svg_to_raster(svg_path, output_png, output_width)
        info["fallback_reason"] = f"{type(cairo_error).__name__}: {cairo_error}"
        return info
    except Exception as fallback_exc:
        raise RuntimeError(
            "SVG 回栅格化失败。已优先尝试 CairoSVG，并尝试本项目 SVG 专用 OpenCV fallback。"
            f"当前解释器：{sys.executable}。CairoSVG 异常：{type(cairo_error).__name__}: {cairo_error}；"
            f"fallback 异常：{type(fallback_exc).__name__}: {fallback_exc}"
        ) from fallback_exc
