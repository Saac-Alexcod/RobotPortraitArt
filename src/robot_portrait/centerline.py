from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable, Sequence

import cv2
import numpy as np


Point = tuple[int, int]


@dataclass(frozen=True, slots=True)
class CenterlineConfig:
    """Configuration for converting raster line art into robot-friendly strokes."""

    threshold: int = 220
    spur_length_px: float = 3.0
    min_stroke_length_px: float = 2.0
    simplify_epsilon_px: float = 0.8
    max_thinning_iterations: int = 2048

    def __post_init__(self) -> None:
        if not 0 <= int(self.threshold) <= 255:
            raise ValueError("threshold 必须位于 [0, 255]")
        if float(self.spur_length_px) < 0:
            raise ValueError("spur_length_px 不能为负数")
        if float(self.min_stroke_length_px) < 0:
            raise ValueError("min_stroke_length_px 不能为负数")
        if float(self.simplify_epsilon_px) < 0:
            raise ValueError("simplify_epsilon_px 不能为负数")
        if int(self.max_thinning_iterations) <= 0:
            raise ValueError("max_thinning_iterations 必须为正整数")


@dataclass(frozen=True, slots=True)
class Stroke:
    """One pen-down trajectory."""

    points: tuple[Point, ...]
    closed: bool = False

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise ValueError("stroke 至少需要 2 个点")

    @property
    def length_px(self) -> float:
        total = polyline_length(self.points)
        if self.closed:
            total += point_distance(
                self.points[-1],
                self.points[0],
            )
        return float(total)

    @property
    def start(self) -> Point:
        return self.points[0]

    @property
    def end(self) -> Point:
        return self.points[0] if self.closed else self.points[-1]

    def reversed(self) -> "Stroke":
        return Stroke(
            points=tuple(reversed(self.points)),
            closed=self.closed,
        )


@dataclass(frozen=True, slots=True)
class CenterlineResult:
    """Centerline skeleton and ordered robot strokes."""

    skeleton: np.ndarray
    strokes: tuple[Stroke, ...]
    skeleton_pixels: int
    pen_down_length_px: float
    pen_up_length_px: float

    def __post_init__(self) -> None:
        skeleton = np.asarray(self.skeleton)
        if skeleton.ndim != 2 or skeleton.dtype != np.uint8:
            raise ValueError("skeleton 必须是 uint8 二维图像")
        values = set(np.unique(skeleton).tolist())
        if not values.issubset({0, 255}):
            raise ValueError("skeleton 必须是 0/255 二值图")

    def to_metadata(self) -> dict[str, object]:
        return {
            "mode": "centerline",
            "stroke_count": len(self.strokes),
            "skeleton_pixels": int(self.skeleton_pixels),
            "pen_down_length_px": float(self.pen_down_length_px),
            "pen_up_length_px": float(self.pen_up_length_px),
            "closed_stroke_count": sum(
                1 for stroke in self.strokes if stroke.closed
            ),
        }


def point_distance(a: Point, b: Point) -> float:
    return hypot(
        float(a[0] - b[0]),
        float(a[1] - b[1]),
    )


def polyline_length(points: Sequence[Point]) -> float:
    if len(points) < 2:
        return 0.0
    return float(
        sum(
            point_distance(a, b)
            for a, b in zip(points, points[1:])
        )
    )


def raster_to_ink_mask(
    image: np.ndarray,
    *,
    threshold: int = 220,
) -> np.ndarray:
    """Return 0/1 foreground where dark line-art pixels are considered ink."""

    array = np.asarray(image)

    if array.ndim == 3 and array.shape[2] == 3:
        gray = cv2.cvtColor(
            array,
            cv2.COLOR_BGR2GRAY,
        )
    elif array.ndim == 2:
        gray = array
    else:
        raise ValueError(
            "image 必须是 HxW 灰度图或 HxWx3 BGR 图像"
        )

    if gray.dtype != np.uint8 or gray.size == 0:
        raise ValueError("image 必须是非空 uint8 图像")

    if not 0 <= int(threshold) <= 255:
        raise ValueError("threshold 必须位于 [0, 255]")

    return (gray < int(threshold)).astype(np.uint8)


def _transition_count(neighbors: Sequence[np.uint8]) -> int:
    ring = list(neighbors) + [neighbors[0]]
    return sum(
        1
        for a, b in zip(ring, ring[1:])
        if int(a) == 0 and int(b) == 1
    )


def zhang_suen_thinning(
    foreground: np.ndarray,
    *,
    max_iterations: int = 2048,
) -> np.ndarray:
    """Thin a 0/1 foreground mask to a one-pixel skeleton.

    This is the classic two-subiteration Zhang-Suen thinning rule.
    """

    image = np.asarray(foreground)

    if (
        image.ndim != 2
        or image.size == 0
    ):
        raise ValueError("foreground 必须是非空二维数组")

    work = (image > 0).astype(np.uint8)

    # Padding makes border behavior explicit and avoids wraparound.
    padded = np.pad(
        work,
        1,
        mode="constant",
        constant_values=0,
    )

    max_iterations = int(max_iterations)
    if max_iterations <= 0:
        raise ValueError("max_iterations 必须为正整数")

    for _ in range(max_iterations):
        changed = False

        for phase in (0, 1):
            remove: list[tuple[int, int]] = []

            ys, xs = np.nonzero(padded)

            for y, x in zip(ys.tolist(), xs.tolist()):
                if (
                    y == 0
                    or x == 0
                    or y == padded.shape[0] - 1
                    or x == padded.shape[1] - 1
                ):
                    continue

                p2 = padded[y - 1, x]
                p3 = padded[y - 1, x + 1]
                p4 = padded[y, x + 1]
                p5 = padded[y + 1, x + 1]
                p6 = padded[y + 1, x]
                p7 = padded[y + 1, x - 1]
                p8 = padded[y, x - 1]
                p9 = padded[y - 1, x - 1]

                neighbors = (
                    p2, p3, p4, p5,
                    p6, p7, p8, p9,
                )

                count = sum(int(v) for v in neighbors)

                if count < 2 or count > 6:
                    continue

                if _transition_count(neighbors) != 1:
                    continue

                if phase == 0:
                    if int(p2 * p4 * p6) != 0:
                        continue
                    if int(p4 * p6 * p8) != 0:
                        continue
                else:
                    if int(p2 * p4 * p8) != 0:
                        continue
                    if int(p2 * p6 * p8) != 0:
                        continue

                remove.append((y, x))

            if remove:
                changed = True
                for y, x in remove:
                    padded[y, x] = 0

        if not changed:
            break
    else:
        raise RuntimeError(
            "thinning 未在 max_iterations 内收敛"
        )

    return padded[1:-1, 1:-1].astype(np.uint8)


def _raw_neighbors(
    skeleton: np.ndarray,
    point: Point,
) -> list[Point]:
    x, y = point
    height, width = skeleton.shape
    result: list[Point] = []

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue

            nx = x + dx
            ny = y + dy

            if (
                nx < 0
                or ny < 0
                or nx >= width
                or ny >= height
                or skeleton[ny, nx] == 0
            ):
                continue

            # Avoid redundant diagonal shortcut edges at right-angle corners.
            # A genuine diagonal line still connects because both orthogonal
            # bridge pixels are absent.
            if dx != 0 and dy != 0:
                if (
                    skeleton[y, nx] != 0
                    or skeleton[ny, x] != 0
                ):
                    continue

            result.append((nx, ny))

    result.sort(key=lambda p: (p[1], p[0]))
    return result


def skeleton_degrees(
    skeleton: np.ndarray,
) -> dict[Point, int]:
    binary = (np.asarray(skeleton) > 0).astype(np.uint8)
    ys, xs = np.nonzero(binary)
    return {
        (int(x), int(y)): len(
            _raw_neighbors(
                binary,
                (int(x), int(y)),
            )
        )
        for y, x in zip(ys.tolist(), xs.tolist())
    }


def _edge_key(a: Point, b: Point) -> tuple[Point, Point]:
    return (a, b) if a <= b else (b, a)


def _walk_from_endpoint(
    skeleton: np.ndarray,
    start: Point,
    degrees: dict[Point, int],
) -> list[Point]:
    path = [start]
    previous: Point | None = None
    current = start

    while True:
        candidates = [
            p
            for p in _raw_neighbors(skeleton, current)
            if p != previous
        ]

        if not candidates:
            return path

        next_point = candidates[0]
        path.append(next_point)

        previous = current
        current = next_point

        if degrees.get(current, 0) != 2:
            return path


def prune_short_spurs(
    skeleton: np.ndarray,
    *,
    max_length_px: float,
) -> np.ndarray:
    """Remove endpoint-to-junction branches no longer than max_length_px."""

    binary = (np.asarray(skeleton) > 0).astype(np.uint8)

    if float(max_length_px) <= 0:
        return binary

    while True:
        degrees = skeleton_degrees(binary)
        endpoints = sorted(
            (
                p
                for p, degree in degrees.items()
                if degree == 1
            ),
            key=lambda p: (p[1], p[0]),
        )

        to_remove: set[Point] = set()

        for endpoint in endpoints:
            if endpoint in to_remove:
                continue

            path = _walk_from_endpoint(
                binary,
                endpoint,
                degrees,
            )

            if len(path) < 2:
                continue

            terminal = path[-1]
            terminal_degree = degrees.get(
                terminal,
                0,
            )

            if terminal_degree < 3:
                continue

            length = polyline_length(path)

            if length <= float(max_length_px):
                # Keep the junction itself so the main structure survives.
                to_remove.update(path[:-1])

        if not to_remove:
            break

        for x, y in to_remove:
            binary[y, x] = 0

    return binary


def trace_skeleton_strokes(
    skeleton: np.ndarray,
) -> tuple[Stroke, ...]:
    """Trace all skeleton graph edges into open branches and closed loops."""

    binary = (np.asarray(skeleton) > 0).astype(np.uint8)

    if binary.ndim != 2:
        raise ValueError("skeleton 必须是二维数组")

    degrees = skeleton_degrees(binary)
    visited_edges: set[tuple[Point, Point]] = set()
    traced: list[Stroke] = []

    nodes = sorted(
        (
            p
            for p, degree in degrees.items()
            if degree != 2
        ),
        key=lambda p: (p[1], p[0]),
    )

    # Trace branches between endpoints/junctions.
    for node in nodes:
        for neighbor in _raw_neighbors(binary, node):
            edge = _edge_key(node, neighbor)

            if edge in visited_edges:
                continue

            path = [node]
            previous = node
            current = neighbor
            visited_edges.add(edge)

            while True:
                path.append(current)

                if degrees.get(current, 0) != 2:
                    break

                candidates = [
                    p
                    for p in _raw_neighbors(binary, current)
                    if p != previous
                ]

                if not candidates:
                    break

                next_point = candidates[0]
                visited_edges.add(
                    _edge_key(current, next_point)
                )
                previous, current = current, next_point

            if len(path) >= 2:
                traced.append(
                    Stroke(
                        points=tuple(path),
                        closed=False,
                    )
                )

    # Any unvisited edges belong to degree-2 cycles.
    all_points = sorted(
        degrees,
        key=lambda p: (p[1], p[0]),
    )

    for start in all_points:
        for neighbor in _raw_neighbors(binary, start):
            first_edge = _edge_key(start, neighbor)

            if first_edge in visited_edges:
                continue

            path = [start]
            previous = start
            current = neighbor
            visited_edges.add(first_edge)

            guard = 0
            max_guard = max(1, len(degrees) * 2)

            while guard < max_guard:
                guard += 1

                if current == start:
                    break

                path.append(current)

                candidates = [
                    p
                    for p in _raw_neighbors(binary, current)
                    if p != previous
                ]

                if not candidates:
                    break

                unvisited = [
                    p
                    for p in candidates
                    if _edge_key(current, p)
                    not in visited_edges
                ]

                next_point = (
                    unvisited[0]
                    if unvisited
                    else candidates[0]
                )

                edge = _edge_key(current, next_point)

                if (
                    edge in visited_edges
                    and next_point != start
                ):
                    break

                visited_edges.add(edge)
                previous, current = current, next_point

            closed = (
                current == start
                and len(path) >= 3
            )

            if len(path) >= 2:
                traced.append(
                    Stroke(
                        points=tuple(path),
                        closed=closed,
                    )
                )

    return tuple(traced)


def simplify_stroke(
    stroke: Stroke,
    *,
    epsilon_px: float,
) -> Stroke:
    epsilon_px = float(epsilon_px)

    if epsilon_px <= 0 or len(stroke.points) <= 2:
        return stroke

    contour = np.asarray(
        stroke.points,
        dtype=np.float32,
    ).reshape((-1, 1, 2))

    simplified = cv2.approxPolyDP(
        contour,
        epsilon_px,
        stroke.closed,
    ).reshape((-1, 2))

    points = tuple(
        (int(round(float(x))), int(round(float(y))))
        for x, y in simplified
    )

    # approxPolyDP can collapse tiny loops too aggressively.
    if stroke.closed and len(points) < 3:
        return stroke

    if not stroke.closed:
        if points[0] != stroke.points[0]:
            points = (stroke.points[0],) + points
        if points[-1] != stroke.points[-1]:
            points = points + (stroke.points[-1],)

    deduped: list[Point] = []
    for point in points:
        if not deduped or deduped[-1] != point:
            deduped.append(point)

    if len(deduped) < 2:
        return stroke

    return Stroke(
        points=tuple(deduped),
        closed=stroke.closed,
    )


def filter_short_strokes(
    strokes: Iterable[Stroke],
    *,
    min_length_px: float,
) -> tuple[Stroke, ...]:
    return tuple(
        stroke
        for stroke in strokes
        if stroke.length_px >= float(min_length_px)
    )


def _rotate_closed_stroke(
    stroke: Stroke,
    index: int,
) -> Stroke:
    if not stroke.closed:
        return stroke

    points = stroke.points
    index %= len(points)
    rotated = (
        points[index:]
        + points[:index]
    )

    return Stroke(
        points=rotated,
        closed=True,
    )


def order_strokes_greedy(
    strokes: Sequence[Stroke],
    *,
    start_point: Point = (0, 0),
) -> tuple[Stroke, ...]:
    """Greedily choose stroke order and direction to reduce pen-up travel."""

    remaining = list(strokes)
    ordered: list[Stroke] = []
    cursor = start_point

    while remaining:
        best: tuple[
            float,
            int,
            int,
            bool,
        ] | None = None

        for index, stroke in enumerate(remaining):
            if stroke.closed:
                for vertex_index, point in enumerate(stroke.points):
                    candidate = (
                        point_distance(cursor, point),
                        index,
                        vertex_index,
                        False,
                    )
                    if best is None or candidate < best:
                        best = candidate
            else:
                candidate_start = (
                    point_distance(cursor, stroke.start),
                    index,
                    0,
                    False,
                )
                candidate_end = (
                    point_distance(cursor, stroke.end),
                    index,
                    0,
                    True,
                )

                for candidate in (
                    candidate_start,
                    candidate_end,
                ):
                    if best is None or candidate < best:
                        best = candidate

        assert best is not None
        _, index, vertex_index, reverse = best
        chosen = remaining.pop(index)

        if chosen.closed:
            chosen = _rotate_closed_stroke(
                chosen,
                vertex_index,
            )
        elif reverse:
            chosen = chosen.reversed()

        ordered.append(chosen)
        cursor = chosen.end

    return tuple(ordered)


def pen_up_length(
    strokes: Sequence[Stroke],
) -> float:
    if len(strokes) < 2:
        return 0.0

    return float(
        sum(
            point_distance(
                previous.end,
                current.start,
            )
            for previous, current
            in zip(strokes, strokes[1:])
        )
    )


def extract_centerline_strokes(
    image: np.ndarray,
    config: CenterlineConfig | None = None,
) -> CenterlineResult:
    config = config or CenterlineConfig()

    ink = raster_to_ink_mask(
        image,
        threshold=config.threshold,
    )

    skeleton01 = zhang_suen_thinning(
        ink,
        max_iterations=(
            config.max_thinning_iterations
        ),
    )

    skeleton01 = prune_short_spurs(
        skeleton01,
        max_length_px=(
            config.spur_length_px
        ),
    )

    strokes = trace_skeleton_strokes(
        skeleton01
    )

    strokes = tuple(
        simplify_stroke(
            stroke,
            epsilon_px=(
                config.simplify_epsilon_px
            ),
        )
        for stroke in strokes
    )

    strokes = filter_short_strokes(
        strokes,
        min_length_px=(
            config.min_stroke_length_px
        ),
    )

    strokes = order_strokes_greedy(
        strokes
    )

    skeleton = (
        skeleton01
        * 255
    ).astype(
        np.uint8,
        copy=False,
    )

    return CenterlineResult(
        skeleton=skeleton,
        strokes=strokes,
        skeleton_pixels=int(
            np.count_nonzero(
                skeleton01
            )
        ),
        pen_down_length_px=float(
            sum(
                stroke.length_px
                for stroke in strokes
            )
        ),
        pen_up_length_px=(
            pen_up_length(
                strokes
            )
        ),
    )
