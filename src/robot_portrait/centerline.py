from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import acos, degrees, hypot
from typing import Hashable, Iterable, Sequence

import cv2
import numpy as np


Point = tuple[int, int]
_GraphNode = Hashable


@dataclass(frozen=True, slots=True)
class CenterlineConfig:
    """Configuration for converting raster line art into robot-friendly strokes."""

    threshold: int = 220
    spur_length_px: float = 3.0
    min_stroke_length_px: float = 2.0
    simplify_epsilon_px: float = 0.8
    max_thinning_iterations: int = 2048

    # E2.1 graph refinement.
    junction_cluster_radius_px: int = 1
    merge_angle_deg: float = 35.0
    merge_lookahead_px: float = 6.0

    # E2.3 inter-trail travel optimization.
    ordering_2opt_max_passes: int = 16
    ordering_2opt_min_improvement_px: float = 0.01

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
        if int(self.junction_cluster_radius_px) < 0:
            raise ValueError("junction_cluster_radius_px 不能为负数")
        if not 0 <= float(self.merge_angle_deg) <= 180:
            raise ValueError("merge_angle_deg 必须位于 [0, 180]")
        if float(self.merge_lookahead_px) <= 0:
            raise ValueError("merge_lookahead_px 必须为正数")
        if int(self.ordering_2opt_max_passes) < 0:
            raise ValueError("ordering_2opt_max_passes 不能为负数")
        if float(self.ordering_2opt_min_improvement_px) < 0:
            raise ValueError(
                "ordering_2opt_min_improvement_px 不能为负数"
            )


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
class SkeletonGraphStats:
    endpoint_count: int
    junction_pixel_count: int
    junction_cluster_count: int
    graph_node_count: int


@dataclass(frozen=True, slots=True)
class TrailPlanStats:
    input_edge_count: int
    connected_component_count: int
    odd_vertex_count: int
    theoretical_min_trail_count: int
    output_trail_count: int
    edge_reuse_count: int


@dataclass(frozen=True, slots=True)
class RouteOptimizationStats:
    initial_pen_up_length_px: float
    optimized_pen_up_length_px: float
    improvement_px: float
    improvement_ratio: float
    move_count: int
    pass_count: int
    skipped_closed_candidates: int


@dataclass(frozen=True, slots=True)
class CenterlineResult:
    """Centerline skeleton and ordered robot strokes."""

    skeleton: np.ndarray
    strokes: tuple[Stroke, ...]
    skeleton_pixels: int
    pen_down_length_px: float
    pen_up_length_px: float

    # E2.1 diagnostics.
    endpoint_count: int = 0
    junction_pixel_count: int = 0
    junction_cluster_count: int = 0
    graph_node_count: int = 0
    raw_stroke_count: int = 0
    merge_count: int = 0

    connected_component_count: int = 0
    odd_vertex_count: int = 0
    theoretical_min_trail_count: int = 0
    edge_reuse_count: int = 0

    ordering_mode: str = "greedy"
    pen_up_greedy_px: float = 0.0
    pen_up_optimized_px: float = 0.0
    two_opt_improvement_px: float = 0.0
    two_opt_improvement_ratio: float = 0.0
    two_opt_move_count: int = 0
    two_opt_pass_count: int = 0
    two_opt_skipped_closed_candidates: int = 0

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
            "raw_stroke_count": int(self.raw_stroke_count),
            "merge_count": int(self.merge_count),
            "planning_mode": "minimum_lift_trail_cover",
            "connected_component_count": int(
                self.connected_component_count
            ),
            "odd_vertex_count": int(
                self.odd_vertex_count
            ),
            "theoretical_min_trail_count": int(
                self.theoretical_min_trail_count
            ),
            "edge_reuse_count": int(
                self.edge_reuse_count
            ),
            "ordering_mode": self.ordering_mode,
            "pen_up_greedy_px": float(
                self.pen_up_greedy_px
            ),
            "pen_up_optimized_px": float(
                self.pen_up_optimized_px
            ),
            "two_opt_improvement_px": float(
                self.two_opt_improvement_px
            ),
            "two_opt_improvement_ratio": float(
                self.two_opt_improvement_ratio
            ),
            "two_opt_move_count": int(
                self.two_opt_move_count
            ),
            "two_opt_pass_count": int(
                self.two_opt_pass_count
            ),
            "two_opt_skipped_closed_candidates": int(
                self.two_opt_skipped_closed_candidates
            ),
            "skeleton_pixels": int(self.skeleton_pixels),
            "endpoint_count": int(self.endpoint_count),
            "junction_pixel_count": int(self.junction_pixel_count),
            "junction_cluster_count": int(self.junction_cluster_count),
            "graph_node_count": int(self.graph_node_count),
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


def zhang_suen_thinning(
    foreground: np.ndarray,
    *,
    max_iterations: int = 2048,
) -> np.ndarray:
    """Thin a 0/1 foreground mask to a one-pixel Zhang-Suen skeleton.

    E2.1 keeps the same two-subiteration rules as E1 but evaluates each
    subiteration with NumPy slices instead of a Python loop over every
    foreground pixel. This preserves the algorithm while making 512x512
    portrait line art much faster.
    """

    image = np.asarray(foreground)

    if image.ndim != 2 or image.size == 0:
        raise ValueError("foreground 必须是非空二维数组")

    max_iterations = int(max_iterations)
    if max_iterations <= 0:
        raise ValueError("max_iterations 必须为正整数")

    padded = np.pad(
        (image > 0).astype(np.uint8),
        1,
        mode="constant",
        constant_values=0,
    )

    for _ in range(max_iterations):
        changed = False

        for phase in (0, 1):
            p1 = padded[1:-1, 1:-1]
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]

            count = (
                p2 + p3 + p4 + p5
                + p6 + p7 + p8 + p9
            )

            transitions = (
                ((p2 == 0) & (p3 == 1)).astype(np.uint8)
                + ((p3 == 0) & (p4 == 1))
                + ((p4 == 0) & (p5 == 1))
                + ((p5 == 0) & (p6 == 1))
                + ((p6 == 0) & (p7 == 1))
                + ((p7 == 0) & (p8 == 1))
                + ((p8 == 0) & (p9 == 1))
                + ((p9 == 0) & (p2 == 1))
            )

            remove = (
                (p1 == 1)
                & (count >= 2)
                & (count <= 6)
                & (transitions == 1)
            )

            if phase == 0:
                remove &= (
                    (p2 * p4 * p6 == 0)
                    & (p4 * p6 * p8 == 0)
                )
            else:
                remove &= (
                    (p2 * p4 * p8 == 0)
                    & (p2 * p6 * p8 == 0)
                )

            if np.any(remove):
                p1[remove] = 0
                changed = True

        if not changed:
            return padded[1:-1, 1:-1].astype(np.uint8)

    raise RuntimeError(
        "thinning 未在 max_iterations 内收敛"
    )


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
            # Genuine diagonal lines remain connected because both orthogonal
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


def _walk_from_endpoint(
    skeleton: np.ndarray,
    start: Point,
    degrees_map: dict[Point, int],
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

        if degrees_map.get(current, 0) != 2:
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
        degrees_map = skeleton_degrees(binary)
        endpoints = sorted(
            (
                p
                for p, degree in degrees_map.items()
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
                degrees_map,
            )

            if len(path) < 2:
                continue

            terminal = path[-1]
            terminal_degree = degrees_map.get(
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


def _junction_regions(
    skeleton: np.ndarray,
    *,
    radius_px: int,
) -> tuple[
    dict[Point, int],
    dict[int, tuple[Point, ...]],
    dict[Point, int],
    dict[int, Point],
    int,
]:
    """Group a local knot of junction pixels into one logical super-node."""

    binary = (np.asarray(skeleton) > 0).astype(np.uint8)
    degrees_map = skeleton_degrees(binary)

    seed = np.zeros_like(binary, dtype=np.uint8)

    for (x, y), degree in degrees_map.items():
        if degree >= 3:
            seed[y, x] = 1

    junction_pixel_count = int(
        np.count_nonzero(seed)
    )

    if (
        junction_pixel_count > 0
        and int(radius_px) > 0
    ):
        size = 2 * int(radius_px) + 1
        kernel = np.ones(
            (size, size),
            dtype=np.uint8,
        )

        region = cv2.dilate(
            seed,
            kernel,
            iterations=1,
        )

        # Contract only actual skeleton pixels; dilation merely defines the
        # local neighborhood that belongs to the same physical junction.
        region = (
            region
            & binary
        ).astype(np.uint8)

    else:
        region = seed

    count, labels = cv2.connectedComponents(
        region,
        connectivity=8,
    )

    seed_points = {
        point
        for point, degree
        in degrees_map.items()
        if degree >= 3
    }

    clusters: dict[
        int,
        tuple[Point, ...],
    ] = {}
    point_to_cluster: dict[
        Point,
        int,
    ] = {}
    representatives: dict[
        int,
        Point,
    ] = {}

    for label in range(1, count):
        ys, xs = np.where(
            labels == label
        )

        points = tuple(
            sorted(
                (
                    (int(x), int(y))
                    for y, x
                    in zip(
                        ys.tolist(),
                        xs.tolist(),
                    )
                ),
                key=lambda p: (
                    p[1],
                    p[0],
                ),
            )
        )

        if not points:
            continue

        actual_junctions = [
            point
            for point in points
            if point in seed_points
        ]

        representative_pool = (
            actual_junctions
            if actual_junctions
            else list(points)
        )

        mean_x = sum(
            p[0]
            for p in representative_pool
        ) / len(representative_pool)

        mean_y = sum(
            p[1]
            for p in representative_pool
        ) / len(representative_pool)

        representative = min(
            representative_pool,
            key=lambda p: (
                (p[0] - mean_x) ** 2
                + (p[1] - mean_y) ** 2,
                p[1],
                p[0],
            ),
        )

        clusters[label] = points
        representatives[label] = (
            representative
        )

        for point in points:
            point_to_cluster[
                point
            ] = label

    return (
        degrees_map,
        clusters,
        point_to_cluster,
        representatives,
        junction_pixel_count,
    )


def _node_sort_key(
    node: _GraphNode,
    coordinates: dict[
        _GraphNode,
        Point,
    ],
) -> tuple[int, int, str]:
    x, y = coordinates[node]
    return (
        y,
        x,
        repr(node),
    )


def _edge_key(
    a: _GraphNode,
    b: _GraphNode,
) -> frozenset[_GraphNode]:
    return frozenset(
        (a, b)
    )


def _build_contracted_graph(
    skeleton: np.ndarray,
    *,
    junction_cluster_radius_px: int,
) -> tuple[
    dict[_GraphNode, set[_GraphNode]],
    dict[_GraphNode, Point],
    SkeletonGraphStats,
]:
    """Build a graph where each local junction knot is one super-node."""

    binary = (np.asarray(skeleton) > 0).astype(np.uint8)

    (
        degrees_map,
        clusters,
        point_to_cluster,
        representatives,
        junction_pixel_count,
    ) = _junction_regions(
        binary,
        radius_px=(
            junction_cluster_radius_px
        ),
    )

    adjacency: dict[
        _GraphNode,
        set[_GraphNode],
    ] = defaultdict(set)

    coordinates: dict[
        _GraphNode,
        Point,
    ] = {}

    def node_for(
        point: Point,
    ) -> _GraphNode:
        cluster = point_to_cluster.get(
            point
        )

        if cluster is not None:
            return (
                "junction",
                cluster,
            )

        return (
            "pixel",
            point[0],
            point[1],
        )

    for cluster, representative in (
        representatives.items()
    ):
        coordinates[
            (
                "junction",
                cluster,
            )
        ] = representative

    for point in degrees_map:
        node = node_for(
            point
        )

        coordinates.setdefault(
            node,
            point,
        )

        adjacency.setdefault(
            node,
            set(),
        )

        for neighbor in _raw_neighbors(
            binary,
            point,
        ):
            other = node_for(
                neighbor
            )

            if node == other:
                # Internal pixels of one physical junction have already been
                # contracted to the same super-node.
                continue

            adjacency[
                node
            ].add(
                other
            )

            adjacency[
                other
            ].add(
                node
            )

    endpoint_count = sum(
        1
        for node
        in adjacency
        if len(adjacency[node]) == 1
    )

    stats = SkeletonGraphStats(
        endpoint_count=endpoint_count,
        junction_pixel_count=(
            junction_pixel_count
        ),
        junction_cluster_count=len(
            clusters
        ),
        graph_node_count=len(
            adjacency
        ),
    )

    return (
        adjacency,
        coordinates,
        stats,
    )


def _trace_contracted_graph(
    skeleton: np.ndarray,
    *,
    junction_cluster_radius_px: int,
) -> tuple[
    tuple[Stroke, ...],
    SkeletonGraphStats,
]:
    adjacency, coordinates, stats = (
        _build_contracted_graph(
            skeleton,
            junction_cluster_radius_px=(
                junction_cluster_radius_px
            ),
        )
    )

    if not adjacency:
        return (
            tuple(),
            stats,
        )

    visited_edges: set[
        frozenset[_GraphNode]
    ] = set()

    traced: list[
        Stroke
    ] = []

    key = lambda node: _node_sort_key(
        node,
        coordinates,
    )

    nodes = sorted(
        adjacency,
        key=key,
    )

    starts = [
        node
        for node in nodes
        if len(
            adjacency[node]
        ) != 2
    ]

    def points_from_nodes(
        path: Sequence[_GraphNode],
    ) -> tuple[Point, ...]:
        points: list[
            Point
        ] = []

        for node in path:
            point = coordinates[
                node
            ]

            if (
                not points
                or points[-1]
                != point
            ):
                points.append(
                    point
                )

        return tuple(
            points
        )

    # Open branches and chains.
    for start in starts:
        for neighbor in sorted(
            adjacency[start],
            key=key,
        ):
            edge = _edge_key(
                start,
                neighbor,
            )

            if edge in visited_edges:
                continue

            path = [
                start
            ]

            previous = start
            current = neighbor
            visited_edges.add(
                edge
            )

            while True:
                path.append(
                    current
                )

                if len(
                    adjacency[current]
                ) != 2:
                    break

                candidates = [
                    node
                    for node
                    in adjacency[current]
                    if node != previous
                ]

                if not candidates:
                    break

                next_node = candidates[0]
                next_edge = _edge_key(
                    current,
                    next_node,
                )

                if (
                    next_edge
                    in visited_edges
                ):
                    break

                visited_edges.add(
                    next_edge
                )

                previous, current = (
                    current,
                    next_node,
                )

            points = points_from_nodes(
                path
            )

            if len(points) >= 2:
                traced.append(
                    Stroke(
                        points=points,
                        closed=False,
                    )
                )

    # Remaining edges are degree-2 cycles.
    for start in nodes:
        for neighbor in sorted(
            adjacency[start],
            key=key,
        ):
            first_edge = _edge_key(
                start,
                neighbor,
            )

            if (
                first_edge
                in visited_edges
            ):
                continue

            path = [
                start
            ]

            previous = start
            current = neighbor
            visited_edges.add(
                first_edge
            )

            guard = 0
            max_guard = max(
                1,
                len(nodes) * 2,
            )

            while guard < max_guard:
                guard += 1

                if current == start:
                    break

                path.append(
                    current
                )

                candidates = [
                    node
                    for node
                    in adjacency[current]
                    if node != previous
                ]

                if not candidates:
                    break

                unvisited = [
                    node
                    for node
                    in candidates
                    if _edge_key(
                        current,
                        node,
                    )
                    not in visited_edges
                ]

                next_node = (
                    unvisited[0]
                    if unvisited
                    else candidates[0]
                )

                next_edge = _edge_key(
                    current,
                    next_node,
                )

                if (
                    next_edge
                    in visited_edges
                    and next_node
                    != start
                ):
                    break

                visited_edges.add(
                    next_edge
                )

                previous, current = (
                    current,
                    next_node,
                )

            points = points_from_nodes(
                path
            )

            closed = (
                current == start
                and len(points) >= 3
            )

            if len(points) >= 2:
                traced.append(
                    Stroke(
                        points=points,
                        closed=closed,
                    )
                )

    return (
        tuple(traced),
        stats,
    )


def trace_skeleton_strokes(
    skeleton: np.ndarray,
    *,
    junction_cluster_radius_px: int = 1,
) -> tuple[Stroke, ...]:
    """Trace skeleton graph edges after contracting local junction knots."""

    binary = (np.asarray(skeleton) > 0).astype(np.uint8)

    if binary.ndim != 2:
        raise ValueError("skeleton 必须是二维数组")

    strokes, _ = (
        _trace_contracted_graph(
            binary,
            junction_cluster_radius_px=(
                junction_cluster_radius_px
            ),
        )
    )

    return strokes


def simplify_stroke(
    stroke: Stroke,
    *,
    epsilon_px: float,
) -> Stroke:
    epsilon_px = float(
        epsilon_px
    )

    if (
        epsilon_px <= 0
        or len(
            stroke.points
        ) <= 2
    ):
        return stroke

    contour = np.asarray(
        stroke.points,
        dtype=np.float32,
    ).reshape(
        (-1, 1, 2)
    )

    simplified = cv2.approxPolyDP(
        contour,
        epsilon_px,
        stroke.closed,
    ).reshape(
        (-1, 2)
    )

    points = tuple(
        (
            int(
                round(
                    float(x)
                )
            ),
            int(
                round(
                    float(y)
                )
            ),
        )
        for x, y
        in simplified
    )

    if (
        stroke.closed
        and len(points) < 3
    ):
        return stroke

    if not stroke.closed:
        if (
            points[0]
            != stroke.points[0]
        ):
            points = (
                stroke.points[0],
            ) + points

        if (
            points[-1]
            != stroke.points[-1]
        ):
            points = points + (
                stroke.points[-1],
            )

    deduped: list[
        Point
    ] = []

    for point in points:
        if (
            not deduped
            or deduped[-1]
            != point
        ):
            deduped.append(
                point
            )

    if len(deduped) < 2:
        return stroke

    return Stroke(
        points=tuple(
            deduped
        ),
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
        if stroke.length_px
        >= float(min_length_px)
    )


def _stroke_outward_direction(
    stroke: Stroke,
    shared: Point,
    *,
    lookahead_px: float,
) -> tuple[float, float] | None:
    if stroke.closed:
        return None

    if stroke.start == shared:
        points = stroke.points
    elif stroke.end == shared:
        points = tuple(
            reversed(
                stroke.points
            )
        )
    else:
        return None

    origin = points[0]
    traveled = 0.0
    target = origin

    for a, b in zip(
        points,
        points[1:],
    ):
        traveled += point_distance(
            a,
            b,
        )

        target = b

        if (
            traveled
            >= float(
                lookahead_px
            )
        ):
            break

    dx = float(
        target[0]
        - origin[0]
    )
    dy = float(
        target[1]
        - origin[1]
    )

    norm = hypot(
        dx,
        dy,
    )

    if norm <= 1e-12:
        return None

    return (
        dx / norm,
        dy / norm,
    )


def _straight_deviation_deg(
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    dot = (
        first[0] * second[0]
        + first[1] * second[1]
    )

    dot = min(
        1.0,
        max(
            -1.0,
            dot,
        ),
    )

    angle = degrees(
        acos(
            dot
        )
    )

    return abs(
        180.0
        - angle
    )


def _merge_two_strokes(
    first: Stroke,
    second: Stroke,
    *,
    shared: Point,
) -> Stroke:
    if first.closed or second.closed:
        raise ValueError(
            "不能合并已闭合 stroke"
        )

    if first.end != shared:
        if first.start == shared:
            first = (
                first.reversed()
            )
        else:
            raise ValueError(
                "first 不经过 shared 端点"
            )

    if second.start != shared:
        if second.end == shared:
            second = (
                second.reversed()
            )
        else:
            raise ValueError(
                "second 不经过 shared 端点"
            )

    points = (
        first.points
        + second.points[1:]
    )

    closed = (
        len(points) >= 4
        and points[0]
        == points[-1]
    )

    if closed:
        points = points[:-1]

    return Stroke(
        points=points,
        closed=closed,
    )


def merge_collinear_strokes(
    strokes: Sequence[Stroke],
    *,
    max_turn_deviation_deg: float,
    lookahead_px: float,
) -> tuple[
    tuple[Stroke, ...],
    int,
]:
    """Continue the most collinear stroke pair through shared junctions.

    At a T junction this typically joins the straight-through pair and leaves
    the side branch as a separate pen-down stroke. At an X junction it can
    join two independent straight-through pairs. Nearby endpoints that do not
    share the same contracted graph node are never connected.
    """

    max_turn_deviation_deg = float(
        max_turn_deviation_deg
    )
    lookahead_px = float(
        lookahead_px
    )

    if not 0 <= max_turn_deviation_deg <= 180:
        raise ValueError(
            "max_turn_deviation_deg 必须位于 [0, 180]"
        )

    if lookahead_px <= 0:
        raise ValueError(
            "lookahead_px 必须为正数"
        )

    current = list(
        strokes
    )
    merge_count = 0

    while True:
        endpoint_index: dict[
            Point,
            list[int],
        ] = defaultdict(
            list
        )

        for index, stroke in enumerate(
            current
        ):
            if stroke.closed:
                continue

            endpoint_index[
                stroke.start
            ].append(
                index
            )

            if (
                stroke.end
                != stroke.start
            ):
                endpoint_index[
                    stroke.end
                ].append(
                    index
                )

        candidates: list[
            tuple[
                float,
                int,
                int,
                int,
                int,
                Point,
            ]
        ] = []

        for shared, indices in (
            endpoint_index.items()
        ):
            if len(indices) < 2:
                continue

            for left_position in range(
                len(indices)
            ):
                for right_position in range(
                    left_position + 1,
                    len(indices),
                ):
                    left = indices[
                        left_position
                    ]
                    right = indices[
                        right_position
                    ]

                    if left == right:
                        continue

                    left_direction = (
                        _stroke_outward_direction(
                            current[left],
                            shared,
                            lookahead_px=(
                                lookahead_px
                            ),
                        )
                    )

                    right_direction = (
                        _stroke_outward_direction(
                            current[right],
                            shared,
                            lookahead_px=(
                                lookahead_px
                            ),
                        )
                    )

                    if (
                        left_direction
                        is None
                        or right_direction
                        is None
                    ):
                        continue

                    deviation = (
                        _straight_deviation_deg(
                            left_direction,
                            right_direction,
                        )
                    )

                    if (
                        deviation
                        <= max_turn_deviation_deg
                    ):
                        candidates.append(
                            (
                                deviation,
                                shared[1],
                                shared[0],
                                left,
                                right,
                                shared,
                            )
                        )

        if not candidates:
            break

        candidates.sort()

        used: set[
            int
        ] = set()

        replacements: dict[
            int,
            Stroke,
        ] = {}

        removed: set[
            int
        ] = set()

        for (
            _,
            _,
            _,
            left,
            right,
            shared,
        ) in candidates:
            if (
                left in used
                or right in used
            ):
                continue

            merged = _merge_two_strokes(
                current[left],
                current[right],
                shared=shared,
            )

            used.update(
                (
                    left,
                    right,
                )
            )

            removed.update(
                (
                    left,
                    right,
                )
            )

            # Keep the replacement at the lower original position for
            # deterministic output ordering before final robot ordering.
            keeper = min(
                left,
                right,
            )

            replacements[
                keeper
            ] = merged

            merge_count += 1

        if not removed:
            break

        next_strokes: list[
            Stroke
        ] = []

        for index, stroke in enumerate(
            current
        ):
            replacement = replacements.get(
                index
            )

            if replacement is not None:
                next_strokes.append(
                    replacement
                )
                continue

            if index not in removed:
                next_strokes.append(
                    stroke
                )

        current = next_strokes

    return (
        tuple(current),
        merge_count,
    )



@dataclass(frozen=True, slots=True)
class _PlanningEdge:
    edge_id: int
    start: Hashable
    end: Hashable
    stroke: Stroke | None
    virtual: bool = False


@dataclass(frozen=True, slots=True)
class _TraversedEdge:
    edge_id: int
    start: Hashable
    end: Hashable
    virtual: bool


def _stroke_graph_components(
    strokes: Sequence[Stroke],
) -> tuple[
    tuple[tuple[int, ...], ...],
    dict[Point, int],
]:
    if not strokes:
        return tuple(), {}

    incident: dict[Point, list[int]] = defaultdict(list)
    degree: dict[Point, int] = defaultdict(int)

    for edge_id, stroke in enumerate(strokes):
        start = stroke.start
        end = stroke.end

        incident[start].append(edge_id)
        degree[start] += 1

        if end == start:
            incident[start].append(edge_id)
            degree[start] += 1
        else:
            incident[end].append(edge_id)
            degree[end] += 1

    edge_seen: set[int] = set()
    components: list[tuple[int, ...]] = []

    for seed_edge in range(len(strokes)):
        if seed_edge in edge_seen:
            continue

        pending = [seed_edge]
        component_edges: set[int] = set()

        while pending:
            edge_id = pending.pop()

            if edge_id in component_edges:
                continue

            component_edges.add(edge_id)
            edge_seen.add(edge_id)

            stroke = strokes[edge_id]

            for vertex in (stroke.start, stroke.end):
                for neighbor_edge in incident[vertex]:
                    if neighbor_edge not in component_edges:
                        pending.append(neighbor_edge)

        components.append(tuple(sorted(component_edges)))

    return tuple(components), dict(degree)


def _hierholzer_augmented(
    edges: Sequence[_PlanningEdge],
    *,
    start_vertex: Hashable,
) -> tuple[_TraversedEdge, ...]:
    adjacency: dict[Hashable, list[int]] = defaultdict(list)
    edge_lookup = {edge.edge_id: edge for edge in edges}

    for edge in edges:
        adjacency[edge.start].append(edge.edge_id)
        adjacency[edge.end].append(edge.edge_id)

    for edge_ids in adjacency.values():
        edge_ids.sort(reverse=True)

    used: set[int] = set()
    stack_vertices: list[Hashable] = [start_vertex]
    stack_edges: list[_TraversedEdge] = []
    circuit_reversed: list[_TraversedEdge] = []

    while stack_vertices:
        vertex = stack_vertices[-1]
        edge_id: int | None = None

        while adjacency[vertex]:
            candidate = adjacency[vertex].pop()
            if candidate not in used:
                edge_id = candidate
                break

        if edge_id is None:
            stack_vertices.pop()
            if stack_edges:
                circuit_reversed.append(stack_edges.pop())
            continue

        used.add(edge_id)
        edge = edge_lookup[edge_id]

        if edge.start == vertex:
            other = edge.end
        elif edge.end == vertex:
            other = edge.start
        else:
            raise RuntimeError(
                "Hierholzer adjacency 与 edge endpoint 不一致"
            )

        stack_vertices.append(other)
        stack_edges.append(
            _TraversedEdge(
                edge_id=edge_id,
                start=vertex,
                end=other,
                virtual=edge.virtual,
            )
        )

    if len(used) != len(edges):
        raise RuntimeError("Hierholzer 未覆盖全部边")

    return tuple(reversed(circuit_reversed))


def _oriented_segment_points(
    stroke: Stroke,
    *,
    start: Point,
    end: Point,
) -> tuple[Point, ...]:
    if stroke.closed:
        if start != end or start != stroke.start:
            raise ValueError("闭合 stroke 的规划端点不一致")
        return stroke.points + (stroke.points[0],)

    if stroke.start == start and stroke.end == end:
        return stroke.points

    if stroke.start == end and stroke.end == start:
        return tuple(reversed(stroke.points))

    raise ValueError("stroke 无法按规划方向定向")


def _assemble_trail(
    traversed: Sequence[_TraversedEdge],
    *,
    edge_to_stroke: dict[int, Stroke],
) -> Stroke:
    if not traversed:
        raise ValueError("trail 至少需要一条真实边")

    points: list[Point] = []

    for item in traversed:
        if item.virtual:
            raise ValueError("assemble_trail 不接受 virtual edge")

        if not (
            isinstance(item.start, tuple)
            and len(item.start) == 2
            and all(isinstance(v, int) for v in item.start)
            and isinstance(item.end, tuple)
            and len(item.end) == 2
            and all(isinstance(v, int) for v in item.end)
        ):
            raise ValueError("真实 edge 不应连接 virtual super-node")

        start: Point = (int(item.start[0]), int(item.start[1]))
        end: Point = (int(item.end[0]), int(item.end[1]))

        segment = _oriented_segment_points(
            edge_to_stroke[item.edge_id],
            start=start,
            end=end,
        )

        if not points:
            points.extend(segment)
        else:
            if points[-1] != segment[0]:
                raise RuntimeError(
                    "trail 中相邻 edge 未共享同一 graph vertex"
                )
            points.extend(segment[1:])

    deduped: list[Point] = []
    for point in points:
        if not deduped or deduped[-1] != point:
            deduped.append(point)

    if len(deduped) < 2:
        raise RuntimeError("trail 合并后不足两个点")

    closed = len(deduped) >= 4 and deduped[0] == deduped[-1]

    if closed:
        deduped.pop()

    return Stroke(points=tuple(deduped), closed=closed)


def plan_minimum_lift_trails(
    strokes: Sequence[Stroke],
) -> tuple[
    tuple[Stroke, ...],
    TrailPlanStats,
]:
    """Minimum-count edge-disjoint trail cover for an undirected multigraph."""

    strokes = tuple(strokes)

    if not strokes:
        return (
            tuple(),
            TrailPlanStats(
                input_edge_count=0,
                connected_component_count=0,
                odd_vertex_count=0,
                theoretical_min_trail_count=0,
                output_trail_count=0,
                edge_reuse_count=0,
            ),
        )

    components, global_degree = _stroke_graph_components(strokes)
    planned: list[Stroke] = []
    odd_vertex_total = 0
    theoretical_total = 0
    real_edge_use_count = 0

    for component_index, component_edge_ids in enumerate(components):
        component_vertices: set[Point] = set()

        for edge_id in component_edge_ids:
            component_vertices.add(strokes[edge_id].start)
            component_vertices.add(strokes[edge_id].end)

        odd_vertices = sorted(
            (
                vertex
                for vertex in component_vertices
                if global_degree[vertex] % 2 == 1
            ),
            key=lambda p: (p[1], p[0]),
        )

        odd_vertex_total += len(odd_vertices)
        theoretical_total += max(1, len(odd_vertices) // 2)

        planning_edges: list[_PlanningEdge] = []
        edge_to_stroke: dict[int, Stroke] = {}
        next_edge_id = 0

        for original_edge_id in component_edge_ids:
            stroke = strokes[original_edge_id]
            planning_edges.append(
                _PlanningEdge(
                    edge_id=next_edge_id,
                    start=stroke.start,
                    end=stroke.end,
                    stroke=stroke,
                    virtual=False,
                )
            )
            edge_to_stroke[next_edge_id] = stroke
            next_edge_id += 1

        if odd_vertices:
            super_vertex = ("__trail_super__", component_index)

            for odd_vertex in odd_vertices:
                planning_edges.append(
                    _PlanningEdge(
                        edge_id=next_edge_id,
                        start=super_vertex,
                        end=odd_vertex,
                        stroke=None,
                        virtual=True,
                    )
                )
                next_edge_id += 1

            circuit = _hierholzer_augmented(
                planning_edges,
                start_vertex=super_vertex,
            )

            current: list[_TraversedEdge] = []

            for item in circuit:
                if item.virtual:
                    if current:
                        planned.append(
                            _assemble_trail(
                                current,
                                edge_to_stroke=edge_to_stroke,
                            )
                        )
                        real_edge_use_count += len(current)
                        current = []
                    continue

                current.append(item)

            if current:
                planned.append(
                    _assemble_trail(
                        current,
                        edge_to_stroke=edge_to_stroke,
                    )
                )
                real_edge_use_count += len(current)

        else:
            start_vertex = min(
                component_vertices,
                key=lambda p: (p[1], p[0]),
            )

            circuit = _hierholzer_augmented(
                planning_edges,
                start_vertex=start_vertex,
            )

            real_items = tuple(
                item for item in circuit if not item.virtual
            )

            if real_items:
                planned.append(
                    _assemble_trail(
                        real_items,
                        edge_to_stroke=edge_to_stroke,
                    )
                )
                real_edge_use_count += len(real_items)

    edge_reuse_count = max(0, real_edge_use_count - len(strokes))

    stats = TrailPlanStats(
        input_edge_count=len(strokes),
        connected_component_count=len(components),
        odd_vertex_count=odd_vertex_total,
        theoretical_min_trail_count=theoretical_total,
        output_trail_count=len(planned),
        edge_reuse_count=edge_reuse_count,
    )

    if stats.output_trail_count != stats.theoretical_min_trail_count:
        raise RuntimeError(
            "minimum-lift trail cover 未达到理论最小 trail 数"
        )

    if real_edge_use_count != len(strokes):
        raise RuntimeError(
            "minimum-lift trail cover 未恰好覆盖全部真实边一次"
        )

    return tuple(planned), stats

def _rotate_closed_stroke(
    stroke: Stroke,
    index: int,
) -> Stroke:
    if not stroke.closed:
        return stroke

    points = stroke.points
    index %= len(
        points
    )

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

    remaining = list(
        strokes
    )

    ordered: list[
        Stroke
    ] = []

    cursor = start_point

    while remaining:
        best: tuple[
            float,
            int,
            int,
            bool,
        ] | None = None

        for index, stroke in enumerate(
            remaining
        ):
            if stroke.closed:
                for (
                    vertex_index,
                    point,
                ) in enumerate(
                    stroke.points
                ):
                    candidate = (
                        point_distance(
                            cursor,
                            point,
                        ),
                        index,
                        vertex_index,
                        False,
                    )

                    if (
                        best is None
                        or candidate < best
                    ):
                        best = candidate

            else:
                candidate_start = (
                    point_distance(
                        cursor,
                        stroke.start,
                    ),
                    index,
                    0,
                    False,
                )

                candidate_end = (
                    point_distance(
                        cursor,
                        stroke.end,
                    ),
                    index,
                    0,
                    True,
                )

                for candidate in (
                    candidate_start,
                    candidate_end,
                ):
                    if (
                        best is None
                        or candidate < best
                    ):
                        best = candidate

        assert best is not None

        (
            _,
            index,
            vertex_index,
            reverse,
        ) = best

        chosen = remaining.pop(
            index
        )

        if chosen.closed:
            chosen = (
                _rotate_closed_stroke(
                    chosen,
                    vertex_index,
                )
            )
        elif reverse:
            chosen = (
                chosen.reversed()
            )

        ordered.append(
            chosen
        )
        cursor = chosen.end

    return tuple(
        ordered
    )


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
            in zip(
                strokes,
                strokes[1:],
            )
        )
    )



def _reverse_open_route_slice(
    route: Sequence[Stroke],
    left: int,
    right: int,
) -> list[Stroke]:
    """Reverse route[left:right+1] and each open stroke direction."""

    replacement = [
        stroke.reversed()
        for stroke in reversed(
            route[left : right + 1]
        )
    ]

    return (
        list(route[:left])
        + replacement
        + list(route[right + 1 :])
    )


def optimize_stroke_order_2opt(
    strokes: Sequence[Stroke],
    *,
    max_passes: int = 16,
    min_improvement_px: float = 0.01,
) -> tuple[
    tuple[Stroke, ...],
    RouteOptimizationStats,
]:
    """Improve inter-trail pen-up travel with deterministic open-path 2-opt.

    The objective is exactly ``pen_up_length``: travel between consecutive
    pen-down trails. There is no return-to-origin term.

    For an all-open slice, reversing the slice order and every stroke
    direction keeps every internal connector distance unchanged because
    Euclidean distance is symmetric. Therefore each 2-opt candidate only
    changes the left and/or right boundary connector and can be evaluated
    exactly in O(1).

    Closed strokes are valid inputs but a candidate slice containing one is
    skipped because ``Stroke.reversed()`` changes a closed path's anchor
    vertex; then the O(1) boundary formula would no longer be exact.
    """

    max_passes = int(max_passes)
    min_improvement_px = float(min_improvement_px)

    if max_passes < 0:
        raise ValueError("max_passes 不能为负数")
    if min_improvement_px < 0:
        raise ValueError("min_improvement_px 不能为负数")

    route = list(strokes)
    initial = pen_up_length(route)

    if len(route) < 2 or max_passes == 0:
        stats = RouteOptimizationStats(
            initial_pen_up_length_px=float(initial),
            optimized_pen_up_length_px=float(initial),
            improvement_px=0.0,
            improvement_ratio=0.0,
            move_count=0,
            pass_count=0,
            skipped_closed_candidates=0,
        )
        return tuple(route), stats

    move_count = 0
    pass_count = 0
    skipped_closed = 0

    for _ in range(max_passes):
        pass_count += 1

        # Prefix sum makes "does this slice contain a closed stroke?" O(1).
        closed_prefix = [0]
        for stroke in route:
            closed_prefix.append(
                closed_prefix[-1]
                + int(stroke.closed)
            )

        best_delta = -min_improvement_px
        best_left: int | None = None
        best_right: int | None = None

        count = len(route)

        for left in range(count):
            for right in range(left, count):
                # Reversing the whole route cannot change a free-endpoint
                # path objective, so skip the zero-delta candidate.
                if left == 0 and right == count - 1:
                    continue

                if (
                    closed_prefix[right + 1]
                    - closed_prefix[left]
                ) > 0:
                    skipped_closed += 1
                    continue

                old_cost = 0.0
                new_cost = 0.0

                if left > 0:
                    previous = route[left - 1]
                    first = route[left]
                    last = route[right]

                    old_cost += point_distance(
                        previous.end,
                        first.start,
                    )

                    # last.reversed().start == last.end for an open stroke.
                    new_cost += point_distance(
                        previous.end,
                        last.end,
                    )

                if right < count - 1:
                    first = route[left]
                    last = route[right]
                    following = route[right + 1]

                    old_cost += point_distance(
                        last.end,
                        following.start,
                    )

                    # first.reversed().end == first.start for an open stroke.
                    new_cost += point_distance(
                        first.start,
                        following.start,
                    )

                delta = new_cost - old_cost

                if delta < best_delta:
                    best_delta = delta
                    best_left = left
                    best_right = right

        if best_left is None or best_right is None:
            break

        route = _reverse_open_route_slice(
            route,
            best_left,
            best_right,
        )
        move_count += 1

    optimized = pen_up_length(route)

    # Guard against numerical or implementation regressions: 2-opt is an
    # improving-only local search and must never make the objective worse.
    if optimized > initial + 1e-7:
        raise RuntimeError(
            "2-opt 优化后 pen-up 距离反而增加"
        )

    improvement = max(
        0.0,
        float(initial - optimized),
    )

    improvement_ratio = (
        improvement / initial
        if initial > 0
        else 0.0
    )

    stats = RouteOptimizationStats(
        initial_pen_up_length_px=float(initial),
        optimized_pen_up_length_px=float(optimized),
        improvement_px=float(improvement),
        improvement_ratio=float(improvement_ratio),
        move_count=move_count,
        pass_count=pass_count,
        skipped_closed_candidates=skipped_closed,
    )

    return tuple(route), stats

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
        max_iterations=config.max_thinning_iterations,
    )

    skeleton01 = prune_short_spurs(
        skeleton01,
        max_length_px=config.spur_length_px,
    )

    raw_strokes, graph_stats = _trace_contracted_graph(
        skeleton01,
        junction_cluster_radius_px=config.junction_cluster_radius_px,
    )

    filtered = filter_short_strokes(
        raw_strokes,
        min_length_px=config.min_stroke_length_px,
    )

    simplified_edges = tuple(
        simplify_stroke(
            stroke,
            epsilon_px=config.simplify_epsilon_px,
        )
        for stroke in filtered
    )

    simplified_edges = filter_short_strokes(
        simplified_edges,
        min_length_px=config.min_stroke_length_px,
    )

    # Retain E2.1 local merge count as a diagnostic for the report.
    _, local_merge_count = merge_collinear_strokes(
        simplified_edges,
        max_turn_deviation_deg=config.merge_angle_deg,
        lookahead_px=config.merge_lookahead_px,
    )

    trails, trail_stats = plan_minimum_lift_trails(
        simplified_edges
    )

    greedy_strokes = order_strokes_greedy(trails)

    strokes, route_stats = optimize_stroke_order_2opt(
        greedy_strokes,
        max_passes=config.ordering_2opt_max_passes,
        min_improvement_px=(
            config.ordering_2opt_min_improvement_px
        ),
    )

    skeleton = (skeleton01 * 255).astype(
        np.uint8,
        copy=False,
    )

    return CenterlineResult(
        skeleton=skeleton,
        strokes=strokes,
        skeleton_pixels=int(np.count_nonzero(skeleton01)),
        pen_down_length_px=float(
            sum(stroke.length_px for stroke in strokes)
        ),
        pen_up_length_px=pen_up_length(strokes),
        endpoint_count=graph_stats.endpoint_count,
        junction_pixel_count=graph_stats.junction_pixel_count,
        junction_cluster_count=graph_stats.junction_cluster_count,
        graph_node_count=graph_stats.graph_node_count,
        raw_stroke_count=len(raw_strokes),
        merge_count=local_merge_count,
        connected_component_count=trail_stats.connected_component_count,
        odd_vertex_count=trail_stats.odd_vertex_count,
        theoretical_min_trail_count=(
            trail_stats.theoretical_min_trail_count
        ),
        edge_reuse_count=trail_stats.edge_reuse_count,
        ordering_mode="greedy+2opt",
        pen_up_greedy_px=(
            route_stats.initial_pen_up_length_px
        ),
        pen_up_optimized_px=(
            route_stats.optimized_pen_up_length_px
        ),
        two_opt_improvement_px=(
            route_stats.improvement_px
        ),
        two_opt_improvement_ratio=(
            route_stats.improvement_ratio
        ),
        two_opt_move_count=route_stats.move_count,
        two_opt_pass_count=route_stats.pass_count,
        two_opt_skipped_closed_candidates=(
            route_stats.skipped_closed_candidates
        ),
    )
