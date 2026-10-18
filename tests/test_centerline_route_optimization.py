from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


from robot_portrait.centerline import (
    CenterlineConfig,
    Stroke,
    extract_centerline_strokes,
    optimize_stroke_order_2opt,
    pen_up_length,
)


def _point_stroke(x: int, y: int) -> Stroke:
    # A one-pixel segment behaves like a route node while remaining a valid
    # non-zero open stroke.
    return Stroke(
        points=((x, y), (x + 1, y))
    )


def test_two_opt_improves_deliberately_crossed_route():
    route = (
        _point_stroke(0, 0),
        _point_stroke(100, 100),
        _point_stroke(0, 100),
        _point_stroke(100, 0),
    )

    before = pen_up_length(route)
    optimized, stats = optimize_stroke_order_2opt(
        route,
        max_passes=16,
        min_improvement_px=1e-9,
    )
    after = pen_up_length(optimized)

    assert after < before
    assert stats.improvement_px == pytest.approx(before - after)
    assert stats.move_count >= 1


def test_two_opt_can_flip_single_stroke_orientation():
    route = (
        Stroke(points=((0, 0), (1, 0))),
        Stroke(points=((20, 0), (10, 0))),
        Stroke(points=((21, 0), (22, 0))),
    )

    optimized, stats = optimize_stroke_order_2opt(
        route,
        max_passes=8,
        min_improvement_px=1e-9,
    )

    assert pen_up_length(optimized) < pen_up_length(route)
    assert stats.move_count >= 1


def test_two_opt_can_improve_free_endpoint_with_prefix_reversal():
    route = (
        _point_stroke(50, 0),
        _point_stroke(0, 0),
        _point_stroke(10, 0),
    )

    optimized, _ = optimize_stroke_order_2opt(
        route,
        max_passes=8,
        min_improvement_px=1e-9,
    )

    assert pen_up_length(optimized) <= pen_up_length(route)


def test_two_opt_preserves_stroke_count_and_pen_down_geometry():
    route = (
        Stroke(points=((0, 0), (10, 0), (20, 0))),
        Stroke(points=((80, 20), (90, 20))),
        Stroke(points=((20, 40), (30, 40))),
        Stroke(points=((60, 0), (70, 0))),
    )

    before_down = sum(stroke.length_px for stroke in route)

    optimized, _ = optimize_stroke_order_2opt(
        route,
        max_passes=16,
    )

    after_down = sum(stroke.length_px for stroke in optimized)

    assert len(optimized) == len(route)
    assert after_down == pytest.approx(before_down)


def test_two_opt_is_deterministic():
    route = (
        _point_stroke(0, 0),
        _point_stroke(100, 100),
        _point_stroke(0, 100),
        _point_stroke(100, 0),
        _point_stroke(50, 50),
    )

    first, first_stats = optimize_stroke_order_2opt(route)
    second, second_stats = optimize_stroke_order_2opt(route)

    assert first == second
    assert first_stats == second_stats


def test_two_opt_never_degrades_already_good_route():
    route = (
        _point_stroke(0, 0),
        _point_stroke(10, 0),
        _point_stroke(20, 0),
        _point_stroke(30, 0),
    )

    optimized, stats = optimize_stroke_order_2opt(route)

    assert pen_up_length(optimized) <= pen_up_length(route) + 1e-9
    assert stats.optimized_pen_up_length_px <= (
        stats.initial_pen_up_length_px + 1e-9
    )


def test_closed_stroke_candidates_are_safely_skipped():
    closed = Stroke(
        points=((20, 20), (30, 20), (30, 30), (20, 30)),
        closed=True,
    )
    route = (
        _point_stroke(0, 0),
        closed,
        _point_stroke(60, 0),
    )

    optimized, stats = optimize_stroke_order_2opt(
        route,
        max_passes=4,
    )

    assert len(optimized) == 3
    assert stats.skipped_closed_candidates > 0
    assert pen_up_length(optimized) <= pen_up_length(route) + 1e-9


def test_zero_passes_returns_route_unchanged():
    route = (
        _point_stroke(0, 0),
        _point_stroke(100, 100),
        _point_stroke(0, 100),
    )

    optimized, stats = optimize_stroke_order_2opt(
        route,
        max_passes=0,
    )

    assert optimized == route
    assert stats.move_count == 0
    assert stats.pass_count == 0


def test_two_opt_rejects_invalid_parameters():
    with pytest.raises(ValueError, match="max_passes"):
        optimize_stroke_order_2opt(
            (_point_stroke(0, 0),),
            max_passes=-1,
        )

    with pytest.raises(ValueError, match="min_improvement"):
        optimize_stroke_order_2opt(
            (_point_stroke(0, 0),),
            min_improvement_px=-1,
        )


def test_centerline_config_rejects_invalid_two_opt_values():
    with pytest.raises(ValueError, match="ordering_2opt_max_passes"):
        CenterlineConfig(
            ordering_2opt_max_passes=-1
        )

    with pytest.raises(
        ValueError,
        match="ordering_2opt_min_improvement",
    ):
        CenterlineConfig(
            ordering_2opt_min_improvement_px=-0.1
        )


def test_extraction_reports_two_opt_metadata_and_consistent_pen_up():
    image = np.full(
        (96, 96, 3),
        255,
        dtype=np.uint8,
    )

    # Several disconnected strokes force inter-trail travel.
    cv2.line(image, (5, 10), (30, 10), (0, 0, 0), 3)
    cv2.line(image, (65, 15), (90, 15), (0, 0, 0), 3)
    cv2.line(image, (8, 75), (35, 75), (0, 0, 0), 3)
    cv2.line(image, (65, 80), (90, 80), (0, 0, 0), 3)

    result = extract_centerline_strokes(
        image,
        CenterlineConfig(
            spur_length_px=0,
            min_stroke_length_px=1,
            simplify_epsilon_px=0,
            ordering_2opt_max_passes=16,
            ordering_2opt_min_improvement_px=0,
        ),
    )

    metadata = result.to_metadata()

    assert metadata["ordering_mode"] == "greedy+2opt"
    assert metadata["pen_up_optimized_px"] == pytest.approx(
        result.pen_up_length_px
    )
    assert metadata["pen_up_optimized_px"] <= (
        metadata["pen_up_greedy_px"] + 1e-9
    )
    assert metadata["two_opt_improvement_px"] >= 0
    assert 0 <= metadata["two_opt_improvement_ratio"] <= 1


def test_two_opt_large_route_smoke():
    route = tuple(
        _point_stroke(
            (index * 37) % 211,
            (index * 83) % 197,
        )
        for index in range(80)
    )

    optimized, stats = optimize_stroke_order_2opt(
        route,
        max_passes=8,
    )

    assert len(optimized) == len(route)
    assert stats.move_count <= 8
    assert pen_up_length(optimized) <= pen_up_length(route) + 1e-9
