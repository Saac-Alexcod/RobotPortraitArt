from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


from robot_portrait.centerline import (
    Stroke,
    plan_minimum_lift_trails,
)


def _length(strokes):
    return sum(stroke.length_px for stroke in strokes)


def test_single_open_edge_is_one_trail():
    edges = (Stroke(points=((0, 0), (10, 0))),)

    trails, stats = plan_minimum_lift_trails(edges)

    assert len(trails) == 1
    assert stats.theoretical_min_trail_count == 1
    assert stats.odd_vertex_count == 2
    assert stats.edge_reuse_count == 0
    assert _length(trails) == pytest.approx(_length(edges))


def test_t_graph_reaches_two_trail_theoretical_minimum():
    c = (10, 10)
    edges = (
        Stroke(points=((0, 10), c)),
        Stroke(points=(c, (20, 10))),
        Stroke(points=(c, (10, 20))),
    )

    trails, stats = plan_minimum_lift_trails(edges)

    assert stats.odd_vertex_count == 4
    assert stats.theoretical_min_trail_count == 2
    assert len(trails) == 2
    assert stats.edge_reuse_count == 0
    assert _length(trails) == pytest.approx(_length(edges))


def test_x_graph_reaches_two_trail_theoretical_minimum():
    c = (10, 10)
    edges = (
        Stroke(points=((0, 10), c)),
        Stroke(points=(c, (20, 10))),
        Stroke(points=((10, 0), c)),
        Stroke(points=(c, (10, 20))),
    )

    trails, stats = plan_minimum_lift_trails(edges)

    assert stats.odd_vertex_count == 4
    assert stats.theoretical_min_trail_count == 2
    assert len(trails) == 2
    assert stats.edge_reuse_count == 0


def test_even_cycle_becomes_one_closed_trail():
    edges = (
        Stroke(points=((0, 0), (10, 0))),
        Stroke(points=((10, 0), (10, 10))),
        Stroke(points=((10, 10), (0, 10))),
        Stroke(points=((0, 10), (0, 0))),
    )

    trails, stats = plan_minimum_lift_trails(edges)

    assert stats.odd_vertex_count == 0
    assert stats.theoretical_min_trail_count == 1
    assert len(trails) == 1
    assert trails[0].closed
    assert _length(trails) == pytest.approx(_length(edges))


def test_disconnected_path_and_cycle_need_two_trails():
    edges = (
        Stroke(points=((0, 0), (10, 0))),
        Stroke(points=((50, 50), (60, 50))),
        Stroke(points=((60, 50), (60, 60))),
        Stroke(points=((60, 60), (50, 60))),
        Stroke(points=((50, 60), (50, 50))),
    )

    trails, stats = plan_minimum_lift_trails(edges)

    assert stats.connected_component_count == 2
    assert stats.odd_vertex_count == 2
    assert stats.theoretical_min_trail_count == 2
    assert len(trails) == 2


def test_parallel_edges_are_supported_as_multigraph():
    edges = (
        Stroke(points=((0, 0), (10, 0), (20, 0))),
        Stroke(points=((0, 0), (10, 1), (20, 0))),
    )

    trails, stats = plan_minimum_lift_trails(edges)

    assert stats.connected_component_count == 1
    assert stats.odd_vertex_count == 0
    assert stats.theoretical_min_trail_count == 1
    assert len(trails) == 1
    assert trails[0].closed
    assert stats.edge_reuse_count == 0


def test_nearby_unshared_endpoints_remain_separate_components():
    edges = (
        Stroke(points=((0, 0), (10, 0))),
        Stroke(points=((11, 0), (20, 0))),
    )

    trails, stats = plan_minimum_lift_trails(edges)

    assert stats.connected_component_count == 2
    assert stats.theoretical_min_trail_count == 2
    assert len(trails) == 2


def test_planning_is_deterministic():
    c = (10, 10)
    edges = (
        Stroke(points=((0, 10), c)),
        Stroke(points=(c, (20, 10))),
        Stroke(points=((10, 0), c)),
        Stroke(points=(c, (10, 20))),
        Stroke(points=(c, (18, 18))),
    )

    first, first_stats = plan_minimum_lift_trails(edges)
    second, second_stats = plan_minimum_lift_trails(edges)

    assert first == second
    assert first_stats == second_stats


def test_empty_input_has_zero_trails():
    trails, stats = plan_minimum_lift_trails(tuple())

    assert trails == tuple()
    assert stats.input_edge_count == 0
    assert stats.connected_component_count == 0
    assert stats.odd_vertex_count == 0
    assert stats.theoretical_min_trail_count == 0
    assert stats.output_trail_count == 0
    assert stats.edge_reuse_count == 0


def test_complex_graph_preserves_total_edge_length_without_reuse():
    c = (20, 20)
    edges = (
        Stroke(points=(c, (30, 20))),
        Stroke(points=((30, 20), (30, 30))),
        Stroke(points=((30, 30), (20, 30))),
        Stroke(points=((20, 30), c)),
        Stroke(points=(c, (10, 20))),
        Stroke(points=((10, 20), (10, 10))),
        Stroke(points=((10, 10), (20, 10))),
        Stroke(points=((20, 10), c)),
        Stroke(points=(c, (20, 0))),
        Stroke(points=(c, (40, 20))),
        Stroke(points=(c, (20, 40))),
        Stroke(points=(c, (0, 20))),
    )

    trails, stats = plan_minimum_lift_trails(edges)

    assert stats.edge_reuse_count == 0
    assert stats.output_trail_count == stats.theoretical_min_trail_count
    assert _length(trails) == pytest.approx(_length(edges))
