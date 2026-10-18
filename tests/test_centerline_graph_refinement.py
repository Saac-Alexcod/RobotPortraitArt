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
    merge_collinear_strokes,
    trace_skeleton_strokes,
    zhang_suen_thinning,
)


def _multi_pixel_x_junction() -> np.ndarray:
    skeleton = np.zeros(
        (20, 20),
        dtype=np.uint8,
    )

    # A small 2x2-ish knot with four genuine long arms. Without junction
    # contraction, local degree-2 pixels around the knot become micro-edges.
    skeleton[8, 7] = 1
    skeleton[8, 8] = 1
    skeleton[9, 7] = 1
    skeleton[9, 8] = 1

    skeleton[2:8, 7] = 1
    skeleton[10:17, 8] = 1
    skeleton[8, 2:7] = 1
    skeleton[9, 9:17] = 1

    return skeleton


def test_junction_cluster_contracts_local_knot_to_four_branches():
    skeleton = _multi_pixel_x_junction()

    strokes = trace_skeleton_strokes(
        skeleton,
        junction_cluster_radius_px=1,
    )

    assert len(strokes) == 4
    assert all(
        not stroke.closed
        for stroke in strokes
    )


def test_t_junction_continuation_merges_straight_pair():
    skeleton = np.zeros(
        (41, 41),
        dtype=np.uint8,
    )

    skeleton[8, 8:33] = 1
    skeleton[8:34, 20] = 1

    raw = trace_skeleton_strokes(
        skeleton,
        junction_cluster_radius_px=1,
    )

    assert len(raw) == 3

    merged, count = merge_collinear_strokes(
        raw,
        max_turn_deviation_deg=20.0,
        lookahead_px=6.0,
    )

    assert count == 1
    assert len(merged) == 2
    assert any(
        stroke.start == (8, 8)
        and stroke.end == (32, 8)
        or stroke.start == (32, 8)
        and stroke.end == (8, 8)
        for stroke in merged
    )


def test_x_junction_pairs_two_straight_continuations():
    skeleton = _multi_pixel_x_junction()

    raw = trace_skeleton_strokes(
        skeleton,
        junction_cluster_radius_px=1,
    )

    merged, count = merge_collinear_strokes(
        raw,
        max_turn_deviation_deg=35.0,
        lookahead_px=5.0,
    )

    assert len(raw) == 4
    assert count == 2
    assert len(merged) == 2


def test_y_junction_is_not_forced_into_bad_continuation():
    shared = (20, 20)

    strokes = (
        Stroke(
            points=(
                shared,
                (20, 10),
            )
        ),
        Stroke(
            points=(
                shared,
                (11, 25),
            )
        ),
        Stroke(
            points=(
                shared,
                (29, 25),
            )
        ),
    )

    merged, count = merge_collinear_strokes(
        strokes,
        max_turn_deviation_deg=25.0,
        lookahead_px=5.0,
    )

    assert count == 0
    assert len(merged) == 3


def test_nearby_but_unshared_endpoints_are_never_connected():
    strokes = (
        Stroke(
            points=(
                (0, 0),
                (10, 0),
            )
        ),
        Stroke(
            points=(
                (11, 0),
                (20, 0),
            )
        ),
    )

    merged, count = merge_collinear_strokes(
        strokes,
        max_turn_deviation_deg=45.0,
        lookahead_px=5.0,
    )

    assert count == 0
    assert merged == strokes


def test_extract_metadata_reports_graph_refinement_counts():
    image = np.full(
        (64, 64, 3),
        255,
        dtype=np.uint8,
    )

    cv2.line(
        image,
        (8, 20),
        (55, 20),
        (0, 0, 0),
        3,
    )

    cv2.line(
        image,
        (32, 20),
        (32, 55),
        (0, 0, 0),
        3,
    )

    result = extract_centerline_strokes(
        image,
        CenterlineConfig(
            spur_length_px=0,
            min_stroke_length_px=1,
            simplify_epsilon_px=0,
            merge_angle_deg=25.0,
        ),
    )

    metadata = result.to_metadata()

    assert metadata["raw_stroke_count"] >= metadata["stroke_count"]
    assert metadata["merge_count"] >= 1
    assert metadata["endpoint_count"] >= 3
    assert metadata["junction_pixel_count"] >= 1
    assert metadata["junction_cluster_count"] >= 1
    assert metadata["graph_node_count"] > 0


def test_vectorized_thinning_is_deterministic_and_single_pixel_wide():
    mask = np.zeros(
        (64, 80),
        dtype=np.uint8,
    )

    mask[20:40, 10:70] = 1

    first = zhang_suen_thinning(
        mask
    )
    second = zhang_suen_thinning(
        mask
    )

    assert np.array_equal(
        first,
        second,
    )

    for x in range(
        20,
        60,
    ):
        assert int(
            first[:, x].sum()
        ) <= 1


def test_zero_merge_angle_still_allows_exactly_straight_pair():
    shared = (10, 10)

    strokes = (
        Stroke(
            points=(
                (0, 10),
                shared,
            )
        ),
        Stroke(
            points=(
                shared,
                (20, 10),
            )
        ),
    )

    merged, count = merge_collinear_strokes(
        strokes,
        max_turn_deviation_deg=0,
        lookahead_px=4,
    )

    assert count == 1
    assert len(merged) == 1
    assert merged[0].length_px == pytest.approx(20.0)


def test_e21_configuration_rejects_invalid_graph_values():
    with pytest.raises(
        ValueError,
        match="junction_cluster_radius",
    ):
        CenterlineConfig(
            junction_cluster_radius_px=-1
        )

    with pytest.raises(
        ValueError,
        match="merge_angle",
    ):
        CenterlineConfig(
            merge_angle_deg=181
        )

    with pytest.raises(
        ValueError,
        match="merge_lookahead",
    ):
        CenterlineConfig(
            merge_lookahead_px=0
        )
