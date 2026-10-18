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
    order_strokes_greedy,
    pen_up_length,
    prune_short_spurs,
    raster_to_ink_mask,
    skeleton_degrees,
    trace_skeleton_strokes,
    zhang_suen_thinning,
)


def _canvas(
    width: int = 96,
    height: int = 96,
) -> np.ndarray:
    return np.full(
        (height, width, 3),
        255,
        dtype=np.uint8,
    )


def test_thick_horizontal_line_becomes_one_open_centerline():
    image = _canvas()
    cv2.line(
        image,
        (12, 48),
        (83, 48),
        (0, 0, 0),
        9,
        cv2.LINE_8,
    )

    result = extract_centerline_strokes(
        image,
        CenterlineConfig(
            spur_length_px=0,
            min_stroke_length_px=5,
            simplify_epsilon_px=0.5,
        ),
    )

    assert len(result.strokes) == 1
    stroke = result.strokes[0]
    assert stroke.closed is False
    assert stroke.length_px > 60
    ys = [p[1] for p in stroke.points]
    assert max(ys) - min(ys) <= 2


def test_thin_diagonal_line_stays_connected():
    image = _canvas()
    cv2.line(
        image,
        (10, 10),
        (80, 80),
        (0, 0, 0),
        1,
        cv2.LINE_8,
    )

    result = extract_centerline_strokes(
        image,
        CenterlineConfig(
            spur_length_px=0,
            min_stroke_length_px=5,
            simplify_epsilon_px=0,
        ),
    )

    assert len(result.strokes) == 1
    assert result.strokes[0].closed is False
    assert result.strokes[0].length_px > 90


def test_l_shape_is_one_open_stroke():
    skeleton = np.zeros((32, 32), dtype=np.uint8)
    skeleton[5:25, 8] = 1
    skeleton[24, 8:24] = 1

    strokes = trace_skeleton_strokes(skeleton)

    assert len(strokes) == 1
    assert strokes[0].closed is False
    assert strokes[0].start != strokes[0].end


def test_t_junction_traces_three_branches_without_losing_edges():
    skeleton = np.zeros((41, 41), dtype=np.uint8)
    skeleton[8, 8:33] = 1
    skeleton[8:34, 20] = 1

    strokes = trace_skeleton_strokes(skeleton)

    assert len(strokes) == 3
    assert all(not stroke.closed for stroke in strokes)

    endpoints = [
        point
        for point, degree in skeleton_degrees(skeleton).items()
        if degree == 1
    ]
    junctions = [
        point
        for point, degree in skeleton_degrees(skeleton).items()
        if degree >= 3
    ]

    assert len(endpoints) == 3
    assert len(junctions) == 1


def test_closed_ring_is_traced_as_closed_stroke():
    skeleton = np.zeros((32, 32), dtype=np.uint8)

    # Diamond loop is naturally one-pixel wide in 8-connectivity.
    points = np.array(
        [
            [16, 4],
            [28, 16],
            [16, 28],
            [4, 16],
        ],
        dtype=np.int32,
    )

    cv2.polylines(
        skeleton,
        [points],
        True,
        1,
        1,
    )

    strokes = trace_skeleton_strokes(skeleton)

    assert len(strokes) == 1
    assert strokes[0].closed is True
    assert strokes[0].length_px > 50


def test_short_spur_is_pruned_but_main_stroke_survives():
    skeleton = np.zeros((48, 48), dtype=np.uint8)
    skeleton[5:43, 24] = 1
    skeleton[22, 24:27] = 1

    before = skeleton_degrees(skeleton)
    assert sum(1 for d in before.values() if d == 1) == 3

    pruned = prune_short_spurs(
        skeleton,
        max_length_px=3.0,
    )

    after = skeleton_degrees(pruned)

    assert pruned[22, 26] == 0
    assert pruned[5, 24] == 1
    assert pruned[42, 24] == 1
    assert sum(1 for d in after.values() if d == 1) == 2


def test_greedy_order_can_reverse_strokes_and_reduce_pen_up():
    strokes = (
        Stroke(
            points=((90, 0), (100, 0)),
        ),
        Stroke(
            points=((10, 0), (20, 0)),
        ),
        Stroke(
            points=((60, 0), (50, 0)),
        ),
    )

    naive = pen_up_length(strokes)

    ordered = order_strokes_greedy(
        strokes,
        start_point=(0, 0),
    )

    assert ordered[0].start == (10, 0)
    assert pen_up_length(ordered) < naive


def test_result_metadata_contains_robot_motion_metrics():
    image = _canvas()
    cv2.line(
        image,
        (10, 20),
        (80, 20),
        (0, 0, 0),
        5,
    )
    cv2.line(
        image,
        (10, 70),
        (80, 70),
        (0, 0, 0),
        5,
    )

    result = extract_centerline_strokes(image)
    metadata = result.to_metadata()

    assert metadata["mode"] == "centerline"
    assert metadata["stroke_count"] == 2
    assert metadata["skeleton_pixels"] > 0
    assert metadata["pen_down_length_px"] > 100
    assert metadata["pen_up_length_px"] > 0


def test_raster_to_ink_mask_uses_dark_pixels_as_foreground():
    image = np.array(
        [
            [0, 219, 220, 255],
        ],
        dtype=np.uint8,
    )

    mask = raster_to_ink_mask(
        image,
        threshold=220,
    )

    assert np.array_equal(
        mask,
        np.array(
            [
                [1, 1, 0, 0],
            ],
            dtype=np.uint8,
        ),
    )


def test_zhang_suen_thins_filled_bar_to_single_pixel_width():
    mask = np.zeros((40, 50), dtype=np.uint8)
    mask[15:25, 8:42] = 1

    skeleton = zhang_suen_thinning(mask)

    assert skeleton.sum() < mask.sum() / 3

    # The center region should not retain a thick 2-D block.
    for x in range(15, 35):
        assert int(skeleton[:, x].sum()) <= 1


def test_configuration_rejects_invalid_values():
    with pytest.raises(ValueError, match="threshold"):
        CenterlineConfig(threshold=300)

    with pytest.raises(ValueError, match="spur_length"):
        CenterlineConfig(spur_length_px=-1)

    with pytest.raises(ValueError, match="min_stroke"):
        CenterlineConfig(min_stroke_length_px=-1)

    with pytest.raises(ValueError, match="simplify"):
        CenterlineConfig(simplify_epsilon_px=-1)

    with pytest.raises(ValueError, match="max_thinning"):
        CenterlineConfig(max_thinning_iterations=0)
