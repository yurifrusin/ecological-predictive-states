from __future__ import annotations

import mujoco
import numpy as np
import pytest

from epsbench.annotations import (
    FLOW_FIXED_POINT_SCALE,
    AnalyticCamera,
    TransportReasonCode,
    compute_analytic_transport,
    focal_scales_from_vertical_fov,
)

WIDTH = 80
HEIGHT = 60
FOV_DEGREES = 60.0
IDENTITY_ROTATION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def _camera(x: float = 0.0, z: float = 0.0) -> AnalyticCamera:
    return AnalyticCamera(
        world_position=(x, 0.0, z),
        world_rotation_row_major=IDENTITY_ROTATION,
        vertical_field_of_view_degrees=FOV_DEGREES,
    )


def _plane_model(
    with_foreground: bool = False,
    plane_half_extent: float = 20.0,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    foreground = (
        '<geom name="foreground" type="box" pos="0 0 -2" size="0.35 0.8 0.8"/>'
        if with_foreground
        else ""
    )
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody>"
        f'<geom name="background" type="plane" pos="0 0 -5" '
        f'size="{plane_half_extent} {plane_half_extent} 0.1"/>'
        f"{foreground}</worldbody></mujoco>"
    )
    return model, mujoco.MjData(model)


def test_fronto_parallel_plane_lateral_translation_matches_closed_form() -> None:
    model, data = _plane_model()
    translation = 0.1
    transport = compute_analytic_transport(
        model,
        data,
        (0,),
        WIDTH,
        HEIGHT,
        _camera(),
        _camera(x=translation),
    )
    focal_x, _ = focal_scales_from_vertical_fov(WIDTH, HEIGHT, FOV_DEGREES)
    expected_pixels = -focal_x * translation / 5.0
    expected_fixed = int(np.rint(expected_pixels * FLOW_FIXED_POINT_SCALE))
    valid_vectors = transport.forward.vectors_fixed[transport.forward.validity == 1]
    assert np.all(valid_vectors[:, 0] == expected_fixed)
    assert np.all(valid_vectors[:, 1] == 0)
    assert expected_fixed / FLOW_FIXED_POINT_SCALE == pytest.approx(
        expected_pixels,
        abs=1.0 / (2.0 * FLOW_FIXED_POINT_SCALE),
    )


def test_fronto_parallel_plane_forward_translation_matches_radial_closed_form() -> None:
    model, data = _plane_model()
    translation = 1.0
    transport = compute_analytic_transport(
        model,
        data,
        (0,),
        WIDTH,
        HEIGHT,
        _camera(),
        _camera(z=-translation),
    )
    rows, columns = np.indices((HEIGHT, WIDTH), dtype=np.float64)
    expected_x = (columns + 0.5 - WIDTH / 2.0) * translation / (5.0 - translation)
    expected_y = (rows + 0.5 - HEIGHT / 2.0) * translation / (5.0 - translation)
    expected_fixed = np.rint(
        np.stack((expected_x, expected_y), axis=-1) * FLOW_FIXED_POINT_SCALE
    ).astype(np.int32)
    valid = transport.forward.validity == 1
    assert np.array_equal(transport.forward.vectors_fixed[valid], expected_fixed[valid])


def test_static_camera_produces_valid_zero_transport() -> None:
    model, data = _plane_model()
    transport = compute_analytic_transport(
        model,
        data,
        (0,),
        WIDTH,
        HEIGHT,
        _camera(),
        _camera(),
    )
    assert np.all(transport.forward.validity == 1)
    assert np.all(transport.forward.reasons == TransportReasonCode.VALID_TRANSPORT)
    assert np.all(transport.forward.vectors_fixed == 0)


def test_finite_visual_plane_rejects_source_hits_outside_xy_extent() -> None:
    model, data = _plane_model(plane_half_extent=0.5)
    transport = compute_analytic_transport(
        model,
        data,
        (0,),
        WIDTH,
        HEIGHT,
        _camera(),
        _camera(),
    )
    outside = transport.before_surface_assignment == -1
    assert np.any(outside)
    assert np.any(~outside)
    assert np.all(
        transport.forward.reasons[outside] == TransportReasonCode.NO_CONTROLLED_SOURCE_SURFACE
    )
    assert np.all(transport.forward.validity[outside] == 0)
    assert np.all(transport.forward.vectors_fixed[outside] == 0)


def test_finite_visual_plane_edge_participates_in_analytic_boundary_band() -> None:
    model, data = _plane_model(plane_half_extent=0.5)
    transport = compute_analytic_transport(
        model,
        data,
        (0,),
        WIDTH,
        HEIGHT,
        _camera(),
        _camera(),
    )
    controlled = transport.before_surface_assignment == 0
    boundary = transport.before_boundary_ambiguous
    assert np.any(boundary & controlled)
    assert np.any(boundary & ~controlled)
    assert np.all(
        transport.forward.reasons[boundary & controlled]
        == TransportReasonCode.ANALYTIC_BOUNDARY_AMBIGUOUS
    )


def test_finite_plane_extent_changes_label_defining_transport_arrays() -> None:
    small_model, small_data = _plane_model(plane_half_extent=0.5)
    large_model, large_data = _plane_model(plane_half_extent=1.0)
    small = compute_analytic_transport(
        small_model,
        small_data,
        (0,),
        WIDTH,
        HEIGHT,
        _camera(),
        _camera(),
    )
    large = compute_analytic_transport(
        large_model,
        large_data,
        (0,),
        WIDTH,
        HEIGHT,
        _camera(),
        _camera(),
    )
    assert not np.array_equal(small.forward.validity, large.forward.validity)
    assert not np.array_equal(small.forward.reasons, large.forward.reasons)


def test_frame_exit_has_explicit_reason_and_canonical_zero_vector() -> None:
    model, data = _plane_model()
    transport = compute_analytic_transport(
        model,
        data,
        (0,),
        WIDTH,
        HEIGHT,
        _camera(),
        _camera(x=20.0),
    )
    exited = transport.forward.reasons == TransportReasonCode.TARGET_OUT_OF_FRAME
    assert np.any(exited)
    assert np.all(transport.forward.validity[exited] == 0)
    assert np.all(transport.forward.vectors_fixed[exited] == 0)


def test_foreground_surface_causes_target_occlusion() -> None:
    model, data = _plane_model(with_foreground=True)
    transport = compute_analytic_transport(
        model,
        data,
        (0, 1),
        WIDTH,
        HEIGHT,
        _camera(x=-1.5),
        _camera(),
    )
    occluded = transport.forward.reasons == TransportReasonCode.OCCLUDED_AT_TARGET
    assert np.any(occluded)
    assert np.all(transport.forward.validity[occluded] == 0)
    assert np.all(transport.forward.vectors_fixed[occluded] == 0)


def test_analytic_assignment_discontinuity_is_boundary_ambiguous() -> None:
    model, data = _plane_model(with_foreground=True)
    transport = compute_analytic_transport(
        model,
        data,
        (0, 1),
        WIDTH,
        HEIGHT,
        _camera(x=-1.5),
        _camera(),
    )
    ambiguous = transport.forward.reasons == TransportReasonCode.ANALYTIC_BOUNDARY_AMBIGUOUS
    assert np.any(transport.before_boundary_ambiguous)
    assert np.any(ambiguous)
    assert np.all(transport.forward.validity[ambiguous] == 0)


def test_integer_pixel_transport_uses_backward_flow_at_the_correspondence() -> None:
    model, data = _plane_model()
    focal_x, _ = focal_scales_from_vertical_fov(WIDTH, HEIGHT, FOV_DEGREES)
    one_pixel_translation = 5.0 / focal_x
    transport = compute_analytic_transport(
        model,
        data,
        (0,),
        WIDTH,
        HEIGHT,
        _camera(),
        _camera(x=one_pixel_translation),
    )
    source_rows, source_columns = np.nonzero(transport.forward.validity == 1)
    forward_fixed = transport.forward.vectors_fixed[source_rows, source_columns]
    assert np.all(forward_fixed[:, 0] == -FLOW_FIXED_POINT_SCALE)
    assert np.all(forward_fixed[:, 1] == 0)
    destination_rows = source_rows + forward_fixed[:, 1] // FLOW_FIXED_POINT_SCALE
    destination_columns = source_columns + forward_fixed[:, 0] // FLOW_FIXED_POINT_SCALE
    assert np.all(transport.backward.validity[destination_rows, destination_columns] == 1)
    backward_at_correspondence = transport.backward.vectors_fixed[
        destination_rows,
        destination_columns,
    ]
    round_trip_fixed = forward_fixed + backward_at_correspondence
    assert np.all(round_trip_fixed == 0)
    assert np.all(np.abs(round_trip_fixed / FLOW_FIXED_POINT_SCALE) <= 1.0 / FLOW_FIXED_POINT_SCALE)


def test_continuous_radial_inverse_is_evaluated_at_expanded_correspondence() -> None:
    depth = 5.0
    forward_translation = 1.0
    source_offsets = np.asarray((-18.5, -7.25, 3.5, 16.75), dtype=np.float64)
    destination_offsets = source_offsets * depth / (depth - forward_translation)
    forward_flow = destination_offsets - source_offsets
    backward_at_destination = (
        destination_offsets * (depth - forward_translation) / depth - destination_offsets
    )
    assert np.allclose(
        forward_flow + backward_at_destination,
        0.0,
        atol=np.finfo(np.float64).eps * 8,
        rtol=0.0,
    )
