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


def _plane_model(with_foreground: bool = False) -> tuple[mujoco.MjModel, mujoco.MjData]:
    foreground = (
        '<geom name="foreground" type="box" pos="0 0 -2" size="0.35 0.8 0.8"/>'
        if with_foreground
        else ""
    )
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody>"
        '<geom name="background" type="plane" pos="0 0 -5" size="20 20 0.1"/>'
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


def test_forward_backward_plane_transport_is_consistent_within_quantisation_tolerance() -> None:
    model, data = _plane_model()
    transport = compute_analytic_transport(
        model,
        data,
        (0,),
        WIDTH,
        HEIGHT,
        _camera(),
        _camera(x=0.1),
    )
    forward = transport.forward.vectors_fixed.astype(np.float64) / FLOW_FIXED_POINT_SCALE
    backward = transport.backward.vectors_fixed.astype(np.float64) / FLOW_FIXED_POINT_SCALE
    mutually_valid = (transport.forward.validity == 1) & (transport.backward.validity == 1)
    error = np.abs(forward[mutually_valid] + backward[mutually_valid])
    assert np.all(error <= 1.0 / FLOW_FIXED_POINT_SCALE)
