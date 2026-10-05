"""Hand-derivable geometry checks; no native capture or ecological state construction."""

import json
from pathlib import Path

import numpy as np
import pytest

from epsbench.diagnostics.causal_history_fixture import (
    Box,
    Camera,
    Scene,
    box_distance,
    check_candidate,
    plane_distance,
)


def test_basis_pixel_centre_projection() -> None:
    camera = Camera(width=4, height=2)
    np.testing.assert_allclose(camera.basis @ camera.basis.T, np.eye(3), atol=1e-15)
    points = camera.origin(0) + 2 * camera.rays()
    pixel, inside = camera.project(points, 0)
    row, col = np.indices((2, 4))
    np.testing.assert_allclose(pixel, np.stack((col, row), axis=-1), atol=1e-15)
    assert inside.all()


def test_box_front_side_edge_parallel_inside() -> None:
    box = Box((0, 1, 0), (1, 2, 1))
    origin = np.array([-1.0, 0.0, 0.5])
    rays = np.array([[1, 1, 0], [1, 0.75, 0], [0, 1, 0], [1, 2, 0]], dtype=float)
    np.testing.assert_allclose(box_distance(origin, rays, box), [1, 4 / 3, np.inf, 1])
    inside = np.array([0.5, 1.5, 0.5])
    assert box_distance(inside, np.array([[1.0, 0.0, 0.0]]), box)[0] == 0.5


def test_finite_support_plane() -> None:
    rays = np.array([[0, 0, -1], [2, 0, -1], [3, 0, -1], [0, 0, 1], [1, 0, 0]], dtype=float)
    np.testing.assert_allclose(
        plane_distance(np.array([0.0, 0.0, 1.0]), rays, (2, 3)), [1, 1, np.inf, np.inf, np.inf]
    )


def test_completed_transport_rejects_occluded_source() -> None:
    camera = Camera(width=20, height=20, up_y=0)
    scene = Scene(
        camera, Box((-0.5, 0.75, 0), (0.5, 0.85, 1.8)), Box((-1.5, 2.45, 0), (1.5, 2.55, 2.1))
    )
    labels, _ = scene.frame(1.5)
    _, validity = scene.transport(1.5, 0)
    assert np.any((labels == 3) & ~validity)
    vectors, identity_validity = scene.transport(1.5, 1.5)
    assert np.any((labels == 3) & identity_validity)
    assert not vectors.any()


def test_fixed_candidate_reports_without_native_claim() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "configs/development/causal_history_fixture_design_v1.json"
    )
    report = check_candidate(path)
    assert report["pose_count"] == 24
    assert report["native_rgb_equality"] == report["native_state_equality"] == "UNPROVEN"
    assert len(report["pairs"]) == 3
    assert report["status"] == "ANALYTIC_RELATIONS_SATISFIED"
    assert [p["target_symmetric_difference"] for p in report["pairs"]] == [210, 201, 210]
    # A failed relation is preserved in the report, never silently repaired.
    assert bool(report["failures"]) == (report["status"] == "ANALYTIC_FAILED")


def test_hand_derived_projection_scale_and_signed_motion() -> None:
    # With upright camera, forward y, focal length 60/tan(30 degrees):
    # point x=1,y=1,z=.5 is depth 4 and right of the centre; moving right
    # by .25 produces exactly -.25*focal/4 horizontal pixels.
    camera = Camera(up_y=0)
    points = np.array([[1.0, 1.0, 0.5]])
    first, _ = camera.project(points, 0)
    second, _ = camera.project(points, 0.25)
    focal = 60 * np.sqrt(3)
    np.testing.assert_allclose(first, [[79.5 + focal / 4, 59.5]], atol=1e-12)
    np.testing.assert_allclose(second - first, [[-0.25 * focal / 4, 0]], atol=1e-12)


def test_failed_synthetic_relation_is_retained(tmp_path: Path) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "configs/development/causal_history_fixture_design_v1.json"
    )
    data = json.loads(source.read_text(encoding="utf-8-sig"))
    # Synthetic degenerate equal boxes test failure handling, not candidate fitting.
    for pair in data["pairs"]:
        pair["background_x"][1] = pair["background_x"][0]
    path = tmp_path / "degenerate.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    report = check_candidate(path)
    assert report["status"] == "ANALYTIC_FAILED"
    assert all(f"pair{pair}:target_masks_different" in report["failures"] for pair in (1, 2, 3))


@pytest.mark.parametrize("field", ["target", "decision"])
@pytest.mark.parametrize("value", [-1, 0, 1, 999, 1.5, True, False, None, "3"])
def test_invalid_timing_fails_before_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "configs/development/causal_history_fixture_design_v1.json"
    )
    data = json.loads(source.read_text(encoding="utf-8-sig"))
    data["pairs"][0][field] = value
    path = tmp_path / "invalid-timing.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    def deny_geometry(*args: object) -> None:
        pytest.fail("invalid timing reached geometry calculation")

    monkeypatch.setattr(Scene, "frame", deny_geometry)
    monkeypatch.setattr(Scene, "transport", deny_geometry)
    report = check_candidate(path)
    assert report["status"] == "ANALYTIC_FAILED"
    assert report["failures"] == ["pair1:invalid_target_timing"]
    assert report["pairs"] == []


@pytest.mark.parametrize("field", ["target", "decision"])
def test_missing_timing_fails_closed(tmp_path: Path, field: str) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "configs/development/causal_history_fixture_design_v1.json"
    )
    data = json.loads(source.read_text(encoding="utf-8-sig"))
    del data["pairs"][0][field]
    path = tmp_path / "missing-timing.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert check_candidate(path)["failures"] == ["pair1:invalid_target_timing"]


@pytest.mark.parametrize("value", [1, -1, True, False, None, "0", [], float("nan"), float("inf")])
def test_unsupported_plane_height_fails_before_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: object
) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "configs/development/causal_history_fixture_design_v1.json"
    )
    data = json.loads(source.read_text(encoding="utf-8-sig"))
    data["support"]["z"] = value
    path = tmp_path / "invalid-plane.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    def deny_geometry(*args: object) -> None:
        pytest.fail("unsupported plane height reached geometry calculation")

    monkeypatch.setattr(Scene, "frame", deny_geometry)
    monkeypatch.setattr(Scene, "transport", deny_geometry)
    report = check_candidate(path)
    assert report["status"] == "ANALYTIC_FAILED"
    assert report["failures"] == ["support:unsupported_z"]
    assert report["pairs"] == []


def test_missing_plane_height_fails_closed(tmp_path: Path) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "configs/development/causal_history_fixture_design_v1.json"
    )
    data = json.loads(source.read_text(encoding="utf-8-sig"))
    del data["support"]["z"]
    path = tmp_path / "missing-plane.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert check_candidate(path)["failures"] == ["support:unsupported_z"]
