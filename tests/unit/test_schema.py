import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from epsbench.schema import (
    CameraInstrumentation,
    DatasetManifest,
    OcclusionRelation,
    TransitionRecord,
    VisibilityState,
)
from epsbench.utils.canonical import canonical_json_bytes


def test_transition_and_manifest_round_trip(smoke_dataset: Path) -> None:
    manifest_bytes = (smoke_dataset / "manifest.json").read_bytes().rstrip(b"\n")
    manifest = DatasetManifest.model_validate_json(manifest_bytes)
    assert canonical_json_bytes(manifest) == manifest_bytes
    transition_path = smoke_dataset / manifest.episodes[0].transition.path
    transition_bytes = transition_path.read_bytes().rstrip(b"\n")
    transition = TransitionRecord.model_validate_json(transition_bytes)
    assert canonical_json_bytes(transition) == transition_bytes


def test_invalid_visibility_fraction_fails() -> None:
    with pytest.raises(ValidationError):
        VisibilityState(
            surface_id="surface-0123456789abcdef",
            before_visible_pixels=1,
            after_visible_pixels=1,
            before_projected_image_fraction=1.01,
            after_projected_image_fraction=0.5,
        )


def test_duplicate_surface_identifiers_fail(smoke_dataset: Path) -> None:
    manifest = DatasetManifest.model_validate_json(
        (smoke_dataset / "manifest.json").read_text(encoding="utf-8")
    )
    transition_path = smoke_dataset / manifest.episodes[0].transition.path
    payload = json.loads(transition_path.read_text(encoding="utf-8"))
    payload["surfaces"][1]["surface_id"] = payload["surfaces"][0]["surface_id"]
    with pytest.raises(ValidationError):
        TransitionRecord.model_validate_json(json.dumps(payload))


def test_malformed_occlusion_relation_fails() -> None:
    with pytest.raises(ValidationError):
        OcclusionRelation(
            occluder_surface_id="surface-0123456789abcdef",
            occluded_surface_id="surface-0123456789abcdef",
            frame_indices=(0,),
        )


@pytest.mark.parametrize(
    ("position", "rotation"),
    [
        ((float("nan"), 0.0, 0.0), (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)),
        ((0.0, 0.0, 0.0), (float("inf"), 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)),
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0)),
    ],
)
def test_nonfinite_or_nonorthonormal_camera_pose_fails(
    position: tuple[float, float, float],
    rotation: tuple[float, ...],
) -> None:
    with pytest.raises(ValidationError):
        CameraInstrumentation(
            frame_index=0,
            camera_world_position=position,
            camera_world_rotation_row_major=rotation,
        )
