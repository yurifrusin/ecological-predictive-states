from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from epsbench.data import DatasetLoader, PermissionDeniedError, validate_dataset
from epsbench.data.identity import (
    compute_renderer_execution_provenance_hash,
    compute_source_provenance_hash,
)
from epsbench.schema import (
    CorridorInstrumentation,
    DatasetManifest,
    ModalityPermissionSet,
    SceneFamily,
    TransitionRecord,
    UnavailableOcclusionAnnotation,
    parse_privileged_instrumentation_json,
)

EXPECTED_CORRIDOR_SURFACES = {
    "corridor_floor",
    "corridor_left_surface",
    "corridor_right_surface",
    "corridor_end_surface",
}


def _manifest(root: Path) -> DatasetManifest:
    return DatasetManifest.model_validate_json((root / "manifest.json").read_text(encoding="utf-8"))


def _transition(root: Path, episode_index: int) -> TransitionRecord:
    manifest = _manifest(root)
    return TransitionRecord.model_validate_json(
        (root / manifest.episodes[episode_index].transition.path).read_text(encoding="utf-8")
    )


def _instrumentation(root: Path, episode_index: int) -> CorridorInstrumentation:
    manifest = _manifest(root)
    instrumentation = parse_privileged_instrumentation_json(
        (root / manifest.episodes[episode_index].privileged_instrumentation.path).read_text(
            encoding="utf-8"
        )
    )
    assert isinstance(instrumentation, CorridorInstrumentation)
    return instrumentation


def test_corridor_regeneration_is_byte_identical(
    deterministic_corridor_datasets: tuple[Path, Path],
) -> None:
    first, second = deterministic_corridor_datasets
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()
    for episode_index in (0, 1):
        first_manifest = validate_dataset(first)
        second_manifest = validate_dataset(second)
        first_transition = first / first_manifest.episodes[episode_index].transition.path
        second_transition = second / second_manifest.episodes[episode_index].transition.path
        assert first_transition.read_bytes() == second_transition.read_bytes()


def test_corridor_source_and_renderer_provenance_are_bound(corridor_dataset: Path) -> None:
    manifest = validate_dataset(corridor_dataset)
    assert compute_source_provenance_hash(manifest.source_provenance) == (
        manifest.source_provenance_sha256
    )
    assert compute_renderer_execution_provenance_hash(manifest.renderer_provenance) == (
        manifest.renderer_execution_provenance_sha256
    )


def test_episode_seed_controls_corridor_geometry_and_content_identity(
    corridor_dataset: Path,
) -> None:
    manifest = validate_dataset(corridor_dataset)
    first = _instrumentation(corridor_dataset, 0)
    second = _instrumentation(corridor_dataset, 1)
    assert first.generation_seeds.episode_seed != second.generation_seeds.episode_seed
    assert first.sampled_geometry != second.sampled_geometry
    assert manifest.episodes[0].scene_content_sha256 != manifest.episodes[1].scene_content_sha256
    assert (
        manifest.episodes[0].ecological_label_sha256 != manifest.episodes[1].ecological_label_sha256
    )


def test_corridor_modalities_align_and_every_required_surface_is_visible(
    corridor_dataset: Path,
) -> None:
    loader = DatasetLoader(corridor_dataset, ModalityPermissionSet.all_modalities())
    instrumentation = _instrumentation(corridor_dataset, 0)
    transition = _transition(corridor_dataset, 0)
    assert set(instrumentation.raw_geom_ids) == EXPECTED_CORRIDOR_SURFACES
    assert set(instrumentation.apparatus_surface_names) == EXPECTED_CORRIDOR_SURFACES
    declared_labels = {surface.segmentation_label for surface in transition.surfaces}
    observed_labels: set[int] = set()
    for frame_index in (0, 1):
        rgb = loader.read_rgb(0, frame_index)
        depth = loader.read_depth(0, frame_index)
        segmentation = loader.read_segmentation(0, frame_index)
        assert rgb.shape[:2] == depth.shape == segmentation.shape
        assert np.all(segmentation != 0)
        observed_labels.update(int(value) for value in np.unique(segmentation) if value != 0)
    assert observed_labels == declared_labels


def test_corridor_surface_mapping_is_exact_bijective_and_episode_local(
    corridor_dataset: Path,
) -> None:
    first_transition = _transition(corridor_dataset, 0)
    second_transition = _transition(corridor_dataset, 1)
    first = _instrumentation(corridor_dataset, 0)
    raw_ids = set(first.raw_geom_ids.values())
    opaque_ids = set(first.raw_to_opaque_surface_ids.values())
    assert set(first.raw_to_opaque_surface_ids) == {str(raw_id) for raw_id in raw_ids}
    assert len(raw_ids) == len(opaque_ids) == len(EXPECTED_CORRIDOR_SURFACES)
    assert opaque_ids == {surface.surface_id for surface in first_transition.surfaces}
    assert {surface.surface_id for surface in first_transition.surfaces}.isdisjoint(
        {surface.surface_id for surface in second_transition.surfaces}
    )


def test_corridor_forward_camera_displacement_is_truthful(corridor_dataset: Path) -> None:
    loader = DatasetLoader(corridor_dataset, ModalityPermissionSet.all_modalities())
    action = loader.read_action(0)
    before = loader.read_camera_world_transform(0, 0)
    after = loader.read_camera_world_transform(0, 1)
    observed = np.subtract(after.camera_world_position, before.camera_world_position)
    assert action.name == "forward"
    assert observed == pytest.approx((0.0, action.delta_forward, 0.0))
    assert after.camera_world_rotation_row_major == pytest.approx(
        before.camera_world_rotation_row_major
    )


def test_corridor_ordinary_record_excludes_semantics_metrics_and_scene_family(
    corridor_dataset: Path,
) -> None:
    transition = _transition(corridor_dataset, 0)
    encoded = transition.model_dump_json().encode("utf-8")
    assert all(name.encode("utf-8") not in encoded for name in EXPECTED_CORRIDOR_SURFACES)
    assert b"sampled_geometry" not in encoded
    assert b"scene_family" not in encoded
    loader = DatasetLoader(corridor_dataset, ModalityPermissionSet.ecological_only())
    ecological_payload = loader.read_ecological_transition(0).model_dump(mode="json")
    assert "scene_family" not in ecological_payload
    assert "sampled_geometry" not in ecological_payload


def test_corridor_appearance_changes_rgb_only(
    corridor_appearance_datasets: tuple[Path, Path],
) -> None:
    base_path, alternate_path = corridor_appearance_datasets
    base = validate_dataset(base_path)
    alternate = validate_dataset(alternate_path)
    base_transition = _transition(base_path, 0)
    alternate_transition = _transition(alternate_path, 0)
    assert base.scene_family == alternate.scene_family == SceneFamily.CORRIDOR
    assert base.episodes[0].rgb_logical_sha256 != alternate.episodes[0].rgb_logical_sha256
    assert base.episodes[0].ecological_label_sha256 == (
        alternate.episodes[0].ecological_label_sha256
    )
    assert base.episodes[0].scene_content_sha256 == alternate.episodes[0].scene_content_sha256
    assert _instrumentation(base_path, 0).sampled_geometry == (
        _instrumentation(alternate_path, 0).sampled_geometry
    )
    assert base_transition.action == alternate_transition.action
    assert base_transition.surfaces == alternate_transition.surfaces
    assert base_transition.before.segmentation == alternate_transition.before.segmentation
    assert base_transition.after.segmentation == alternate_transition.after.segmentation
    assert base_transition.region_correspondence == alternate_transition.region_correspondence
    assert base_transition.region_mask_changes == alternate_transition.region_mask_changes
    assert base_transition.ecological_visibility_events == (
        alternate_transition.ecological_visibility_events
    )
    assert isinstance(base_transition.occlusion, UnavailableOcclusionAnnotation)
    assert base_transition.occlusion == alternate_transition.occlusion


def test_corridor_visibility_and_occlusion_claims_are_typed_unavailable(
    corridor_dataset: Path,
) -> None:
    transition = _transition(corridor_dataset, 0)
    assert transition.ecological_visibility_events.status == "unavailable"
    assert transition.ecological_visibility_events.reason_category == (
        "oriented_boundary_ownership_unavailable"
    )
    assert isinstance(transition.occlusion, UnavailableOcclusionAnnotation)
    assert transition.occlusion.reason_category == (
        "oriented_corridor_occlusion_oracle_unavailable"
    )


def test_ecological_permissions_deny_corridor_privileged_data_before_open(
    corridor_dataset: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = DatasetLoader(corridor_dataset, ModalityPermissionSet.ecological_only())

    def fail_if_opened(episode_index: int) -> None:
        raise AssertionError(f"instrumentation opened for episode {episode_index}")

    monkeypatch.setattr(loader, "_instrumentation", fail_if_opened)
    with pytest.raises(PermissionDeniedError):
        loader.read_sampled_corridor_geometry(0)
    with pytest.raises(PermissionDeniedError):
        loader.read_semantic_surface_names(0)
    with pytest.raises(PermissionDeniedError):
        loader.read_raw_mujoco_geom_ids(0)
    with pytest.raises(PermissionDeniedError):
        loader.read_raw_world_coordinates(0)
    with pytest.raises(PermissionDeniedError):
        loader.read_scene_family()

    def fail_if_transition_opened(episode_index: int) -> None:
        raise AssertionError(f"transition opened for episode {episode_index}")

    monkeypatch.setattr(loader, "_transition", fail_if_transition_opened)
    with pytest.raises(PermissionDeniedError):
        loader.read_depth(0, 0)
    with pytest.raises(PermissionDeniedError):
        loader.read_camera_world_transform(0, 0)
