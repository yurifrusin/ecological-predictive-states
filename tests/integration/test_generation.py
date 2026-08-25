from pathlib import Path

import numpy as np

from epsbench.data import DatasetLoader, validate_dataset
from epsbench.schema import DatasetManifest, ModalityPermissionSet, TransitionRecord


def _transition(root: Path, episode_index: int) -> TransitionRecord:
    manifest = DatasetManifest.model_validate_json(
        (root / "manifest.json").read_text(encoding="utf-8")
    )
    episode = manifest.episodes[episode_index]
    return TransitionRecord.model_validate_json(
        (root / episode.transition.path).read_text(encoding="utf-8")
    )


def test_same_seed_has_identical_manifests_and_annotations(
    deterministic_datasets: tuple[Path, Path],
) -> None:
    first, second = deterministic_datasets
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()
    for episode_index in (0, 1):
        first_manifest = validate_dataset(first)
        second_manifest = validate_dataset(second)
        first_transition = first / first_manifest.episodes[episode_index].transition.path
        second_transition = second / second_manifest.episodes[episode_index].transition.path
        assert first_transition.read_bytes() == second_transition.read_bytes()


def test_rgb_depth_and_segmentation_align(smoke_dataset: Path) -> None:
    loader = DatasetLoader(smoke_dataset, ModalityPermissionSet.all_modalities())
    for frame_index in (0, 1):
        rgb = loader.read_rgb(0, frame_index)
        depth = loader.read_depth(0, frame_index)
        segmentation = loader.read_segmentation(0, frame_index)
        assert rgb.shape[:2] == depth.shape == segmentation.shape


def test_segmentation_labels_are_declared_and_remapped(smoke_dataset: Path) -> None:
    transition = _transition(smoke_dataset, 0)
    loader = DatasetLoader(smoke_dataset, ModalityPermissionSet.all_modalities())
    declared = {surface.segmentation_label for surface in transition.surfaces}
    observed: set[int] = set()
    for frame_index in (0, 1):
        segmentation = loader.read_segmentation(0, frame_index)
        observed.update(int(value) for value in np.unique(segmentation) if value != 0)
    assert observed == declared


def test_appearance_change_changes_rgb_but_not_ecological_labels(
    appearance_datasets: tuple[Path, Path],
) -> None:
    base_path, alternate_path = appearance_datasets
    base = validate_dataset(base_path)
    alternate = validate_dataset(alternate_path)
    assert base.episodes[0].rgb_logical_sha256 != alternate.episodes[0].rgb_logical_sha256
    assert base.episodes[0].ecological_label_sha256 == alternate.episodes[0].ecological_label_sha256
    assert _transition(base_path, 0).surfaces == _transition(alternate_path, 0).surfaces


def test_surface_identifiers_are_episode_local(smoke_dataset: Path) -> None:
    first_ids = {surface.surface_id for surface in _transition(smoke_dataset, 0).surfaces}
    second_ids = {surface.surface_id for surface in _transition(smoke_dataset, 1).surfaces}
    assert first_ids.isdisjoint(second_ids)


def test_smoke_dataset_validates(smoke_dataset: Path) -> None:
    manifest = validate_dataset(smoke_dataset)
    assert len(manifest.episodes) == 2
