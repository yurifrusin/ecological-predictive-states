from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from epsbench.appearance import load_appearance_registry, load_evaluation_seed_registry
from epsbench.config import load_config
from epsbench.data import (
    DatasetLoader,
    PermissionDeniedError,
    generate_dataset,
    validate_dataset,
)
from epsbench.schema import DatasetManifest, ModalityPermissionSet, TransitionRecord


@pytest.fixture(scope="module")
def appearance_profile_matrix(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[tuple[str, str], tuple[Path, DatasetManifest, TransitionRecord]]:
    root = tmp_path_factory.mktemp("appearance-profile-matrix")
    registry = load_appearance_registry(Path("configs/appearance_candidates_v0.yaml"))
    seeds = load_evaluation_seed_registry(Path("configs/evaluation_seed_candidates_v0.yaml"))
    candidate_seed = seeds.candidate_episode_seeds[0]
    result: dict[tuple[str, str], tuple[Path, DatasetManifest, TransitionRecord]] = {}
    for config_path in (Path("configs/benchmark_v0.yaml"), Path("configs/corridor_v0.yaml")):
        base_config = load_config(config_path)
        for profile in registry.profiles:
            config = type(base_config).model_validate(
                {
                    **base_config.model_dump(mode="python"),
                    "seed": candidate_seed,
                    "appearance": {
                        "registry_version": registry.registry_version,
                        "profile_id": profile.profile_id,
                    },
                }
            )
            output = root / f"{config.scene_family.value}--{profile.profile_id}"
            manifest = generate_dataset(
                config,
                1,
                output,
                appearance_registry=registry,
                seed_registry=seeds,
            )
            validate_dataset(output)
            transition = TransitionRecord.model_validate_json(
                (output / manifest.episodes[0].transition.path).read_bytes()
            )
            result[(config.scene_family.value, profile.profile_id)] = (
                output,
                manifest,
                transition,
            )
    return result


def _arrays(root: Path, transition: TransitionRecord, role: str) -> tuple[Any, Any]:
    before = getattr(transition.before, role)
    after = getattr(transition.after, role)
    if role == "rgb":
        from PIL import Image

        with Image.open(root / before.path) as image:
            before_array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        with Image.open(root / after.path) as image:
            after_array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        return before_array, after_array
    return (
        np.load(root / before.path, allow_pickle=False),
        np.load(root / after.path, allow_pickle=False),
    )


def test_ecological_projection_contains_no_profile_or_style_slot(smoke_dataset: Path) -> None:
    loader = DatasetLoader(smoke_dataset, ModalityPermissionSet.ecological_only())
    payload = loader.read_ecological_transition(0).model_dump_json()
    assert "profile" not in payload
    assert "style-slot" not in payload
    assert "appearance" not in payload


def test_appearance_control_denial_happens_before_instrumentation_is_opened(
    smoke_dataset: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = DatasetLoader(smoke_dataset, ModalityPermissionSet.ecological_only())

    def opened(_: int) -> None:
        raise AssertionError("protected instrumentation was opened")

    monkeypatch.setattr(loader, "_instrumentation", opened)
    with pytest.raises(PermissionDeniedError, match="appearance_control"):
        loader.read_appearance_control(0)


def test_appearance_control_requires_both_private_permissions(smoke_dataset: Path) -> None:
    all_modalities = ModalityPermissionSet.all_modalities()
    loader = DatasetLoader(smoke_dataset, all_modalities)
    record = loader.read_appearance_control(0)
    assert record.profile.profile_id == "legacy_solid_base_v1"
    assert record.freeze_status == "candidate_not_frozen"


def test_every_profile_generates_validates_and_preserves_matched_control_structure(
    appearance_profile_matrix: dict[
        tuple[str, str], tuple[Path, DatasetManifest, TransitionRecord]
    ],
) -> None:
    registry = load_appearance_registry(Path("configs/appearance_candidates_v0.yaml"))
    assert len(appearance_profile_matrix) == 20
    for scene in ("single_occluder", "corridor"):
        for profile in registry.profiles:
            root, manifest, transition = appearance_profile_matrix[(scene, profile.profile_id)]
            control_root, control_manifest, control_transition = appearance_profile_matrix[
                (scene, profile.matched_control_profile_id)
            ]
            episode = manifest.episodes[0]
            control_episode = control_manifest.episodes[0]
            for field in (
                "scene_content_sha256",
                "analytic_transport_sha256",
                "oriented_boundary_sha256",
                "visibility_event_sha256",
                "ecological_label_sha256",
            ):
                assert getattr(episode, field) == getattr(control_episode, field)
            assert transition.action == control_transition.action
            assert transition.occlusion == control_transition.occlusion
            for role in ("depth", "segmentation"):
                observed = _arrays(root, transition, role)
                control = _arrays(control_root, control_transition, role)
                assert all(
                    np.array_equal(first, second)
                    for first, second in zip(observed, control, strict=True)
                )
            if profile.freeze_eligible:
                observed_rgb = _arrays(root, transition, "rgb")
                control_rgb = _arrays(control_root, control_transition, "rgb")
                assert all(
                    not np.array_equal(first, second)
                    for first, second in zip(observed_rgb, control_rgb, strict=True)
                )
