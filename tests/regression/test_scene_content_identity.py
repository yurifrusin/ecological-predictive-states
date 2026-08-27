from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from epsbench.config import AppearanceConfig, CorridorConfig, SingleOccluderConfig
from epsbench.data import DatasetValidationError, validate_dataset
from epsbench.data.identity import (
    compute_content_provenance_binding,
    compute_corridor_scene_content_hash,
    compute_dataset_logical_hash,
    compute_single_occluder_scene_content_hash,
    corridor_scene_content_domain,
    single_occluder_scene_content_domain,
)
from epsbench.schema import CorridorSampledGeometry, DatasetManifest
from epsbench.sim import sample_corridor_geometry
from epsbench.utils.canonical import canonical_json_bytes, write_canonical_json

EXPECTED_SINGLE_OCCLUDER_SCENE_CONTENT_HASH = (
    "57e8b2f1df42d0a3d13e2fe3403d903a24bf7d8b5b7ce410291afcd091fdc351"
)
EXPECTED_CORRIDOR_SCENE_CONTENT_HASHES = (
    "16a1f5a214577c3d189be6a2c0e47eb4ee3dbe0dfd913b4739b3c6b8832b9525",
    "1de8d8a877a29695958917251e76044bc267616bd4da5050ac49ea909e65bffd",
)


def _single_with(
    config: SingleOccluderConfig,
    **updates: object,
) -> SingleOccluderConfig:
    return SingleOccluderConfig.model_validate({**config.model_dump(mode="python"), **updates})


def _corridor_with(
    config: CorridorConfig,
    **updates: object,
) -> CorridorConfig:
    return CorridorConfig.model_validate({**config.model_dump(mode="python"), **updates})


def _geometry_with(
    geometry: CorridorSampledGeometry,
    **updates: object,
) -> CorridorSampledGeometry:
    return CorridorSampledGeometry.model_validate({**geometry.model_dump(mode="python"), **updates})


def test_single_occluder_content_is_seed_and_appearance_independent_and_complete(
    benchmark_config: SingleOccluderConfig,
) -> None:
    alternate = _single_with(
        benchmark_config,
        seed=benchmark_config.seed + 1,
        appearance=AppearanceConfig(
            registry_version="appearance_candidate_registry_v0",
            profile_id="legacy_solid_alternate_v1",
        ).model_dump(mode="python"),
    )
    assert compute_single_occluder_scene_content_hash(benchmark_config) == (
        EXPECTED_SINGLE_OCCLUDER_SCENE_CONTENT_HASH
    )
    assert compute_single_occluder_scene_content_hash(alternate) == (
        EXPECTED_SINGLE_OCCLUDER_SCENE_CONTENT_HASH
    )

    domain = single_occluder_scene_content_domain(benchmark_config)
    assert domain["apparatus_version"] == "single_occluder_v1"
    assert domain["surfaces"] == {
        "support_surface": {
            "type": "plane",
            "position": [0.0, 0.0, 0.0],
            "size": [4.0, 7.0, 0.1],
        },
        "background_surface": {
            "type": "box",
            "position": [0.0, 2.5, 1.05],
            "half_size": [2.2, 0.05, 1.05],
        },
        "occluding_surface": {
            "type": "box",
            "position": [0.0, 0.8, 0.9],
            "half_size": [0.55, 0.05, 0.9],
        },
    }
    assert domain["camera"] == {
        "before_position": [-0.35, -3.0, 1.25],
        "after_position": [0.35, -3.0, 1.25],
        "orientation_rule": "xyaxes_1_0_0_0_0.16_1",
        "field_of_view_degrees": 55.0,
    }
    assert domain["action"] == benchmark_config.action.model_dump(mode="json")
    encoded = canonical_json_bytes(domain)
    for excluded in (
        b"episode_seed",
        b"appearance",
        b"opaque",
        b"segmentation_label",
        b"renderer",
        b"provenance",
    ):
        assert excluded not in encoded


def test_corridor_content_is_seed_history_appearance_and_remapping_independent(
    corridor_config: CorridorConfig,
) -> None:
    geometry = sample_corridor_geometry(corridor_config, episode_seed=20260826)
    alternate = _corridor_with(
        corridor_config,
        seed=corridor_config.seed + 1,
        appearance=AppearanceConfig(
            registry_version="appearance_candidate_registry_v0",
            profile_id="legacy_solid_alternate_v1",
        ).model_dump(mode="python"),
    )
    assert compute_corridor_scene_content_hash(corridor_config, geometry) == (
        compute_corridor_scene_content_hash(alternate, geometry)
    )
    encoded = canonical_json_bytes(corridor_scene_content_domain(corridor_config, geometry))
    for excluded in (
        b"episode_seed",
        b"appearance",
        b"opaque",
        b"segmentation_label",
        b"renderer",
        b"provenance",
    ):
        assert excluded not in encoded


def test_corridor_content_changes_with_geometry_camera_fov_and_action(
    corridor_config: CorridorConfig,
) -> None:
    geometry = sample_corridor_geometry(corridor_config, episode_seed=20260826)
    baseline = compute_corridor_scene_content_hash(corridor_config, geometry)
    geometry_variants = (
        _geometry_with(geometry, width=geometry.width + 0.01),
        _geometry_with(geometry, length=geometry.length + 0.01),
        _geometry_with(
            geometry,
            camera_lateral_position=geometry.camera_lateral_position + 0.01,
        ),
        _geometry_with(
            geometry,
            field_of_view_degrees=geometry.field_of_view_degrees + 1.0,
        ),
    )
    assert all(
        compute_corridor_scene_content_hash(corridor_config, variant) != baseline
        for variant in geometry_variants
    )

    changed_action = {
        **corridor_config.action.model_dump(mode="python"),
        "delta_forward": corridor_config.action.delta_forward + 0.1,
    }
    action_config = _corridor_with(corridor_config, action=changed_action)
    action_geometry = _geometry_with(
        geometry,
        camera_after_forward_position=geometry.camera_after_forward_position + 0.1,
    )
    assert compute_corridor_scene_content_hash(action_config, action_geometry) != baseline


def test_generated_scene_content_hashes_match_exact_regressions(
    smoke_dataset: Path,
    corridor_dataset: Path,
) -> None:
    single = DatasetManifest.model_validate_json(
        (smoke_dataset / "manifest.json").read_text(encoding="utf-8")
    )
    corridor = DatasetManifest.model_validate_json(
        (corridor_dataset / "manifest.json").read_text(encoding="utf-8")
    )
    assert {episode.scene_content_sha256 for episode in single.episodes} == {
        EXPECTED_SINGLE_OCCLUDER_SCENE_CONTENT_HASH
    }
    assert tuple(episode.scene_content_sha256 for episode in corridor.episodes) == (
        EXPECTED_CORRIDOR_SCENE_CONTENT_HASHES
    )


@pytest.mark.parametrize("fixture_name", ["smoke_dataset", "corridor_dataset"])
def test_validator_independently_recomputes_scene_content_domain(
    fixture_name: str,
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    source = request.getfixturevalue(fixture_name)
    assert isinstance(source, Path)
    broken = tmp_path / fixture_name
    shutil.copytree(source, broken)
    manifest = DatasetManifest.model_validate_json(
        (broken / "manifest.json").read_text(encoding="utf-8")
    )
    episodes = list(manifest.episodes)
    episodes[0] = episodes[0].model_copy(update={"scene_content_sha256": "0" * 64})
    provisional = DatasetManifest.model_validate(
        {**manifest.model_dump(mode="python"), "episodes": tuple(episodes)}
    )
    dataset_hash = compute_dataset_logical_hash(provisional)
    changed = DatasetManifest.model_validate(
        {
            **provisional.model_dump(mode="python"),
            "dataset_logical_sha256": dataset_hash,
            "content_provenance_binding_sha256": compute_content_provenance_binding(
                dataset_hash,
                provisional.source_provenance_sha256,
                provisional.renderer_execution_provenance_sha256,
            ),
        }
    )
    write_canonical_json(broken / "manifest.json", changed)

    with pytest.raises(DatasetValidationError, match="scene-content hash"):
        validate_dataset(broken)
