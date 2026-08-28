from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from epsbench.data import DatasetLoader, DatasetValidationError, validate_dataset
from epsbench.schema import ArtifactRecord, ModalityPermissionSet
from epsbench.utils.canonical import canonical_json_bytes
from tests.dataset_mutations import (
    commit_episode_payloads,
    commit_resolved_config_payload,
    load_episode_payloads,
    load_manifest,
    rewrite_array_artifact,
)


def _copy_dataset(source: Path, tmp_path: Path, name: str) -> Path:
    target = tmp_path / name
    shutil.copytree(source, target)
    return target


@pytest.mark.parametrize("fixture_name", ["smoke_dataset", "corridor_dataset"])
@pytest.mark.parametrize("corruption", ["raw_id_permutation", "position", "size"])
def test_compiled_apparatus_contract_rejects_hash_rebuilt_false_instrumentation(
    fixture_name: str,
    corruption: str,
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    source = request.getfixturevalue(fixture_name)
    assert isinstance(source, Path)
    broken = _copy_dataset(source, tmp_path, f"{fixture_name}-{corruption}")
    transition, instrumentation = load_episode_payloads(broken, 0)
    names = tuple(instrumentation["raw_geom_ids"])
    first, second = names[:2]
    if corruption == "raw_id_permutation":
        first_id = instrumentation["raw_geom_ids"][first]
        instrumentation["raw_geom_ids"][first] = instrumentation["raw_geom_ids"][second]
        instrumentation["raw_geom_ids"][second] = first_id
    elif corruption == "position":
        instrumentation["raw_geom_world_positions"][first][0] += 0.01
    else:
        instrumentation["raw_geom_compiled_sizes"][first][0] += 0.01
    commit_episode_payloads(broken, 0, transition, instrumentation)

    expected = "identifiers" if corruption == "raw_id_permutation" else "geometry|position|size"
    with pytest.raises(DatasetValidationError, match=expected):
        validate_dataset(broken)


@pytest.mark.parametrize("fixture_name", ["smoke_dataset", "corridor_dataset"])
def test_resolved_render_dimensions_are_independently_bound_to_every_frame(
    fixture_name: str,
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    source = request.getfixturevalue(fixture_name)
    assert isinstance(source, Path)
    broken = _copy_dataset(source, tmp_path, f"{fixture_name}-render-dimensions")
    manifest = load_manifest(broken)
    config_path = broken / manifest.resolved_config.path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["render"]["width"] += 1
    commit_resolved_config_payload(broken, config)

    with pytest.raises(DatasetValidationError, match="frame dimensions"):
        validate_dataset(broken)


@pytest.mark.parametrize("fixture_name", ["smoke_dataset", "corridor_dataset"])
def test_scene_specific_raster_dimensions_equal_resolved_render_configuration(
    fixture_name: str,
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    source = request.getfixturevalue(fixture_name)
    assert isinstance(source, Path)
    broken = _copy_dataset(source, tmp_path, f"{fixture_name}-privileged-raster")
    transition, instrumentation = load_episode_payloads(broken, 0)
    if fixture_name == "smoke_dataset":
        evidence = instrumentation["occlusion_oracle"]["frames"][0]
        expected_error = "counterfactual segmentation"
        record_payload = evidence["counterfactual_segmentation"]
    else:
        evidence = instrumentation["raw_segmentation_frames"][0]
        expected_error = "corridor raw segmentation"
        record_payload = evidence["raw_segmentation"]
    record = ArtifactRecord.model_validate_json(canonical_json_bytes(record_payload))
    array = np.load(broken / record.path, allow_pickle=False)
    changed_record = rewrite_array_artifact(broken, record, array[:-1, :])
    if fixture_name == "smoke_dataset":
        evidence["counterfactual_segmentation"] = changed_record.model_dump(mode="json")
    else:
        evidence["raw_segmentation"] = changed_record.model_dump(mode="json")
    commit_episode_payloads(broken, 0, transition, instrumentation)

    with pytest.raises(DatasetValidationError, match=expected_error):
        validate_dataset(broken)


def test_ecological_loader_has_no_public_manifest_surface(smoke_dataset: Path) -> None:
    ecological = DatasetLoader(smoke_dataset, ModalityPermissionSet.ecological_only())
    public_names = {name for name in dir(ecological) if not name.startswith("_")}
    assert "manifest" not in public_names
    assert not hasattr(ecological, "manifest")
    with pytest.raises(PermissionError, match="modality permission denied"):
        ecological.read_dataset_manifest()

    privileged = DatasetLoader(smoke_dataset, ModalityPermissionSet.all_modalities())
    assert privileged.read_dataset_manifest() == load_manifest(smoke_dataset)


@pytest.mark.skipif(os.name == "nt", reason="POSIX symbolic-link regression")
def test_external_manifest_symlink_is_rejected_by_validation_and_loading(
    smoke_dataset: Path,
    tmp_path: Path,
) -> None:
    broken = _copy_dataset(smoke_dataset, tmp_path, "external-manifest-symlink")
    manifest_path = broken / "manifest.json"
    external = tmp_path / "external-manifest.json"
    external.write_bytes(manifest_path.read_bytes())
    manifest_path.unlink()
    manifest_path.symlink_to(external)

    with pytest.raises(DatasetValidationError, match="symbolic-link alias"):
        validate_dataset(broken)
    with pytest.raises(ValueError, match="symbolic-link alias"):
        DatasetLoader(broken, ModalityPermissionSet.all_modalities())


def test_external_manifest_hardlink_is_rejected_by_validation_and_loading(
    smoke_dataset: Path,
    tmp_path: Path,
) -> None:
    broken = _copy_dataset(smoke_dataset, tmp_path, "external-manifest-hardlink")
    manifest_path = broken / "manifest.json"
    external = tmp_path / "external-hardlink-target.json"
    external.write_bytes(manifest_path.read_bytes())
    manifest_path.unlink()
    os.link(external, manifest_path)

    with pytest.raises(DatasetValidationError, match="hard-link alias"):
        validate_dataset(broken)
    with pytest.raises(ValueError, match="hard-link alias"):
        DatasetLoader(broken, ModalityPermissionSet.all_modalities())


@pytest.mark.skipif(os.name == "nt", reason="POSIX special-file regression")
def test_special_file_manifest_is_rejected_before_reading(
    smoke_dataset: Path,
    tmp_path: Path,
) -> None:
    broken = _copy_dataset(smoke_dataset, tmp_path, "special-file-manifest")
    manifest_path = broken / "manifest.json"
    manifest_path.unlink()
    mkfifo = getattr(os, "mkfifo", None)
    assert mkfifo is not None
    mkfifo(manifest_path)

    with pytest.raises(DatasetValidationError, match="regular file"):
        validate_dataset(broken)
    with pytest.raises(ValueError, match="regular file"):
        DatasetLoader(broken, ModalityPermissionSet.all_modalities())
