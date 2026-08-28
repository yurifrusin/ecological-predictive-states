from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from epsbench.appearance import (
    load_appearance_registry,
    load_evaluation_seed_registry,
    seed_registry_hash,
)
from epsbench.config import load_config
from epsbench.data import DatasetValidationError, generate_dataset, validate_dataset
from epsbench.schema import parse_privileged_instrumentation_json
from epsbench.utils.canonical import write_canonical_json
from tests.dataset_mutations import _write_manifest, load_manifest, rewrite_json_artifact


def test_validation_is_independent_of_the_ambient_repository_registry(
    smoke_dataset: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty_working_directory = tmp_path / "no-repository-configs"
    empty_working_directory.mkdir()
    monkeypatch.chdir(empty_working_directory)
    validated = validate_dataset(smoke_dataset)
    assert validated.evaluation_seed_registry_snapshot.path == (
        "evaluation_seed_registry_snapshot.json"
    )


def test_dataset_snapshot_reconstructs_candidate_schedule_and_instance(
    tmp_path: Path,
) -> None:
    appearance_registry = load_appearance_registry(Path("configs/appearance_candidates_v0.yaml"))
    seed_registry = load_evaluation_seed_registry(
        Path("configs/evaluation_seed_candidates_v0.yaml")
    )
    base = load_config(Path("configs/benchmark_v0.yaml"))
    candidate_root = seed_registry.candidate_episode_seeds[2]
    config = type(base).model_validate({**base.model_dump(mode="python"), "seed": candidate_root})
    output = tmp_path / "candidate-dataset"
    manifest = generate_dataset(
        config,
        1,
        output,
        appearance_registry=appearance_registry,
        seed_registry=seed_registry,
    )
    validate_dataset(output)
    instrumentation = parse_privileged_instrumentation_json(
        (output / manifest.episodes[0].privileged_instrumentation.path).read_bytes()
    )
    appearance = instrumentation.appearance
    assert manifest.evaluation_seed_registry_sha256 == seed_registry_hash(seed_registry)
    assert appearance.evaluation_seed_registry_sha256 == seed_registry_hash(seed_registry)
    assert appearance.evaluation_seed_registry_version == seed_registry.registry_version
    assert appearance.assignment_schedule_source == ("snapshotted_evaluation_seed_registry_v1")
    assert appearance.assignment_root_seed == candidate_root
    assert appearance.seeds.candidate_schedule_index == 2
    assert appearance.assignment_schedule_posture == "candidate_registry_index"


def test_fully_rehashed_seed_registry_snapshot_mutation_fails(
    smoke_dataset: Path,
    tmp_path: Path,
) -> None:
    broken = tmp_path / "snapshot-corruption"
    shutil.copytree(smoke_dataset, broken)
    manifest = load_manifest(broken)
    snapshot_path = broken / manifest.evaluation_seed_registry_snapshot.path
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    payload["candidate_episode_seeds"] = list(reversed(payload["candidate_episode_seeds"]))
    snapshot_record = rewrite_json_artifact(
        broken,
        manifest.evaluation_seed_registry_snapshot,
        payload,
    )
    changed = manifest.model_copy(
        update={
            "evaluation_seed_registry_snapshot": snapshot_record,
            "evaluation_seed_registry_sha256": snapshot_record.logical_sha256,
        }
    )
    _write_manifest(broken, changed, list(changed.episodes))
    with pytest.raises(DatasetValidationError, match="seed registry snapshot"):
        validate_dataset(broken)


def test_fully_rehashed_declared_seed_registry_hash_mutation_fails(
    smoke_dataset: Path,
    tmp_path: Path,
) -> None:
    broken = tmp_path / "declared-hash-corruption"
    shutil.copytree(smoke_dataset, broken)
    manifest = load_manifest(broken)
    changed = manifest.model_copy(update={"evaluation_seed_registry_sha256": "0" * 64})
    _write_manifest(broken, changed, list(changed.episodes))
    with pytest.raises(DatasetValidationError, match="seed registry hash mismatch"):
        validate_dataset(broken)


def test_seed_registry_snapshot_symlink_is_rejected_before_open(
    smoke_dataset: Path,
    tmp_path: Path,
) -> None:
    # EPS-ER11-0002
    broken = tmp_path / "seed-snapshot-symlink"
    shutil.copytree(smoke_dataset, broken)
    manifest = load_manifest(broken)
    snapshot = broken / manifest.evaluation_seed_registry_snapshot.path
    external = tmp_path / "external-seed-snapshot.json"
    external.write_bytes(snapshot.read_bytes())
    snapshot.unlink()
    try:
        snapshot.symlink_to(external)
    except OSError as error:
        pytest.skip(f"symbolic links are unavailable: {error}")
    with pytest.raises(DatasetValidationError, match="symbolic-link alias"):
        validate_dataset(broken)


def test_seed_registry_snapshot_external_hardlink_is_rejected_before_open(
    smoke_dataset: Path,
    tmp_path: Path,
) -> None:
    broken = tmp_path / "seed-snapshot-hardlink"
    shutil.copytree(smoke_dataset, broken)
    manifest = load_manifest(broken)
    snapshot = broken / manifest.evaluation_seed_registry_snapshot.path
    external = tmp_path / "external-seed-hardlink.json"
    external.write_bytes(snapshot.read_bytes())
    snapshot.unlink()
    os.link(external, snapshot)
    with pytest.raises(DatasetValidationError, match="hard-link alias"):
        validate_dataset(broken)


def test_seed_registry_snapshot_special_file_is_rejected_before_open(
    smoke_dataset: Path,
    tmp_path: Path,
) -> None:
    broken = tmp_path / "seed-snapshot-special"
    shutil.copytree(smoke_dataset, broken)
    manifest = load_manifest(broken)
    snapshot = broken / manifest.evaluation_seed_registry_snapshot.path
    snapshot.unlink()
    snapshot.mkdir()
    with pytest.raises(DatasetValidationError, match="regular file"):
        validate_dataset(broken)


def test_seed_registry_snapshot_duplicate_path_is_rejected(
    smoke_dataset: Path,
    tmp_path: Path,
) -> None:
    broken = tmp_path / "seed-snapshot-duplicate"
    shutil.copytree(smoke_dataset, broken)
    manifest = load_manifest(broken)
    changed = manifest.model_copy(
        update={"evaluation_seed_registry_snapshot": manifest.appearance_registry_snapshot}
    )
    _write_manifest(broken, changed, list(changed.episodes))
    with pytest.raises(DatasetValidationError, match="duplicate artifact path"):
        validate_dataset(broken)


def test_seed_registry_snapshot_root_escape_is_rejected_by_schema(
    smoke_dataset: Path,
    tmp_path: Path,
) -> None:
    broken = tmp_path / "seed-snapshot-root-escape"
    shutil.copytree(smoke_dataset, broken)
    manifest_path = broken / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["evaluation_seed_registry_snapshot"]["path"] = "../external-seeds.json"
    write_canonical_json(manifest_path, payload)
    with pytest.raises(DatasetValidationError, match="manifest failed schema validation"):
        validate_dataset(broken)
