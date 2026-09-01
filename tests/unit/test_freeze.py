"""Prospective, non-rendering tests for Appearance Benchmark Input Freeze v0."""

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from epsbench.appearance import (
    AppearanceRevision1Registry,
    FinalEvaluationSeedRegistry,
    appearance_profile_hash,
    appearance_registry_hash,
    load_appearance_registry_any,
    parse_seed_registry,
    profile_by_id,
    seed_registry_hash,
)
from epsbench.audit import _dataset_cell
from epsbench.config import load_config
from epsbench.freeze import (
    EXCLUDED_PROFILE_IDS,
    LOGICAL_DOMAINS,
    SELECTED_PROFILE_IDS,
    BenchmarkDefinition,
    FreezeDefinitionLock,
    _atomic_publish_receipt,
    _domain_hash,
    _legacy_control_membership_domain,
    _readiness_summary,
    _selected_membership_domain,
    _source_identity_from_dataset,
    benchmark_definition_hash,
    create_definition_lock_payload,
    evaluation_seed_registry_hash,
    load_benchmark_definition,
    load_final_evaluation_seeds,
    validate_definition_lock,
)
from epsbench.revision import validate_definition_lock as validate_revision1_lock
from epsbench.utils.canonical import canonical_json_bytes, sha256_file

DEFINITION = Path("configs/appearance_benchmark_freeze_v0.yaml")
SEEDS = Path("configs/appearance_benchmark_v0_evaluation_episode_seeds.yaml")
LOCK = Path("configs/appearance_benchmark_freeze_v0_lock.json")
REVISION = Path("configs/appearance_candidates_revision1.yaml")
SINGLE = Path("configs/benchmark_v0.yaml")
CORRIDOR = Path("configs/corridor_v0.yaml")

PROTECTED_FILE_HASHES = {
    "configs/appearance_candidates_v0.yaml": (
        "94ce33d78430a66fb654ceb12f1eab669194fe665c586821e1a19582fde737e7"
    ),
    "configs/evaluation_seed_candidates_v0.yaml": (
        "7263438f5f25d791a06cf6e802c00d907e678c79f8bdd499bbbffc2c8feaa2e0"
    ),
    "configs/appearance_candidates_revision1.yaml": (
        "25ca9cac05bc6d0bc4415bd1dd4f2a7fac7419ddaf52b5ba93f6d9390158d645"
    ),
    "configs/appearance_revision1_qualification_seeds_v0.yaml": (
        "d90798acd02464234fdc08f1f22e6481ebaed11203b754b32d0b437535ffae4a"
    ),
    "configs/appearance_candidate_revision1_lock.json": (
        "201603df9a69eb612cc71168319e17a2c7f130cb4cf978bc2a5a53c8df6749f5"
    ),
}


def _definition_payload() -> dict[str, object]:
    return load_benchmark_definition(DEFINITION).model_dump(mode="json")


def _seed_payload() -> dict[str, object]:
    return load_final_evaluation_seeds(SEEDS).model_dump(mode="json")


def test_protected_canonical_definition_files_are_byte_identical() -> None:
    assert {path: sha256_file(Path(path)) for path in PROTECTED_FILE_HASHES} == (
        PROTECTED_FILE_HASHES
    )


def test_protected_revision1_roots_and_profile_hashes_remain_exact() -> None:
    revision = load_appearance_registry_any(REVISION)
    assert isinstance(revision, AppearanceRevision1Registry)
    assert appearance_registry_hash(revision) == (
        "81da1bb9414e7c53e42bbf65b198aa61d8bb7ed3f81eb2bf5813a1edff245ac4"
    )
    definition = load_benchmark_definition(DEFINITION)
    for role in definition.profiles:
        assert appearance_profile_hash(profile_by_id(revision, role.profile_id)) == (
            role.profile_sha256
        )
    revision_lock = validate_revision1_lock(
        Path("configs/appearance_candidates_v0.yaml"),
        Path("configs/evaluation_seed_candidates_v0.yaml"),
        REVISION,
        Path("configs/appearance_revision1_qualification_seeds_v0.yaml"),
        Path("configs/appearance_candidate_revision1_lock.json"),
        Path("configs/appearance_revision1_baseline_failure_analysis.json"),
    )
    assert revision_lock["definition_lock_sha256"] == (
        "71d2ed7a9f45c55bf17ec518c08b5d0b827a7cf0ae2cc3c339dc09197c55f633"
    )


def test_role_map_is_exact_and_semantically_distinct() -> None:
    definition = load_benchmark_definition(DEFINITION)
    selected = [item for item in definition.profiles if item.selection_status == "selected"]
    excluded = [
        item
        for item in definition.profiles
        if item.selection_status == "excluded_negative_evidence"
    ]
    assert tuple(item.profile_id for item in selected) == SELECTED_PROFILE_IDS
    assert tuple(item.profile_id for item in excluded) == EXCLUDED_PROFILE_IDS
    assert sum(item.primary_id_reference for item in definition.profiles) == 1
    assert sum(item.training_appearance_eligible for item in definition.profiles) == 1
    assert sum(item.held_out_appearance_ood for item in definition.profiles) == 4
    checker_high = next(
        item for item in definition.profiles if item.profile_id == "revision1_checker_high_v1"
    )
    combined = next(
        item for item in definition.profiles if item.profile_id == "revision1_combined_stress_v1"
    )
    assert checker_high.secondary_diagnostic_control_profile_id == "revision1_checker_low_v1"
    assert checker_high.benchmark_primary_reference_profile_id == (
        "revision1_balanced_reference_v1"
    )
    assert combined.role.value == "held_out_combined_appearance_ood"


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("duplicate", None),
        ("unknown_role", "unknown_role"),
        ("select_excluded", "selected"),
        ("training_ood", True),
        ("wrong_reference", "revision1_checker_low_v1"),
    ],
)
def test_role_map_mutations_fail_closed(mutation: str, value: object) -> None:
    payload = deepcopy(_definition_payload())
    profiles = payload["profiles"]
    assert isinstance(profiles, list)
    if mutation == "duplicate":
        profiles[1] = deepcopy(profiles[0])
    elif mutation == "unknown_role":
        profiles[1]["role"] = value
    elif mutation == "select_excluded":
        profiles[5]["selection_status"] = value
    elif mutation == "training_ood":
        profiles[1]["training_appearance_eligible"] = value
    else:
        profiles[1]["benchmark_primary_reference_profile_id"] = value
    with pytest.raises(ValueError):
        BenchmarkDefinition.model_validate_json(json.dumps(payload))


def test_final_evaluation_seed_registry_is_exact_unique_disjoint_and_locked() -> None:
    seeds = load_final_evaluation_seeds(SEEDS)
    assert isinstance(seeds, FinalEvaluationSeedRegistry)
    assert seeds.indices == tuple(range(16))
    assert len(set(seeds.candidate_episode_seeds)) == 16
    assert evaluation_seed_registry_hash(seeds) == (
        "6747d234aa5e843e1a09013b978c0e8ce55b4c71c0f3bb820cee46176b618652"
    )
    assert seed_registry_hash(seeds) != evaluation_seed_registry_hash(seeds)
    lock = validate_definition_lock(DEFINITION, SEEDS, LOCK, REVISION, SINGLE, CORRIDOR)
    assert lock["evaluation_episode_seed_registry_sha256"] == evaluation_seed_registry_hash(seeds)


def test_final_seed_registry_keeps_distinct_artifact_and_freeze_identities(
    tmp_path: Path,
) -> None:
    seeds = load_final_evaluation_seeds(SEEDS)
    revision = load_appearance_registry_any(REVISION)
    assert isinstance(revision, AppearanceRevision1Registry)
    config = load_config(SINGLE)
    profile = profile_by_id(revision, "revision1_balanced_reference_v1")
    packet = tmp_path / "packet"
    packet.mkdir()

    cell = _dataset_cell(
        tmp_path / "dataset",
        tmp_path / "repeat",
        packet,
        config,
        profile,
        seeds.indices[0],
        seeds.candidate_episode_seeds[0],
        revision,
        seeds,
        cell_id_prefix="final-evaluation-regression",
        retain_source_evidence=True,
    )

    assert cell["generation_status"] == "success"
    assert cell["evaluation_seed_registry_sha256"] == seed_registry_hash(seeds)
    assert cell["evaluation_seed_registry_sha256"] != evaluation_seed_registry_hash(seeds)
    assert (packet / cell["source_evidence"]["dataset_path"]).is_dir()


@pytest.mark.parametrize(
    "mutation",
    ["reorder", "remove", "add", "duplicate", "replace", "boolean", "float", "negative"],
)
def test_final_evaluation_seed_registry_mutations_fail_closed(mutation: str) -> None:
    payload = deepcopy(_seed_payload())
    values = payload["candidate_episode_seeds"]
    assert isinstance(values, list)
    if mutation == "reorder":
        values[0], values[1] = values[1], values[0]
    elif mutation == "remove":
        values.pop()
    elif mutation == "add":
        values.append(1)
    elif mutation == "duplicate":
        values[1] = values[0]
    elif mutation == "replace":
        values[0] = 1
    elif mutation == "boolean":
        values[0] = True
    elif mutation == "float":
        values[0] = 1.0
    else:
        values[0] = -1
    with pytest.raises(ValueError):
        parse_seed_registry(payload)


def test_definition_lock_and_membership_are_independently_recomputed() -> None:
    definition = load_benchmark_definition(DEFINITION)
    seeds = load_final_evaluation_seeds(SEEDS)
    revision = load_appearance_registry_any(REVISION)
    assert isinstance(revision, AppearanceRevision1Registry)
    expected = create_definition_lock_payload(
        definition,
        seeds,
        revision,
        (load_config(SINGLE), load_config(CORRIDOR)),
    )
    actual = json.loads(LOCK.read_text(encoding="utf-8"))
    assert actual == expected
    assert canonical_json_bytes(actual) + b"\n" == LOCK.read_bytes()
    FreezeDefinitionLock.model_validate_json(canonical_json_bytes(actual))
    assert len(_selected_membership_domain(definition, seeds)) == 160
    assert len(_legacy_control_membership_domain(definition, seeds)) == 32
    assert benchmark_definition_hash(definition) == actual["benchmark_definition_sha256"]


def test_replacement_lock_preserves_and_supersedes_exact_historical_lock() -> None:
    lock = validate_definition_lock(DEFINITION, SEEDS, LOCK, REVISION, SINGLE, CORRIDOR)
    assert lock["schema_version"] == "appearance_benchmark_freeze_definition_lock_v1"
    assert lock["reviewed_implementation_head"] == ("8b34b78d5488af7103119697a286cfb8757cc125")
    assert lock["superseded_definition_lock_commit"] == ("1a5929307dfcba1d726c650f5e1ce68771f66801")
    assert lock["superseded_definition_lock_sha256"] == (
        "a373a4742a6b5a3057b820b7d925c2e19fbddc0a83cc45384b8b81ea4e8b1464"
    )
    assert lock["supersession_finding_id"] == "EPS-ER17-0004"


def test_every_corrected_logical_domain_is_unique_and_domain_separated() -> None:
    assert len(LOGICAL_DOMAINS) == len(set(LOGICAL_DOMAINS.values()))
    payload = {"same": "payload"}
    roots = {_domain_hash(name, payload) for name in LOGICAL_DOMAINS}
    assert len(roots) == len(LOGICAL_DOMAINS)


def test_model_result_renderer_selection_dependency_is_normative_and_unsatisfied() -> None:
    dependency = load_benchmark_definition(
        DEFINITION
    ).renderer_policy.model_result_renderer_selection_dependency
    assert dependency.selection_required_before_comparative_model_result_access is True
    assert dependency.primary_model_result_renderer is None
    assert dependency.other_renderer_model_result_classification is None
    assert dependency.permitted_other_renderer_classifications == (
        "replication",
        "robustness",
        "sensitivity",
        "unsupported",
    )
    assert dependency.selection_may_not_depend_on_observed_comparative_results is True
    assert dependency.averaging_or_aggregation_may_not_depend_on_observed_results is True
    assert dependency.aggregation_rule_must_be_preregistered_prospectively is True
    assert dependency.comparative_model_result_access_authorised is False


def test_source_pairing_identity_is_reconstructed_and_appearance_invariant(
    appearance_datasets: tuple[Path, Path], benchmark_config: Any
) -> None:
    base, alternate = appearance_datasets
    base_identity, _ = _source_identity_from_dataset(base, benchmark_config)
    alternate_identity, _ = _source_identity_from_dataset(alternate, benchmark_config)
    assert base_identity["schema_version"] == "appearance_benchmark_source_identity_v1"
    assert (
        base_identity["source_identity_root_sha256"]
        == (alternate_identity["source_identity_root_sha256"])
    )
    assert base_identity["dataset_logical_sha256"] != alternate_identity["dataset_logical_sha256"]


def test_renderer_selection_dependency_mutations_fail_closed() -> None:
    payload = deepcopy(_definition_payload())
    renderer = payload["renderer_policy"]
    assert isinstance(renderer, dict)
    dependency = renderer["model_result_renderer_selection_dependency"]
    assert isinstance(dependency, dict)
    dependency["selection_may_not_depend_on_observed_comparative_results"] = False
    with pytest.raises(ValueError):
        BenchmarkDefinition.model_validate_json(json.dumps(payload))


def test_receipt_publication_is_atomic_exclusive_and_outside_packet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = tmp_path / "packet"
    packet.mkdir()
    publication = tmp_path / "publication"
    publication.mkdir()
    payload = b'{"canonical":true}\n'

    destination = publication / "receipt.json"
    _atomic_publish_receipt(payload, destination, packet)
    assert destination.read_bytes() == payload
    assert destination.stat().st_nlink == 1

    existing = publication / "existing.json"
    existing.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        _atomic_publish_receipt(payload, existing, packet)
    assert existing.read_bytes() == b"existing"

    hardlink_source = publication / "hardlink-source.json"
    hardlink_source.write_bytes(b"alias")
    hardlink_target = publication / "hardlink-target.json"
    os.link(hardlink_source, hardlink_target)
    with pytest.raises(FileExistsError):
        _atomic_publish_receipt(payload, hardlink_target, packet)
    assert hardlink_target.read_bytes() == b"alias"

    inside = packet / "receipt.json"
    with pytest.raises(ValueError, match="inside its source packet"):
        _atomic_publish_receipt(payload, inside, packet)
    assert not inside.exists()

    alias = tmp_path / "packet-alias"
    try:
        alias.symlink_to(packet, target_is_directory=True)
    except OSError:
        pass
    else:
        with pytest.raises(ValueError, match="non-alias directory"):
            _atomic_publish_receipt(payload, alias / "receipt.json", packet)

    racing = publication / "racing.json"
    real_link = os.link

    def concurrent_creator(source: Any, target: Any, *args: Any, **kwargs: Any) -> None:
        Path(target).write_bytes(b"racing-winner")
        real_link(source, target, *args, **kwargs)

    monkeypatch.setattr(os, "link", concurrent_creator)
    with pytest.raises(FileExistsError):
        _atomic_publish_receipt(payload, racing, packet)
    assert racing.read_bytes() == b"racing-winner"


def test_all_cells_both_renderers_readiness_is_indivisible() -> None:
    definition = load_benchmark_definition(DEFINITION)
    local = [
        {
            "profile_id": role.profile_id,
            "profile_sha256": role.profile_sha256,
            "benchmark_role": role.role.value,
            "expected_cell_count": 32,
            "observed_cell_count": 32,
            "cell_counts": {"admitted": 32},
            "renderer_apparatus_qualified": True,
        }
        for role in definition.profiles[:5]
    ]
    roots = {"root": "a"}
    counterpart: dict[str, Any] = {
        "environment_id": "ubuntu_osmesa_locked",
        "profiles": deepcopy(local),
        "portable_apparatus_roots": roots,
    }
    ready = _readiness_summary(definition, "windows_wgl_locked", local, counterpart, roots)
    assert ready["freeze_candidate_status"] == "ready_for_dual_review"
    assert ready["seed_set_disposition"] == "eligible_for_owner_freeze_if_approved"
    counterpart["profiles"][0]["renderer_apparatus_qualified"] = False
    failed = _readiness_summary(definition, "windows_wgl_locked", local, counterpart, roots)
    assert failed["freeze_candidate_status"] == "not_ready"
    assert failed["seed_set_disposition"] == "retired_after_failed_freeze_qualification"
