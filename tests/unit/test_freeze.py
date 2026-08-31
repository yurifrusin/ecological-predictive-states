"""Prospective, non-rendering tests for Appearance Benchmark Input Freeze v0."""

import json
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
from epsbench.config import load_config
from epsbench.freeze import (
    EXCLUDED_PROFILE_IDS,
    SELECTED_PROFILE_IDS,
    BenchmarkDefinition,
    FreezeDefinitionLock,
    _legacy_control_membership_domain,
    _readiness_summary,
    _selected_membership_domain,
    benchmark_definition_hash,
    create_definition_lock_payload,
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
    assert seed_registry_hash(seeds) == (
        "eb3ca6083af203b325a16612590f0f2e56efdf4b99f1019148f01bd2caab94b2"
    )
    lock = validate_definition_lock(DEFINITION, SEEDS, LOCK, REVISION, SINGLE, CORRIDOR)
    assert lock["evaluation_episode_seed_registry_sha256"] == seed_registry_hash(seeds)


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
