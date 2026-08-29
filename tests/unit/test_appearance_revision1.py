from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from epsbench.appearance import (
    CANONICAL_PROFILE_IDS,
    REVISION1_PROFILE_IDS,
    AppearanceRevision1Registry,
    QualificationSeedRegistry,
    appearance_profile_hash,
    appearance_registry_hash,
    load_appearance_registry,
    load_appearance_registry_any,
    load_evaluation_seed_registry,
    load_seed_registry,
    profile_by_id,
    resolve_appearance,
    seed_registry_hash,
    validate_axis_isolation,
)
from epsbench.data.provenance import collect_source_provenance
from epsbench.revision import _validate_lock_commit_snapshot, validate_definition_lock
from epsbench.utils.canonical import canonical_json_bytes, sha256_file

BASELINE_REGISTRY = Path("configs/appearance_candidates_v0.yaml")
DESIGN_SEEDS = Path("configs/evaluation_seed_candidates_v0.yaml")
REVISION_REGISTRY = Path("configs/appearance_candidates_revision1.yaml")
QUALIFICATION_SEEDS = Path("configs/appearance_revision1_qualification_seeds_v0.yaml")
DEFINITION_LOCK = Path("configs/appearance_candidate_revision1_lock.json")
BASELINE_ANALYSIS = Path("configs/appearance_revision1_baseline_failure_analysis.json")


def test_protected_registry_and_design_seed_bytes_and_hashes_are_exact() -> None:
    assert sha256_file(BASELINE_REGISTRY) == (
        "94ce33d78430a66fb654ceb12f1eab669194fe665c586821e1a19582fde737e7"
    )
    assert sha256_file(DESIGN_SEEDS) == (
        "7263438f5f25d791a06cf6e802c00d907e678c79f8bdd499bbbffc2c8feaa2e0"
    )
    assert appearance_registry_hash(load_appearance_registry(BASELINE_REGISTRY)) == (
        "f90feb3cf9d798ab61c3adbb8d2b276c5d2cb131b95c5a9187df151f9b801d60"
    )
    assert seed_registry_hash(load_evaluation_seed_registry(DESIGN_SEEDS)) == (
        "6c6816ae1f6657a710631f08a435cca4c623e81efc71ac4e6330a44338313482"
    )


def test_revision_registry_preserves_all_original_profile_hashes() -> None:
    baseline = load_appearance_registry(BASELINE_REGISTRY)
    revision = load_appearance_registry_any(REVISION_REGISTRY)
    assert isinstance(revision, AppearanceRevision1Registry)
    assert revision.revision1_admission_profile_ids == REVISION1_PROFILE_IDS
    for profile_id in CANONICAL_PROFILE_IDS:
        assert appearance_profile_hash(profile_by_id(revision, profile_id)) == (
            appearance_profile_hash(profile_by_id(baseline, profile_id))
        )
    validate_axis_isolation(revision)


def test_revision_profile_roles_are_exact_and_axis_isolated() -> None:
    revision = load_appearance_registry_any(REVISION_REGISTRY)
    assert isinstance(revision, AppearanceRevision1Registry)
    profiles = {
        profile_id: profile_by_id(revision, profile_id) for profile_id in REVISION1_PROFILE_IDS
    }
    assert profiles["revision1_balanced_reference_v1"].matched_control_profile_id == (
        "legacy_solid_base_v1"
    )
    assert [tag.value for tag in profiles["revision1_balanced_reference_v1"].axis_tags] == [
        "colour",
        "illumination",
        "combined",
    ]
    assert [tag.value for tag in profiles["revision1_colour_shift_v1"].axis_tags] == ["colour"]
    assert [tag.value for tag in profiles["revision1_checker_low_v1"].axis_tags] == [
        "texture_family"
    ]
    assert [tag.value for tag in profiles["revision1_checker_high_v1"].axis_tags] == [
        "texture_frequency"
    ]
    assert profiles["revision1_checker_high_v1"].texture.cycles_per_tile == 4
    assert profiles["revision1_checker_low_v1"].texture.cycles_per_tile == 1
    assert [tag.value for tag in profiles["revision1_stripes_low_v1"].axis_tags] == [
        "texture_family"
    ]
    assert [tag.value for tag in profiles["revision1_illumination_shift_v1"].axis_tags] == [
        "illumination"
    ]
    assert all(profile.freeze_eligible for profile in profiles.values())
    assert all(
        profile.style_assignment_rule.value == "balanced_cyclic_permutation_v1"
        for profile in profiles.values()
    )


def test_qualification_seeds_are_exact_unique_disjoint_and_order_bound() -> None:
    qualification = load_seed_registry(QUALIFICATION_SEEDS)
    assert isinstance(qualification, QualificationSeedRegistry)
    assert qualification.candidate_episode_seeds == (
        13263716994628839049,
        10419656982453996572,
        17335017240715108969,
        6808163546807211275,
        10340178108013510301,
        2958807714473405968,
        1453940245817783884,
        12368970943374979214,
    )
    design = load_evaluation_seed_registry(DESIGN_SEEDS)
    assert not set(qualification.candidate_episode_seeds) & set(design.candidate_episode_seeds)
    assert seed_registry_hash(qualification) == (
        "247db21c869703f5604e63678f0cf614f0b88040a05ce42c61629ddf712cefd5"
    )
    changed = qualification.model_dump(mode="python")
    changed["candidate_episode_seeds"] = tuple(reversed(changed["candidate_episode_seeds"]))
    with pytest.raises(ValidationError):
        QualificationSeedRegistry.model_validate(changed)


def test_revision_instance_binds_ambient_registry_and_partition_seed_identity() -> None:
    revision = load_appearance_registry_any(REVISION_REGISTRY)
    qualification = load_seed_registry(QUALIFICATION_SEEDS)
    assert isinstance(revision, AppearanceRevision1Registry)
    assert isinstance(qualification, QualificationSeedRegistry)
    profile = profile_by_id(revision, "revision1_checker_low_v1")
    plan = resolve_appearance(
        revision,
        profile.profile_id,
        "single_occluder",
        ("support_surface", "occluding_surface", "background_surface"),
        1,
        qualification.candidate_episode_seeds[0],
        qualification,
    )
    assert plan.record.appearance_instance_version == "appearance_instance_v4"
    assert plan.record.registry_version == "appearance_candidate_registry_v2"
    assert plan.record.evaluation_seed_registry_version == qualification.registry_version
    assert plan.record.assignment_schedule_source == (
        "snapshotted_revision_partition_seed_registry_v1"
    )
    assert plan.light_ambient == "0.35 0.35 0.35"


def test_definition_lock_recomputes_and_rejects_fully_rehashed_profile_mutation(
    tmp_path: Path,
) -> None:
    lock = validate_definition_lock(
        BASELINE_REGISTRY,
        DESIGN_SEEDS,
        REVISION_REGISTRY,
        QUALIFICATION_SEEDS,
        DEFINITION_LOCK,
        BASELINE_ANALYSIS,
    )
    assert lock["definition_lock_sha256"] == (
        "71d2ed7a9f45c55bf17ec518c08b5d0b827a7cf0ae2cc3c339dc09197c55f633"
    )
    mutated = json.loads(DEFINITION_LOCK.read_text(encoding="utf-8"))
    mutated["revision1_profile_sha256"]["revision1_colour_shift_v1"] = "0" * 64
    domain = dict(mutated)
    domain.pop("definition_lock_sha256")
    from epsbench.audit import _hash_json

    mutated["definition_lock_sha256"] = _hash_json(domain)
    path = tmp_path / "lock.json"
    path.write_bytes(canonical_json_bytes(mutated))
    with pytest.raises(ValueError, match="profile or control identity differs"):
        validate_definition_lock(
            BASELINE_REGISTRY,
            DESIGN_SEEDS,
            REVISION_REGISTRY,
            QUALIFICATION_SEEDS,
            path,
            BASELINE_ANALYSIS,
        )


def test_definition_lock_commit_accepts_canonical_json_source_provenance() -> None:
    lock = json.loads(DEFINITION_LOCK.read_text(encoding="utf-8"))
    source = collect_source_provenance(Path.cwd()).model_dump(mode="json")
    _validate_lock_commit_snapshot(
        "914550ce4e3a819dcbcd0bd5390e3c6034af5bf6",
        lock,
        source,
    )
