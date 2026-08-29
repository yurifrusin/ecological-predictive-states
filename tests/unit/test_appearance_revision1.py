from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from epsbench.appearance import (
    CANONICAL_PROFILE_IDS,
    REVISION1_PROFILE_IDS,
    AppearanceInstanceRecord,
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
from epsbench.audit import _hash_json
from epsbench.data.provenance import collect_source_provenance
from epsbench.revision import (
    REPORT_FILES,
    REVISION_AUDIT_SCHEMA_VERSION,
    REVISION_FREEZE_STATUS,
    REVISION_ROOT_FIELDS,
    REVISION_ROOT_SCHEMA_VERSION,
    SNAPSHOT_FILES,
    _partition_membership_domain,
    _profile_admission_outcome_domain,
    _renderer_local_partition_outcome_domain,
    _validate_baseline_analysis_snapshot,
    _validate_lock_commit_snapshot,
    validate_definition_lock,
    validate_revision_audit,
)
from epsbench.utils.canonical import canonical_json_bytes, sha256_file, write_canonical_json

BASELINE_REGISTRY = Path("configs/appearance_candidates_v0.yaml")
DESIGN_SEEDS = Path("configs/evaluation_seed_candidates_v0.yaml")
REVISION_REGISTRY = Path("configs/appearance_candidates_revision1.yaml")
QUALIFICATION_SEEDS = Path("configs/appearance_revision1_qualification_seeds_v0.yaml")
DEFINITION_LOCK = Path("configs/appearance_candidate_revision1_lock.json")
BASELINE_ANALYSIS = Path("configs/appearance_revision1_baseline_failure_analysis.json")
LOCK_COMMIT = "914550ce4e3a819dcbcd0bd5390e3c6034af5bf6"
PROFILE_OUTCOME_ROOT = "8496521a261367e4fb1da340fc6a58b007eb9b2955fb8cafa44c2c8db0532d5b"


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
    canonical_record = canonical_json_bytes(plan.record)
    assert AppearanceInstanceRecord.model_validate_json(canonical_record) == plan.record


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
    analysis = json.loads(BASELINE_ANALYSIS.read_text(encoding="utf-8"))
    source = collect_source_provenance(Path.cwd()).model_dump(mode="json")
    _validate_lock_commit_snapshot(
        LOCK_COMMIT,
        lock,
        analysis,
        source,
    )


def _rehash_renderer_analysis(analysis: dict[str, Any]) -> None:
    renderer_domain = dict(analysis)
    renderer_domain.pop("renderer_specific_failure_analysis_sha256")
    analysis["renderer_specific_failure_analysis_sha256"] = _hash_json(renderer_domain)


@pytest.mark.parametrize(
    ("field", "mutate"),
    (
        (
            "diagnostic summary",
            lambda analysis: analysis["diagnostic_summary"].__setitem__(
                "high_source_variation_but_rendered_failure_occurrences", 215
            ),
        ),
        (
            "causal answer",
            lambda analysis: analysis["causal_diagnostic_answers"][0].__setitem__(
                "answer", "Fully rehashed but prospectively different diagnosis."
            ),
        ),
        (
            "source packet root",
            lambda analysis: analysis.__setitem__("source_packet_logical_root_sha256", "0" * 64),
        ),
        (
            "renderer fingerprint",
            lambda analysis: analysis.__setitem__(
                "renderer_fingerprints", ["fully-rehashed-different-renderer"]
            ),
        ),
    ),
)
def test_fully_rehashed_analysis_content_cannot_differ_from_lock_commit(
    field: str,
    mutate: Any,
) -> None:
    lock = json.loads(DEFINITION_LOCK.read_text(encoding="utf-8"))
    analysis = json.loads(BASELINE_ANALYSIS.read_text(encoding="utf-8"))
    portable_root = analysis["baseline_failure_analysis_sha256"]
    mutate(analysis)
    _rehash_renderer_analysis(analysis)
    assert analysis["baseline_failure_analysis_sha256"] == portable_root
    _validate_baseline_analysis_snapshot(analysis, lock)
    with pytest.raises(ValueError, match="baseline analysis differs from the exact lock commit"):
        _validate_lock_commit_snapshot(
            LOCK_COMMIT,
            lock,
            analysis,
            collect_source_provenance(Path.cwd()).model_dump(mode="json"),
        )


@pytest.mark.parametrize(
    "child_field",
    (
        "profile_failure_matrix_sha256",
        "surface_failure_matrix_sha256",
        "threshold_margin_summary_sha256",
    ),
)
def test_fully_rehashed_analysis_child_root_must_match_lock(child_field: str) -> None:
    lock = json.loads(DEFINITION_LOCK.read_text(encoding="utf-8"))
    analysis = json.loads(BASELINE_ANALYSIS.read_text(encoding="utf-8"))
    analysis[child_field] = "0" * 64
    _rehash_renderer_analysis(analysis)
    with pytest.raises(ValueError, match="child-root binding differs"):
        _validate_baseline_analysis_snapshot(analysis, lock)


def test_portable_valid_analysis_with_false_renderer_root_is_rejected() -> None:
    lock = json.loads(DEFINITION_LOCK.read_text(encoding="utf-8"))
    analysis = json.loads(BASELINE_ANALYSIS.read_text(encoding="utf-8"))
    analysis["renderer_specific_failure_analysis_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="renderer-specific root mismatch"):
        _validate_baseline_analysis_snapshot(analysis, lock)


def _profile_summary_for_reviewed_disposition() -> dict[str, Any]:
    revision = load_appearance_registry_any(REVISION_REGISTRY)
    assert isinstance(revision, AppearanceRevision1Registry)
    admitted = {
        "revision1_balanced_reference_v1",
        "revision1_colour_shift_v1",
        "revision1_checker_low_v1",
        "revision1_checker_high_v1",
        "revision1_combined_stress_v1",
    }
    profiles = []
    for profile_id in REVISION1_PROFILE_IDS:
        profile = profile_by_id(revision, profile_id)
        profile_admitted = profile_id in admitted
        profiles.append(
            {
                "profile_id": profile_id,
                "profile_sha256": appearance_profile_hash(profile),
                "matched_control_profile_id": profile.matched_control_profile_id,
                "design": {"cell_count": 16, "set_admitted": profile_admitted},
                "qualification": {"cell_count": 16, "set_admitted": profile_admitted},
                "revision1_profile_admitted": profile_admitted,
            }
        )
    return {"profiles": profiles}


def _minimal_candidate_cell(partition: str = "design") -> dict[str, Any]:
    revision = load_appearance_registry_any(REVISION_REGISTRY)
    assert isinstance(revision, AppearanceRevision1Registry)
    profile_id = "revision1_stripes_low_v1"
    profile = profile_by_id(revision, profile_id)
    return {
        "partition": partition,
        "cell_id": f"{partition}--corridor--{profile_id}--seed-0",
        "profile_id": profile_id,
        "appearance_profile_sha256": appearance_profile_hash(profile),
        "scene_family": "corridor",
        "seed_index": 0,
        "candidate_seed": 6594827050443514047,
        "control_cell_id": f"{partition}--corridor--revision1_checker_low_v1--seed-0",
        "generation_status": "success",
        "admission_checks": {"texture_variation": False},
        "admission_status": "rejected",
        "rejection_reasons": ["rendered texture variation below threshold"],
        "frame_metrics": {"renderer_local_rgb_metric": 0.0},
    }


def test_renderer_local_stripe_difference_does_not_change_portable_roots() -> None:
    revision = load_appearance_registry_any(REVISION_REGISTRY)
    assert isinstance(revision, AppearanceRevision1Registry)
    wgl = [_minimal_candidate_cell()]
    osmesa = deepcopy(wgl)
    osmesa[0]["admission_checks"] = {"texture_variation": True}
    osmesa[0]["admission_status"] = "admitted"
    osmesa[0]["rejection_reasons"] = []
    osmesa[0]["frame_metrics"] = {"renderer_local_rgb_metric": 0.1}

    assert _hash_json(_partition_membership_domain(wgl, revision, "design")) == _hash_json(
        _partition_membership_domain(osmesa, revision, "design")
    )
    profile_summary = _profile_summary_for_reviewed_disposition()
    assert _hash_json(_profile_admission_outcome_domain(profile_summary)) == PROFILE_OUTCOME_ROOT
    assert _hash_json(_renderer_local_partition_outcome_domain(wgl, "design")) != _hash_json(
        _renderer_local_partition_outcome_domain(osmesa, "design")
    )


def test_qualification_outcomes_are_renderer_local_even_when_current_values_match() -> None:
    wgl = [_minimal_candidate_cell("qualification")]
    osmesa = deepcopy(wgl)
    assert _hash_json(_renderer_local_partition_outcome_domain(wgl, "qualification")) == _hash_json(
        _renderer_local_partition_outcome_domain(osmesa, "qualification")
    )


def test_profile_disposition_change_changes_portable_profile_root() -> None:
    profile_summary = _profile_summary_for_reviewed_disposition()
    changed = deepcopy(profile_summary)
    changed["profiles"][0]["revision1_profile_admitted"] = False
    assert _hash_json(_profile_admission_outcome_domain(profile_summary)) == PROFILE_OUTCOME_ROOT
    assert _hash_json(_profile_admission_outcome_domain(changed)) != PROFILE_OUTCOME_ROOT


def test_reviewed_wgl_and_osmesa_profile_disposition_roots_match_exactly() -> None:
    wgl = _profile_summary_for_reviewed_disposition()
    osmesa = deepcopy(wgl)
    stripe_index = next(
        index
        for index, profile in enumerate(wgl["profiles"])
        if profile["profile_id"] == "revision1_stripes_low_v1"
    )
    wgl["profiles"][stripe_index]["design"]["cell_counts"] = {
        "admitted": 11,
        "rejected": 5,
    }
    osmesa["profiles"][stripe_index]["design"]["cell_counts"] = {
        "admitted": 9,
        "rejected": 7,
    }
    assert _hash_json(_profile_admission_outcome_domain(wgl)) == PROFILE_OUTCOME_ROOT
    assert _hash_json(_profile_admission_outcome_domain(osmesa)) == PROFILE_OUTCOME_ROOT


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("cell_id", "design--corridor--revision1_stripes_low_v1--seed-1"),
        ("candidate_seed", 1),
        ("control_cell_id", "design--corridor--revision1_balanced_reference_v1--seed-0"),
        ("appearance_profile_sha256", "0" * 64),
    ),
)
def test_membership_identity_change_changes_portable_root(field: str, replacement: Any) -> None:
    revision = load_appearance_registry_any(REVISION_REGISTRY)
    assert isinstance(revision, AppearanceRevision1Registry)
    original = [_minimal_candidate_cell()]
    changed = deepcopy(original)
    changed[0][field] = replacement
    assert _hash_json(_partition_membership_domain(original, revision, "design")) != _hash_json(
        _partition_membership_domain(changed, revision, "design")
    )


def test_renderer_metrics_and_reasons_cannot_enter_portable_membership_root() -> None:
    revision = load_appearance_registry_any(REVISION_REGISTRY)
    assert isinstance(revision, AppearanceRevision1Registry)
    original = [_minimal_candidate_cell()]
    changed = deepcopy(original)
    changed[0]["frame_metrics"] = {"renderer_local_rgb_metric": 999.0}
    changed[0]["rejection_reasons"] = ["different renderer-local reason"]
    assert _hash_json(_partition_membership_domain(original, revision, "design")) == _hash_json(
        _partition_membership_domain(changed, revision, "design")
    )


def _minimal_revision_packet() -> dict[str, Any]:
    logical = {
        "schema_version": REVISION_AUDIT_SCHEMA_VERSION,
        "root_schema_version": REVISION_ROOT_SCHEMA_VERSION,
        "freeze_status": REVISION_FREEZE_STATUS,
        "candidate_definition_lock_commit": LOCK_COMMIT,
        "source_provenance": {},
        "roots": {field: "0" * 64 for field in REVISION_ROOT_FIELDS},
        "snapshot_file_sha256": {field: "0" * 64 for field in SNAPSHOT_FILES},
        "report_file_sha256": {field: "0" * 64 for field in REPORT_FILES},
        "contact_sheet_manifest": {},
        "matrix_counts": {},
        "revision1_profile_ids": list(REVISION1_PROFILE_IDS),
        "design_matrix_cell_count": 112,
        "qualification_matrix_cell_count": 112,
        "control_cell_count": 32,
        "scene_families": ["single_occluder", "corridor"],
        "final_split": None,
        "final_evaluation_seeds": None,
        "full_gate_0b_complete": False,
        "benchmark_frozen": False,
    }
    return {**logical, "complete_packet_root_sha256": _hash_json(logical)}


def test_fully_rehashed_old_root_schema_is_rejected(tmp_path: Path) -> None:
    packet = _minimal_revision_packet()
    packet["root_schema_version"] = "appearance_candidate_revision_root_domains_v0"
    logical = dict(packet)
    logical.pop("complete_packet_root_sha256")
    packet["complete_packet_root_sha256"] = _hash_json(logical)
    write_canonical_json(tmp_path / "revision1_candidate_packet.json", packet)
    with pytest.raises(ValueError, match="unauthorised status claim"):
        validate_revision_audit(tmp_path)


def test_fully_rehashed_root_field_name_corruption_is_rejected(tmp_path: Path) -> None:
    packet = _minimal_revision_packet()
    packet["roots"]["design_partition_outcome_root_sha256"] = packet["roots"].pop(
        "design_partition_membership_root_sha256"
    )
    logical = dict(packet)
    logical.pop("complete_packet_root_sha256")
    packet["complete_packet_root_sha256"] = _hash_json(logical)
    write_canonical_json(tmp_path / "revision1_candidate_packet.json", packet)
    with pytest.raises(ValueError, match="root schema is not strict"):
        validate_revision_audit(tmp_path)
