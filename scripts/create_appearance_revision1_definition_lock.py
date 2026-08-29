"""Create the immutable prospective definition-lock inputs before qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from epsbench.appearance import (
    REVISION1_PROFILE_IDS,
    AppearanceRegistry,
    AppearanceRevision1Registry,
    EvaluationSeedRegistry,
    QualificationSeedRegistry,
    appearance_profile_hash,
    appearance_registry_hash,
    load_appearance_registry_any,
    load_evaluation_seed_registry,
    load_seed_registry,
    profile_by_id,
    seed_registry_hash,
    validate_axis_isolation,
)
from epsbench.failure_analysis import CANONICAL_BASE, CANONICAL_ROOTS, validate_failure_analysis
from epsbench.utils.canonical import canonical_json_bytes, sha256_bytes, write_canonical_json

ROOT = Path(__file__).resolve().parents[1]
BASELINE_REGISTRY_PATH = ROOT / "configs/appearance_candidates_v0.yaml"
DESIGN_SEEDS_PATH = ROOT / "configs/evaluation_seed_candidates_v0.yaml"
REVISION_REGISTRY_PATH = ROOT / "configs/appearance_candidates_revision1.yaml"
QUALIFICATION_SEEDS_PATH = ROOT / "configs/appearance_revision1_qualification_seeds_v0.yaml"
COMMITTED_ANALYSIS_PATH = ROOT / "configs/appearance_revision1_baseline_failure_analysis.json"
LOCK_PATH = ROOT / "configs/appearance_candidate_revision1_lock.json"

RATIONALE = {
    "revision1_balanced_reference_v1": (
        "Adds identity-bound ambient fill and a moderate-luminance balanced palette to address "
        "the cross-slot corridor side-wall lower-exposure failure while remaining a solid, "
        "non-legacy reference."
    ),
    "revision1_colour_shift_v1": (
        "Changes only the prospectively selected moderate-luminance palette to retain a "
        "colour-only intervention with exposure margin."
    ),
    "revision1_checker_low_v1": (
        "Introduces a one-cycle checker to reduce rendered-scale loss on small projected "
        "surfaces without weakening the texture threshold."
    ),
    "revision1_checker_high_v1": (
        "Uses exactly four cycles as the required 4:1 frequency-only partner, below the "
        "diagnosed 16-cycle minification regime."
    ),
    "revision1_stripes_low_v1": (
        "Changes only checker to one-cycle axis-aligned stripes so family sensitivity is not "
        "confounded by frequency, palette, material, light, or assignment."
    ),
    "revision1_illumination_shift_v1": (
        "Changes only the bound illumination domain to a second ambient/directional balance "
        "designed to retain lateral-wall exposure."
    ),
    "revision1_combined_stress_v1": (
        "Prospectively composes the locked colour shift, four-cycle checker, and illumination "
        "shift; it is not assembled from qualification outcomes."
    ),
}


def _hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if type(payload) is not dict:
        raise ValueError(f"JSON object required: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    args = parser.parse_args()
    analysis = validate_failure_analysis(args.analysis)

    baseline = load_appearance_registry_any(BASELINE_REGISTRY_PATH)
    design = load_evaluation_seed_registry(DESIGN_SEEDS_PATH)
    revision = load_appearance_registry_any(REVISION_REGISTRY_PATH)
    qualification = load_seed_registry(QUALIFICATION_SEEDS_PATH)
    if not isinstance(baseline, AppearanceRegistry):
        raise ValueError("canonical baseline registry type is invalid")
    if not isinstance(design, EvaluationSeedRegistry):
        raise ValueError("canonical design seed registry type is invalid")
    if not isinstance(revision, AppearanceRevision1Registry):
        raise ValueError("Revision 1 registry type is invalid")
    if not isinstance(qualification, QualificationSeedRegistry):
        raise ValueError("qualification seed registry type is invalid")
    validate_axis_isolation(baseline)
    validate_axis_isolation(revision)
    for profile_id in (profile.profile_id for profile in baseline.profiles):
        if appearance_profile_hash(profile_by_id(baseline, profile_id)) != appearance_profile_hash(
            profile_by_id(revision, profile_id)
        ):
            raise ValueError(f"canonical profile changed in Revision 1 registry: {profile_id}")

    write_canonical_json(COMMITTED_ANALYSIS_PATH, analysis)
    profiles = [profile_by_id(revision, profile_id) for profile_id in REVISION1_PROFILE_IDS]
    thresholds = profiles[0].non_degeneracy_thresholds.model_dump(mode="json")
    if any(
        profile.non_degeneracy_thresholds.model_dump(mode="json") != thresholds
        for profile in profiles
    ):
        raise ValueError("Revision 1 profiles do not share the canonical thresholds")
    lock_domain = {
        "schema_version": "appearance_candidate_revision1_definition_lock_v0",
        "canonical_base_sha": CANONICAL_BASE,
        "baseline": {
            "appearance_registry_sha256": appearance_registry_hash(baseline),
            "design_seed_registry_sha256": seed_registry_hash(design),
            "portable_roots": CANONICAL_ROOTS,
            "matrix": {
                "successful": 160,
                "admitted": 44,
                "rejected": 116,
                "rejected_profiles": 10,
                "admitted_profile_ids": [],
            },
            "failure_analysis_sha256": analysis["baseline_failure_analysis_sha256"],
            "failure_analysis_child_roots": {
                "profile_failure_matrix_sha256": analysis["profile_failure_matrix_sha256"],
                "surface_failure_matrix_sha256": analysis["surface_failure_matrix_sha256"],
                "threshold_margin_summary_sha256": analysis["threshold_margin_summary_sha256"],
            },
        },
        "revision1_registry_sha256": appearance_registry_hash(revision),
        "revision1_profile_ids": list(REVISION1_PROFILE_IDS),
        "revision1_profile_sha256": {
            profile.profile_id: appearance_profile_hash(profile) for profile in profiles
        },
        "matched_controls": {
            profile.profile_id: profile.matched_control_profile_id for profile in profiles
        },
        "qualification_seed_registry_sha256": seed_registry_hash(qualification),
        "design_seed_registry_version": design.registry_version,
        "design_seeds": list(design.candidate_episode_seeds),
        "qualification_seed_registry_version": qualification.registry_version,
        "qualification_seeds": list(qualification.candidate_episode_seeds),
        "admission_thresholds": thresholds,
        "prospective_design_targets": {
            "mean_luminance_target_band": [0.1, 0.9],
            "textured_luminance_standard_deviation_target_minimum": 0.04,
            "normalized_controlled_rgb_mad_target_minimum": 0.04,
            "targets_are_advisory_not_admission_criteria": True,
        },
        "profile_rationale": RATIONALE,
        "version_matrix": {
            "baseline_registry": "appearance_candidate_registry_v1",
            "revision_registry": "appearance_candidate_registry_v2",
            "baseline_profile": "appearance_profile_v2",
            "revision_profile": "appearance_profile_v3",
            "configuration": "0.1.0-dev.5",
            "dataset_manifest": "0.1.0-dev.8",
            "privileged_instrumentation": "0.1.0-dev.13",
            "appearance_instance": "appearance_instance_v4",
            "revision_audit": "appearance_candidate_revision_audit_v0",
        },
        "method_versions": {
            "texture_generator": "repository_procedural_texture_v1",
            "style_assignment": "balanced_cyclic_permutation_v1",
            "surface_repeat": "mujoco_geom_local_uv_repeat_v1",
            "ambient_fill_binding": "explicit_profile_ambient_rgb_v1",
            "assignment_schedule_source": ("snapshotted_revision_partition_seed_registry_v1"),
        },
        "qualification_started": False,
        "final_split": None,
        "final_evaluation_seeds": None,
        "benchmark_frozen": False,
        "full_gate_0b_complete": False,
        "scientific_result": None,
    }
    lock = {**lock_domain, "definition_lock_sha256": _hash_json(lock_domain)}
    write_canonical_json(LOCK_PATH, lock)


if __name__ == "__main__":
    main()
