"""Prospectively locked Appearance Benchmark Input Freeze v0 apparatus."""

from __future__ import annotations

import json
import math
import platform
import shutil
import socket
import stat
import subprocess
import tempfile
import time
import zipfile
from collections import Counter
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, cast

import numpy as np
import yaml
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from epsbench.appearance import (
    REVISION1_PROFILE_IDS,
    TEXTURE_RESOLUTION,
    AdmissionThresholds,
    AppearanceInstanceRecord,
    AppearanceRevision1Registry,
    FinalEvaluationSeedRegistry,
    appearance_profile_hash,
    appearance_registry_hash,
    assignment_balance,
    load_appearance_registry_any,
    load_seed_registry,
    parse_appearance_registry,
    parse_seed_registry,
    profile_by_id,
    resolve_appearance,
    seed_registry_hash,
    validate_axis_isolation,
)
from epsbench.audit import (
    SCENE_FAMILIES,
    _admission_evidence_domain,
    _atomic_no_replace_directory,
    _contact_sheet_image,
    _dataset_cell,
    _evaluate_cell,
    _load_frame,
    _PacketArtifactRegistry,
    _source_texture_diagnostics,
    _validate_cell_schema,
)
from epsbench.config import BenchmarkConfig, load_config, parse_config
from epsbench.data.identity import (
    analytic_transport_domain,
    corridor_scene_content_domain,
    oriented_boundary_domain,
    single_occluder_scene_content_domain,
    visibility_event_domain,
)
from epsbench.data.paths import UnsafeOwnedFileError, open_owned_regular_file
from epsbench.data.provenance import collect_source_provenance
from epsbench.data.publication import atomic_publish_owned_bytes
from epsbench.data.validate import validate_dataset
from epsbench.schema import (
    AvailableDenseOpticalTransport,
    AvailableEcologicalVisibilityEvents,
    AvailableOrientedBoundaryOwnership,
    CorridorSampledGeometry,
    DatasetManifest,
    RendererProvenance,
    SourceProvenance,
    TransitionRecord,
    canonical_github_repository_identity,
    parse_privileged_instrumentation_json,
)
from epsbench.utils.canonical import (
    canonical_json_bytes,
    logical_array_hash,
    sha256_bytes,
    sha256_file,
    write_canonical_json,
)

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
GitObject = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]

BENCHMARK_DEFINITION_VERSION = "appearance_benchmark_input_definition_v1"
FREEZE_LOCK_VERSION = "appearance_benchmark_freeze_definition_lock_v1"
FREEZE_PACKET_VERSION = "appearance_benchmark_freeze_candidate_v1"
FREEZE_ROOT_VERSION = "appearance_benchmark_freeze_root_domains_v1"
FREEZE_CONTACT_SHEET_VERSION = "appearance_benchmark_freeze_contact_sheet_manifest_v1"
RENDERER_RECEIPT_VERSION = "appearance_benchmark_renderer_qualification_receipt_v1"
PUBLICATION_RECORD_VERSION = "appearance_benchmark_public_ci_packet_record_v1"
SOURCE_IDENTITY_VERSION = "appearance_benchmark_source_identity_v1"
THRESHOLD_MARGIN_VERSION = "appearance_benchmark_threshold_margin_summary_v1"
RENDERER_SELECTION_DEPENDENCY_VERSION = "model_result_renderer_selection_dependency_v1"
DOMAIN_ENVELOPE_VERSION = "epsbench_logical_domain_envelope_v1"
CANONICAL_BASE = "08179fbdce909e9a0d6dbb2939c58f7ab5d0d8a7"
REVIEWED_IMPLEMENTATION_HEAD = "8b34b78d5488af7103119697a286cfb8757cc125"
SUPERSEDED_FREEZE_LOCK_COMMIT = "1a5929307dfcba1d726c650f5e1ce68771f66801"
SUPERSEDED_FREEZE_LOCK_ROOT = "a373a4742a6b5a3057b820b7d925c2e19fbddc0a83cc45384b8b81ea4e8b1464"
REVISION1_APPROVED_HEAD = "da37a729bc4af00ea83c9460c3849307171bba69"
REVISION1_LOCK_COMMIT = "914550ce4e3a819dcbcd0bd5390e3c6034af5bf6"
REVISION1_LOCK_ROOT = "71d2ed7a9f45c55bf17ec518c08b5d0b827a7cf0ae2cc3c339dc09197c55f633"
REVISION1_REGISTRY_ROOT = "81da1bb9414e7c53e42bbf65b198aa61d8bb7ed3f81eb2bf5813a1edff245ac4"
REVISION1_ADMISSION_ROOT = "8496521a261367e4fb1da340fc6a58b007eb9b2955fb8cafa44c2c8db0532d5b"
PRIMARY_REFERENCE = "revision1_balanced_reference_v1"
SELECTED_PROFILE_IDS = (
    PRIMARY_REFERENCE,
    "revision1_colour_shift_v1",
    "revision1_checker_low_v1",
    "revision1_checker_high_v1",
    "revision1_combined_stress_v1",
)
EXCLUDED_PROFILE_IDS = (
    "revision1_stripes_low_v1",
    "revision1_illumination_shift_v1",
)
LEGACY_CONTROL_PROFILE_ID = "legacy_solid_base_v1"
LOCK_PATH = Path("configs/appearance_benchmark_freeze_v0_lock.json")
CANONICAL_REPOSITORY_IDENTITY = "yurifrusin/ecological-predictive-states"

LOGICAL_DOMAINS = {
    "benchmark_definition": "epsbench.appearance_benchmark.v1.benchmark_definition",
    "profile_roles": "epsbench.appearance_benchmark.v1.profile_roles",
    "selected_profile_set": "epsbench.appearance_benchmark.v1.selected_profile_set",
    "excluded_candidate_set": "epsbench.appearance_benchmark.v1.excluded_candidate_set",
    "evaluation_seed_registry": "epsbench.appearance_benchmark.v1.evaluation_seed_registry",
    "admission_thresholds": "epsbench.appearance_benchmark.v1.admission_thresholds",
    "training_evaluation_policy": "epsbench.appearance_benchmark.v1.training_evaluation_policy",
    "renderer_selection_dependency": (
        "epsbench.appearance_benchmark.v1.renderer_selection_dependency"
    ),
    "selected_matrix_membership": ("epsbench.appearance_benchmark.v1.selected_matrix_membership"),
    "legacy_control_membership": ("epsbench.appearance_benchmark.v1.legacy_control_membership"),
    "scene_family_membership": "epsbench.appearance_benchmark.v1.scene_family_membership",
    "definition_lock": "epsbench.appearance_benchmark.v1.definition_lock",
    "procedural_assets": "epsbench.appearance_benchmark.v1.procedural_assets",
    "appearance_assignments": "epsbench.appearance_benchmark.v1.appearance_assignments",
    "portable_source_identities": ("epsbench.appearance_benchmark.v1.portable_source_identities"),
    "within_renderer_invariance": ("epsbench.appearance_benchmark.v1.within_renderer_invariance"),
    "selected_outcomes": "epsbench.appearance_benchmark.v1.selected_outcomes",
    "control_outcomes": "epsbench.appearance_benchmark.v1.control_outcomes",
    "renderer_ecological_labels": ("epsbench.appearance_benchmark.v1.renderer_ecological_labels"),
    "renderer_audit": "epsbench.appearance_benchmark.v1.renderer_audit",
    "renderer_contact_sheets": "epsbench.appearance_benchmark.v1.renderer_contact_sheets",
    "renderer_source_evidence": "epsbench.appearance_benchmark.v1.renderer_source_evidence",
    "profile_readiness": "epsbench.appearance_benchmark.v1.profile_readiness",
    "threshold_margins": "epsbench.appearance_benchmark.v1.threshold_margins",
    "complete_packet": "epsbench.appearance_benchmark.v1.complete_packet",
    "renderer_receipt": "epsbench.appearance_benchmark.v1.renderer_receipt",
    "public_packet_tree": "epsbench.appearance_benchmark.v1.public_packet_tree",
    "source_identity_root": "epsbench.appearance_benchmark.v1.source_identity_root",
    "sampled_geometry": "epsbench.appearance_benchmark.v1.sampled_geometry",
    "camera_trajectory": "epsbench.appearance_benchmark.v1.camera_trajectory",
    "executed_action": "epsbench.appearance_benchmark.v1.executed_action",
    "surface_remapping": "epsbench.appearance_benchmark.v1.surface_remapping",
    "scene_content": "epsbench.appearance_benchmark.v1.scene_content",
    "analytic_transport": "epsbench.appearance_benchmark.v1.analytic_transport",
    "oriented_boundary_ownership": ("epsbench.appearance_benchmark.v1.oriented_boundary_ownership"),
    "visibility_events": "epsbench.appearance_benchmark.v1.visibility_events",
    "public_occlusion_relation": ("epsbench.appearance_benchmark.v1.public_occlusion_relation"),
}


def _domain_envelope(domain_key: str, payload: Any) -> dict[str, Any]:
    try:
        domain = LOGICAL_DOMAINS[domain_key]
    except KeyError as error:
        raise FreezeError(f"unknown logical domain: {domain_key}") from error
    return {
        "schema_version": DOMAIN_ENVELOPE_VERSION,
        "domain": domain,
        "payload": payload,
    }


def _domain_hash(domain_key: str, payload: Any) -> str:
    return sha256_bytes(canonical_json_bytes(_domain_envelope(domain_key, payload)))


class FreezeError(ValueError):
    """Raised when a freeze definition, lock, receipt, or packet is invalid."""


class ImmutableArtifactBindingError(FreezeError):
    """Raised when live artifact authority cannot be bound to the consumed archive bytes."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


_FREEZE_PORTABLE_METRIC_DECIMAL_PLACES = 12


def _portable_freeze_metric_value(value: Any) -> Any:
    """Canonicalize PR #17 derived floats without changing historical audit packets."""

    if type(value) is float:
        return float(round(value, _FREEZE_PORTABLE_METRIC_DECIMAL_PLACES))
    if type(value) is list:
        return [_portable_freeze_metric_value(item) for item in value]
    if type(value) is dict:
        return {key: _portable_freeze_metric_value(item) for key, item in value.items()}
    return value


def _portable_freeze_source_texture_diagnostics(
    diagnostics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize signed real-FFT aliases and last-bit derived floats for this freeze."""

    normalized = cast(list[dict[str, Any]], _portable_freeze_metric_value(diagnostics))
    for item in normalized:
        dominant = item["dominant_spectrum_index"]
        dominant[0] = min(dominant[0], TEXTURE_RESOLUTION - dominant[0])
    return normalized


def _first_json_difference(
    stored: Any,
    recomputed: Any,
    path: str = "$",
) -> tuple[str, Any, Any]:
    """Return one strict JSON difference for actionable fail-closed diagnostics."""

    if type(stored) is not type(recomputed):
        return path, stored, recomputed
    if isinstance(stored, dict):
        for key in sorted(set(stored) | set(recomputed)):
            child_path = f"{path}.{key}"
            if key not in stored:
                return child_path, "<missing>", recomputed[key]
            if key not in recomputed:
                return child_path, stored[key], "<missing>"
            if canonical_json_bytes(stored[key]) != canonical_json_bytes(recomputed[key]):
                return _first_json_difference(stored[key], recomputed[key], child_path)
    elif isinstance(stored, list):
        if len(stored) != len(recomputed):
            return f"{path}.length", len(stored), len(recomputed)
        for index, (stored_item, recomputed_item) in enumerate(
            zip(stored, recomputed, strict=True)
        ):
            child_path = f"{path}[{index}]"
            if canonical_json_bytes(stored_item) != canonical_json_bytes(recomputed_item):
                return _first_json_difference(stored_item, recomputed_item, child_path)
    return path, stored, recomputed


class StrictFreezeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class BenchmarkRole(StrEnum):
    DEVELOPMENT_REFERENCE = "development_reference"
    HELD_OUT_COLOUR_OOD = "held_out_colour_ood"
    HELD_OUT_TEXTURE_PRESENCE_OOD = "held_out_texture_presence_ood"
    HELD_OUT_HIGH_FREQUENCY_CHECKER_OOD = "held_out_high_frequency_checker_ood"
    HELD_OUT_COMBINED_APPEARANCE_OOD = "held_out_combined_appearance_ood"
    EXCLUDED_REJECTED_CANDIDATE = "excluded_rejected_candidate"


ROLE_BY_PROFILE = {
    PRIMARY_REFERENCE: BenchmarkRole.DEVELOPMENT_REFERENCE,
    "revision1_colour_shift_v1": BenchmarkRole.HELD_OUT_COLOUR_OOD,
    "revision1_checker_low_v1": BenchmarkRole.HELD_OUT_TEXTURE_PRESENCE_OOD,
    "revision1_checker_high_v1": BenchmarkRole.HELD_OUT_HIGH_FREQUENCY_CHECKER_OOD,
    "revision1_combined_stress_v1": BenchmarkRole.HELD_OUT_COMBINED_APPEARANCE_OOD,
    "revision1_stripes_low_v1": BenchmarkRole.EXCLUDED_REJECTED_CANDIDATE,
    "revision1_illumination_shift_v1": BenchmarkRole.EXCLUDED_REJECTED_CANDIDATE,
}


class Revision1Authority(StrictFreezeModel):
    approved_implementation_head: Literal["da37a729bc4af00ea83c9460c3849307171bba69"]
    definition_lock_commit: Literal["914550ce4e3a819dcbcd0bd5390e3c6034af5bf6"]
    definition_lock_sha256: Literal[
        "71d2ed7a9f45c55bf17ec518c08b5d0b827a7cf0ae2cc3c339dc09197c55f633"
    ]
    registry_sha256: Literal["81da1bb9414e7c53e42bbf65b198aa61d8bb7ed3f81eb2bf5813a1edff245ac4"]
    profile_admission_root_sha256: Literal[
        "8496521a261367e4fb1da340fc6a58b007eb9b2955fb8cafa44c2c8db0532d5b"
    ]


class FrozenProfileRole(StrictFreezeModel):
    profile_id: str
    profile_sha256: Sha256
    role: BenchmarkRole
    selection_status: Literal["selected", "excluded_negative_evidence"]
    primary_id_reference: bool
    training_appearance_eligible: bool
    held_out_appearance_ood: bool
    benchmark_primary_reference_profile_id: str | None
    candidate_admission_matched_control_profile_id: str
    secondary_diagnostic_control_profile_id: str | None
    canonical_revision1_admitted: bool

    @model_validator(mode="after")
    def posture_matches_exact_role(self) -> FrozenProfileRole:
        expected_role = ROLE_BY_PROFILE.get(self.profile_id)
        if expected_role is None or self.role != expected_role:
            raise ValueError("profile role is not the exact authorised mapping")
        selected = self.profile_id in SELECTED_PROFILE_IDS
        if self.selection_status != ("selected" if selected else "excluded_negative_evidence"):
            raise ValueError("profile selection posture differs from the exact role map")
        if self.primary_id_reference != (self.profile_id == PRIMARY_REFERENCE):
            raise ValueError("balanced reference must be the sole primary ID reference")
        if self.training_appearance_eligible != (self.profile_id == PRIMARY_REFERENCE):
            raise ValueError("balanced reference must be the sole training-eligible appearance")
        if self.held_out_appearance_ood != (selected and self.profile_id != PRIMARY_REFERENCE):
            raise ValueError("held-out OOD posture differs from the exact role map")
        if self.benchmark_primary_reference_profile_id != (PRIMARY_REFERENCE if selected else None):
            raise ValueError("benchmark primary-reference relationship differs")
        expected_secondary = (
            "revision1_checker_low_v1" if self.profile_id == "revision1_checker_high_v1" else None
        )
        if self.secondary_diagnostic_control_profile_id != expected_secondary:
            raise ValueError("secondary checker-frequency relationship differs")
        if self.canonical_revision1_admitted != selected:
            raise ValueError("canonical Revision 1 disposition differs")
        return self


class CheckerDiagnostic(StrictFreezeModel):
    candidate_profile_id: Literal["revision1_checker_high_v1"]
    control_profile_id: Literal["revision1_checker_low_v1"]
    interpretation: Literal["frequency_only_relative_to_checker_low"]


class SceneMembership(StrictFreezeModel):
    scene_family: Literal["single_occluder", "corridor"]
    config_path: Literal["configs/benchmark_v0.yaml", "configs/corridor_v0.yaml"]
    config_logical_sha256: Sha256


class PairingPolicy(StrictFreezeModel):
    pair_key_fields: tuple[Literal["scene_family"], Literal["evaluation_episode_root"]]
    identical_apparatus_fields: tuple[
        Literal["sampled_geometry"],
        Literal["camera_trajectory"],
        Literal["executed_action"],
        Literal["surface_remapping"],
        Literal["scene_content_identity"],
        Literal["analytic_transport"],
        Literal["oriented_boundary_ownership"],
        Literal["visibility_events"],
        Literal["public_occlusion_relation"],
    ]
    primary_reference_rule: Literal[
        "every_selected_ood_uses_balanced_reference_same_scene_and_root"
    ]
    candidate_admission_control_rule: Literal[
        "preserve_revision1_matched_control_same_scene_and_root"
    ]
    secondary_diagnostic_rule: Literal["checker_high_versus_checker_low_same_scene_and_root"]


class EvaluationUsePolicy(StrictFreezeModel):
    public_evaluation_roots_are_evaluation_only: Literal[True]
    public_does_not_mean_development_eligible: Literal[True]
    held_out_ood_profiles_are_evaluation_only: Literal[True]
    development_reference_is_sole_training_appearance_eligible_profile: Literal[True]
    training_on_final_evaluation_roots_prohibited: Literal[True]
    tuning_on_final_evaluation_roots_prohibited: Literal[True]
    checkpoint_selection_from_final_evaluation_results_prohibited: Literal[True]
    held_out_ood_training_or_tuning_prohibited: Literal[True]
    post_result_role_changes_prohibited: Literal[True]
    difficult_root_replacement_prohibited: Literal[True]
    original_design_roots_posture: Literal["apparatus_design_evidence_not_automatically_training"]
    revision1_qualification_roots_posture: Literal[
        "candidate_qualification_evidence_not_automatically_training"
    ]
    training_episode_roots: None
    validation_episode_roots: None
    model_random_seeds: None


class RendererEnvironment(StrictFreezeModel):
    environment_id: Literal["windows_wgl_locked", "ubuntu_osmesa_locked"]
    fingerprint: RendererProvenance

    @model_validator(mode="after")
    def fingerprint_matches_environment(self) -> RendererEnvironment:
        expected = {
            "windows_wgl_locked": {
                "mujoco_version": "3.12.0",
                "numpy_version": "2.4.6",
                "renderer": "mujoco.Renderer",
                "backend": "wgl-default",
                "operating_system": "Windows",
            },
            "ubuntu_osmesa_locked": {
                "mujoco_version": "3.12.0",
                "numpy_version": "2.4.6",
                "renderer": "mujoco.Renderer",
                "backend": "osmesa",
                "operating_system": "Linux",
            },
        }[self.environment_id]
        if self.fingerprint.model_dump(mode="json") != expected:
            raise ValueError("renderer fingerprint differs from the locked environment")
        return self


class RendererSelectionDependency(StrictFreezeModel):
    schema_version: Literal["model_result_renderer_selection_dependency_v1"]
    selection_required_before_comparative_model_result_access: Literal[True]
    primary_model_result_renderer: None
    other_renderer_model_result_classification: None
    permitted_other_renderer_classifications: tuple[
        Literal["replication"],
        Literal["robustness"],
        Literal["sensitivity"],
        Literal["unsupported"],
    ]
    selection_may_not_depend_on_observed_comparative_results: Literal[True]
    averaging_or_aggregation_may_not_depend_on_observed_results: Literal[True]
    aggregation_rule_must_be_preregistered_prospectively: Literal[True]
    aggregation_rule: None
    comparative_model_result_access_authorised: Literal[False]
    dependency_status: Literal["required_not_yet_satisfied"]

    @model_validator(mode="after")
    def classification_domain_is_exact(self) -> RendererSelectionDependency:
        if self.permitted_other_renderer_classifications != (
            "replication",
            "robustness",
            "sensitivity",
            "unsupported",
        ):
            raise ValueError("model-result renderer classification domain differs")
        return self


class RendererPolicy(StrictFreezeModel):
    supported_apparatus_environments: tuple[RendererEnvironment, RendererEnvironment]
    every_selected_profile_all_cells_both_environments_required: Literal[True]
    portable_apparatus_roots_must_match: Literal[True]
    renderer_local_roots_are_reported_not_compared: Literal[True]
    renderer_specific_profiles_or_thresholds_prohibited: Literal[True]
    portable_identity_backend_dispatch_prohibited: Literal[True]
    model_result_renderer_selection_dependency: RendererSelectionDependency

    @model_validator(mode="after")
    def environments_are_exact_and_ordered(self) -> RendererPolicy:
        if tuple(item.environment_id for item in self.supported_apparatus_environments) != (
            "windows_wgl_locked",
            "ubuntu_osmesa_locked",
        ):
            raise ValueError("locked renderer environments are not exact and ordered")
        return self


class FailurePolicy(StrictFreezeModel):
    selected_profile_set_is_indivisible: Literal[True]
    any_selected_cell_failure_makes_candidate_not_ready: Literal[True]
    any_locked_renderer_failure_makes_candidate_not_ready: Literal[True]
    failed_seed_set_disposition: Literal["retired_after_failed_freeze_qualification"]
    successful_seed_set_disposition: Literal["eligible_for_owner_freeze_if_approved"]
    failed_roots_may_not_be_reused_for_revised_profiles: Literal[True]
    future_revised_profile_set_requires_owner_authorised_root_and_namespace: Literal[True]


class FreezeScope(StrictFreezeModel):
    checkpoint_fields: tuple[str, ...]
    deferred_gate_0d_0e_fields: tuple[str, ...]
    complete_experimental_preregistration: Literal[False]
    benchmark_frozen: Literal[False]
    model_protocol_frozen: Literal[False]
    full_gate_0b_complete: Literal[False]
    gate_0c_authorised: Literal[False]
    gate_0d_authorised: Literal[False]
    scientific_result: None


class BenchmarkDefinition(StrictFreezeModel):
    schema_version: Literal["appearance_benchmark_input_definition_v1"]
    canonical_base_sha: Literal["08179fbdce909e9a0d6dbb2939c58f7ab5d0d8a7"]
    review_profile: Literal["DUAL_REVIEW"]
    evidence_class: Literal["PUBLIC_REPOSITORY_ONLY"]
    closeout_boundary: Literal["BENCHMARK_OR_PREREGISTRATION_FREEZE"]
    revision1_authority: Revision1Authority
    profiles: tuple[
        FrozenProfileRole,
        FrozenProfileRole,
        FrozenProfileRole,
        FrozenProfileRole,
        FrozenProfileRole,
        FrozenProfileRole,
        FrozenProfileRole,
    ]
    primary_benchmark_reference_profile_id: Literal["revision1_balanced_reference_v1"]
    secondary_checker_frequency_diagnostic: CheckerDiagnostic
    scene_families: tuple[SceneMembership, SceneMembership]
    admission_thresholds: AdmissionThresholds
    pairing_policy: PairingPolicy
    evaluation_use_policy: EvaluationUsePolicy
    renderer_policy: RendererPolicy
    failure_policy: FailurePolicy
    freeze_scope: FreezeScope

    @model_validator(mode="after")
    def membership_is_exact(self) -> BenchmarkDefinition:
        if tuple(profile.profile_id for profile in self.profiles) != (
            *SELECTED_PROFILE_IDS,
            *EXCLUDED_PROFILE_IDS,
        ):
            raise ValueError("benchmark profile membership or order differs")
        if tuple(scene.scene_family for scene in self.scene_families) != SCENE_FAMILIES:
            raise ValueError("benchmark scene-family membership differs")
        if tuple(scene.config_path for scene in self.scene_families) != (
            "configs/benchmark_v0.yaml",
            "configs/corridor_v0.yaml",
        ):
            raise ValueError("benchmark scene configuration membership differs")
        return self


class FreezeDefinitionLock(StrictFreezeModel):
    schema_version: Literal["appearance_benchmark_freeze_definition_lock_v1"]
    canonical_base_sha: Literal["08179fbdce909e9a0d6dbb2939c58f7ab5d0d8a7"]
    reviewed_implementation_head: Literal["8b34b78d5488af7103119697a286cfb8757cc125"]
    superseded_definition_lock_commit: Literal["1a5929307dfcba1d726c650f5e1ce68771f66801"]
    superseded_definition_lock_sha256: Literal[
        "a373a4742a6b5a3057b820b7d925c2e19fbddc0a83cc45384b8b81ea4e8b1464"
    ]
    supersession_finding_id: Literal["EPS-ER17-0004"]
    supersession_posture: Literal[
        "historical_lock_preserved_replacement_lock_authoritative_for_corrected_candidate"
    ]
    revision1_approved_implementation_head: Literal["da37a729bc4af00ea83c9460c3849307171bba69"]
    revision1_definition_lock_commit: Literal["914550ce4e3a819dcbcd0bd5390e3c6034af5bf6"]
    revision1_definition_lock_sha256: Literal[
        "71d2ed7a9f45c55bf17ec518c08b5d0b827a7cf0ae2cc3c339dc09197c55f633"
    ]
    revision1_registry_sha256: Literal[
        "81da1bb9414e7c53e42bbf65b198aa61d8bb7ed3f81eb2bf5813a1edff245ac4"
    ]
    revision1_profile_admission_root_sha256: Literal[
        "8496521a261367e4fb1da340fc6a58b007eb9b2955fb8cafa44c2c8db0532d5b"
    ]
    benchmark_definition_sha256: Sha256
    profile_role_root_sha256: Sha256
    selected_profile_set_root_sha256: Sha256
    excluded_candidate_set_root_sha256: Sha256
    selected_profiles: tuple[dict[str, Any], ...]
    excluded_profiles: tuple[dict[str, Any], ...]
    primary_reference_profile_id: Literal["revision1_balanced_reference_v1"]
    secondary_checker_frequency_diagnostic: CheckerDiagnostic
    scene_families: tuple[Literal["single_occluder"], Literal["corridor"]]
    scene_config_sha256: dict[str, Sha256]
    evaluation_episode_roots: tuple[int, ...]
    evaluation_episode_seed_registry_sha256: Sha256
    admission_thresholds: AdmissionThresholds
    admission_threshold_identity_sha256: Sha256
    paired_geometry_action_generation_rule: PairingPolicy
    supported_renderer_environment_policy: RendererPolicy
    training_evaluation_exclusion_policy: EvaluationUsePolicy
    training_evaluation_policy_root_sha256: Sha256
    renderer_selection_dependency_root_sha256: Sha256
    selected_matrix_membership_root_sha256: Sha256
    legacy_control_membership_root_sha256: Sha256
    expected_unique_selected_cell_count: Literal[160]
    expected_unique_control_cell_count: Literal[32]
    expected_total_cell_count: Literal[192]
    expected_selected_cells_per_profile: Literal[32]
    on_failure_seed_retirement_rule: FailurePolicy
    schema_versions: dict[str, str]
    method_versions: dict[str, str]
    qualification_started: Literal[False]
    freeze_candidate_status: Literal["unqualified"]
    benchmark_frozen: Literal[False]
    final_split_authority: Literal["none"]
    model_protocol_frozen: Literal[False]
    full_gate_0b_complete: Literal[False]
    gate_0c_authorised: Literal[False]
    gate_0d_authorised: Literal[False]
    scientific_result: None
    definition_lock_sha256: Sha256


class RendererOutcomeRow(StrictFreezeModel):
    cell_id: str
    profile_id: str
    benchmark_role: str
    scene_family: Literal["single_occluder", "corridor"]
    seed_index: int = Field(ge=0, le=15)
    evaluation_episode_root: int = Field(ge=0, lt=2**64)
    admission_status: Literal["admitted", "rejected"]


class RendererReadinessRow(StrictFreezeModel):
    profile_id: str
    profile_sha256: Sha256
    benchmark_role: str
    expected_cell_count: Literal[32]
    observed_cell_count: Literal[32]
    cell_counts: dict[str, int]
    renderer_apparatus_qualified: bool

    @model_validator(mode="after")
    def counts_are_exact(self) -> RendererReadinessRow:
        if (
            not self.cell_counts
            or set(self.cell_counts) - {"admitted", "rejected"}
            or sum(self.cell_counts.values()) != self.observed_cell_count
            or any(type(value) is not int or value < 0 for value in self.cell_counts.values())
        ):
            raise ValueError("renderer readiness counts are not exact")
        return self


class ThresholdMarginMetric(StrictFreezeModel):
    count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    near_threshold_count_abs_margin_le_0_005: int = Field(ge=0)
    minimum_margin: float | None
    maximum_margin: float | None

    @model_validator(mode="after")
    def values_are_finite_and_ordered(self) -> ThresholdMarginMetric:
        values = (self.minimum_margin, self.maximum_margin)
        if any(value is not None and not math.isfinite(value) for value in values):
            raise ValueError("threshold margins must be finite")
        if self.count == 0:
            if values != (None, None):
                raise ValueError("empty threshold margin cannot report extrema")
        elif (
            self.minimum_margin is None
            or self.maximum_margin is None
            or self.minimum_margin > self.maximum_margin
            or self.failure_count > self.count
            or self.near_threshold_count_abs_margin_le_0_005 > self.count
        ):
            raise ValueError("threshold margin values are inconsistent")
        return self


class ThresholdMarginSummary(StrictFreezeModel):
    schema_version: Literal["appearance_benchmark_threshold_margin_summary_v1"]
    metrics: dict[str, ThresholdMarginMetric]
    threshold_margin_root_sha256: Sha256

    @model_validator(mode="after")
    def metric_domain_is_exact(self) -> ThresholdMarginSummary:
        expected = {
            "changed_controlled_pixel_fraction",
            "normalized_controlled_rgb_mad",
            "visible_surface_mean_luminance_lower",
            "visible_surface_mean_luminance_upper",
            "textured_surface_luminance_standard_deviation",
        }
        dumped = {name: value.model_dump(mode="json") for name, value in self.metrics.items()}
        if set(self.metrics) != expected or self.threshold_margin_root_sha256 != _domain_hash(
            "threshold_margins", dumped
        ):
            raise ValueError("threshold margin schema or identity differs")
        return self


class PublicQualificationEvidence(StrictFreezeModel):
    evidence_class: Literal["PUBLIC_REPOSITORY_ONLY"]
    availability: Literal["repository_or_ci_artifact"]
    packet_schema_version: Literal["appearance_benchmark_freeze_candidate_v1"]
    packet_root_schema_version: Literal["appearance_benchmark_freeze_root_domains_v1"]
    packet_file_sha256: Sha256
    packet_tree_root_sha256: Sha256
    packet_artifact_count: int = Field(gt=0)


class RendererQualificationReceipt(StrictFreezeModel):
    schema_version: Literal["appearance_benchmark_renderer_qualification_receipt_v1"]
    environment_id: Literal["windows_wgl_locked", "ubuntu_osmesa_locked"]
    renderer_fingerprint: RendererProvenance
    qualification_source_provenance: SourceProvenance
    qualification_source_commit: GitObject
    qualification_source_tree: GitObject
    replacement_definition_lock_commit: GitObject
    replacement_definition_lock_sha256: Sha256
    benchmark_definition_sha256: Sha256
    evaluation_episode_seed_registry_sha256: Sha256
    complete_packet_root_sha256: Sha256
    selected_cell_count: Literal[160]
    control_cell_count: Literal[32]
    total_cell_count: Literal[192]
    selected_matrix_counts: dict[str, int]
    control_matrix_counts: dict[str, int]
    selected_outcome_map: tuple[RendererOutcomeRow, ...]
    control_outcome_map: tuple[RendererOutcomeRow, ...]
    profiles: tuple[
        RendererReadinessRow,
        RendererReadinessRow,
        RendererReadinessRow,
        RendererReadinessRow,
        RendererReadinessRow,
    ]
    portable_definition_roots: dict[str, Sha256]
    portable_apparatus_roots: dict[str, Sha256]
    renderer_local_roots: dict[str, Sha256]
    threshold_margin_summary: ThresholdMarginSummary
    public_qualification_evidence: PublicQualificationEvidence
    benchmark_frozen: Literal[False]
    scientific_result: None
    receipt_sha256: Sha256


class PublicPacketPublicationRecord(StrictFreezeModel):
    schema_version: Literal["appearance_benchmark_public_ci_packet_record_v1"]
    evidence_class: Literal["PUBLIC_REPOSITORY_ONLY"]
    repository: Literal["yurifrusin/ecological-predictive-states"]
    source_commit: GitObject
    source_tree: GitObject
    workflow_run_id: int = Field(gt=0)
    workflow_run_attempt: int = Field(gt=0)
    workflow_name: str
    workflow_path: str
    workflow_run_url: str
    workflow_run_created_at_utc: str
    job_database_id: int = Field(gt=0)
    job_name: str
    job_api_url: str
    job_html_url: str
    artifact_id: int = Field(gt=0)
    artifact_name: str
    artifact_api_url: str
    artifact_url: str
    artifact_archive_download_url: str
    artifact_digest_sha256: Sha256
    artifact_size_in_bytes: int = Field(gt=0)
    artifact_created_at_utc: str
    artifact_expires_at_utc: str
    artifact_expired_at_record_creation: Literal[False]
    retention_days: Literal[90]
    retention_posture: Literal[
        "github_actions_immutable_90_day_artifact_unless_repository_run_or_owner_deletes_earlier"
    ]
    public_access_posture: Literal["public_repository_authenticated_actions_artifact"]
    packet_identity: PublicQualificationEvidence
    complete_packet_root_sha256: Sha256
    record_sha256: Sha256

    @model_validator(mode="after")
    def timestamps_and_identity_are_exact(self) -> PublicPacketPublicationRecord:
        try:
            run_created = datetime.fromisoformat(
                self.workflow_run_created_at_utc.replace("Z", "+00:00")
            )
            artifact_created = datetime.fromisoformat(
                self.artifact_created_at_utc.replace("Z", "+00:00")
            )
            expires = datetime.fromisoformat(self.artifact_expires_at_utc.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("publication record timestamps are invalid") from error
        if (
            run_created.tzinfo is None
            or artifact_created.tzinfo is None
            or expires.tzinfo is None
            or artifact_created < run_created
            or expires <= artifact_created
            or expires - run_created != timedelta(days=90)
            or datetime.now(UTC) >= expires.astimezone(UTC)
        ):
            raise ValueError(
                "publication record is expired or lacks exact 90-day workflow-run retention"
            )
        domain = self.model_dump(mode="json")
        declared = domain.pop("record_sha256")
        if sha256_bytes(canonical_json_bytes(domain)) != declared:
            raise ValueError("publication record identity differs")
        return self


def _parse_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if type(payload) is not dict:
        raise FreezeError(f"strict YAML object required: {path}")
    return payload


def load_benchmark_definition(path: Path) -> BenchmarkDefinition:
    try:
        return BenchmarkDefinition.model_validate_json(json.dumps(_parse_yaml(path)))
    except Exception as error:
        raise FreezeError("appearance benchmark definition is invalid") from error


def load_final_evaluation_seeds(path: Path) -> FinalEvaluationSeedRegistry:
    try:
        registry = load_seed_registry(path)
    except Exception as error:
        raise FreezeError("final evaluation episode-seed registry is invalid") from error
    if not isinstance(registry, FinalEvaluationSeedRegistry):
        raise FreezeError("final evaluation episode-seed registry has the wrong type")
    return registry


def benchmark_definition_hash(definition: BenchmarkDefinition) -> str:
    return _domain_hash("benchmark_definition", definition.model_dump(mode="json"))


def evaluation_seed_registry_hash(seeds: FinalEvaluationSeedRegistry) -> str:
    return _domain_hash("evaluation_seed_registry", seeds.model_dump(mode="json"))


def _role_domain(definition: BenchmarkDefinition) -> list[dict[str, Any]]:
    return [profile.model_dump(mode="json") for profile in definition.profiles]


def _selected_set_domain(definition: BenchmarkDefinition) -> list[dict[str, Any]]:
    return [
        {
            "profile_id": profile.profile_id,
            "profile_sha256": profile.profile_sha256,
            "role": profile.role.value,
        }
        for profile in definition.profiles
        if profile.selection_status == "selected"
    ]


def _excluded_set_domain(definition: BenchmarkDefinition) -> list[dict[str, Any]]:
    return [
        {
            "profile_id": profile.profile_id,
            "profile_sha256": profile.profile_sha256,
            "role": profile.role.value,
            "canonical_revision1_admitted": profile.canonical_revision1_admitted,
            "revision1_profile_admission_root_sha256": (
                definition.revision1_authority.profile_admission_root_sha256
            ),
        }
        for profile in definition.profiles
        if profile.selection_status == "excluded_negative_evidence"
    ]


def _cell_id(scene: str, profile_id: str, seed_index: int) -> str:
    return f"final-evaluation--{scene}--{profile_id}--seed-{seed_index}"


def _selected_membership_domain(
    definition: BenchmarkDefinition, seeds: FinalEvaluationSeedRegistry
) -> list[dict[str, Any]]:
    roles = {profile.profile_id: profile for profile in definition.profiles}
    rows = []
    for scene in definition.scene_families:
        for profile_id in SELECTED_PROFILE_IDS:
            role = roles[profile_id]
            for seed_index, episode_root in zip(
                seeds.indices, seeds.candidate_episode_seeds, strict=True
            ):
                rows.append(
                    {
                        "cell_id": _cell_id(scene.scene_family, profile_id, seed_index),
                        "profile_id": profile_id,
                        "profile_sha256": role.profile_sha256,
                        "benchmark_role": role.role.value,
                        "scene_family": scene.scene_family,
                        "scene_config_sha256": scene.config_logical_sha256,
                        "seed_index": seed_index,
                        "evaluation_episode_root": episode_root,
                        "benchmark_reference_cell_id": _cell_id(
                            scene.scene_family, PRIMARY_REFERENCE, seed_index
                        ),
                        "candidate_admission_control_cell_id": _cell_id(
                            scene.scene_family,
                            role.candidate_admission_matched_control_profile_id,
                            seed_index,
                        ),
                    }
                )
    return rows


def _legacy_control_membership_domain(
    definition: BenchmarkDefinition, seeds: FinalEvaluationSeedRegistry
) -> list[dict[str, Any]]:
    return [
        {
            "cell_id": _cell_id(scene.scene_family, LEGACY_CONTROL_PROFILE_ID, seed_index),
            "profile_id": LEGACY_CONTROL_PROFILE_ID,
            "scene_family": scene.scene_family,
            "scene_config_sha256": scene.config_logical_sha256,
            "seed_index": seed_index,
            "evaluation_episode_root": episode_root,
        }
        for scene in definition.scene_families
        for seed_index, episode_root in zip(
            seeds.indices, seeds.candidate_episode_seeds, strict=True
        )
    ]


def _lock_domain(lock: dict[str, Any]) -> dict[str, Any]:
    domain = dict(lock)
    domain.pop("definition_lock_sha256", None)
    return domain


def create_definition_lock_payload(
    definition: BenchmarkDefinition,
    seeds: FinalEvaluationSeedRegistry,
    revision: AppearanceRevision1Registry,
    configs: tuple[BenchmarkConfig, BenchmarkConfig],
) -> dict[str, Any]:
    """Build the prospective lock without observing final-root rendering."""

    validate_axis_isolation(revision)
    if appearance_registry_hash(revision) != REVISION1_REGISTRY_ROOT:
        raise FreezeError("Revision 1 registry root differs from protected evidence")
    roles = {profile.profile_id: profile for profile in definition.profiles}
    for profile_id in REVISION1_PROFILE_IDS:
        actual = profile_by_id(revision, profile_id)
        role = roles[profile_id]
        if appearance_profile_hash(actual) != role.profile_sha256:
            raise FreezeError(f"protected profile hash differs: {profile_id}")
        if actual.matched_control_profile_id != role.candidate_admission_matched_control_profile_id:
            raise FreezeError(f"candidate matched control differs: {profile_id}")
    if tuple(config.scene_family.value for config in configs) != SCENE_FAMILIES:
        raise FreezeError("scene configurations must be single_occluder then corridor")
    for scene, config in zip(definition.scene_families, configs, strict=True):
        if sha256_bytes(canonical_json_bytes(config)) != scene.config_logical_sha256:
            raise FreezeError(f"scene configuration identity differs: {scene.scene_family}")
    thresholds = profile_by_id(revision, PRIMARY_REFERENCE).non_degeneracy_thresholds
    if definition.admission_thresholds != thresholds:
        raise FreezeError("benchmark definition changes protected admission thresholds")
    selected = _selected_set_domain(definition)
    excluded = _excluded_set_domain(definition)
    domain: dict[str, Any] = {
        "schema_version": FREEZE_LOCK_VERSION,
        "canonical_base_sha": CANONICAL_BASE,
        "reviewed_implementation_head": REVIEWED_IMPLEMENTATION_HEAD,
        "superseded_definition_lock_commit": SUPERSEDED_FREEZE_LOCK_COMMIT,
        "superseded_definition_lock_sha256": SUPERSEDED_FREEZE_LOCK_ROOT,
        "supersession_finding_id": "EPS-ER17-0004",
        "supersession_posture": (
            "historical_lock_preserved_replacement_lock_authoritative_for_corrected_candidate"
        ),
        "revision1_approved_implementation_head": REVISION1_APPROVED_HEAD,
        "revision1_definition_lock_commit": REVISION1_LOCK_COMMIT,
        "revision1_definition_lock_sha256": REVISION1_LOCK_ROOT,
        "revision1_registry_sha256": REVISION1_REGISTRY_ROOT,
        "revision1_profile_admission_root_sha256": REVISION1_ADMISSION_ROOT,
        "benchmark_definition_sha256": benchmark_definition_hash(definition),
        "profile_role_root_sha256": _domain_hash("profile_roles", _role_domain(definition)),
        "selected_profile_set_root_sha256": _domain_hash("selected_profile_set", selected),
        "excluded_candidate_set_root_sha256": _domain_hash("excluded_candidate_set", excluded),
        "selected_profiles": selected,
        "excluded_profiles": excluded,
        "primary_reference_profile_id": PRIMARY_REFERENCE,
        "secondary_checker_frequency_diagnostic": (
            definition.secondary_checker_frequency_diagnostic.model_dump(mode="json")
        ),
        "scene_families": list(SCENE_FAMILIES),
        "scene_config_sha256": {
            scene.scene_family: scene.config_logical_sha256 for scene in definition.scene_families
        },
        "evaluation_episode_roots": list(seeds.candidate_episode_seeds),
        "evaluation_episode_seed_registry_sha256": evaluation_seed_registry_hash(seeds),
        "admission_thresholds": definition.admission_thresholds.model_dump(mode="json"),
        "admission_threshold_identity_sha256": _domain_hash(
            "admission_thresholds", definition.admission_thresholds.model_dump(mode="json")
        ),
        "paired_geometry_action_generation_rule": definition.pairing_policy.model_dump(mode="json"),
        "supported_renderer_environment_policy": definition.renderer_policy.model_dump(mode="json"),
        "training_evaluation_exclusion_policy": definition.evaluation_use_policy.model_dump(
            mode="json"
        ),
        "training_evaluation_policy_root_sha256": _domain_hash(
            "training_evaluation_policy", definition.evaluation_use_policy.model_dump(mode="json")
        ),
        "renderer_selection_dependency_root_sha256": _domain_hash(
            "renderer_selection_dependency",
            definition.renderer_policy.model_result_renderer_selection_dependency.model_dump(
                mode="json"
            ),
        ),
        "selected_matrix_membership_root_sha256": _domain_hash(
            "selected_matrix_membership", _selected_membership_domain(definition, seeds)
        ),
        "legacy_control_membership_root_sha256": _domain_hash(
            "legacy_control_membership", _legacy_control_membership_domain(definition, seeds)
        ),
        "expected_unique_selected_cell_count": 160,
        "expected_unique_control_cell_count": 32,
        "expected_total_cell_count": 192,
        "expected_selected_cells_per_profile": 32,
        "on_failure_seed_retirement_rule": definition.failure_policy.model_dump(mode="json"),
        "schema_versions": {
            "benchmark_definition": BENCHMARK_DEFINITION_VERSION,
            "evaluation_episode_seed_registry": seeds.registry_version,
            "freeze_definition_lock": FREEZE_LOCK_VERSION,
            "freeze_candidate_packet": FREEZE_PACKET_VERSION,
            "freeze_root_domains": FREEZE_ROOT_VERSION,
            "contact_sheet_manifest": FREEZE_CONTACT_SHEET_VERSION,
            "renderer_receipt": RENDERER_RECEIPT_VERSION,
            "source_identity": SOURCE_IDENTITY_VERSION,
            "threshold_margin_summary": THRESHOLD_MARGIN_VERSION,
            "logical_domain_envelope": DOMAIN_ENVELOPE_VERSION,
            "appearance_registry": "appearance_candidate_registry_v2",
            "appearance_profile": "appearance_profile_v3",
            "configuration": "0.1.0-dev.5",
            "dataset_manifest": "0.1.0-dev.8",
            "appearance_instance": "appearance_instance_v4",
        },
        "method_versions": {
            "seed_derivation": "derive_seed_v1",
            "texture_generator": "repository_procedural_texture_v1",
            "style_assignment": "balanced_cyclic_permutation_v1",
            "surface_repeat": "mujoco_geom_local_uv_repeat_v1",
            "assignment_schedule_source": "snapshotted_revision_partition_seed_registry_v1",
            "pairing": "independent_source_evidence_same_scene_same_root_v2",
            "readiness": "all_cells_both_locked_renderers_v2",
            "seed_retirement": "entire_registry_after_any_failed_cell_v1",
        },
        "qualification_started": False,
        "freeze_candidate_status": "unqualified",
        "benchmark_frozen": False,
        "final_split_authority": "none",
        "model_protocol_frozen": False,
        "full_gate_0b_complete": False,
        "gate_0c_authorised": False,
        "gate_0d_authorised": False,
        "scientific_result": None,
    }
    payload = {**domain, "definition_lock_sha256": _domain_hash("definition_lock", domain)}
    FreezeDefinitionLock.model_validate_json(canonical_json_bytes(payload))
    return payload


def _read_canonical_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FreezeError(f"invalid JSON object: {path}") from error
    if type(payload) is not dict or raw != canonical_json_bytes(payload) + b"\n":
        raise FreezeError(f"canonical JSON object required: {path}")
    return payload


def validate_definition_lock(
    benchmark_definition_path: Path,
    evaluation_seeds_path: Path,
    definition_lock_path: Path,
    revision_registry_path: Path,
    single_config_path: Path,
    corridor_config_path: Path,
) -> dict[str, Any]:
    definition = load_benchmark_definition(benchmark_definition_path)
    seeds = load_final_evaluation_seeds(evaluation_seeds_path)
    revision = load_appearance_registry_any(revision_registry_path)
    if not isinstance(revision, AppearanceRevision1Registry):
        raise FreezeError("freeze definition requires the exact Revision 1 registry")
    configs = (load_config(single_config_path), load_config(corridor_config_path))
    expected = create_definition_lock_payload(definition, seeds, revision, configs)
    actual = _read_canonical_json(definition_lock_path)
    try:
        FreezeDefinitionLock.model_validate_json(canonical_json_bytes(actual))
    except Exception as error:
        raise FreezeError("freeze definition lock schema is invalid") from error
    if (
        actual != expected
        or _domain_hash("definition_lock", _lock_domain(actual)) != actual["definition_lock_sha256"]
    ):
        raise FreezeError("freeze definition lock differs from independent recomputation")
    return actual


def _definition_lock_commit(path: Path = LOCK_PATH) -> str:
    relative = path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    result = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", relative],
        capture_output=True,
        text=True,
        check=True,
    )
    commits = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(commits) != 1:
        raise FreezeError("replacement freeze definition lock commit is unavailable")
    commit = commits[0]
    subject = subprocess.run(
        ["git", "show", "-s", "--format=%s", commit],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if subject != "feat: replace appearance benchmark freeze v0 lock domains":
        raise FreezeError("replacement freeze definition lock commit subject is not exact")
    if commit == SUPERSEDED_FREEZE_LOCK_COMMIT:
        raise FreezeError("historical freeze lock cannot serve as the replacement lock")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        capture_output=True,
        check=False,
    ).returncode:
        raise FreezeError("replacement freeze definition lock commit is not an ancestor of HEAD")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", SUPERSEDED_FREEZE_LOCK_COMMIT, commit],
        capture_output=True,
        check=False,
    ).returncode:
        raise FreezeError("replacement lock does not preserve the historical lock in ancestry")
    return commit


PORTABLE_DEFINITION_ROOT_FIELDS = (
    "benchmark_definition_sha256",
    "profile_role_root_sha256",
    "selected_profile_set_root_sha256",
    "excluded_candidate_set_root_sha256",
    "evaluation_episode_seed_registry_sha256",
    "scene_family_membership_root_sha256",
    "selected_matrix_membership_root_sha256",
    "legacy_control_membership_root_sha256",
    "freeze_definition_lock_sha256",
    "training_evaluation_policy_root_sha256",
    "renderer_selection_dependency_root_sha256",
)
PORTABLE_APPARATUS_ROOT_FIELDS = (
    "procedural_asset_root_sha256",
    "appearance_assignment_root_sha256",
    "portable_source_identity_root_sha256",
    "within_renderer_invariance_outcome_root_sha256",
)
RENDERER_LOCAL_ROOT_FIELDS = (
    "renderer_local_selected_cell_outcome_root_sha256",
    "renderer_local_control_cell_outcome_root_sha256",
    "renderer_local_ecological_label_root_sha256",
    "renderer_specific_audit_root_sha256",
    "renderer_specific_contact_sheet_root_sha256",
    "renderer_local_source_evidence_root_sha256",
)
SNAPSHOT_FILES = (
    "benchmark_definition_snapshot.json",
    "revision_registry_snapshot.json",
    "evaluation_episode_seed_registry_snapshot.json",
    "freeze_definition_lock.json",
    "single_scene_config_snapshot.json",
    "corridor_scene_config_snapshot.json",
)
REPORT_FILES = (
    "selected_profile_matrix.json",
    "legacy_control_matrix.json",
    "profile_readiness_summary.json",
    "negative_evidence.json",
    "contact_sheet_manifest.json",
)

SOURCE_IDENTITY_FIELDS = (
    "sampled_geometry_identity_sha256",
    "camera_trajectory_identity_sha256",
    "executed_action_identity_sha256",
    "surface_remapping_identity_sha256",
    "scene_content_identity_sha256",
    "analytic_transport_identity_sha256",
    "oriented_boundary_ownership_identity_sha256",
    "visibility_event_identity_sha256",
    "public_occlusion_relation_identity_sha256",
)


def _git_tree(commit: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", f"{commit}^{{tree}}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if len(result) != 40 or any(character not in "0123456789abcdef" for character in result):
        raise FreezeError("qualification source tree is not a canonical Git tree identity")
    return result


def _source_identity_from_dataset(
    source_root: Path,
    config: BenchmarkConfig,
) -> tuple[dict[str, Any], DatasetManifest]:
    """Reconstruct every portable pairing identity from retained immutable source evidence."""

    validate_dataset(source_root)
    try:
        manifest = DatasetManifest.model_validate_json((source_root / "manifest.json").read_bytes())
    except Exception as error:
        raise FreezeError("retained source-evidence manifest is invalid") from error
    if len(manifest.episodes) != 1:
        raise FreezeError("retained source evidence must contain exactly one episode")
    episode = manifest.episodes[0]
    try:
        transition = TransitionRecord.model_validate_json(
            (source_root / episode.transition.path).read_bytes()
        )
        instrumentation = parse_privileged_instrumentation_json(
            (source_root / episode.privileged_instrumentation.path).read_bytes()
        )
    except Exception as error:
        raise FreezeError("retained typed source evidence is invalid") from error
    if not isinstance(transition.analytic_optical_transport, AvailableDenseOpticalTransport):
        raise FreezeError("retained analytic transport evidence is unavailable")
    if not isinstance(
        transition.oriented_boundary_ownership, AvailableOrientedBoundaryOwnership
    ) or not isinstance(
        transition.ecological_visibility_events, AvailableEcologicalVisibilityEvents
    ):
        raise FreezeError("retained boundary or visibility evidence is unavailable")
    camera_records = [
        _read_canonical_json(source_root / frame.camera_world_transform.path)
        for frame in (transition.before, transition.after)
    ]
    sampled_geometry = getattr(instrumentation, "sampled_geometry", None)
    sampled_geometry_domain = {
        "positions": instrumentation.raw_geom_world_positions,
        "sizes": instrumentation.raw_geom_compiled_sizes,
        "types": instrumentation.raw_geom_types,
        "rotations": instrumentation.raw_geom_world_rotations_row_major,
        "sampled_geometry": (
            sampled_geometry.model_dump(mode="json") if sampled_geometry is not None else None
        ),
    }
    if config.scene_family.value == "single_occluder":
        scene_content = single_occluder_scene_content_domain(config)  # type: ignore[arg-type]
    else:
        if not isinstance(sampled_geometry, CorridorSampledGeometry):
            raise FreezeError("corridor source evidence lacks sampled geometry")
        scene_content = corridor_scene_content_domain(config, sampled_geometry)  # type: ignore[arg-type]
    roots = {
        "sampled_geometry_identity_sha256": _domain_hash(
            "sampled_geometry", sampled_geometry_domain
        ),
        "camera_trajectory_identity_sha256": _domain_hash("camera_trajectory", camera_records),
        "executed_action_identity_sha256": _domain_hash(
            "executed_action", transition.action.model_dump(mode="json")
        ),
        "surface_remapping_identity_sha256": _domain_hash(
            "surface_remapping", instrumentation.raw_to_opaque_surface_ids
        ),
        "scene_content_identity_sha256": _domain_hash("scene_content", scene_content),
        "analytic_transport_identity_sha256": _domain_hash(
            "analytic_transport", analytic_transport_domain(transition.analytic_optical_transport)
        ),
        "oriented_boundary_ownership_identity_sha256": _domain_hash(
            "oriented_boundary_ownership",
            oriented_boundary_domain(transition.oriented_boundary_ownership),
        ),
        "visibility_event_identity_sha256": _domain_hash(
            "visibility_events",
            visibility_event_domain(transition.ecological_visibility_events),
        ),
        "public_occlusion_relation_identity_sha256": _domain_hash(
            "public_occlusion_relation", transition.occlusion.model_dump(mode="json")
        ),
    }
    identity = {
        "schema_version": SOURCE_IDENTITY_VERSION,
        "dataset_logical_sha256": manifest.dataset_logical_sha256,
        **roots,
        "source_identity_root_sha256": _domain_hash("source_identity_root", roots),
    }
    return identity, manifest


def _legacy_source_claims_from_dataset(source_root: Path) -> dict[str, str]:
    """Reconstruct every retained compatibility hash instead of trusting matrix claims."""

    validate_dataset(source_root)
    try:
        manifest = DatasetManifest.model_validate_json((source_root / "manifest.json").read_bytes())
        episode = manifest.episodes[0]
        transition = TransitionRecord.model_validate_json(
            (source_root / episode.transition.path).read_bytes()
        )
        instrumentation = parse_privileged_instrumentation_json(
            (source_root / episode.privileged_instrumentation.path).read_bytes()
        )
    except Exception as error:
        raise FreezeError("retained compatibility source evidence is invalid") from error
    camera_records = [
        _read_canonical_json(source_root / frame.camera_world_transform.path)
        for frame in (transition.before, transition.after)
    ]
    sampled_geometry = getattr(instrumentation, "sampled_geometry", None)
    geometry_domain = {
        "positions": instrumentation.raw_geom_world_positions,
        "sizes": instrumentation.raw_geom_compiled_sizes,
        "types": instrumentation.raw_geom_types,
        "rotations": instrumentation.raw_geom_world_rotations_row_major,
        "sampled_geometry": (
            sampled_geometry.model_dump(mode="json") if sampled_geometry is not None else None
        ),
    }
    return {
        "geometry_sha256": sha256_bytes(canonical_json_bytes(geometry_domain)),
        "camera_trajectory_sha256": sha256_bytes(canonical_json_bytes(camera_records)),
        "action_sha256": sha256_bytes(canonical_json_bytes(transition.action)),
        "opaque_remapping_sha256": sha256_bytes(
            canonical_json_bytes(instrumentation.raw_to_opaque_surface_ids)
        ),
        "scene_content_sha256": episode.scene_content_sha256,
        "analytic_transport_sha256": episode.analytic_transport_sha256,
        "oriented_boundary_sha256": episode.oriented_boundary_sha256,
        "visibility_event_sha256": episode.visibility_event_sha256,
        "occlusion_sha256": sha256_bytes(canonical_json_bytes(transition.occlusion)),
    }


def _validate_source_evidence(
    packet_root: Path,
    cell: dict[str, Any],
    config: BenchmarkConfig,
    artifacts: _PacketArtifactRegistry,
) -> dict[str, Any]:
    record = cell.get("source_evidence")
    expected_record_fields = {
        "schema_version",
        "dataset_path",
        "manifest_file_sha256",
        "dataset_logical_sha256",
        "file_count",
        "file_manifest",
    }
    if (
        type(record) is not dict
        or set(record) != expected_record_fields
        or record["schema_version"] != "appearance_benchmark_source_dataset_evidence_v1"
        or type(record["dataset_path"]) is not str
        or type(record["file_count"]) is not int
        or record["file_count"] <= 0
        or type(record["file_manifest"]) is not list
    ):
        raise FreezeError("freeze retained source-evidence declaration is not strict")
    expected_path = f"source_evidence/{cell['cell_id']}"
    if record["dataset_path"] != expected_path:
        raise FreezeError("freeze retained source-evidence path differs")
    source_root = packet_root / expected_path
    identity, manifest = _source_identity_from_dataset(source_root, config)
    expected_config = type(config).model_validate(
        {
            **config.model_dump(mode="python"),
            "schema_version": "0.1.0-dev.5",
            "seed": cell["candidate_seed"],
            "appearance": {
                "registry_version": "appearance_candidate_registry_v2",
                "profile_id": cell["profile_id"],
            },
        }
    )
    retained_config = parse_config(
        _read_canonical_json(source_root / manifest.resolved_config.path)
    )
    if (
        sha256_file(source_root / "manifest.json") != record["manifest_file_sha256"]
        or manifest.dataset_logical_sha256 != record["dataset_logical_sha256"]
        or manifest.dataset_logical_sha256 != identity["dataset_logical_sha256"]
        or manifest.root_seed != cell["candidate_seed"]
        or manifest.scene_family.value != cell["scene_family"]
        or manifest.appearance_profile_id != cell["profile_id"]
        or retained_config != expected_config
        or manifest.config_logical_sha256 != sha256_bytes(canonical_json_bytes(expected_config))
    ):
        raise FreezeError("freeze retained source-evidence binding differs")
    files = sorted(path for path in source_root.rglob("*") if path.is_file())
    if len(files) != record["file_count"]:
        raise FreezeError("freeze retained source-evidence file count differs")
    actual_manifest = []
    for path in files:
        relative = path.relative_to(packet_root).as_posix()
        with artifacts.claim(relative, f"source-evidence:{cell['cell_id']}:{relative}") as owned:
            actual_manifest.append(
                {
                    "path": path.relative_to(source_root).as_posix(),
                    "file_sha256": sha256_bytes(owned.payload),
                    "byte_count": owned.byte_count,
                }
            )
    if record["file_manifest"] != actual_manifest:
        raise FreezeError("freeze retained source-evidence file manifest differs")
    if cell.get("source_identity") != identity:
        raise FreezeError("freeze source identity differs from independent reconstruction")
    legacy = _legacy_source_claims_from_dataset(source_root)
    if any(cell.get(field) != value for field, value in legacy.items()):
        raise FreezeError("legacy source claim differs from retained source evidence")
    return identity


def _freeze_cell_without_fields(cell: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(cell))
    if type(payload) is not dict:
        raise FreezeError("freeze cell copy is not an object")
    for field in (
        "benchmark_role",
        "benchmark_reference_cell_id",
        "candidate_admission_control_cell_id",
        "benchmark_pair_checks",
        "source_evidence",
        "source_identity",
    ):
        payload.pop(field, None)
    return payload


def _pair_checks(cell: dict[str, Any], reference: dict[str, Any]) -> dict[str, bool]:
    names = (
        "same_scene_and_evaluation_root",
        "sampled_geometry_equality",
        "camera_trajectory_equality",
        "executed_action_equality",
        "surface_remapping_equality",
        "scene_content_identity_equality",
        "analytic_transport_identity_equality",
        "oriented_boundary_ownership_identity_equality",
        "visibility_event_identity_equality",
        "public_occlusion_relation_identity_equality",
        "ecological_label_equality_within_renderer",
        "depth_equality_within_renderer",
        "segmentation_equality_within_renderer",
    )
    if cell["generation_status"] != "success" or reference["generation_status"] != "success":
        return dict.fromkeys(names, False)
    identity = cell["source_identity"]
    reference_identity = reference["source_identity"]
    return {
        "same_scene_and_evaluation_root": cell["scene_family"] == reference["scene_family"]
        and cell["candidate_seed"] == reference["candidate_seed"],
        **{
            check_name: identity[field_name] == reference_identity[field_name]
            for check_name, field_name in (
                ("sampled_geometry_equality", "sampled_geometry_identity_sha256"),
                ("camera_trajectory_equality", "camera_trajectory_identity_sha256"),
                ("executed_action_equality", "executed_action_identity_sha256"),
                ("surface_remapping_equality", "surface_remapping_identity_sha256"),
                ("scene_content_identity_equality", "scene_content_identity_sha256"),
                ("analytic_transport_identity_equality", "analytic_transport_identity_sha256"),
                (
                    "oriented_boundary_ownership_identity_equality",
                    "oriented_boundary_ownership_identity_sha256",
                ),
                ("visibility_event_identity_equality", "visibility_event_identity_sha256"),
                (
                    "public_occlusion_relation_identity_equality",
                    "public_occlusion_relation_identity_sha256",
                ),
            )
        },
        "ecological_label_equality_within_renderer": cell["ecological_label_sha256"]
        == reference["ecological_label_sha256"],
        "depth_equality_within_renderer": cell["depth_logical_sha256"]
        == reference["depth_logical_sha256"],
        "segmentation_equality_within_renderer": cell["segmentation_logical_sha256"]
        == reference["segmentation_logical_sha256"],
    }


def _evaluate_freeze_cell(
    packet_root: Path,
    cell: dict[str, Any],
    control: dict[str, Any],
    profile: Any,
    *,
    artifact_registry: _PacketArtifactRegistry | None = None,
    frame_cache: dict[tuple[str, str], tuple[Any, Any, Any]] | None = None,
) -> None:
    """Evaluate admission with reconstructed typed source identities as sole authority."""

    _evaluate_cell(
        packet_root,
        cell,
        control,
        profile,
        artifact_registry=artifact_registry,
        frame_cache=frame_cache,
    )
    if "frame_metrics" in cell:
        cell["frame_metrics"] = _portable_freeze_metric_value(cell["frame_metrics"])
    if cell["generation_status"] != "success" or control["generation_status"] != "success":
        return
    identity = cell["source_identity"]
    control_identity = control["source_identity"]
    structural_fields = SOURCE_IDENTITY_FIELDS
    portable_fields = (
        "scene_content_identity_sha256",
        "analytic_transport_identity_sha256",
        "oriented_boundary_ownership_identity_sha256",
        "visibility_event_identity_sha256",
        "public_occlusion_relation_identity_sha256",
        "executed_action_identity_sha256",
    )
    checks = cell["admission_checks"]
    checks["structural_invariance"] = all(
        identity[field] == control_identity[field] for field in structural_fields
    )
    checks["portable_analytic_identity_equality"] = all(
        identity[field] == control_identity[field] for field in portable_fields
    )
    reasons = [name for name, passed in checks.items() if not passed]
    cell["admission_status"] = "admitted" if not reasons else "rejected"
    cell["rejection_reasons"] = reasons


def _portable_source_identity(cell: dict[str, Any]) -> dict[str, Any]:
    identity = cell["source_identity"]
    return {
        "schema_version": identity["schema_version"],
        **{field: identity[field] for field in SOURCE_IDENTITY_FIELDS},
        "source_identity_root_sha256": identity["source_identity_root_sha256"],
    }


def _local_profile_readiness(
    selected: list[dict[str, Any]], definition: BenchmarkDefinition
) -> list[dict[str, Any]]:
    roles = {profile.profile_id: profile for profile in definition.profiles}
    rows = []
    for profile_id in SELECTED_PROFILE_IDS:
        cells = [cell for cell in selected if cell["profile_id"] == profile_id]
        rows.append(
            {
                "profile_id": profile_id,
                "profile_sha256": roles[profile_id].profile_sha256,
                "benchmark_role": roles[profile_id].role.value,
                "expected_cell_count": 32,
                "observed_cell_count": len(cells),
                "cell_counts": dict(Counter(cell["admission_status"] for cell in cells)),
                "renderer_apparatus_qualified": len(cells) == 32
                and all(
                    cell["admission_status"] == "admitted"
                    and all(cell["benchmark_pair_checks"].values())
                    for cell in cells
                ),
            }
        )
    return rows


def _renderer_environment_id(definition: BenchmarkDefinition, renderer: dict[str, Any]) -> str:
    for environment in definition.renderer_policy.supported_apparatus_environments:
        if renderer == environment.fingerprint.model_dump(mode="json"):
            return environment.environment_id
    raise FreezeError("observed renderer is not one of the two locked environments")


def _readiness_summary(
    definition: BenchmarkDefinition,
    environment_id: str,
    local_rows: list[dict[str, Any]],
    counterpart: dict[str, Any] | None,
    portable_apparatus_roots: dict[str, str],
) -> dict[str, Any]:
    local = {row["profile_id"]: row for row in local_rows}
    other = (
        {row["profile_id"]: row for row in counterpart["profiles"]}
        if counterpart is not None
        else {}
    )
    other_environment = counterpart["environment_id"] if counterpart is not None else None
    if other_environment == environment_id:
        raise FreezeError("counterpart receipt duplicates the local renderer")
    portable_match = (
        counterpart["portable_apparatus_roots"] == portable_apparatus_roots
        if counterpart is not None
        else None
    )
    profiles = []
    for profile_id in SELECTED_PROFILE_IDS:
        role = next(item for item in definition.profiles if item.profile_id == profile_id)
        local_value = local[profile_id]["renderer_apparatus_qualified"]
        other_value = other[profile_id]["renderer_apparatus_qualified"] if other else None
        windows = (
            local_value
            if environment_id == "windows_wgl_locked"
            else other_value
            if other_environment == "windows_wgl_locked"
            else None
        )
        osmesa = (
            local_value
            if environment_id == "ubuntu_osmesa_locked"
            else other_value
            if other_environment == "ubuntu_osmesa_locked"
            else None
        )
        cross = bool(windows and osmesa and portable_match) if counterpart is not None else None
        profiles.append(
            {
                "profile_id": profile_id,
                "profile_sha256": role.profile_sha256,
                "benchmark_role": role.role.value,
                "expected_cell_count_per_renderer": 32,
                "windows_wgl_apparatus_qualified": windows,
                "ubuntu_osmesa_apparatus_qualified": osmesa,
                "cross_renderer_apparatus_qualified": cross,
            }
        )
    local_all = all(row["renderer_apparatus_qualified"] for row in local_rows)
    if not local_all:
        status = "not_ready"
        disposition = "retired_after_failed_freeze_qualification"
    elif counterpart is None:
        status = "qualification_incomplete"
        disposition = "pending_second_renderer_qualification"
    elif portable_match and all(
        row["cross_renderer_apparatus_qualified"] is True for row in profiles
    ):
        status = "ready_for_dual_review"
        disposition = "eligible_for_owner_freeze_if_approved"
    else:
        status = "not_ready"
        disposition = "retired_after_failed_freeze_qualification"
    return {
        "schema_version": FREEZE_PACKET_VERSION,
        "local_environment_id": environment_id,
        "counterpart_environment_id": other_environment,
        "cross_renderer_complete": counterpart is not None,
        "portable_apparatus_roots_match": portable_match,
        "profiles": profiles,
        "freeze_candidate_status": status,
        "seed_set_disposition": disposition,
        "benchmark_frozen": False,
    }


def _portable_readiness_domain(summary: dict[str, Any]) -> list[dict[str, Any]] | None:
    if not summary["cross_renderer_complete"]:
        return None
    return [
        {
            "profile_id": row["profile_id"],
            "profile_sha256": row["profile_sha256"],
            "benchmark_role": row["benchmark_role"],
            "expected_cell_count_per_renderer": row["expected_cell_count_per_renderer"],
            "windows_wgl_apparatus_qualified": row["windows_wgl_apparatus_qualified"],
            "ubuntu_osmesa_apparatus_qualified": row["ubuntu_osmesa_apparatus_qualified"],
            "cross_renderer_apparatus_qualified": row["cross_renderer_apparatus_qualified"],
        }
        for row in summary["profiles"]
    ]


def _negative_evidence(
    definition: BenchmarkDefinition, selected: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "schema_version": FREEZE_PACKET_VERSION,
        "excluded_rejected_candidates": _excluded_set_domain(definition),
        "benchmark_limitations": [
            "no_admitted_stripe_family_only_condition",
            "no_admitted_illumination_only_condition",
        ],
        "selected_profile_rejected_cells": [
            {
                "cell_id": cell["cell_id"],
                "profile_id": cell["profile_id"],
                "scene_family": cell["scene_family"],
                "seed_index": cell["seed_index"],
                "evaluation_episode_root": cell["candidate_seed"],
                "rejection_reasons": cell["rejection_reasons"],
                "failure_type": cell.get("failure_type"),
                "failure_message": cell.get("failure_message"),
            }
            for cell in selected
            if cell["admission_status"] == "rejected"
        ],
    }


def _threshold_margin_summary(
    selected: list[dict[str, Any]], revision: AppearanceRevision1Registry
) -> dict[str, Any]:
    threshold = profile_by_id(revision, PRIMARY_REFERENCE).non_degeneracy_thresholds
    margins: dict[str, list[float]] = {
        "changed_controlled_pixel_fraction": [],
        "normalized_controlled_rgb_mad": [],
        "visible_surface_mean_luminance_lower": [],
        "visible_surface_mean_luminance_upper": [],
        "textured_surface_luminance_standard_deviation": [],
    }
    for cell in selected:
        if cell["generation_status"] != "success":
            continue
        textured = cell["appearance_instance"]["profile"]["texture"]["family"] != "solid"
        for frame in cell["frame_metrics"].values():
            margins["changed_controlled_pixel_fraction"].append(
                frame["changed_controlled_pixel_fraction"]
                - threshold.changed_controlled_pixel_fraction_minimum
            )
            margins["normalized_controlled_rgb_mad"].append(
                frame["normalized_controlled_rgb_mad"]
                - threshold.normalized_controlled_rgb_mad_minimum
            )
            for surface in frame["surface_diagnostics"]:
                margins["visible_surface_mean_luminance_lower"].append(
                    surface["luminance_mean"] - threshold.visible_surface_mean_luminance_minimum
                )
                margins["visible_surface_mean_luminance_upper"].append(
                    threshold.visible_surface_mean_luminance_maximum - surface["luminance_mean"]
                )
                if textured and surface["pixel_count"] >= threshold.textured_surface_minimum_pixels:
                    margins["textured_surface_luminance_standard_deviation"].append(
                        surface["luminance_standard_deviation"]
                        - threshold.textured_surface_luminance_std_minimum
                    )
    metrics = {
        name: {
            "count": len(values),
            "failure_count": sum(value < 0 for value in values),
            "near_threshold_count_abs_margin_le_0_005": sum(
                abs(value) <= 0.005 for value in values
            ),
            "minimum_margin": min(values) if values else None,
            "maximum_margin": max(values) if values else None,
        }
        for name, values in margins.items()
    }
    return {
        "schema_version": THRESHOLD_MARGIN_VERSION,
        "metrics": metrics,
        "threshold_margin_root_sha256": _domain_hash("threshold_margins", metrics),
    }


def _contact_sheets(packet_root: Path, selected: list[dict[str, Any]]) -> dict[str, Any]:
    directory = packet_root / "representative_contact_sheets"
    directory.mkdir()
    records = []
    for profile_id in SELECTED_PROFILE_IDS:
        profile_cells = [cell for cell in selected if cell["profile_id"] == profile_id]
        for scene in SCENE_FAMILIES:
            sheet = _contact_sheet_image(packet_root, profile_cells, scene)
            relative = f"representative_contact_sheets/{profile_id}--{scene}--seed-0.png"
            path = packet_root / relative
            sheet.save(path, format="PNG")
            pixels = np.asarray(sheet, dtype=np.uint8)
            records.append(
                {
                    "profile_id": profile_id,
                    "scene_family": scene,
                    "seed_index": 0,
                    "path": relative,
                    "media_type": "image/png",
                    "mode": "RGB",
                    "dimensions": list(sheet.size),
                    "dtype": str(pixels.dtype),
                    "shape": list(pixels.shape),
                    "logical_sha256": logical_array_hash(pixels),
                    "file_sha256": sha256_file(path),
                    "byte_count": path.stat().st_size,
                }
            )
    return {"schema_version": FREEZE_CONTACT_SHEET_VERSION, "sheets": records}


def _root_domains(
    definition: BenchmarkDefinition,
    seeds: FinalEvaluationSeedRegistry,
    lock: dict[str, Any],
    selected: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    contact_manifest: dict[str, Any],
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    successful = [cell for cell in [*controls, *selected] if cell["generation_status"] == "success"]
    definition_roots = {
        "benchmark_definition_sha256": benchmark_definition_hash(definition),
        "profile_role_root_sha256": _domain_hash("profile_roles", _role_domain(definition)),
        "selected_profile_set_root_sha256": _domain_hash(
            "selected_profile_set", _selected_set_domain(definition)
        ),
        "excluded_candidate_set_root_sha256": _domain_hash(
            "excluded_candidate_set", _excluded_set_domain(definition)
        ),
        "evaluation_episode_seed_registry_sha256": evaluation_seed_registry_hash(seeds),
        "scene_family_membership_root_sha256": _domain_hash(
            "scene_family_membership",
            [scene.model_dump(mode="json") for scene in definition.scene_families],
        ),
        "selected_matrix_membership_root_sha256": _domain_hash(
            "selected_matrix_membership", _selected_membership_domain(definition, seeds)
        ),
        "legacy_control_membership_root_sha256": _domain_hash(
            "legacy_control_membership", _legacy_control_membership_domain(definition, seeds)
        ),
        "freeze_definition_lock_sha256": lock["definition_lock_sha256"],
        "training_evaluation_policy_root_sha256": _domain_hash(
            "training_evaluation_policy", definition.evaluation_use_policy.model_dump(mode="json")
        ),
        "renderer_selection_dependency_root_sha256": _domain_hash(
            "renderer_selection_dependency",
            definition.renderer_policy.model_result_renderer_selection_dependency.model_dump(
                mode="json"
            ),
        ),
    }
    procedural = [
        {
            "cell_id": cell["cell_id"],
            "profile_sha256": cell["appearance_profile_sha256"],
            "textures": cell["appearance_instance"]["textures"],
        }
        for cell in successful
    ]
    assignments = [
        {
            "cell_id": cell["cell_id"],
            "appearance_instance_sha256": cell["appearance_instance_sha256"],
            "seed_registry_sha256": cell["evaluation_seed_registry_sha256"],
            "schedule_source": cell["appearance_assignment_schedule_source"],
            "schedule_index": cell["appearance_instance"]["seeds"]["candidate_schedule_index"],
            "style_assignment": cell["appearance_instance"]["style_assignment"],
        }
        for cell in successful
    ]
    portable = [
        {"cell_id": cell["cell_id"], **_portable_source_identity(cell)} for cell in successful
    ]
    invariance = [
        {
            "cell_id": cell["cell_id"],
            **_portable_source_identity(cell),
            "candidate_control_invariance": {
                name: cell["admission_checks"][name]
                for name in (
                    "structural_invariance",
                    "portable_analytic_identity_equality",
                    "ecological_label_equality",
                    "depth_segmentation_invariance",
                    "determinism",
                )
            },
            "benchmark_reference_pair_invariance": cell["benchmark_pair_checks"],
        }
        for cell in selected
        if cell["generation_status"] == "success"
    ]
    apparatus_roots = {
        "procedural_asset_root_sha256": _domain_hash("procedural_assets", procedural),
        "appearance_assignment_root_sha256": _domain_hash("appearance_assignments", assignments),
        "portable_source_identity_root_sha256": _domain_hash(
            "portable_source_identities", portable
        ),
        "within_renderer_invariance_outcome_root_sha256": _domain_hash(
            "within_renderer_invariance", invariance
        ),
    }
    selected_outcomes = [
        {
            "cell_id": cell["cell_id"],
            "profile_id": cell["profile_id"],
            "scene_family": cell["scene_family"],
            "seed_index": cell["seed_index"],
            "evaluation_episode_root": cell["candidate_seed"],
            "admission_status": cell["admission_status"],
            "admission_checks": cell["admission_checks"],
            "benchmark_pair_checks": cell["benchmark_pair_checks"],
            "rejection_reasons": cell["rejection_reasons"],
        }
        for cell in selected
    ]
    control_outcomes = [
        {
            "cell_id": cell["cell_id"],
            "scene_family": cell["scene_family"],
            "seed_index": cell["seed_index"],
            "evaluation_episode_root": cell["candidate_seed"],
            "admission_status": cell["admission_status"],
            "admission_checks": cell["admission_checks"],
            "rejection_reasons": cell["rejection_reasons"],
        }
        for cell in controls
    ]
    renderer_labels = [
        {"cell_id": cell["cell_id"], "ecological_label_sha256": cell["ecological_label_sha256"]}
        for cell in successful
    ]
    renderer_evidence = [
        {
            "cell_id": cell["cell_id"],
            "renderer_provenance": cell["renderer_provenance"],
            "rgb_logical_sha256": cell["rgb_logical_sha256"],
            "depth_logical_sha256": cell["depth_logical_sha256"],
            "segmentation_logical_sha256": cell["segmentation_logical_sha256"],
            "frame_metrics": cell["frame_metrics"],
            "admission_status": cell["admission_status"],
        }
        for cell in successful
    ]
    renderer_roots = {
        "renderer_local_selected_cell_outcome_root_sha256": _domain_hash(
            "selected_outcomes", selected_outcomes
        ),
        "renderer_local_control_cell_outcome_root_sha256": _domain_hash(
            "control_outcomes", control_outcomes
        ),
        "renderer_local_ecological_label_root_sha256": _domain_hash(
            "renderer_ecological_labels", renderer_labels
        ),
        "renderer_specific_audit_root_sha256": _domain_hash("renderer_audit", renderer_evidence),
        "renderer_specific_contact_sheet_root_sha256": _domain_hash(
            "renderer_contact_sheets", contact_manifest
        ),
        "renderer_local_source_evidence_root_sha256": _domain_hash(
            "renderer_source_evidence",
            [
                {
                    "cell_id": cell["cell_id"],
                    "source_evidence": cell["source_evidence"],
                    "source_identity": cell["source_identity"],
                }
                for cell in successful
            ],
        ),
    }
    return definition_roots, apparatus_roots, renderer_roots


def _validate_lock_commit_snapshots(
    commit: str,
    definition: BenchmarkDefinition,
    seeds: FinalEvaluationSeedRegistry,
    lock: dict[str, Any],
    source_provenance: dict[str, Any],
) -> None:
    if commit != _definition_lock_commit():
        raise FreezeError("packet lock commit is not the prospective replacement lock")
    historical_raw = subprocess.run(
        [
            "git",
            "show",
            f"{SUPERSEDED_FREEZE_LOCK_COMMIT}:configs/appearance_benchmark_freeze_v0_lock.json",
        ],
        capture_output=True,
        check=True,
    ).stdout
    historical = json.loads(historical_raw.decode("utf-8"))
    if (
        historical_raw != canonical_json_bytes(historical) + b"\n"
        or historical.get("definition_lock_sha256") != SUPERSEDED_FREEZE_LOCK_ROOT
    ):
        raise FreezeError("historical freeze lock was not preserved exactly")
    expected = {
        "configs/appearance_benchmark_freeze_v0.yaml": definition.model_dump(mode="json"),
        "configs/appearance_benchmark_v0_evaluation_episode_seeds.yaml": seeds.model_dump(
            mode="json"
        ),
        "configs/appearance_benchmark_freeze_v0_lock.json": lock,
    }
    for relative, value in expected.items():
        raw = subprocess.run(
            ["git", "show", f"{commit}:{relative}"], capture_output=True, check=True
        ).stdout
        committed = (
            json.loads(raw.decode("utf-8"))
            if relative.endswith(".json")
            else yaml.safe_load(raw.decode("utf-8"))
        )
        if relative.endswith(".json") and raw != canonical_json_bytes(committed) + b"\n":
            raise FreezeError("committed freeze lock is not canonical JSON")
        if canonical_json_bytes(committed) != canonical_json_bytes(value):
            raise FreezeError(f"definition-lock commit snapshot differs: {relative}")
    try:
        provenance = SourceProvenance.model_validate_json(canonical_json_bytes(source_provenance))
    except Exception as error:
        raise FreezeError("qualification source provenance is invalid") from error
    if (
        provenance.git_commit is None
        or provenance.git_dirty is not False
        or provenance.git_repository != CANONICAL_REPOSITORY_IDENTITY
    ):
        raise FreezeError("qualification requires exact Git source provenance")
    if source_provenance != collect_source_provenance(Path.cwd()).model_dump(mode="json"):
        raise FreezeError("qualification source provenance is not truthful")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, provenance.git_commit],
        capture_output=True,
        check=False,
    ).returncode:
        raise FreezeError("qualification source predates the immutable freeze lock")
    _git_tree(provenance.git_commit)


def _outcome_rows(
    cells: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "cell_id": cell["cell_id"],
            "profile_id": cell["profile_id"],
            "benchmark_role": cell["benchmark_role"],
            "scene_family": cell["scene_family"],
            "seed_index": cell["seed_index"],
            "evaluation_episode_root": cell["candidate_seed"],
            "admission_status": cell["admission_status"],
        }
        for cell in cells
    ]


def _validate_receipt_outcome_membership(
    payload: dict[str, Any],
    definition: BenchmarkDefinition,
    seeds: FinalEvaluationSeedRegistry,
) -> None:
    selected_expected = _selected_membership_domain(definition, seeds)
    control_expected = _legacy_control_membership_domain(definition, seeds)
    selected = payload["selected_outcome_map"]
    controls = payload["control_outcome_map"]
    if len(selected) != 160 or len(controls) != 32:
        raise FreezeError("renderer receipt outcome maps are incomplete")
    for actual, expected in zip(selected, selected_expected, strict=True):
        if (
            actual["cell_id"] != expected["cell_id"]
            or actual["profile_id"] != expected["profile_id"]
            or actual["benchmark_role"] != expected["benchmark_role"]
            or actual["scene_family"] != expected["scene_family"]
            or actual["seed_index"] != expected["seed_index"]
            or actual["evaluation_episode_root"] != expected["evaluation_episode_root"]
        ):
            raise FreezeError("renderer receipt selected outcome membership differs")
    for actual, expected in zip(controls, control_expected, strict=True):
        if (
            actual["cell_id"] != expected["cell_id"]
            or actual["profile_id"] != LEGACY_CONTROL_PROFILE_ID
            or actual["benchmark_role"] != "legacy_apparatus_control"
            or actual["scene_family"] != expected["scene_family"]
            or actual["seed_index"] != expected["seed_index"]
            or actual["evaluation_episode_root"] != expected["evaluation_episode_root"]
        ):
            raise FreezeError("renderer receipt control outcome membership differs")


def validate_renderer_receipt_payload(
    payload: dict[str, Any],
    definition: BenchmarkDefinition,
    seeds: FinalEvaluationSeedRegistry,
    lock: dict[str, Any],
) -> dict[str, Any]:
    """Validate one typed receipt against independent prospective definition evidence."""

    try:
        model = RendererQualificationReceipt.model_validate_json(canonical_json_bytes(payload))
    except Exception as error:
        raise FreezeError("renderer receipt schema is not strict and typed") from error
    validated = model.model_dump(mode="json")
    domain = dict(validated)
    declared = domain.pop("receipt_sha256")
    if _domain_hash("renderer_receipt", domain) != declared:
        raise FreezeError("renderer receipt identity differs")
    expected_environment = next(
        environment
        for environment in definition.renderer_policy.supported_apparatus_environments
        if environment.environment_id == validated["environment_id"]
    )
    if validated["renderer_fingerprint"] != expected_environment.fingerprint.model_dump(
        mode="json"
    ):
        raise FreezeError("renderer receipt fingerprint differs from its exact environment")
    provenance = validated["qualification_source_provenance"]
    if (
        provenance["git_commit"] != validated["qualification_source_commit"]
        or provenance["git_repository"] != CANONICAL_REPOSITORY_IDENTITY
        or provenance["git_dirty"] is not False
        or provenance["dirty_diff_sha256"] is not None
        or _git_tree(validated["qualification_source_commit"])
        != validated["qualification_source_tree"]
    ):
        raise FreezeError("renderer receipt source revision or tree binding differs")
    if subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            validated["replacement_definition_lock_commit"],
            validated["qualification_source_commit"],
        ],
        capture_output=True,
        check=False,
    ).returncode:
        raise FreezeError("renderer qualification source predates the replacement lock")
    if (
        validated["replacement_definition_lock_commit"] != _definition_lock_commit()
        or validated["replacement_definition_lock_sha256"] != lock["definition_lock_sha256"]
        or validated["benchmark_definition_sha256"] != benchmark_definition_hash(definition)
        or validated["evaluation_episode_seed_registry_sha256"]
        != evaluation_seed_registry_hash(seeds)
        or validated["portable_definition_roots"]["benchmark_definition_sha256"]
        != benchmark_definition_hash(definition)
        or validated["portable_definition_roots"]["evaluation_episode_seed_registry_sha256"]
        != evaluation_seed_registry_hash(seeds)
        or validated["portable_definition_roots"]["freeze_definition_lock_sha256"]
        != lock["definition_lock_sha256"]
    ):
        raise FreezeError("renderer receipt prospective definition binding differs")
    if set(validated["portable_definition_roots"]) != set(PORTABLE_DEFINITION_ROOT_FIELDS):
        raise FreezeError("renderer receipt definition root schema differs")
    if set(validated["portable_apparatus_roots"]) != set(PORTABLE_APPARATUS_ROOT_FIELDS):
        raise FreezeError("renderer receipt apparatus root schema differs")
    if set(validated["renderer_local_roots"]) != set(RENDERER_LOCAL_ROOT_FIELDS):
        raise FreezeError("renderer receipt local root schema differs")
    _validate_receipt_outcome_membership(validated, definition, seeds)
    selected_counts = dict(
        Counter(row["admission_status"] for row in validated["selected_outcome_map"])
    )
    control_counts = dict(
        Counter(row["admission_status"] for row in validated["control_outcome_map"])
    )
    roles = {profile.profile_id: profile for profile in definition.profiles}
    if (
        validated["selected_matrix_counts"] != selected_counts
        or validated["control_matrix_counts"] != control_counts
        or [row["profile_id"] for row in validated["profiles"]] != list(SELECTED_PROFILE_IDS)
        or any(
            row["profile_sha256"] != roles[row["profile_id"]].profile_sha256
            or row["benchmark_role"] != roles[row["profile_id"]].role.value
            or row["cell_counts"]
            != dict(
                Counter(
                    outcome["admission_status"]
                    for outcome in validated["selected_outcome_map"]
                    if outcome["profile_id"] == row["profile_id"]
                )
            )
            for row in validated["profiles"]
        )
    ):
        raise FreezeError("renderer receipt outcomes, roles, counts, or readiness differ")
    return validated


def load_renderer_receipt(
    path: Path,
    definition: BenchmarkDefinition,
    seeds: FinalEvaluationSeedRegistry,
    lock: dict[str, Any],
) -> dict[str, Any]:
    return validate_renderer_receipt_payload(_read_canonical_json(path), definition, seeds, lock)


def create_freeze_audit(
    benchmark_definition_path: Path,
    evaluation_seeds_path: Path,
    definition_lock_path: Path,
    revision_registry_path: Path,
    single_config_path: Path,
    corridor_config_path: Path,
    output: Path,
    counterpart_evidence_root: Path | None = None,
) -> dict[str, Any]:
    """Render all 192 final-root cells and publish one atomic qualification packet."""

    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError(f"freeze audit output is not empty: {output}")
        output.rmdir()
    lock = validate_definition_lock(
        benchmark_definition_path,
        evaluation_seeds_path,
        definition_lock_path,
        revision_registry_path,
        single_config_path,
        corridor_config_path,
    )
    lock_commit = _definition_lock_commit(definition_lock_path)
    definition = load_benchmark_definition(benchmark_definition_path)
    seeds = load_final_evaluation_seeds(evaluation_seeds_path)
    revision = load_appearance_registry_any(revision_registry_path)
    if not isinstance(revision, AppearanceRevision1Registry):
        raise FreezeError("freeze audit requires the exact Revision 1 registry")
    configs = (load_config(single_config_path), load_config(corridor_config_path))
    counterpart = (
        _load_counterpart_evidence(counterpart_evidence_root, definition, seeds, lock)
        if counterpart_evidence_root is not None
        else None
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    started = time.monotonic()
    try:
        write_canonical_json(staging / "benchmark_definition_snapshot.json", definition)
        write_canonical_json(staging / "revision_registry_snapshot.json", revision)
        write_canonical_json(staging / "evaluation_episode_seed_registry_snapshot.json", seeds)
        write_canonical_json(staging / "freeze_definition_lock.json", lock)
        write_canonical_json(staging / "single_scene_config_snapshot.json", configs[0])
        write_canonical_json(staging / "corridor_scene_config_snapshot.json", configs[1])
        if counterpart is not None:
            write_canonical_json(
                staging / "counterpart_renderer_receipt.json", counterpart["receipt"]
            )
            write_canonical_json(
                staging / "counterpart_publication_record.json",
                counterpart["publication_record"],
            )
        work = staging / "_temporary_datasets"
        work.mkdir()
        selected: list[dict[str, Any]] = []
        controls: list[dict[str, Any]] = []
        by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
        role_by_id = {profile.profile_id: profile for profile in definition.profiles}
        for config in configs:
            scene = config.scene_family.value
            for profile_id in (LEGACY_CONTROL_PROFILE_ID, *SELECTED_PROFILE_IDS):
                profile = profile_by_id(revision, profile_id)
                for seed_index, episode_root in zip(
                    seeds.indices, seeds.candidate_episode_seeds, strict=True
                ):
                    cell_id = _cell_id(scene, profile_id, seed_index)
                    cell = _dataset_cell(
                        work / cell_id,
                        work / f"{cell_id}--repeat",
                        staging,
                        config,
                        profile,
                        seed_index,
                        episode_root,
                        revision,
                        seeds,
                        cell_id_prefix="final-evaluation",
                        retain_source_evidence=True,
                    )
                    if cell["generation_status"] == "success":
                        cell["source_texture_diagnostics"] = (
                            _portable_freeze_source_texture_diagnostics(
                                cell["source_texture_diagnostics"]
                            )
                        )
                        source_root = staging / cell["source_evidence"]["dataset_path"]
                        cell["source_identity"], _ = _source_identity_from_dataset(
                            source_root, config
                        )
                    cell["benchmark_role"] = (
                        "legacy_apparatus_control"
                        if profile_id == LEGACY_CONTROL_PROFILE_ID
                        else role_by_id[profile_id].role.value
                    )
                    cell["benchmark_reference_cell_id"] = _cell_id(
                        scene, PRIMARY_REFERENCE, seed_index
                    )
                    cell["candidate_admission_control_cell_id"] = _cell_id(
                        scene, profile.matched_control_profile_id, seed_index
                    )
                    by_key[(scene, profile_id, seed_index)] = cell
                    (controls if profile_id == LEGACY_CONTROL_PROFILE_ID else selected).append(cell)
        for cell in controls:
            profile = profile_by_id(revision, cell["profile_id"])
            _evaluate_freeze_cell(staging, cell, cell, profile)
            cell["benchmark_pair_checks"] = _pair_checks(cell, cell)
        for cell in selected:
            profile = profile_by_id(revision, cell["profile_id"])
            control = by_key[
                (cell["scene_family"], profile.matched_control_profile_id, cell["seed_index"])
            ]
            _evaluate_freeze_cell(staging, cell, control, profile)
            reference = by_key[(cell["scene_family"], PRIMARY_REFERENCE, cell["seed_index"])]
            cell["benchmark_pair_checks"] = _pair_checks(cell, reference)
        shutil.rmtree(work)
        contact_manifest = _contact_sheets(staging, selected)
        write_canonical_json(staging / "contact_sheet_manifest.json", contact_manifest)
        definition_roots, apparatus_roots, renderer_roots = _root_domains(
            definition, seeds, lock, selected, controls, contact_manifest
        )
        for field in (
            "benchmark_definition_sha256",
            "profile_role_root_sha256",
            "selected_profile_set_root_sha256",
            "excluded_candidate_set_root_sha256",
            "evaluation_episode_seed_registry_sha256",
            "selected_matrix_membership_root_sha256",
            "legacy_control_membership_root_sha256",
            "freeze_definition_lock_sha256",
            "training_evaluation_policy_root_sha256",
            "renderer_selection_dependency_root_sha256",
        ):
            lock_field = (
                "definition_lock_sha256" if field.startswith("freeze_definition") else field
            )
            if definition_roots[field] != lock[lock_field]:
                raise FreezeError(f"generated definition root differs: {field}")
        successful = [
            cell for cell in [*controls, *selected] if cell["generation_status"] == "success"
        ]
        if len(successful) != 192:
            raise FreezeError("freeze matrix has a generation or validation failure")
        renderer = successful[0]["renderer_provenance"]
        if any(cell["renderer_provenance"] != renderer for cell in successful):
            raise FreezeError("freeze packet mixes renderer environments")
        environment_id = _renderer_environment_id(definition, renderer)
        if counterpart is not None:
            if counterpart["replacement_definition_lock_commit"] != lock_commit:
                raise FreezeError("counterpart receipt uses another lock commit")
            if counterpart["replacement_definition_lock_sha256"] != lock["definition_lock_sha256"]:
                raise FreezeError("counterpart receipt uses another lock root")
            if counterpart["portable_definition_roots"] != definition_roots:
                raise FreezeError("counterpart definition roots differ")
        local_rows = _local_profile_readiness(selected, definition)
        readiness = _readiness_summary(
            definition, environment_id, local_rows, counterpart, apparatus_roots
        )
        negative = _negative_evidence(definition, selected)
        reports = {
            "selected_profile_matrix.json": {
                "schema_version": FREEZE_PACKET_VERSION,
                "cells": selected,
            },
            "legacy_control_matrix.json": {
                "schema_version": FREEZE_PACKET_VERSION,
                "cells": controls,
            },
            "profile_readiness_summary.json": readiness,
            "negative_evidence.json": negative,
        }
        for name, report in reports.items():
            write_canonical_json(staging / name, report)
        portable_readiness = _portable_readiness_domain(readiness)
        threshold_summary = _threshold_margin_summary(selected, revision)
        snapshot_names = list(SNAPSHOT_FILES)
        if counterpart is not None:
            snapshot_names.extend(
                (
                    "counterpart_renderer_receipt.json",
                    "counterpart_publication_record.json",
                )
            )
        logical_domain = {
            "schema_version": FREEZE_PACKET_VERSION,
            "root_schema_version": FREEZE_ROOT_VERSION,
            "replacement_definition_lock_commit": lock_commit,
            "source_provenance": collect_source_provenance(Path.cwd()).model_dump(mode="json"),
            "renderer_environment_id": environment_id,
            "renderer_fingerprint": renderer,
            "portable_definition_roots": definition_roots,
            "portable_apparatus_roots": apparatus_roots,
            "portable_profile_readiness_root_sha256": (
                _domain_hash("profile_readiness", portable_readiness)
                if portable_readiness is not None
                else None
            ),
            "renderer_local_roots": renderer_roots,
            "snapshot_file_sha256": {name: sha256_file(staging / name) for name in snapshot_names},
            "report_file_sha256": {name: sha256_file(staging / name) for name in REPORT_FILES},
            "selected_cell_count": len(selected),
            "control_cell_count": len(controls),
            "total_cell_count": len(selected) + len(controls),
            "selected_matrix_counts": dict(Counter(cell["admission_status"] for cell in selected)),
            "control_matrix_counts": dict(Counter(cell["admission_status"] for cell in controls)),
            "local_profile_readiness": local_rows,
            "threshold_margin_summary": threshold_summary,
            "freeze_candidate_status": readiness["freeze_candidate_status"],
            "seed_set_disposition": readiness["seed_set_disposition"],
            "owner_approval": None,
            "closeout_authority": None,
            "benchmark_frozen": False,
            "primary_model_result_renderer": None,
            "model_protocol_frozen": False,
            "full_gate_0b_complete": False,
            "gate_0c_authorised": False,
            "gate_0d_authorised": False,
            "scientific_result": None,
        }
        packet = {
            **logical_domain,
            "complete_packet_root_sha256": _domain_hash("complete_packet", logical_domain),
        }
        write_canonical_json(staging / "freeze_candidate_packet.json", packet)
        (staging / "run.json").write_bytes(
            (
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(UTC).isoformat(),
                        "hostname": socket.gethostname(),
                        "operating_system": platform.platform(),
                        "wall_clock_seconds": time.monotonic() - started,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
        )
        validate_freeze_audit(staging, counterpart_evidence_root)
        _atomic_no_replace_directory(staging, output)
        return packet
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _public_packet_evidence(packet_root: Path, packet: dict[str, Any]) -> dict[str, Any]:
    files = sorted(
        path for path in packet_root.rglob("*") if path.is_file() and path.name != "run.json"
    )
    manifest = [
        {
            "path": path.relative_to(packet_root).as_posix(),
            "file_sha256": sha256_file(path),
            "byte_count": path.stat().st_size,
        }
        for path in files
    ]
    packet_path = packet_root / "freeze_candidate_packet.json"
    return {
        "evidence_class": "PUBLIC_REPOSITORY_ONLY",
        "availability": "repository_or_ci_artifact",
        "packet_schema_version": packet["schema_version"],
        "packet_root_schema_version": packet["root_schema_version"],
        "packet_file_sha256": sha256_file(packet_path),
        "packet_tree_root_sha256": _domain_hash("public_packet_tree", manifest),
        "packet_artifact_count": len(files),
    }


def _ordinary_json_object(path: Path, role: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FreezeError(f"{role} is unavailable or invalid") from error
    if type(payload) is not dict:
        raise FreezeError(f"{role} must be a JSON object")
    return payload


def _artifact_digest(value: Any) -> str:
    if type(value) is not str:
        raise FreezeError("CI artifact digest is unavailable")
    digest = value.removeprefix("sha256:")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise FreezeError("CI artifact digest is not canonical SHA-256")
    return digest


def _is_canonical_repository_reference(value: Any) -> bool:
    if type(value) is not str:
        return False
    try:
        return canonical_github_repository_identity(value) == CANONICAL_REPOSITORY_IDENTITY
    except ValueError:
        return False


def _owned_archive_payload(path: Path, role: str) -> bytes:
    try:
        with open_owned_regular_file(path.parent, path.name) as owned:
            return owned.payload
    except UnsafeOwnedFileError as error:
        raise ImmutableArtifactBindingError(
            "artifact_archive_unavailable",
            f"{role} archive is not one stable owned regular file",
        ) from error


def _verify_artifact_archive_payload(
    payload: bytes,
    artifact_metadata: dict[str, Any],
    role: str,
) -> None:
    if artifact_metadata.get("expired") is not False:
        raise ImmutableArtifactBindingError(
            "artifact_expired",
            f"{role} artifact is expired or its live expiry status is unavailable",
        )
    try:
        expected_digest = _artifact_digest(artifact_metadata.get("digest"))
    except FreezeError as error:
        raise ImmutableArtifactBindingError(
            "artifact_digest_unavailable",
            f"{role} live artifact digest is unavailable",
        ) from error
    if type(artifact_metadata.get("size_in_bytes")) is not int:
        raise ImmutableArtifactBindingError(
            "artifact_size_unavailable",
            f"{role} live artifact size is unavailable",
        )
    if len(payload) != artifact_metadata["size_in_bytes"]:
        raise ImmutableArtifactBindingError(
            "artifact_archive_size_mismatch",
            f"{role} archive bytes differ from the live artifact size",
        )
    if sha256_bytes(payload) != expected_digest:
        raise ImmutableArtifactBindingError(
            "artifact_archive_digest_mismatch",
            f"{role} archive bytes differ from the live immutable artifact digest",
        )


def _safe_extract_artifact_archive(
    payload: bytes,
    destination: Path,
    role: str,
    *,
    expected_members: set[str] | None = None,
) -> set[str]:
    """Extract one already digest-verified ZIP into a new isolated directory."""

    if destination.exists():
        raise ImmutableArtifactBindingError(
            "artifact_extraction_destination_exists",
            f"{role} extraction destination already exists",
        )
    try:
        archive = zipfile.ZipFile(BytesIO(payload))
    except (OSError, zipfile.BadZipFile) as error:
        raise ImmutableArtifactBindingError(
            "artifact_archive_invalid",
            f"{role} archive is not a valid ZIP",
        ) from error
    with archive:
        members: dict[str, zipfile.ZipInfo] = {}
        casefolded: set[str] = set()
        try:
            infos = archive.infolist()
        except (OSError, zipfile.BadZipFile) as error:
            raise ImmutableArtifactBindingError(
                "artifact_archive_invalid",
                f"{role} archive member table is invalid",
            ) from error
        if len(infos) > 10_000 or sum(info.file_size for info in infos) > 1_073_741_824:
            raise ImmutableArtifactBindingError(
                "artifact_archive_resource_limit",
                f"{role} archive exceeds the bounded member or extracted-byte limit",
            )
        for info in infos:
            name = info.filename
            logical = PurePosixPath(name)
            unix_mode = (info.external_attr >> 16) & 0xFFFF
            unix_kind = stat.S_IFMT(unix_mode)
            if (
                not name
                or "\\" in name
                or "\x00" in name
                or name.endswith("/")
                or info.is_dir()
                or info.flag_bits & 0x1
                or logical.is_absolute()
                or logical.as_posix() != name
                or ".." in logical.parts
                or any(not part or part == "." or ":" in part for part in logical.parts)
                or unix_kind not in {0, stat.S_IFREG}
            ):
                raise ImmutableArtifactBindingError(
                    "artifact_archive_unsafe_member",
                    f"{role} archive contains a non-regular or unsafe member",
                )
            folded = name.casefold()
            if name in members or folded in casefolded:
                raise ImmutableArtifactBindingError(
                    "artifact_archive_colliding_member",
                    f"{role} archive contains duplicate or case-colliding members",
                )
            members[name] = info
            casefolded.add(folded)
        for name in members:
            parent = PurePosixPath(name).parent
            while parent != PurePosixPath("."):
                if parent.as_posix().casefold() in casefolded:
                    raise ImmutableArtifactBindingError(
                        "artifact_archive_colliding_member",
                        f"{role} archive contains a file/directory path collision",
                    )
                parent = parent.parent
        member_names = set(members)
        if expected_members is not None and member_names != expected_members:
            raise ImmutableArtifactBindingError(
                "artifact_archive_member_set_mismatch",
                f"{role} archive member set is not the exact authorised set",
            )

        destination.mkdir(parents=False, exist_ok=False)
        for name in sorted(members):
            target = destination.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with archive.open(members[name], "r") as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
            except (OSError, RuntimeError, zipfile.BadZipFile) as error:
                raise ImmutableArtifactBindingError(
                    "artifact_archive_extraction_failed",
                    f"{role} archive member could not be extracted exactly",
                ) from error
            try:
                extracted_size = target.stat().st_size
            except OSError as error:
                raise ImmutableArtifactBindingError(
                    "artifact_archive_extraction_failed",
                    f"{role} extracted member is unavailable",
                ) from error
            if extracted_size != members[name].file_size:
                raise ImmutableArtifactBindingError(
                    "artifact_archive_extraction_failed",
                    f"{role} extracted member size differs",
                )
        return member_names


def _validate_packet_archive_binding(
    packet_root: Path,
    artifact_archive_path: Path,
    artifact_metadata: dict[str, Any],
    counterpart_evidence_root: Path | None,
    role: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    archive_payload = _owned_archive_payload(artifact_archive_path, role)
    _verify_artifact_archive_payload(archive_payload, artifact_metadata, role)
    with tempfile.TemporaryDirectory(prefix="epsbench-verified-artifact-") as temporary:
        extracted_root = Path(temporary) / "packet"
        _safe_extract_artifact_archive(archive_payload, extracted_root, role)
        archive_packet = validate_freeze_audit(extracted_root, counterpart_evidence_root)
        archive_identity = _public_packet_evidence(extracted_root, archive_packet)
        local_packet = validate_freeze_audit(packet_root, counterpart_evidence_root)
        if archive_packet != local_packet or archive_identity != _public_packet_evidence(
            packet_root, local_packet
        ):
            raise ImmutableArtifactBindingError(
                "artifact_archive_packet_mismatch",
                f"{role} digest-verified archive differs from the supplied packet",
            )
        return archive_packet, archive_identity


def _matching_workflow_job(
    jobs_metadata: dict[str, Any],
    *,
    job_name: str,
    workflow_run_id: int,
    workflow_run_attempt: int,
    source_commit: str,
) -> dict[str, Any]:
    jobs = jobs_metadata.get("jobs")
    expected_endpoint = (
        "https://api.github.com/repos/yurifrusin/ecological-predictive-states/actions/"
        f"runs/{workflow_run_id}/attempts/{workflow_run_attempt}/jobs?per_page=100"
    )
    if (
        type(jobs) is not list
        or jobs_metadata.get("resolved_workflow_run_id") != workflow_run_id
        or jobs_metadata.get("resolved_workflow_run_attempt") != workflow_run_attempt
        or jobs_metadata.get("resolved_head_sha") != source_commit
        or jobs_metadata.get("resolution_endpoint") != expected_endpoint
    ):
        raise FreezeError("workflow jobs metadata is incomplete")
    matches = [
        job
        for job in jobs
        if type(job) is dict
        and job.get("name") == job_name
        and job.get("run_id") == workflow_run_id
        and job.get("head_sha") == source_commit
    ]
    if len(matches) != 1 or type(matches[0].get("id")) is not int:
        raise FreezeError("workflow job identity is absent or ambiguous")
    return cast(dict[str, Any], matches[0])


def create_publication_record(
    packet_root: Path,
    artifact_archive_path: Path,
    artifact_metadata_path: Path,
    workflow_run_metadata_path: Path,
    workflow_jobs_metadata_path: Path,
    *,
    job_name: str,
    artifact_digest_sha256: str,
    artifact_url: str,
    output: Path,
    counterpart_evidence_root: Path | None = None,
) -> dict[str, Any]:
    """Bind a digest-verified immutable CI archive to its validated complete packet."""

    artifact = _ordinary_json_object(artifact_metadata_path, "CI artifact metadata")
    packet, packet_identity = _validate_packet_archive_binding(
        packet_root,
        artifact_archive_path,
        artifact,
        counterpart_evidence_root,
        "published packet",
    )
    run = _ordinary_json_object(workflow_run_metadata_path, "workflow run metadata")
    jobs = _ordinary_json_object(workflow_jobs_metadata_path, "workflow jobs metadata")
    repository = run.get("repository")
    workflow_run = artifact.get("workflow_run")
    head_commit = run.get("head_commit")
    source_commit = packet["source_provenance"]["git_commit"]
    source_tree = _git_tree(source_commit)
    renderer_name = {
        "windows_wgl_locked": "wgl",
        "ubuntu_osmesa_locked": "osmesa",
    }.get(packet["renderer_environment_id"])
    expected_artifact_name = f"appearance-freeze-{renderer_name}-packet-{source_commit}"
    expected_job_name = f"qualify-{renderer_name}"
    if (
        renderer_name is None
        or job_name != expected_job_name
        or type(repository) is not dict
        or not _is_canonical_repository_reference(repository.get("html_url"))
        or repository.get("full_name") != CANONICAL_REPOSITORY_IDENTITY
        or run.get("head_sha") != source_commit
        or type(head_commit) is not dict
        or head_commit.get("id") != source_commit
        or head_commit.get("tree_id") != source_tree
        or type(run.get("id")) is not int
        or type(run.get("run_attempt")) is not int
        or type(run.get("name")) is not str
        or type(run.get("path")) is not str
        or type(run.get("html_url")) is not str
        or type(run.get("created_at")) is not str
        or type(workflow_run) is not dict
        or workflow_run.get("id") != run["id"]
        or workflow_run.get("head_sha") != source_commit
        or artifact.get("expired") is not False
        or type(artifact.get("id")) is not int
        or artifact.get("name") != expected_artifact_name
        or artifact.get("url")
        != (
            f"https://api.github.com/repos/{CANONICAL_REPOSITORY_IDENTITY}/actions/"
            f"artifacts/{artifact.get('id')}"
        )
        or type(artifact.get("size_in_bytes")) is not int
        or artifact["size_in_bytes"] <= 0
        or artifact.get("archive_download_url") != f"{artifact.get('url')}/zip"
        or type(artifact.get("created_at")) is not str
        or type(artifact.get("expires_at")) is not str
    ):
        raise FreezeError("CI artifact or workflow provenance is incomplete")
    digest = _artifact_digest(artifact_digest_sha256)
    if _artifact_digest(artifact.get("digest")) != digest:
        raise FreezeError("CI artifact API and upload digest differ")
    job = _matching_workflow_job(
        jobs,
        job_name=job_name,
        workflow_run_id=run["id"],
        workflow_run_attempt=run["run_attempt"],
        source_commit=source_commit,
    )
    expected_artifact_url = (
        f"https://github.com/yurifrusin/ecological-predictive-states/actions/runs/"
        f"{run['id']}/artifacts/{artifact['id']}"
    )
    if artifact_url != expected_artifact_url:
        raise FreezeError("CI artifact public URL differs from its exact identity")
    domain = {
        "schema_version": PUBLICATION_RECORD_VERSION,
        "evidence_class": "PUBLIC_REPOSITORY_ONLY",
        "repository": CANONICAL_REPOSITORY_IDENTITY,
        "source_commit": source_commit,
        "source_tree": source_tree,
        "workflow_run_id": run["id"],
        "workflow_run_attempt": run["run_attempt"],
        "workflow_name": run["name"],
        "workflow_path": run["path"],
        "workflow_run_url": run["html_url"],
        "workflow_run_created_at_utc": run["created_at"],
        "job_database_id": job["id"],
        "job_name": job_name,
        "job_api_url": job["url"],
        "job_html_url": job["html_url"],
        "artifact_id": artifact["id"],
        "artifact_name": artifact["name"],
        "artifact_api_url": artifact["url"],
        "artifact_url": artifact_url,
        "artifact_archive_download_url": artifact["archive_download_url"],
        "artifact_digest_sha256": digest,
        "artifact_size_in_bytes": artifact["size_in_bytes"],
        "artifact_created_at_utc": artifact["created_at"],
        "artifact_expires_at_utc": artifact["expires_at"],
        "artifact_expired_at_record_creation": False,
        "retention_days": 90,
        "retention_posture": (
            "github_actions_immutable_90_day_artifact_unless_repository_run_or_owner_"
            "deletes_earlier"
        ),
        "public_access_posture": "public_repository_authenticated_actions_artifact",
        "packet_identity": packet_identity,
        "complete_packet_root_sha256": packet["complete_packet_root_sha256"],
    }
    record = {**domain, "record_sha256": sha256_bytes(canonical_json_bytes(domain))}
    PublicPacketPublicationRecord.model_validate_json(canonical_json_bytes(record))
    if output.exists():
        raise FileExistsError(f"publication record already exists: {output}")
    write_canonical_json(output, record)
    return record


def validate_publication_record(
    payload: dict[str, Any],
    packet_root: Path,
    packet: dict[str, Any],
    artifact_metadata: dict[str, Any],
    workflow_run_metadata: dict[str, Any],
    workflow_jobs_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Verify packet publication against a live GitHub Actions metadata resolution."""

    try:
        record = PublicPacketPublicationRecord.model_validate_json(
            canonical_json_bytes(payload)
        ).model_dump(mode="json")
    except Exception as error:
        raise FreezeError("public packet publication record is invalid or expired") from error
    if (
        record["source_commit"] != packet["source_provenance"]["git_commit"]
        or record["source_tree"] != _git_tree(record["source_commit"])
        or record["packet_identity"] != _public_packet_evidence(packet_root, packet)
        or record["complete_packet_root_sha256"] != packet["complete_packet_root_sha256"]
    ):
        raise FreezeError("public packet publication differs from validated packet")
    repository = workflow_run_metadata.get("repository")
    repository_url = repository.get("html_url") if type(repository) is dict else None
    artifact_run = artifact_metadata.get("workflow_run")
    head_commit = workflow_run_metadata.get("head_commit")
    renderer_name = {
        "windows_wgl_locked": "wgl",
        "ubuntu_osmesa_locked": "osmesa",
    }.get(packet["renderer_environment_id"])
    expected_artifact_name = f"appearance-freeze-{renderer_name}-packet-{record['source_commit']}"
    expected_api_url = (
        f"https://api.github.com/repos/{record['repository']}/actions/artifacts/"
        f"{record['artifact_id']}"
    )
    if (
        renderer_name is None
        or record["job_name"] != f"qualify-{renderer_name}"
        or type(repository) is not dict
        or not _is_canonical_repository_reference(repository_url)
        or repository.get("full_name") != record["repository"]
        or workflow_run_metadata.get("id") != record["workflow_run_id"]
        or workflow_run_metadata.get("run_attempt") != record["workflow_run_attempt"]
        or workflow_run_metadata.get("name") != record["workflow_name"]
        or workflow_run_metadata.get("path") != record["workflow_path"]
        or workflow_run_metadata.get("html_url") != record["workflow_run_url"]
        or workflow_run_metadata.get("created_at") != record["workflow_run_created_at_utc"]
        or workflow_run_metadata.get("head_sha") != record["source_commit"]
        or type(head_commit) is not dict
        or head_commit.get("id") != record["source_commit"]
        or head_commit.get("tree_id") != record["source_tree"]
        or type(artifact_run) is not dict
        or artifact_run.get("id") != record["workflow_run_id"]
        or artifact_run.get("head_sha") != record["source_commit"]
        or artifact_metadata.get("id") != record["artifact_id"]
        or artifact_metadata.get("name") != expected_artifact_name
        or record["artifact_name"] != expected_artifact_name
        or artifact_metadata.get("url") != expected_api_url
        or record["artifact_api_url"] != expected_api_url
        or artifact_metadata.get("archive_download_url") != f"{expected_api_url}/zip"
        or record["artifact_archive_download_url"] != f"{expected_api_url}/zip"
        or artifact_metadata.get("size_in_bytes") != record["artifact_size_in_bytes"]
        or artifact_metadata.get("created_at") != record["artifact_created_at_utc"]
        or artifact_metadata.get("expires_at") != record["artifact_expires_at_utc"]
        or artifact_metadata.get("expired") is not False
        or _artifact_digest(artifact_metadata.get("digest")) != record["artifact_digest_sha256"]
        or record["artifact_url"]
        != (
            f"https://github.com/{record['repository']}/actions/runs/"
            f"{record['workflow_run_id']}/artifacts/{record['artifact_id']}"
        )
    ):
        raise FreezeError("live CI artifact resolution differs or is unavailable")
    job = _matching_workflow_job(
        workflow_jobs_metadata,
        job_name=record["job_name"],
        workflow_run_id=record["workflow_run_id"],
        workflow_run_attempt=record["workflow_run_attempt"],
        source_commit=record["source_commit"],
    )
    if job["id"] != record["job_database_id"]:
        raise FreezeError("live workflow job identity differs")
    if job.get("url") != record["job_api_url"] or job.get("html_url") != record["job_html_url"]:
        raise FreezeError("live workflow job URLs differ")
    return record


def _receipt_context(
    packet_root: Path,
) -> tuple[BenchmarkDefinition, FinalEvaluationSeedRegistry, dict[str, Any]]:
    definition = BenchmarkDefinition.model_validate_json(
        (packet_root / "benchmark_definition_snapshot.json").read_bytes()
    )
    seeds = parse_seed_registry(
        _read_canonical_json(packet_root / "evaluation_episode_seed_registry_snapshot.json")
    )
    if not isinstance(seeds, FinalEvaluationSeedRegistry):
        raise FreezeError("renderer receipt packet has the wrong seed registry")
    lock = FreezeDefinitionLock.model_validate_json(
        (packet_root / "freeze_definition_lock.json").read_bytes()
    ).model_dump(mode="json")
    return definition, seeds, lock


def _validate_receipt_against_packet(
    receipt: dict[str, Any],
    packet_root: Path,
    packet: dict[str, Any],
    definition: BenchmarkDefinition,
    seeds: FinalEvaluationSeedRegistry,
    lock: dict[str, Any],
) -> dict[str, Any]:
    validated = validate_renderer_receipt_payload(receipt, definition, seeds, lock)
    selected = _read_canonical_json(packet_root / "selected_profile_matrix.json")["cells"]
    controls = _read_canonical_json(packet_root / "legacy_control_matrix.json")["cells"]
    exact = {
        "environment_id": packet["renderer_environment_id"],
        "renderer_fingerprint": packet["renderer_fingerprint"],
        "qualification_source_provenance": packet["source_provenance"],
        "qualification_source_commit": packet["source_provenance"]["git_commit"],
        "qualification_source_tree": _git_tree(packet["source_provenance"]["git_commit"]),
        "replacement_definition_lock_commit": packet["replacement_definition_lock_commit"],
        "replacement_definition_lock_sha256": packet["portable_definition_roots"][
            "freeze_definition_lock_sha256"
        ],
        "benchmark_definition_sha256": packet["portable_definition_roots"][
            "benchmark_definition_sha256"
        ],
        "evaluation_episode_seed_registry_sha256": packet["portable_definition_roots"][
            "evaluation_episode_seed_registry_sha256"
        ],
        "complete_packet_root_sha256": packet["complete_packet_root_sha256"],
        "selected_cell_count": packet["selected_cell_count"],
        "control_cell_count": packet["control_cell_count"],
        "total_cell_count": packet["total_cell_count"],
        "selected_matrix_counts": packet["selected_matrix_counts"],
        "control_matrix_counts": packet["control_matrix_counts"],
        "selected_outcome_map": _outcome_rows(selected),
        "control_outcome_map": _outcome_rows(controls),
        "profiles": packet["local_profile_readiness"],
        "portable_definition_roots": packet["portable_definition_roots"],
        "portable_apparatus_roots": packet["portable_apparatus_roots"],
        "renderer_local_roots": packet["renderer_local_roots"],
        "threshold_margin_summary": packet["threshold_margin_summary"],
        "public_qualification_evidence": _public_packet_evidence(packet_root, packet),
    }
    if any(validated[field] != value for field, value in exact.items()):
        raise FreezeError("renderer receipt differs from independently validated packet evidence")
    return validated


_COUNTERPART_BUNDLE_JSON_FILES = (
    "packet_live_artifact.json",
    "packet_live_workflow_run.json",
    "packet_live_workflow_jobs.json",
    "evidence_live_artifact.json",
    "evidence_live_workflow_run.json",
    "evidence_live_workflow_jobs.json",
)
_COUNTERPART_BUNDLE_ARCHIVE_FILES = (
    "packet_artifact.zip",
    "evidence_artifact.zip",
)
_EVIDENCE_ARCHIVE_MEMBERS = {"renderer_receipt.json", "publication_record.json"}


def _load_counterpart_bundle(evidence_root: Path) -> dict[str, Any]:
    artifacts = _PacketArtifactRegistry(evidence_root)
    bundle: dict[str, Any] = {}
    try:
        for name in _COUNTERPART_BUNDLE_JSON_FILES:
            with artifacts.claim(name, f"counterpart-bundle:{name}") as owned:
                bundle[name] = _decode_canonical_object(owned.payload, name)
        for name in _COUNTERPART_BUNDLE_ARCHIVE_FILES:
            with artifacts.claim(name, f"counterpart-bundle:{name}") as owned:
                bundle[name] = owned.payload
        artifacts.assert_exact_tree()
    except ImmutableArtifactBindingError:
        raise
    except (OSError, ValueError) as error:
        raise ImmutableArtifactBindingError(
            "counterpart_bundle_invalid",
            "counterpart evidence bundle is incomplete, aliased, or contains extra paths",
        ) from error
    return bundle


def _validate_evidence_artifact_metadata(
    artifact: dict[str, Any],
    run: dict[str, Any],
    jobs: dict[str, Any],
    publication: dict[str, Any],
    environment_id: str,
) -> None:
    repository = run.get("repository")
    artifact_run = artifact.get("workflow_run")
    repository_url = repository.get("html_url") if type(repository) is dict else None
    renderer_name = {
        "windows_wgl_locked": "wgl",
        "ubuntu_osmesa_locked": "osmesa",
    }.get(environment_id)
    expected_name = f"appearance-freeze-{renderer_name}-evidence-{publication['source_commit']}"
    expected_api_url = (
        f"https://api.github.com/repos/{CANONICAL_REPOSITORY_IDENTITY}/actions/artifacts/"
        f"{artifact.get('id')}"
    )
    expected_archive_url = f"{expected_api_url}/zip"
    if (
        renderer_name is None
        or type(repository) is not dict
        or not _is_canonical_repository_reference(repository_url)
        or repository.get("full_name") != CANONICAL_REPOSITORY_IDENTITY
        or run.get("id") != publication["workflow_run_id"]
        or run.get("run_attempt") != publication["workflow_run_attempt"]
        or run.get("name") != publication["workflow_name"]
        or run.get("path") != publication["workflow_path"]
        or run.get("html_url") != publication["workflow_run_url"]
        or run.get("created_at") != publication["workflow_run_created_at_utc"]
        or run.get("head_sha") != publication["source_commit"]
        or type(run.get("head_commit")) is not dict
        or run["head_commit"].get("id") != publication["source_commit"]
        or run["head_commit"].get("tree_id") != publication["source_tree"]
        or type(artifact_run) is not dict
        or artifact_run.get("id") != publication["workflow_run_id"]
        or artifact_run.get("head_sha") != publication["source_commit"]
        or type(artifact.get("id")) is not int
        or artifact["id"] <= 0
        or artifact["id"] == publication["artifact_id"]
        or artifact.get("name") != expected_name
        or artifact.get("url") != expected_api_url
        or artifact.get("archive_download_url") != expected_archive_url
        or type(artifact.get("created_at")) is not str
        or type(artifact.get("expires_at")) is not str
        or artifact.get("expired") is not False
    ):
        raise ImmutableArtifactBindingError(
            "evidence_artifact_metadata_mismatch",
            "evidence archive live repository, run, job, head, or artifact identity differs",
        )
    job = _matching_workflow_job(
        jobs,
        job_name=publication["job_name"],
        workflow_run_id=publication["workflow_run_id"],
        workflow_run_attempt=publication["workflow_run_attempt"],
        source_commit=publication["source_commit"],
    )
    if (
        job["id"] != publication["job_database_id"]
        or job.get("url") != publication["job_api_url"]
        or job.get("html_url") != publication["job_html_url"]
    ):
        raise ImmutableArtifactBindingError(
            "evidence_artifact_job_mismatch",
            "evidence archive live workflow job differs from the packet publication authority",
        )
    try:
        run_created = datetime.fromisoformat(
            publication["workflow_run_created_at_utc"].replace("Z", "+00:00")
        )
        artifact_created = datetime.fromisoformat(artifact["created_at"].replace("Z", "+00:00"))
        expires = datetime.fromisoformat(artifact["expires_at"].replace("Z", "+00:00"))
    except ValueError as error:
        raise ImmutableArtifactBindingError(
            "evidence_artifact_timestamp_invalid",
            "evidence archive live timestamps are invalid",
        ) from error
    if (
        run_created.tzinfo is None
        or artifact_created.tzinfo is None
        or expires.tzinfo is None
        or artifact_created < run_created
        or expires - run_created != timedelta(days=90)
        or datetime.now(UTC) >= expires.astimezone(UTC)
    ):
        raise ImmutableArtifactBindingError(
            "evidence_artifact_expired",
            "evidence archive is expired or lacks exact workflow-run retention",
        )


def _load_counterpart_evidence(
    evidence_root: Path,
    definition: BenchmarkDefinition,
    seeds: FinalEvaluationSeedRegistry,
    lock: dict[str, Any],
) -> dict[str, Any]:
    """Bind raw live archives to extracted bytes before admitting a counterpart renderer."""

    bundle = _load_counterpart_bundle(evidence_root)
    packet_metadata = cast(dict[str, Any], bundle["packet_live_artifact.json"])
    evidence_metadata = cast(dict[str, Any], bundle["evidence_live_artifact.json"])
    packet_archive = cast(bytes, bundle["packet_artifact.zip"])
    evidence_archive = cast(bytes, bundle["evidence_artifact.zip"])
    _verify_artifact_archive_payload(packet_archive, packet_metadata, "counterpart packet")
    _verify_artifact_archive_payload(evidence_archive, evidence_metadata, "counterpart evidence")

    with tempfile.TemporaryDirectory(prefix="epsbench-counterpart-evidence-") as temporary:
        packet_root = Path(temporary) / "packet"
        extracted_evidence = Path(temporary) / "evidence"
        _safe_extract_artifact_archive(packet_archive, packet_root, "counterpart packet")
        _safe_extract_artifact_archive(
            evidence_archive,
            extracted_evidence,
            "counterpart evidence",
            expected_members=_EVIDENCE_ARCHIVE_MEMBERS,
        )
        counterpart_packet = validate_freeze_audit(packet_root)
        counterpart_definition, counterpart_seeds, counterpart_lock = _receipt_context(packet_root)
        if (
            counterpart_definition != definition
            or counterpart_seeds != seeds
            or counterpart_lock != lock
        ):
            raise FreezeError("counterpart packet scientific inputs differ")
        receipt = _read_canonical_json(extracted_evidence / "renderer_receipt.json")
        validated_receipt = _validate_receipt_against_packet(
            receipt,
            packet_root,
            counterpart_packet,
            counterpart_definition,
            counterpart_seeds,
            counterpart_lock,
        )
        publication = _read_canonical_json(extracted_evidence / "publication_record.json")
        validated_publication = validate_publication_record(
            publication,
            packet_root,
            counterpart_packet,
            packet_metadata,
            cast(dict[str, Any], bundle["packet_live_workflow_run.json"]),
            cast(dict[str, Any], bundle["packet_live_workflow_jobs.json"]),
        )
        _validate_evidence_artifact_metadata(
            evidence_metadata,
            cast(dict[str, Any], bundle["evidence_live_workflow_run.json"]),
            cast(dict[str, Any], bundle["evidence_live_workflow_jobs.json"]),
            validated_publication,
            counterpart_packet["renderer_environment_id"],
        )
        selected = _read_canonical_json(packet_root / "selected_profile_matrix.json")["cells"]
        local_rows = _local_profile_readiness(selected, counterpart_definition)
        if local_rows != counterpart_packet["local_profile_readiness"]:
            raise FreezeError("counterpart packet readiness reconstruction differs")
        return {
            "environment_id": counterpart_packet["renderer_environment_id"],
            "replacement_definition_lock_commit": counterpart_packet[
                "replacement_definition_lock_commit"
            ],
            "replacement_definition_lock_sha256": counterpart_packet["portable_definition_roots"][
                "freeze_definition_lock_sha256"
            ],
            "profiles": local_rows,
            "portable_definition_roots": counterpart_packet["portable_definition_roots"],
            "portable_apparatus_roots": counterpart_packet["portable_apparatus_roots"],
            "renderer_local_roots": counterpart_packet["renderer_local_roots"],
            "threshold_margin_summary": counterpart_packet["threshold_margin_summary"],
            "selected_outcome_map": _outcome_rows(selected),
            "receipt": validated_receipt,
            "publication_record": validated_publication,
        }


def _reject_aliased_publication_parent(parent: Path) -> tuple[Path, tuple[int, int]]:
    try:
        unresolved = parent.absolute()
        parent_stat = unresolved.lstat()
    except OSError as error:
        raise FreezeError("renderer receipt publication parent must already exist") from error
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if (
        stat.S_ISLNK(parent_stat.st_mode)
        or bool(getattr(parent_stat, "st_file_attributes", 0) & reparse_flag)
        or not stat.S_ISDIR(parent_stat.st_mode)
    ):
        raise FreezeError("renderer receipt publication parent must be a non-alias directory")
    resolved = unresolved.resolve(strict=True)
    cursor = resolved
    while cursor != cursor.parent:
        current = cursor.lstat()
        if stat.S_ISLNK(current.st_mode) or bool(
            getattr(current, "st_file_attributes", 0) & reparse_flag
        ):
            raise FreezeError("renderer receipt publication path contains an alias")
        cursor = cursor.parent
    return resolved, (parent_stat.st_dev, parent_stat.st_ino)


def _atomic_publish_receipt(
    payload: bytes,
    output: Path,
    packet_root: Path,
    *,
    revalidate_packet: Any | None = None,
    boundary_hook: Any | None = None,
) -> None:
    if ".." in output.parts or output.name in {"", ".", ".."}:
        raise FreezeError("renderer receipt publication path contains traversal")
    parent, parent_identity = _reject_aliased_publication_parent(output.parent)
    destination = parent / output.name
    resolved_packet = packet_root.resolve(strict=True)
    if destination.resolve(strict=False).is_relative_to(resolved_packet):
        raise FreezeError("renderer receipt may not be published inside its source packet")

    def verify_packet() -> None:
        if revalidate_packet is not None:
            revalidate_packet()

    try:
        options: dict[str, Any] = {
            "expected_parent_identity": parent_identity,
            "post_publish_check": verify_packet,
        }
        if boundary_hook is not None:
            options["boundary_hook"] = boundary_hook
        atomic_publish_owned_bytes(parent, output.name, payload, **options)
    finally:
        verify_packet()


def create_renderer_receipt(
    packet_root: Path,
    output: Path,
    counterpart_evidence_root: Path | None = None,
) -> dict[str, Any]:
    packet = validate_freeze_audit(packet_root, counterpart_evidence_root)
    definition, seeds, lock = _receipt_context(packet_root)
    selected = _read_canonical_json(packet_root / "selected_profile_matrix.json")["cells"]
    controls = _read_canonical_json(packet_root / "legacy_control_matrix.json")["cells"]
    source_commit = packet["source_provenance"]["git_commit"]
    domain = {
        "schema_version": RENDERER_RECEIPT_VERSION,
        "environment_id": packet["renderer_environment_id"],
        "renderer_fingerprint": packet["renderer_fingerprint"],
        "qualification_source_provenance": packet["source_provenance"],
        "qualification_source_commit": source_commit,
        "qualification_source_tree": _git_tree(source_commit),
        "replacement_definition_lock_commit": packet["replacement_definition_lock_commit"],
        "replacement_definition_lock_sha256": packet["portable_definition_roots"][
            "freeze_definition_lock_sha256"
        ],
        "benchmark_definition_sha256": packet["portable_definition_roots"][
            "benchmark_definition_sha256"
        ],
        "evaluation_episode_seed_registry_sha256": packet["portable_definition_roots"][
            "evaluation_episode_seed_registry_sha256"
        ],
        "complete_packet_root_sha256": packet["complete_packet_root_sha256"],
        "selected_cell_count": packet["selected_cell_count"],
        "control_cell_count": packet["control_cell_count"],
        "total_cell_count": packet["total_cell_count"],
        "selected_matrix_counts": packet["selected_matrix_counts"],
        "control_matrix_counts": packet["control_matrix_counts"],
        "selected_outcome_map": _outcome_rows(selected),
        "control_outcome_map": _outcome_rows(controls),
        "profiles": packet["local_profile_readiness"],
        "portable_definition_roots": packet["portable_definition_roots"],
        "portable_apparatus_roots": packet["portable_apparatus_roots"],
        "renderer_local_roots": packet["renderer_local_roots"],
        "threshold_margin_summary": packet["threshold_margin_summary"],
        "public_qualification_evidence": _public_packet_evidence(packet_root, packet),
        "benchmark_frozen": False,
        "scientific_result": None,
    }
    receipt = {**domain, "receipt_sha256": _domain_hash("renderer_receipt", domain)}
    _validate_receipt_against_packet(receipt, packet_root, packet, definition, seeds, lock)
    encoded = canonical_json_bytes(receipt) + b"\n"
    _atomic_publish_receipt(
        encoded,
        output,
        packet_root,
        revalidate_packet=lambda: validate_freeze_audit(packet_root, counterpart_evidence_root),
    )
    published = _read_canonical_json(output)
    _validate_receipt_against_packet(published, packet_root, packet, definition, seeds, lock)
    return published


def _decode_canonical_object(payload: bytes, role: str) -> dict[str, Any]:
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FreezeError(f"freeze packet JSON is invalid: {role}") from error
    if type(decoded) is not dict or payload != canonical_json_bytes(decoded) + b"\n":
        raise FreezeError(f"freeze packet JSON is not canonical: {role}")
    return decoded


def _claim_run_metadata(artifacts: _PacketArtifactRegistry) -> None:
    with artifacts.claim("run.json", "volatile-run-metadata") as owned:
        try:
            run = json.loads(owned.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FreezeError("freeze run metadata is invalid") from error
        if owned.payload != (json.dumps(run, indent=2, sort_keys=True) + "\n").encode("utf-8"):
            raise FreezeError("freeze run metadata encoding is not exact")
    if type(run) is not dict or set(run) != {
        "generated_at_utc",
        "hostname",
        "operating_system",
        "wall_clock_seconds",
    }:
        raise FreezeError("freeze run metadata schema is not strict")
    try:
        generated_at = datetime.fromisoformat(run["generated_at_utc"])
        elapsed = run["wall_clock_seconds"]
        valid = (
            generated_at.tzinfo is not None
            and all(
                type(run[field]) is str and bool(run[field])
                for field in ("hostname", "operating_system")
            )
            and type(elapsed) in {int, float}
            and math.isfinite(elapsed)
            and elapsed >= 0
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise FreezeError("freeze run metadata values are invalid")


def _validate_contact_sheets(
    packet_root: Path,
    selected: list[dict[str, Any]],
    manifest: dict[str, Any],
    artifacts: _PacketArtifactRegistry,
    frame_cache: dict[tuple[str, str], tuple[Any, Any, Any]],
) -> None:
    if (
        set(manifest) != {"schema_version", "sheets"}
        or manifest["schema_version"] != FREEZE_CONTACT_SHEET_VERSION
    ):
        raise FreezeError("freeze contact-sheet manifest is not strict")
    records = manifest["sheets"]
    if type(records) is not list or len(records) != 10:
        raise FreezeError("freeze contact-sheet manifest is incomplete")
    index = 0
    for profile_id in SELECTED_PROFILE_IDS:
        profile_cells = [cell for cell in selected if cell["profile_id"] == profile_id]
        for scene in SCENE_FAMILIES:
            record = records[index]
            index += 1
            fields = {
                "profile_id",
                "scene_family",
                "seed_index",
                "path",
                "media_type",
                "mode",
                "dimensions",
                "dtype",
                "shape",
                "logical_sha256",
                "file_sha256",
                "byte_count",
            }
            if type(record) is not dict or set(record) != fields:
                raise FreezeError("freeze contact-sheet record is not strict")
            if (
                record["profile_id"] != profile_id
                or record["scene_family"] != scene
                or record["seed_index"] != 0
            ):
                raise FreezeError("freeze contact-sheet identity differs")
            with artifacts.claim(
                record["path"],
                f"contact:{profile_id}:{scene}",
                expected_file_sha256=record["file_sha256"],
                expected_byte_count=record["byte_count"],
            ) as owned:
                with Image.open(BytesIO(owned.payload)) as image:
                    image.load()
                    actual = np.asarray(image, dtype=np.uint8).copy()
            expected = np.asarray(
                _contact_sheet_image(packet_root, profile_cells, scene, frame_cache=frame_cache),
                dtype=np.uint8,
            )
            if (
                record["media_type"] != "image/png"
                or record["mode"] != "RGB"
                or record["dimensions"] != [actual.shape[1], actual.shape[0]]
                or record["dtype"] != str(actual.dtype)
                or record["shape"] != list(actual.shape)
                or record["logical_sha256"] != logical_array_hash(actual)
                or not np.array_equal(actual, expected)
            ):
                raise FreezeError("freeze contact sheet differs from reconstruction")


def validate_freeze_audit(
    packet_root: Path,
    counterpart_evidence_root: Path | None = None,
) -> dict[str, Any]:
    """Independently validate snapshots, cells, pairings, readiness, roots, and tree."""

    artifacts = _PacketArtifactRegistry(packet_root)
    with artifacts.claim("freeze_candidate_packet.json", "freeze-candidate-packet") as owned:
        packet = _decode_canonical_object(owned.payload, "freeze_candidate_packet.json")
    logical_fields = {
        "schema_version",
        "root_schema_version",
        "replacement_definition_lock_commit",
        "source_provenance",
        "renderer_environment_id",
        "renderer_fingerprint",
        "portable_definition_roots",
        "portable_apparatus_roots",
        "portable_profile_readiness_root_sha256",
        "renderer_local_roots",
        "snapshot_file_sha256",
        "report_file_sha256",
        "selected_cell_count",
        "control_cell_count",
        "total_cell_count",
        "selected_matrix_counts",
        "control_matrix_counts",
        "local_profile_readiness",
        "threshold_margin_summary",
        "freeze_candidate_status",
        "seed_set_disposition",
        "owner_approval",
        "closeout_authority",
        "benchmark_frozen",
        "primary_model_result_renderer",
        "model_protocol_frozen",
        "full_gate_0b_complete",
        "gate_0c_authorised",
        "gate_0d_authorised",
        "scientific_result",
    }
    if set(packet) != logical_fields | {"complete_packet_root_sha256"}:
        raise FreezeError("freeze candidate packet schema is not strict")
    if (
        packet["schema_version"] != FREEZE_PACKET_VERSION
        or packet["root_schema_version"] != FREEZE_ROOT_VERSION
        or packet["freeze_candidate_status"]
        not in {"qualification_incomplete", "ready_for_dual_review", "not_ready"}
        or packet["seed_set_disposition"]
        not in {
            "pending_second_renderer_qualification",
            "eligible_for_owner_freeze_if_approved",
            "retired_after_failed_freeze_qualification",
        }
        or any(
            packet[field] is not None
            for field in (
                "owner_approval",
                "closeout_authority",
                "primary_model_result_renderer",
                "scientific_result",
            )
        )
        or any(
            packet[field] is not False
            for field in (
                "benchmark_frozen",
                "model_protocol_frozen",
                "full_gate_0b_complete",
                "gate_0c_authorised",
                "gate_0d_authorised",
            )
        )
    ):
        raise FreezeError("freeze packet exceeds implementation authority")
    if set(packet["portable_definition_roots"]) != set(PORTABLE_DEFINITION_ROOT_FIELDS):
        raise FreezeError("portable definition root schema differs")
    if set(packet["portable_apparatus_roots"]) != set(PORTABLE_APPARATUS_ROOT_FIELDS):
        raise FreezeError("portable apparatus root schema differs")
    if set(packet["renderer_local_roots"]) != set(RENDERER_LOCAL_ROOT_FIELDS):
        raise FreezeError("renderer-local root schema differs")
    snapshot_hashes = packet["snapshot_file_sha256"]
    snapshot_names = set(SNAPSHOT_FILES)
    has_counterpart_receipt = "counterpart_renderer_receipt.json" in snapshot_hashes
    has_counterpart_publication = "counterpart_publication_record.json" in snapshot_hashes
    if has_counterpart_receipt != has_counterpart_publication:
        raise FreezeError("counterpart packet evidence snapshots are incomplete")
    has_counterpart = has_counterpart_receipt
    if has_counterpart:
        snapshot_names.update(
            {
                "counterpart_renderer_receipt.json",
                "counterpart_publication_record.json",
            }
        )
    if set(snapshot_hashes) != snapshot_names:
        raise FreezeError("freeze packet snapshot hash domain differs")
    if set(packet["report_file_sha256"]) != set(REPORT_FILES):
        raise FreezeError("freeze packet report hash domain differs")
    snapshots: dict[str, dict[str, Any]] = {}
    for name in sorted(snapshot_names):
        with artifacts.claim(
            name, f"snapshot:{name}", expected_file_sha256=snapshot_hashes[name]
        ) as owned:
            snapshots[name] = _decode_canonical_object(owned.payload, name)
    reports: dict[str, dict[str, Any]] = {}
    for name in REPORT_FILES:
        with artifacts.claim(
            name,
            f"report:{name}",
            expected_file_sha256=packet["report_file_sha256"][name],
        ) as owned:
            reports[name] = _decode_canonical_object(owned.payload, name)
    try:
        definition = BenchmarkDefinition.model_validate_json(
            canonical_json_bytes(snapshots["benchmark_definition_snapshot.json"])
        )
        revision = parse_appearance_registry(snapshots["revision_registry_snapshot.json"])
        seeds = parse_seed_registry(snapshots["evaluation_episode_seed_registry_snapshot.json"])
        configs = (
            parse_config(snapshots["single_scene_config_snapshot.json"]),
            parse_config(snapshots["corridor_scene_config_snapshot.json"]),
        )
        lock = FreezeDefinitionLock.model_validate_json(
            canonical_json_bytes(snapshots["freeze_definition_lock.json"])
        ).model_dump(mode="json")
    except Exception as error:
        raise FreezeError("freeze packet input snapshot is invalid") from error
    if not isinstance(revision, AppearanceRevision1Registry):
        raise FreezeError("freeze packet Revision 1 snapshot is invalid")
    if not isinstance(seeds, FinalEvaluationSeedRegistry):
        raise FreezeError("freeze packet final-root snapshot is invalid")
    if lock != create_definition_lock_payload(definition, seeds, revision, configs):
        raise FreezeError("freeze packet lock differs from recomputation")
    _validate_lock_commit_snapshots(
        packet["replacement_definition_lock_commit"],
        definition,
        seeds,
        lock,
        packet["source_provenance"],
    )
    if has_counterpart and counterpart_evidence_root is None:
        raise FreezeError("counterpart receipt cannot be used without its resolved public packet")
    if not has_counterpart and counterpart_evidence_root is not None:
        raise FreezeError("counterpart packet was supplied to a single-renderer packet")
    counterpart = (
        _load_counterpart_evidence(counterpart_evidence_root, definition, seeds, lock)
        if counterpart_evidence_root is not None
        else None
    )
    if counterpart is not None and (
        snapshots["counterpart_renderer_receipt.json"] != counterpart["receipt"]
        or snapshots["counterpart_publication_record.json"] != counterpart["publication_record"]
    ):
        raise FreezeError("counterpart packet evidence snapshots were substituted")
    raw_selected = reports["selected_profile_matrix.json"].get("cells")
    raw_controls = reports["legacy_control_matrix.json"].get("cells")
    if type(raw_selected) is not list or type(raw_controls) is not list:
        raise FreezeError("freeze matrix cells must be lists")
    selected = cast(list[dict[str, Any]], raw_selected)
    controls = cast(list[dict[str, Any]], raw_controls)
    if len(selected) != 160 or len(controls) != 32:
        raise FreezeError("freeze matrix cell counts are not exact")
    expected_selected = _selected_membership_domain(definition, seeds)
    expected_controls = _legacy_control_membership_domain(definition, seeds)
    if len({cell.get("cell_id") for cell in [*selected, *controls]}) != 192:
        raise FreezeError("freeze matrix cell identities are not unique")
    roles = {profile.profile_id: profile for profile in definition.profiles}
    for cell, expected in zip(selected, expected_selected, strict=True):
        role = roles[expected["profile_id"]]
        if (
            cell.get("cell_id") != expected["cell_id"]
            or cell.get("profile_id") != expected["profile_id"]
            or cell.get("appearance_profile_sha256") != expected["profile_sha256"]
            or cell.get("benchmark_role") != expected["benchmark_role"]
            or cell.get("scene_family") != expected["scene_family"]
            or cell.get("seed_index") != expected["seed_index"]
            or cell.get("candidate_seed") != expected["evaluation_episode_root"]
            or cell.get("benchmark_reference_cell_id") != expected["benchmark_reference_cell_id"]
            or cell.get("candidate_admission_control_cell_id")
            != expected["candidate_admission_control_cell_id"]
            or cell.get("matched_control_profile_id")
            != role.candidate_admission_matched_control_profile_id
        ):
            raise FreezeError("selected freeze cell identity or pairing differs")
    for cell, expected in zip(controls, expected_controls, strict=True):
        if (
            cell.get("cell_id") != expected["cell_id"]
            or cell.get("profile_id") != LEGACY_CONTROL_PROFILE_ID
            or cell.get("scene_family") != expected["scene_family"]
            or cell.get("seed_index") != expected["seed_index"]
            or cell.get("candidate_seed") != expected["evaluation_episode_root"]
            or cell.get("benchmark_role") != "legacy_apparatus_control"
            or cell.get("candidate_admission_control_cell_id") != cell.get("cell_id")
        ):
            raise FreezeError("legacy control cell identity differs")
    all_cells = [*controls, *selected]
    expected_check_names = set(_pair_checks(all_cells[0], all_cells[0]))
    config_by_scene = {config.scene_family.value: config for config in configs}
    for cell in all_cells:
        if cell["generation_status"] == "success":
            _validate_source_evidence(
                packet_root,
                cell,
                config_by_scene[cell["scene_family"]],
                artifacts,
            )
        _validate_cell_schema(_freeze_cell_without_fields(cell))
        checks = cell.get("benchmark_pair_checks")
        if (
            type(checks) is not dict
            or set(checks) != expected_check_names
            or any(type(value) is not bool for value in checks.values())
        ):
            raise FreezeError("benchmark pair-check schema is not strict")
    by_id = {cell["cell_id"]: cell for cell in all_cells}
    frame_cache: dict[tuple[str, str], tuple[Any, Any, Any]] = {}
    for cell in all_cells:
        if cell["generation_status"] == "success":
            for frame_name in ("before", "after"):
                _load_frame(packet_root, cell, frame_name, artifacts, frame_cache)
    for cell in all_cells:
        profile = profile_by_id(revision, cell["profile_id"])
        if cell["generation_status"] == "success":
            surfaces = (
                ("support_surface", "occluding_surface", "background_surface")
                if cell["scene_family"] == "single_occluder"
                else (
                    "corridor_floor",
                    "corridor_left_surface",
                    "corridor_right_surface",
                    "corridor_end_surface",
                )
            )
            render_plan = resolve_appearance(
                revision,
                profile.profile_id,
                cell["scene_family"],
                surfaces,
                cell["episode_seed"],
                cell["candidate_seed"],
                seeds,
            )
            try:
                nested = AppearanceInstanceRecord.model_validate_json(
                    canonical_json_bytes(cell["appearance_instance"])
                )
            except Exception as error:
                raise FreezeError("freeze appearance instance is invalid") from error
            stored_appearance = {
                "appearance_instance": nested.model_dump(mode="json"),
                "appearance_instance_sha256": cell["appearance_instance_sha256"],
                "appearance_profile_sha256": cell["appearance_profile_sha256"],
                "evaluation_seed_registry_sha256": cell["evaluation_seed_registry_sha256"],
                "source_texture_diagnostics": cell["source_texture_diagnostics"],
            }
            recomputed_appearance = {
                "appearance_instance": render_plan.record.model_dump(mode="json"),
                "appearance_instance_sha256": render_plan.record.appearance_instance_sha256,
                "appearance_profile_sha256": render_plan.record.appearance_profile_sha256,
                "evaluation_seed_registry_sha256": seed_registry_hash(seeds),
                "source_texture_diagnostics": _portable_freeze_source_texture_diagnostics(
                    _source_texture_diagnostics(profile, cell["scene_family"], render_plan.record)
                ),
            }
            if canonical_json_bytes(stored_appearance) != canonical_json_bytes(
                recomputed_appearance
            ):
                difference_path, stored_value, recomputed_value = _first_json_difference(
                    stored_appearance, recomputed_appearance
                )
                raise FreezeError(
                    "freeze appearance instance differs from recomputation for "
                    f"{cell['cell_id']} at {difference_path}: "
                    f"stored={stored_value!r}, recomputed={recomputed_value!r}"
                )
            for frame_index, frame_name in enumerate(("before", "after")):
                arrays = frame_cache[(cell["cell_id"], frame_name)]
                for role_name, array in zip(("rgb", "depth", "segmentation"), arrays, strict=True):
                    if (
                        logical_array_hash(array)
                        != cell[f"{role_name}_logical_sha256"][frame_index]
                    ):
                        raise FreezeError("freeze retained array identity differs")
        stored = _admission_evidence_domain(cell)
        recomputed = _freeze_cell_without_fields(cell)
        recomputed["source_identity"] = cell["source_identity"]
        for field in (
            "frame_metrics",
            "admission_checks",
            "admission_status",
            "rejection_reasons",
            "matched_control_failure_type",
            "matched_control_failure_message",
        ):
            recomputed.pop(field, None)
        control = by_id[cell["candidate_admission_control_cell_id"]]
        control_for_evaluation = _freeze_cell_without_fields(control)
        control_for_evaluation["source_identity"] = control["source_identity"]
        _evaluate_freeze_cell(
            packet_root,
            recomputed,
            control_for_evaluation,
            profile,
            artifact_registry=artifacts,
            frame_cache=frame_cache,
        )
        recomputed_admission = _admission_evidence_domain(recomputed)
        if canonical_json_bytes(stored) != canonical_json_bytes(recomputed_admission):
            difference_path, stored_value, recomputed_value = _first_json_difference(
                stored, recomputed_admission
            )
            raise FreezeError(
                "freeze admission evidence differs from recomputation for "
                f"{cell['cell_id']} at {difference_path}: "
                f"stored={stored_value!r}, recomputed={recomputed_value!r}"
            )
        reference = by_id[cell["benchmark_reference_cell_id"]]
        if cell["benchmark_pair_checks"] != _pair_checks(cell, reference):
            raise FreezeError("freeze benchmark pair evidence differs")
    for profile_id in SELECTED_PROFILE_IDS:
        profile = profile_by_id(revision, profile_id)
        for surfaces in (
            ("support_surface", "occluding_surface", "background_surface"),
            (
                "corridor_floor",
                "corridor_left_surface",
                "corridor_right_surface",
                "corridor_end_surface",
            ),
        ):
            balance = assignment_balance(profile, surfaces, seeds.candidate_episode_seeds, seeds)
            if any(max(row.values()) - min(row.values()) > 1 for row in balance.values()):
                raise FreezeError("freeze appearance assignment is not balanced")
    contact_manifest = reports["contact_sheet_manifest.json"]
    expected_definition, expected_apparatus, expected_renderer = _root_domains(
        definition, seeds, lock, selected, controls, contact_manifest
    )
    if packet["portable_definition_roots"] != expected_definition:
        raise FreezeError("freeze portable definition roots differ")
    if packet["portable_apparatus_roots"] != expected_apparatus:
        raise FreezeError("freeze portable apparatus roots differ")
    if packet["renderer_local_roots"] != expected_renderer:
        raise FreezeError("freeze renderer-local roots differ")
    if (
        packet["selected_cell_count"] != 160
        or packet["control_cell_count"] != 32
        or packet["total_cell_count"] != 192
        or packet["selected_matrix_counts"]
        != dict(Counter(cell["admission_status"] for cell in selected))
        or packet["control_matrix_counts"]
        != dict(Counter(cell["admission_status"] for cell in controls))
    ):
        raise FreezeError("freeze matrix count claim differs")
    renderer = next(
        (
            cell["renderer_provenance"]
            for cell in all_cells
            if cell["generation_status"] == "success"
        ),
        None,
    )
    if renderer is None or any(
        cell["renderer_provenance"] != renderer
        for cell in all_cells
        if cell["generation_status"] == "success"
    ):
        raise FreezeError("freeze renderer evidence differs")
    environment_id = _renderer_environment_id(definition, renderer)
    if (
        packet["renderer_environment_id"] != environment_id
        or packet["renderer_fingerprint"] != renderer
    ):
        raise FreezeError("freeze renderer claim differs")
    if counterpart is not None:
        if (
            counterpart["replacement_definition_lock_commit"]
            != packet["replacement_definition_lock_commit"]
            or counterpart["replacement_definition_lock_sha256"]
            != expected_definition["freeze_definition_lock_sha256"]
            or counterpart["portable_definition_roots"] != expected_definition
        ):
            raise FreezeError("counterpart receipt binding differs")
    local_rows = _local_profile_readiness(selected, definition)
    readiness = _readiness_summary(
        definition, environment_id, local_rows, counterpart, expected_apparatus
    )
    if reports["profile_readiness_summary.json"] != readiness:
        raise FreezeError("freeze profile readiness summary differs")
    if reports["negative_evidence.json"] != _negative_evidence(definition, selected):
        raise FreezeError("freeze negative evidence is incomplete")
    if packet["local_profile_readiness"] != local_rows:
        raise FreezeError("freeze local profile readiness differs")
    portable_readiness = _portable_readiness_domain(readiness)
    expected_readiness_root = (
        _domain_hash("profile_readiness", portable_readiness)
        if portable_readiness is not None
        else None
    )
    if packet["portable_profile_readiness_root_sha256"] != expected_readiness_root:
        raise FreezeError("freeze portable readiness root differs")
    if (
        packet["freeze_candidate_status"] != readiness["freeze_candidate_status"]
        or packet["seed_set_disposition"] != readiness["seed_set_disposition"]
        or packet["threshold_margin_summary"] != _threshold_margin_summary(selected, revision)
    ):
        raise FreezeError("freeze disposition or threshold evidence differs")
    _validate_contact_sheets(packet_root, selected, contact_manifest, artifacts, frame_cache)
    logical = {field: packet[field] for field in logical_fields}
    if _domain_hash("complete_packet", logical) != packet["complete_packet_root_sha256"]:
        raise FreezeError("freeze complete packet root differs")
    _claim_run_metadata(artifacts)
    artifacts.assert_exact_tree()
    return packet
