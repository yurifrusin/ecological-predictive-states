"""Prospectively locked Gate 0B Appearance Candidate Revision 1 audit."""

from __future__ import annotations

import json
import platform
import shutil
import socket
import subprocess
import tempfile
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image

from epsbench.appearance import (
    REVISION1_PROFILE_IDS,
    AppearanceInstanceRecord,
    AppearanceRegistry,
    AppearanceRevision1Registry,
    EvaluationSeedRegistry,
    QualificationSeedRegistry,
    appearance_profile_hash,
    appearance_registry_hash,
    assignment_balance,
    load_appearance_registry_any,
    load_evaluation_seed_registry,
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
    _hash_json,
    _load_frame,
    _PacketArtifactRegistry,
    _portable_analytic_identity_domain,
    _source_texture_diagnostics,
    _validate_cell_schema,
)
from epsbench.config import load_config
from epsbench.data.provenance import collect_source_provenance
from epsbench.failure_analysis import CANONICAL_BASE, CANONICAL_ROOTS
from epsbench.schema import SourceProvenance
from epsbench.utils.canonical import (
    canonical_json_bytes,
    logical_array_hash,
    sha256_file,
    write_canonical_json,
)

REVISION_AUDIT_SCHEMA_VERSION = "appearance_candidate_revision_audit_v0"
REVISION_ROOT_SCHEMA_VERSION = "appearance_candidate_revision_root_domains_v0"
REVISION_FREEZE_STATUS = "candidate_revision_packet_only_not_frozen"
REVISION_CONTACT_SHEET_VERSION = "appearance_revision_contact_sheet_manifest_v0"
PARTITIONS = ("design", "qualification")
SNAPSHOT_FILES = (
    "baseline_failure_analysis.json",
    "candidate_definition_lock.json",
    "baseline_registry_snapshot.json",
    "revision_registry_snapshot.json",
    "design_seed_registry_snapshot.json",
    "qualification_seed_registry_snapshot.json",
)
REPORT_FILES = (
    "control_seed_matrix.json",
    "design_seed_matrix.json",
    "qualification_seed_matrix.json",
    "profile_summary.json",
    "negative_evidence.json",
)


class RevisionAuditError(ValueError):
    """Raised when a Revision 1 definition lock or packet is invalid."""


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if type(payload) is not dict:
        raise RevisionAuditError(f"JSON object required: {path}")
    return payload


def _definition_lock_domain(lock: dict[str, Any]) -> dict[str, Any]:
    domain = dict(lock)
    domain.pop("definition_lock_sha256", None)
    return domain


def validate_definition_lock(
    baseline_registry_path: Path,
    design_seeds_path: Path,
    revision_registry_path: Path,
    qualification_seeds_path: Path,
    definition_lock_path: Path,
    baseline_analysis_path: Path | None = None,
) -> dict[str, Any]:
    """Recompute all prospective definitions and the immutable lock root."""

    baseline = load_appearance_registry_any(baseline_registry_path)
    design = load_evaluation_seed_registry(design_seeds_path)
    revision = load_appearance_registry_any(revision_registry_path)
    qualification = load_seed_registry(qualification_seeds_path)
    if not isinstance(baseline, AppearanceRegistry):
        raise RevisionAuditError("definition lock baseline registry is not canonical v1")
    if not isinstance(design, EvaluationSeedRegistry):
        raise RevisionAuditError("definition lock design seed registry is not canonical")
    if not isinstance(revision, AppearanceRevision1Registry):
        raise RevisionAuditError("definition lock Revision 1 registry is not v2")
    if not isinstance(qualification, QualificationSeedRegistry):
        raise RevisionAuditError("definition lock qualification seed registry is invalid")
    validate_axis_isolation(baseline)
    validate_axis_isolation(revision)
    lock = _read_json(definition_lock_path)
    expected_fields = {
        "schema_version",
        "canonical_base_sha",
        "baseline",
        "revision1_registry_sha256",
        "revision1_profile_ids",
        "revision1_profile_sha256",
        "matched_controls",
        "qualification_seed_registry_sha256",
        "design_seed_registry_version",
        "design_seeds",
        "qualification_seed_registry_version",
        "qualification_seeds",
        "admission_thresholds",
        "prospective_design_targets",
        "profile_rationale",
        "version_matrix",
        "method_versions",
        "qualification_started",
        "final_split",
        "final_evaluation_seeds",
        "benchmark_frozen",
        "full_gate_0b_complete",
        "scientific_result",
        "definition_lock_sha256",
    }
    if set(lock) != expected_fields:
        raise RevisionAuditError("definition-lock schema is not strict")
    if lock["schema_version"] != "appearance_candidate_revision1_definition_lock_v0":
        raise RevisionAuditError("definition-lock schema version is unsupported")
    if _hash_json(_definition_lock_domain(lock)) != lock["definition_lock_sha256"]:
        raise RevisionAuditError("definition-lock logical root mismatch")
    if lock["canonical_base_sha"] != CANONICAL_BASE:
        raise RevisionAuditError("definition lock canonical base is not exact")
    if lock["baseline"]["portable_roots"] != CANONICAL_ROOTS:
        raise RevisionAuditError("definition lock changes protected baseline roots")
    if lock["baseline"]["matrix"] != {
        "successful": 160,
        "admitted": 44,
        "rejected": 116,
        "rejected_profiles": 10,
        "admitted_profile_ids": [],
    }:
        raise RevisionAuditError("definition lock changes the protected 160/44/116 result")
    if lock["baseline"]["appearance_registry_sha256"] != appearance_registry_hash(baseline) or lock[
        "baseline"
    ]["design_seed_registry_sha256"] != seed_registry_hash(design):
        raise RevisionAuditError("definition lock baseline registry binding differs")
    if lock["revision1_registry_sha256"] != appearance_registry_hash(revision):
        raise RevisionAuditError("definition lock Revision 1 registry hash differs")
    if lock["qualification_seed_registry_sha256"] != seed_registry_hash(qualification):
        raise RevisionAuditError("definition lock qualification registry hash differs")
    if lock["revision1_profile_ids"] != list(REVISION1_PROFILE_IDS):
        raise RevisionAuditError("definition lock Revision 1 profile set differs")
    expected_profile_hashes = {
        profile_id: appearance_profile_hash(profile_by_id(revision, profile_id))
        for profile_id in REVISION1_PROFILE_IDS
    }
    expected_controls = {
        profile_id: profile_by_id(revision, profile_id).matched_control_profile_id
        for profile_id in REVISION1_PROFILE_IDS
    }
    if (
        lock["revision1_profile_sha256"] != expected_profile_hashes
        or lock["matched_controls"] != expected_controls
    ):
        raise RevisionAuditError("definition lock profile or control identity differs")
    for profile in baseline.profiles:
        if appearance_profile_hash(profile) != appearance_profile_hash(
            profile_by_id(revision, profile.profile_id)
        ):
            raise RevisionAuditError("Revision 1 registry changes a canonical profile")
    if (
        lock["design_seeds"] != list(design.candidate_episode_seeds)
        or lock["qualification_seeds"] != list(qualification.candidate_episode_seeds)
        or lock["design_seed_registry_version"] != design.registry_version
        or lock["qualification_seed_registry_version"] != qualification.registry_version
    ):
        raise RevisionAuditError("definition lock exact seed binding differs")
    expected_thresholds = profile_by_id(
        revision, REVISION1_PROFILE_IDS[0]
    ).non_degeneracy_thresholds.model_dump(mode="json")
    if lock["admission_thresholds"] != expected_thresholds:
        raise RevisionAuditError("definition lock changes canonical admission thresholds")
    if any(
        profile_by_id(revision, profile_id).non_degeneracy_thresholds.model_dump(mode="json")
        != expected_thresholds
        for profile_id in REVISION1_PROFILE_IDS
    ):
        raise RevisionAuditError("Revision 1 profile thresholds are not exact")
    if (
        lock["qualification_started"] is not False
        or lock["final_split"] is not None
        or lock["final_evaluation_seeds"] is not None
        or lock["benchmark_frozen"] is not False
        or lock["full_gate_0b_complete"] is not False
        or lock["scientific_result"] is not None
    ):
        raise RevisionAuditError("definition lock exceeds prospective authority")
    if baseline_analysis_path is not None:
        analysis = _read_json(baseline_analysis_path)
        if (
            analysis.get("baseline_failure_analysis_sha256")
            != lock["baseline"]["failure_analysis_sha256"]
        ):
            raise RevisionAuditError("definition lock baseline analysis binding differs")
    return lock


def _lock_commit(definition_lock_path: Path) -> str:
    relative = definition_lock_path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    command = [
        "git",
        "log",
        "--diff-filter=A",
        "--format=%H",
        "--reverse",
        "--",
        relative,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    commits = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(commits) != 1:
        raise RevisionAuditError("definition lock must have exactly one additive Git commit")
    commit = commits[0]
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        capture_output=True,
        check=False,
    )
    if ancestry.returncode != 0:
        raise RevisionAuditError("definition-lock commit is not an ancestor of qualification HEAD")
    return commit


def _validate_lock_commit_snapshot(
    commit: str,
    lock: dict[str, Any],
    source_provenance: dict[str, Any],
) -> None:
    if type(commit) is not str or len(commit) != 40:
        raise RevisionAuditError("revision packet definition-lock commit is malformed")
    try:
        committed = subprocess.run(
            [
                "git",
                "show",
                f"{commit}:configs/appearance_candidate_revision1_lock.json",
            ],
            capture_output=True,
            check=True,
        ).stdout
        committed_lock = json.loads(committed.decode("utf-8"))
    except Exception as error:
        raise RevisionAuditError("revision packet definition-lock commit is unavailable") from error
    if canonical_json_bytes(committed_lock) != canonical_json_bytes(lock):
        raise RevisionAuditError("revision packet lock differs from the exact lock commit")
    try:
        provenance = SourceProvenance.model_validate_json(canonical_json_bytes(source_provenance))
    except Exception as error:
        raise RevisionAuditError("revision packet source provenance is invalid") from error
    if provenance.git_commit is None:
        raise RevisionAuditError("revision qualification requires exact Git source provenance")
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, provenance.git_commit],
        capture_output=True,
        check=False,
    )
    if ancestry.returncode != 0:
        raise RevisionAuditError("revision qualification source predates the definition lock")


def _revision_cell_id(partition: str, scene: str, profile_id: str, seed_index: int) -> str:
    return f"{partition}--{scene}--{profile_id}--seed-{seed_index}"


def _cell_without_revision_fields(cell: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(cell))
    if type(payload) is not dict:
        raise RevisionAuditError("revision cell copy is not an object")
    payload.pop("partition", None)
    payload.pop("control_cell_id", None)
    return payload


def _root_domains(
    revision: AppearanceRevision1Registry,
    qualification: QualificationSeedRegistry,
    lock: dict[str, Any],
    analysis: dict[str, Any],
    candidates: list[dict[str, Any]],
    controls: list[dict[str, Any]],
) -> dict[str, str]:
    all_cells = [*controls, *candidates]
    successful = [cell for cell in all_cells if cell["generation_status"] == "success"]
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
        {"cell_id": cell["cell_id"], **_portable_analytic_identity_domain(cell)}
        for cell in successful
    ]
    invariance = [
        {
            "cell_id": cell["cell_id"],
            **_portable_analytic_identity_domain(cell),
            **{
                name: cell["admission_checks"][name]
                for name in (
                    "structural_invariance",
                    "portable_analytic_identity_equality",
                    "ecological_label_equality",
                    "depth_segmentation_invariance",
                    "determinism",
                )
            },
        }
        for cell in candidates
        if cell["generation_status"] == "success"
    ]

    def partition_outcomes(partition: str) -> list[dict[str, Any]]:
        return [
            {
                "cell_id": cell["cell_id"],
                "profile_id": cell["profile_id"],
                "scene_family": cell["scene_family"],
                "seed_index": cell["seed_index"],
                "candidate_seed": cell["candidate_seed"],
                "control_cell_id": cell["control_cell_id"],
                "generation_status": cell["generation_status"],
                "admission_checks": cell["admission_checks"],
                "admission_status": cell["admission_status"],
                "rejection_reasons": cell["rejection_reasons"],
            }
            for cell in candidates
            if cell["partition"] == partition
        ]

    renderer_labels = [
        {
            "cell_id": cell["cell_id"],
            "ecological_label_sha256": cell["ecological_label_sha256"],
        }
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
    return {
        "baseline_failure_analysis_root_sha256": analysis["baseline_failure_analysis_sha256"],
        "definition_lock_root_sha256": lock["definition_lock_sha256"],
        "revision1_registry_sha256": appearance_registry_hash(revision),
        "qualification_seed_registry_sha256": seed_registry_hash(qualification),
        "procedural_asset_root_sha256": _hash_json(procedural),
        "appearance_assignment_root_sha256": _hash_json(assignments),
        "portable_analytic_identity_root_sha256": _hash_json(portable),
        "within_renderer_invariance_outcome_root_sha256": _hash_json(invariance),
        "design_partition_outcome_root_sha256": _hash_json(partition_outcomes("design")),
        "qualification_partition_outcome_root_sha256": _hash_json(
            partition_outcomes("qualification")
        ),
        "renderer_local_ecological_label_root_sha256": _hash_json(renderer_labels),
        "renderer_specific_audit_root_sha256": _hash_json(renderer_evidence),
    }


def _profile_summary(candidates: list[dict[str, Any]], revision: Any) -> dict[str, Any]:
    profiles = []
    for profile_id in REVISION1_PROFILE_IDS:
        profile = profile_by_id(revision, profile_id)
        partitions: dict[str, Any] = {}
        for partition in PARTITIONS:
            selected = [
                cell
                for cell in candidates
                if cell["partition"] == partition and cell["profile_id"] == profile_id
            ]
            partitions[partition] = {
                "cell_count": len(selected),
                "cell_counts": dict(Counter(cell["admission_status"] for cell in selected)),
                "set_admitted": len(selected) == 16
                and all(cell["admission_status"] == "admitted" for cell in selected),
            }
        profiles.append(
            {
                "profile_id": profile_id,
                "profile_sha256": appearance_profile_hash(profile),
                "matched_control_profile_id": profile.matched_control_profile_id,
                "axis_tags": [tag.value for tag in profile.axis_tags],
                "design": partitions["design"],
                "qualification": partitions["qualification"],
                "revision1_profile_admitted": (
                    partitions["design"]["set_admitted"]
                    and partitions["qualification"]["set_admitted"]
                ),
            }
        )
    return {"schema_version": REVISION_AUDIT_SCHEMA_VERSION, "profiles": profiles}


def _contact_sheets(packet_root: Path, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    directory = packet_root / "representative_contact_sheets"
    directory.mkdir()
    records = []
    for partition in PARTITIONS:
        selected = [cell for cell in candidates if cell["partition"] == partition]
        for scene in SCENE_FAMILIES:
            sheet = _contact_sheet_image(packet_root, selected, scene)
            relative = f"representative_contact_sheets/{partition}_{scene}_seed_0.png"
            path = packet_root / relative
            sheet.save(path, format="PNG")
            pixels = np.asarray(sheet, dtype=np.uint8)
            records.append(
                {
                    "partition": partition,
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
    return {"schema_version": REVISION_CONTACT_SHEET_VERSION, "sheets": records}


def _negative_evidence(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": REVISION_AUDIT_SCHEMA_VERSION,
        "retention_rule": "all_revision1_rejected_cells_and_failures_retained_v0",
        "rejected_cells": [
            {
                "cell_id": cell["cell_id"],
                "partition": cell["partition"],
                "profile_id": cell["profile_id"],
                "scene_family": cell["scene_family"],
                "seed_index": cell["seed_index"],
                "candidate_seed": cell["candidate_seed"],
                "control_cell_id": cell["control_cell_id"],
                "rejection_reasons": cell["rejection_reasons"],
                "failure_type": cell.get("failure_type"),
                "failure_message": cell.get("failure_message"),
                "matched_control_failure_type": cell.get("matched_control_failure_type"),
                "matched_control_failure_message": cell.get("matched_control_failure_message"),
            }
            for cell in candidates
            if cell["admission_status"] == "rejected"
        ],
    }


def create_revision_audit(
    baseline_registry_path: Path,
    design_seeds_path: Path,
    revision_registry_path: Path,
    qualification_seeds_path: Path,
    definition_lock_path: Path,
    single_config_path: Path,
    corridor_config_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Generate all 224 locked Revision 1 cells and publish one complete packet."""

    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError(f"revision audit output is not empty: {output}")
        output.rmdir()
    analysis_path = (
        definition_lock_path.parent / "appearance_revision1_baseline_failure_analysis.json"
    )
    lock = validate_definition_lock(
        baseline_registry_path,
        design_seeds_path,
        revision_registry_path,
        qualification_seeds_path,
        definition_lock_path,
        analysis_path,
    )
    lock_commit = _lock_commit(definition_lock_path)
    baseline = load_appearance_registry_any(baseline_registry_path)
    design = load_evaluation_seed_registry(design_seeds_path)
    revision = load_appearance_registry_any(revision_registry_path)
    qualification = load_seed_registry(qualification_seeds_path)
    if not isinstance(baseline, AppearanceRegistry):
        raise RevisionAuditError("baseline registry type changed after lock validation")
    if not isinstance(design, EvaluationSeedRegistry):
        raise RevisionAuditError("design registry type changed after lock validation")
    if not isinstance(revision, AppearanceRevision1Registry):
        raise RevisionAuditError("revision registry type changed after lock validation")
    if not isinstance(qualification, QualificationSeedRegistry):
        raise RevisionAuditError("qualification registry type changed after lock validation")
    analysis = _read_json(analysis_path)
    configs = (load_config(single_config_path), load_config(corridor_config_path))
    if tuple(config.scene_family.value for config in configs) != SCENE_FAMILIES:
        raise RevisionAuditError("revision configurations must be single_occluder then corridor")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    started = time.monotonic()
    try:
        shutil.copy2(analysis_path, staging / "baseline_failure_analysis.json")
        shutil.copy2(definition_lock_path, staging / "candidate_definition_lock.json")
        write_canonical_json(staging / "baseline_registry_snapshot.json", baseline)
        write_canonical_json(staging / "revision_registry_snapshot.json", revision)
        write_canonical_json(staging / "design_seed_registry_snapshot.json", design)
        write_canonical_json(staging / "qualification_seed_registry_snapshot.json", qualification)
        work = staging / "_temporary_datasets"
        work.mkdir()
        candidates: list[dict[str, Any]] = []
        controls: list[dict[str, Any]] = []
        all_by_key: dict[tuple[str, str, str, int], dict[str, Any]] = {}
        for partition, seeds in (("design", design), ("qualification", qualification)):
            for config in configs:
                scene = config.scene_family.value
                profile_ids = ("legacy_solid_base_v1", *REVISION1_PROFILE_IDS)
                for profile_id in profile_ids:
                    profile = profile_by_id(revision, profile_id)
                    for seed_index, candidate_seed in zip(
                        seeds.indices, seeds.candidate_episode_seeds, strict=True
                    ):
                        cell_id = _revision_cell_id(partition, scene, profile_id, seed_index)
                        cell = _dataset_cell(
                            work / cell_id,
                            work / f"{cell_id}--repeat",
                            staging,
                            config,
                            profile,
                            seed_index,
                            candidate_seed,
                            revision,
                            seeds,
                            cell_id_prefix=partition,
                        )
                        cell["partition"] = partition
                        cell["control_cell_id"] = _revision_cell_id(
                            partition,
                            scene,
                            profile.matched_control_profile_id,
                            seed_index,
                        )
                        all_by_key[(partition, scene, profile_id, seed_index)] = cell
                        if profile_id == "legacy_solid_base_v1":
                            controls.append(cell)
                        else:
                            candidates.append(cell)
        for cell in controls:
            profile = profile_by_id(revision, cell["profile_id"])
            _evaluate_cell(staging, cell, cell, profile)
        for cell in candidates:
            profile = profile_by_id(revision, cell["profile_id"])
            control = all_by_key[
                (
                    cell["partition"],
                    cell["scene_family"],
                    profile.matched_control_profile_id,
                    cell["seed_index"],
                )
            ]
            _evaluate_cell(staging, cell, control, profile)
        shutil.rmtree(work)
        design_cells = [cell for cell in candidates if cell["partition"] == "design"]
        qualification_cells = [cell for cell in candidates if cell["partition"] == "qualification"]
        profile_summary = _profile_summary(candidates, revision)
        negative = _negative_evidence(candidates)
        reports = {
            "control_seed_matrix.json": {
                "schema_version": REVISION_AUDIT_SCHEMA_VERSION,
                "cells": controls,
            },
            "design_seed_matrix.json": {
                "schema_version": REVISION_AUDIT_SCHEMA_VERSION,
                "partition": "design",
                "cells": design_cells,
            },
            "qualification_seed_matrix.json": {
                "schema_version": REVISION_AUDIT_SCHEMA_VERSION,
                "partition": "qualification",
                "cells": qualification_cells,
            },
            "profile_summary.json": profile_summary,
            "negative_evidence.json": negative,
        }
        for name, report in reports.items():
            write_canonical_json(staging / name, report)
        contact_manifest = _contact_sheets(staging, candidates)
        roots = _root_domains(revision, qualification, lock, analysis, candidates, controls)
        logical_domain = {
            "schema_version": REVISION_AUDIT_SCHEMA_VERSION,
            "root_schema_version": REVISION_ROOT_SCHEMA_VERSION,
            "freeze_status": REVISION_FREEZE_STATUS,
            "candidate_definition_lock_commit": lock_commit,
            "source_provenance": collect_source_provenance(Path.cwd()).model_dump(mode="json"),
            "roots": roots,
            "snapshot_file_sha256": {name: sha256_file(staging / name) for name in SNAPSHOT_FILES},
            "report_file_sha256": {name: sha256_file(staging / name) for name in REPORT_FILES},
            "contact_sheet_manifest": contact_manifest,
            "matrix_counts": {
                "design": dict(Counter(cell["admission_status"] for cell in design_cells)),
                "qualification": dict(
                    Counter(cell["admission_status"] for cell in qualification_cells)
                ),
            },
            "revision1_profile_ids": list(REVISION1_PROFILE_IDS),
            "design_matrix_cell_count": len(design_cells),
            "qualification_matrix_cell_count": len(qualification_cells),
            "control_cell_count": len(controls),
            "scene_families": list(SCENE_FAMILIES),
            "final_split": None,
            "final_evaluation_seeds": None,
            "full_gate_0b_complete": False,
            "benchmark_frozen": False,
        }
        packet = {
            **logical_domain,
            "complete_packet_root_sha256": _hash_json(logical_domain),
        }
        write_canonical_json(staging / "revision1_candidate_packet.json", packet)
        (staging / "run.json").write_text(
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
            + "\n",
            encoding="utf-8",
        )
        validate_revision_audit(staging)
        _atomic_no_replace_directory(staging, output)
        return packet
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _validate_contact_sheets(
    packet_root: Path,
    candidates: list[dict[str, Any]],
    manifest: dict[str, Any],
    artifacts: _PacketArtifactRegistry,
    frame_cache: dict[tuple[str, str], tuple[Any, Any, Any]],
) -> None:
    if (
        set(manifest) != {"schema_version", "sheets"}
        or manifest["schema_version"] != REVISION_CONTACT_SHEET_VERSION
    ):
        raise RevisionAuditError("revision contact-sheet manifest is not strict")
    records = manifest["sheets"]
    if type(records) is not list or len(records) != 4:
        raise RevisionAuditError("revision contact-sheet manifest is incomplete")
    index = 0
    for partition in PARTITIONS:
        selected = [cell for cell in candidates if cell["partition"] == partition]
        for scene in SCENE_FAMILIES:
            record = records[index]
            index += 1
            expected_fields = {
                "partition",
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
            if type(record) is not dict or set(record) != expected_fields:
                raise RevisionAuditError("revision contact-sheet record is not strict")
            if (
                record["partition"] != partition
                or record["scene_family"] != scene
                or record["seed_index"] != 0
            ):
                raise RevisionAuditError("revision contact-sheet identity differs")
            with artifacts.claim(
                record["path"],
                f"contact:{partition}:{scene}",
                expected_file_sha256=record["file_sha256"],
                expected_byte_count=record["byte_count"],
            ) as owned:
                with Image.open(owned.path) as image:
                    actual = np.asarray(image, dtype=np.uint8).copy()
            expected = np.asarray(
                _contact_sheet_image(
                    packet_root,
                    selected,
                    scene,
                    frame_cache=frame_cache,
                ),
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
                raise RevisionAuditError("revision contact sheet differs from reconstruction")


def validate_revision_audit(packet_root: Path) -> dict[str, Any]:
    """Independently validate cell identities, evidence, admissions, roots, and claims."""

    artifacts = _PacketArtifactRegistry(packet_root)
    with artifacts.claim("revision1_candidate_packet.json", "revision-candidate-packet") as owned:
        packet = json.loads(owned.payload.decode("utf-8"))
    logical_fields = {
        "schema_version",
        "root_schema_version",
        "freeze_status",
        "candidate_definition_lock_commit",
        "source_provenance",
        "roots",
        "snapshot_file_sha256",
        "report_file_sha256",
        "contact_sheet_manifest",
        "matrix_counts",
        "revision1_profile_ids",
        "design_matrix_cell_count",
        "qualification_matrix_cell_count",
        "control_cell_count",
        "scene_families",
        "final_split",
        "final_evaluation_seeds",
        "full_gate_0b_complete",
        "benchmark_frozen",
    }
    if type(packet) is not dict or set(packet) != logical_fields | {"complete_packet_root_sha256"}:
        raise RevisionAuditError("revision packet schema is not strict")
    if (
        packet["schema_version"] != REVISION_AUDIT_SCHEMA_VERSION
        or packet["root_schema_version"] != REVISION_ROOT_SCHEMA_VERSION
        or packet["freeze_status"] != REVISION_FREEZE_STATUS
        or packet["final_split"] is not None
        or packet["final_evaluation_seeds"] is not None
        or packet["full_gate_0b_complete"] is not False
        or packet["benchmark_frozen"] is not False
    ):
        raise RevisionAuditError("revision packet makes an unauthorised status claim")
    snapshot_hashes = packet["snapshot_file_sha256"]
    report_hashes = packet["report_file_sha256"]
    if set(snapshot_hashes) != set(SNAPSHOT_FILES) or set(report_hashes) != set(REPORT_FILES):
        raise RevisionAuditError("revision packet file-hash domain is not exact")
    snapshots: dict[str, dict[str, Any]] = {}
    for name in SNAPSHOT_FILES:
        with artifacts.claim(
            name, f"snapshot:{name}", expected_file_sha256=snapshot_hashes[name]
        ) as owned:
            snapshots[name] = json.loads(owned.payload.decode("utf-8"))
    reports: dict[str, dict[str, Any]] = {}
    for name in REPORT_FILES:
        with artifacts.claim(
            name, f"report:{name}", expected_file_sha256=report_hashes[name]
        ) as owned:
            reports[name] = json.loads(owned.payload.decode("utf-8"))
    baseline = parse_appearance_registry(snapshots["baseline_registry_snapshot.json"])
    revision = parse_appearance_registry(snapshots["revision_registry_snapshot.json"])
    design = parse_seed_registry(snapshots["design_seed_registry_snapshot.json"])
    qualification = parse_seed_registry(snapshots["qualification_seed_registry_snapshot.json"])
    lock = snapshots["candidate_definition_lock.json"]
    analysis = snapshots["baseline_failure_analysis.json"]
    if not isinstance(baseline, AppearanceRegistry):
        raise RevisionAuditError("revision packet baseline snapshot type is invalid")
    if not isinstance(revision, AppearanceRevision1Registry):
        raise RevisionAuditError("revision packet registry snapshot type is invalid")
    if not isinstance(design, EvaluationSeedRegistry):
        raise RevisionAuditError("revision packet design snapshot type is invalid")
    if not isinstance(qualification, QualificationSeedRegistry):
        raise RevisionAuditError("revision packet qualification snapshot type is invalid")
    validate_axis_isolation(revision)
    if _hash_json(_definition_lock_domain(lock)) != lock.get("definition_lock_sha256"):
        raise RevisionAuditError("revision packet definition-lock snapshot is invalid")
    if (
        analysis.get("baseline_failure_analysis_sha256")
        != lock["baseline"]["failure_analysis_sha256"]
    ):
        raise RevisionAuditError("revision packet baseline analysis snapshot differs")
    _validate_lock_commit_snapshot(
        packet["candidate_definition_lock_commit"],
        lock,
        packet["source_provenance"],
    )
    if (
        appearance_registry_hash(revision) != lock["revision1_registry_sha256"]
        or seed_registry_hash(qualification) != lock["qualification_seed_registry_sha256"]
    ):
        raise RevisionAuditError("revision packet locked definition identity differs")

    raw_controls = reports["control_seed_matrix.json"].get("cells")
    raw_design_cells = reports["design_seed_matrix.json"].get("cells")
    raw_qualification_cells = reports["qualification_seed_matrix.json"].get("cells")
    if not all(
        type(value) is list for value in (raw_controls, raw_design_cells, raw_qualification_cells)
    ):
        raise RevisionAuditError("revision packet matrix cells must be lists")
    controls = cast(list[dict[str, Any]], raw_controls)
    design_cells = cast(list[dict[str, Any]], raw_design_cells)
    qualification_cells = cast(list[dict[str, Any]], raw_qualification_cells)
    if len(controls) != 32 or len(design_cells) != 112 or len(qualification_cells) != 112:
        raise RevisionAuditError("revision packet matrix has incorrect cell counts")
    candidates = [*design_cells, *qualification_cells]
    if (
        packet["design_matrix_cell_count"] != 112
        or packet["qualification_matrix_cell_count"] != 112
        or packet["control_cell_count"] != 32
    ):
        raise RevisionAuditError("revision packet cell-count claim differs")
    if packet["revision1_profile_ids"] != list(REVISION1_PROFILE_IDS):
        raise RevisionAuditError("revision packet profile membership differs")

    expected_candidates: list[tuple[str, str, str, int, int]] = []
    expected_controls: list[tuple[str, str, str, int, int]] = []
    for partition, seeds in (("design", design), ("qualification", qualification)):
        for scene in SCENE_FAMILIES:
            for seed_index, candidate_seed in zip(
                seeds.indices, seeds.candidate_episode_seeds, strict=True
            ):
                expected_controls.append(
                    (partition, scene, "legacy_solid_base_v1", seed_index, candidate_seed)
                )
            for profile_id in REVISION1_PROFILE_IDS:
                for seed_index, candidate_seed in zip(
                    seeds.indices, seeds.candidate_episode_seeds, strict=True
                ):
                    expected_candidates.append(
                        (partition, scene, profile_id, seed_index, candidate_seed)
                    )
    ordered_candidates = [*design_cells, *qualification_cells]
    for cell, expected in zip(controls, expected_controls, strict=True):
        partition, scene, profile_id, seed_index, candidate_seed = expected
        if (
            cell.get("partition") != partition
            or cell.get("cell_id") != _revision_cell_id(partition, scene, profile_id, seed_index)
            or cell.get("profile_id") != profile_id
            or cell.get("scene_family") != scene
            or cell.get("seed_index") != seed_index
            or cell.get("candidate_seed") != candidate_seed
            or cell.get("control_cell_id") != cell.get("cell_id")
        ):
            raise RevisionAuditError("revision control-cell identity differs")
    for cell, expected in zip(ordered_candidates, expected_candidates, strict=True):
        partition, scene, profile_id, seed_index, candidate_seed = expected
        profile = profile_by_id(revision, profile_id)
        if (
            cell.get("partition") != partition
            or cell.get("cell_id") != _revision_cell_id(partition, scene, profile_id, seed_index)
            or cell.get("profile_id") != profile_id
            or cell.get("scene_family") != scene
            or cell.get("seed_index") != seed_index
            or cell.get("candidate_seed") != candidate_seed
            or cell.get("control_cell_id")
            != _revision_cell_id(partition, scene, profile.matched_control_profile_id, seed_index)
        ):
            raise RevisionAuditError("revision candidate-cell identity differs")
    all_cells = [*controls, *ordered_candidates]
    for cell in all_cells:
        _validate_cell_schema(_cell_without_revision_fields(cell))
    by_id = {cell["cell_id"]: cell for cell in all_cells}
    if len(by_id) != len(all_cells):
        raise RevisionAuditError("revision packet cell identities are not unique")
    frame_cache: dict[tuple[str, str], tuple[Any, Any, Any]] = {}
    for cell in all_cells:
        if cell["generation_status"] == "success":
            for frame_name in ("before", "after"):
                _load_frame(packet_root, cell, frame_name, artifacts, frame_cache)
    for cell in all_cells:
        seeds = design if cell["partition"] == "design" else qualification
        profile = profile_by_id(revision, cell["profile_id"])
        if cell["generation_status"] == "success":
            surface_names = (
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
                surface_names,
                cell["episode_seed"],
                cell["candidate_seed"],
                seeds,
            )
            try:
                nested = AppearanceInstanceRecord.model_validate_json(
                    canonical_json_bytes(cell["appearance_instance"])
                )
            except Exception as error:
                raise RevisionAuditError("revision appearance instance is invalid") from error
            if (
                nested != render_plan.record
                or cell["appearance_instance_sha256"]
                != render_plan.record.appearance_instance_sha256
                or cell["appearance_profile_sha256"] != render_plan.record.appearance_profile_sha256
                or cell["evaluation_seed_registry_sha256"] != seed_registry_hash(seeds)
                or cell["source_texture_diagnostics"]
                != _source_texture_diagnostics(profile, cell["scene_family"], render_plan.record)
            ):
                raise RevisionAuditError("revision appearance instance differs from recomputation")
            for frame_index, frame_name in enumerate(("before", "after")):
                arrays = frame_cache[(cell["cell_id"], frame_name)]
                for role, array in zip(("rgb", "depth", "segmentation"), arrays, strict=True):
                    declared = cell[f"{role}_logical_sha256"]
                    if (
                        type(declared) is not list
                        or len(declared) != 2
                        or logical_array_hash(array) != declared[frame_index]
                    ):
                        raise RevisionAuditError(
                            "revision retained evidence differs from declared logical identity"
                        )
        stored = _admission_evidence_domain(cell)
        recomputed = _cell_without_revision_fields(cell)
        for field in (
            "frame_metrics",
            "admission_checks",
            "admission_status",
            "rejection_reasons",
            "matched_control_failure_type",
            "matched_control_failure_message",
        ):
            recomputed.pop(field, None)
        control = by_id[cell["control_cell_id"]]
        _evaluate_cell(
            packet_root,
            recomputed,
            _cell_without_revision_fields(control),
            profile,
            frame_cache=frame_cache,
        )
        if canonical_json_bytes(stored) != canonical_json_bytes(
            _admission_evidence_domain(recomputed)
        ):
            raise RevisionAuditError("revision admission evidence differs from recomputation")
    for seeds in (design, qualification):
        for profile_id in REVISION1_PROFILE_IDS:
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
                counts = assignment_balance(profile, surfaces, seeds.candidate_episode_seeds, seeds)
                if any(max(row.values()) - min(row.values()) > 1 for row in counts.values()):
                    raise RevisionAuditError("revision assignment schedule is unbalanced")
    expected_summary = _profile_summary(candidates, revision)
    expected_negative = _negative_evidence(candidates)
    if reports["profile_summary.json"] != expected_summary:
        raise RevisionAuditError("revision profile summary differs from matrices")
    if reports["negative_evidence.json"] != expected_negative:
        raise RevisionAuditError("revision negative evidence is incomplete")
    expected_counts = {
        "design": dict(Counter(cell["admission_status"] for cell in design_cells)),
        "qualification": dict(Counter(cell["admission_status"] for cell in qualification_cells)),
    }
    if packet["matrix_counts"] != expected_counts:
        raise RevisionAuditError("revision packet matrix count claim differs")
    expected_roots = _root_domains(revision, qualification, lock, analysis, candidates, controls)
    if packet["roots"] != expected_roots:
        raise RevisionAuditError("revision packet root differs from recomputation")
    _validate_contact_sheets(
        packet_root,
        candidates,
        packet["contact_sheet_manifest"],
        artifacts,
        frame_cache,
    )
    logical = {field: packet[field] for field in logical_fields}
    if _hash_json(logical) != packet["complete_packet_root_sha256"]:
        raise RevisionAuditError("revision complete packet root mismatch")
    actual_files = {
        path.relative_to(packet_root).as_posix()
        for path in packet_root.rglob("*")
        if path.is_file()
    }
    if actual_files != artifacts.paths | {"run.json"}:
        raise RevisionAuditError("revision packet contains missing or additional files")
    return packet
