"""Reproducible Gate 0B Slice 5 appearance-candidate audit packets."""

from __future__ import annotations

import ctypes
import errno
import json
import os
import platform
import shutil
import socket
import stat
import tempfile
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from epsbench.appearance import (
    AppearanceProfile,
    AppearanceRegistry,
    CandidateClass,
    EvaluationSeedRegistry,
    TextureFamily,
    appearance_profile_hash,
    appearance_registry_hash,
    assignment_balance,
    generate_texture,
    load_appearance_registry,
    load_evaluation_seed_registry,
    resolve_appearance,
    seed_registry_hash,
    validate_axis_isolation,
)
from epsbench.config import BenchmarkConfig, load_config
from epsbench.data.generate import generate_dataset
from epsbench.data.paths import UnsafeOwnedFileError, resolve_owned_regular_file
from epsbench.data.provenance import collect_source_provenance
from epsbench.data.validate import validate_dataset
from epsbench.schema import (
    DatasetManifest,
    TransitionRecord,
    parse_privileged_instrumentation_json,
)
from epsbench.utils.canonical import (
    canonical_json_bytes,
    logical_array_hash,
    sha256_bytes,
    sha256_file,
    write_canonical_json,
)
from epsbench.utils.seeding import derive_seed

AUDIT_SCHEMA_VERSION = "appearance_candidate_audit_v2"
ROOT_SCHEMA_VERSION = "appearance_candidate_root_domains_v2"
PACKET_FREEZE_STATUS = "candidate_packet_only_not_frozen"
CONTACT_SHEET_MANIFEST_VERSION = "appearance_contact_sheet_manifest_v1"
SCENE_FAMILIES = ("single_occluder", "corridor")
GOVERNING_DOCUMENTS = (
    "docs/RESEARCH_CHARTER.md",
    "docs/EPS_BENCH_V0.md",
    "docs/MILESTONE_0.md",
    "docs/review-protocol.md",
)


class AppearanceAuditError(ValueError):
    """Raised when an audit packet or publication operation is invalid."""


PACKET_LOGICAL_FIELDS = (
    "schema_version",
    "root_schema_version",
    "freeze_status",
    "source_provenance",
    "governing_document_hashes",
    "roots",
    "report_file_sha256",
    "contact_sheet_manifest",
    "matrix_counts",
    "profile_count",
    "candidate_seed_count",
    "scene_families",
    "final_split",
    "final_evaluation_seeds",
)


class _PacketArtifactRegistry:
    """Require every declared packet artifact to be one uniquely owned regular file."""

    def __init__(self, root: Path) -> None:
        try:
            root_stat = root.lstat()
            self.root = root.resolve(strict=True)
        except OSError as error:
            raise AppearanceAuditError("candidate packet root does not exist") from error
        if root.is_symlink() or not stat.S_ISDIR(root_stat.st_mode):
            raise AppearanceAuditError("candidate packet root must be a non-link directory")
        self.roles: set[str] = set()
        self.paths: set[str] = set()
        self.resolved_paths: set[Path] = set()
        self.file_identities: set[tuple[int, int]] = set()
        self.by_path: dict[str, Path] = {}

    def claim(
        self,
        relative_path: str,
        role: str,
        *,
        expected_file_sha256: str | None = None,
        expected_byte_count: int | None = None,
    ) -> Path:
        if role in self.roles:
            raise AppearanceAuditError(f"duplicate packet artifact role: {role}")
        if relative_path in self.paths:
            raise AppearanceAuditError(f"duplicate packet artifact path: {relative_path}")
        try:
            owned = resolve_owned_regular_file(self.root, relative_path)
        except UnsafeOwnedFileError as error:
            raise AppearanceAuditError(str(error)) from error
        identity = (owned.device, owned.inode)
        if owned.path in self.resolved_paths or identity in self.file_identities:
            raise AppearanceAuditError(f"packet artifact aliases another role: {relative_path}")
        if expected_byte_count is not None and owned.byte_count != expected_byte_count:
            raise AppearanceAuditError(f"packet artifact byte count mismatch: {relative_path}")
        if expected_file_sha256 is not None and sha256_file(owned.path) != expected_file_sha256:
            raise AppearanceAuditError(f"packet artifact hash mismatch: {relative_path}")
        self.roles.add(role)
        self.paths.add(relative_path)
        self.resolved_paths.add(owned.path)
        self.file_identities.add(identity)
        self.by_path[relative_path] = owned.path
        return owned.path

    def path_for(self, relative_path: str) -> Path:
        try:
            return self.by_path[relative_path]
        except KeyError as error:
            raise AppearanceAuditError(
                f"packet artifact was not independently claimed: {relative_path}"
            ) from error


def _packet_logical_domain(packet: dict[str, Any]) -> dict[str, Any]:
    try:
        return {field: packet[field] for field in PACKET_LOGICAL_FIELDS}
    except KeyError as error:
        raise AppearanceAuditError(f"candidate packet field is missing: {error.args[0]}") from error


def _atomic_no_replace_directory(source: Path, destination: Path) -> None:
    """Atomically publish one directory without replacing a racing target."""

    if os.name == "nt":
        os.rename(source, destination)
        return
    if platform.system() != "Linux":
        raise AppearanceAuditError(
            "atomic no-replace directory publication is supported only on locked Windows/Linux"
        )
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise AppearanceAuditError("Linux renameat2 is unavailable; refusing unsafe publication")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), destination)
    raise OSError(error_number, os.strerror(error_number), destination)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AppearanceAuditError(f"JSON object required: {path}")
    return payload


def _hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _cell_id(scene: str, profile_id: str, seed_index: int) -> str:
    return f"{scene}--{profile_id}--seed-{seed_index}"


def _profile_config(
    config: BenchmarkConfig, profile: AppearanceProfile, candidate_seed: int
) -> BenchmarkConfig:
    return type(config).model_validate(
        {
            **config.model_dump(mode="python"),
            "seed": candidate_seed,
            "appearance": {
                "registry_version": "appearance_candidate_registry_v1",
                "profile_id": profile.profile_id,
            },
        }
    )


def _copy_evidence(
    dataset: Path,
    manifest: DatasetManifest,
    destination: Path,
) -> dict[str, Any]:
    transition = TransitionRecord.model_validate_json(
        (dataset / manifest.episodes[0].transition.path).read_bytes()
    )
    destination.mkdir(parents=True)
    evidence: dict[str, Any] = {}
    for frame_name, frame in (("before", transition.before), ("after", transition.after)):
        frame_evidence: dict[str, Any] = {}
        for role, artifact in (
            ("rgb", frame.rgb),
            ("depth", frame.depth),
            ("segmentation", frame.segmentation),
        ):
            suffix = ".png" if role == "rgb" else ".npy"
            target = destination / f"{role}_{frame_name}{suffix}"
            shutil.copy2(dataset / artifact.path, target)
            frame_evidence[role] = {
                "path": target.as_posix(),
                "media_type": artifact.media_type,
                "dtype": artifact.dtype,
                "shape": list(artifact.shape),
                "file_sha256": sha256_file(target),
                "logical_sha256": artifact.logical_sha256,
                "byte_count": target.stat().st_size,
            }
        evidence[frame_name] = frame_evidence
    return evidence


def _evidence_record(cell: dict[str, Any], frame: str, role: str) -> dict[str, Any]:
    try:
        record = cell["evidence"][frame][role]
    except (KeyError, TypeError) as error:
        raise AppearanceAuditError(
            f"audit evidence declaration is incomplete: {cell.get('cell_id')}:{frame}:{role}"
        ) from error
    expected_fields = {
        "path",
        "media_type",
        "dtype",
        "shape",
        "file_sha256",
        "logical_sha256",
        "byte_count",
    }
    if not isinstance(record, dict) or set(record) != expected_fields:
        raise AppearanceAuditError(
            f"audit evidence declaration is not strict: {cell.get('cell_id')}:{frame}:{role}"
        )
    return record


def _claim_cell_evidence(
    artifact_registry: _PacketArtifactRegistry,
    cell: dict[str, Any],
) -> None:
    evidence = cell.get("evidence")
    if not isinstance(evidence, dict) or set(evidence) != {"before", "after"}:
        raise AppearanceAuditError(f"audit evidence frames are not strict: {cell.get('cell_id')}")
    if any(
        not isinstance(evidence[frame], dict)
        or set(evidence[frame]) != {"rgb", "depth", "segmentation"}
        for frame in ("before", "after")
    ):
        raise AppearanceAuditError(f"audit evidence roles are not strict: {cell.get('cell_id')}")
    for frame in ("before", "after"):
        for role in ("rgb", "depth", "segmentation"):
            record = _evidence_record(cell, frame, role)
            path = record["path"]
            file_sha256 = record["file_sha256"]
            byte_count = record["byte_count"]
            if (
                not isinstance(path, str)
                or not isinstance(file_sha256, str)
                or not isinstance(byte_count, int)
            ):
                raise AppearanceAuditError("audit evidence path/hash/size metadata is invalid")
            artifact_registry.claim(
                path,
                f"evidence:{cell['cell_id']}:{frame}:{role}",
                expected_file_sha256=file_sha256,
                expected_byte_count=byte_count,
            )


def _load_frame(
    packet_root: Path,
    cell: dict[str, Any],
    frame: str,
    artifact_registry: _PacketArtifactRegistry | None = None,
) -> tuple[Any, Any, Any]:
    evidence = cell["evidence"][frame]
    paths: dict[str, Path] = {}
    for role in ("rgb", "depth", "segmentation"):
        record = _evidence_record(cell, frame, role)
        expected_media_type = "image/png" if role == "rgb" else "application/x-npy"
        if record["media_type"] != expected_media_type:
            raise AppearanceAuditError(f"audit evidence media type mismatch: {role}")
        if artifact_registry is None:
            try:
                owned = resolve_owned_regular_file(packet_root, record["path"])
            except UnsafeOwnedFileError as error:
                raise AppearanceAuditError(str(error)) from error
            path = owned.path
            if owned.byte_count != record["byte_count"]:
                raise AppearanceAuditError(f"audit evidence byte count mismatch: {record['path']}")
        else:
            path = artifact_registry.path_for(record["path"])
        if sha256_file(path) != record["file_sha256"]:
            raise AppearanceAuditError(f"audit evidence is missing or corrupt: {record['path']}")
        paths[role] = path
    try:
        with Image.open(paths["rgb"]) as image:
            if image.mode != "RGB":
                raise AppearanceAuditError("audit RGB evidence must use RGB mode")
            rgb = np.asarray(image, dtype=np.uint8).copy()
        depth = np.load(paths["depth"], allow_pickle=False)
        segmentation = np.load(paths["segmentation"], allow_pickle=False)
    except (OSError, ValueError) as error:
        raise AppearanceAuditError("audit evidence cannot be decoded") from error
    for role, array in (("rgb", rgb), ("depth", depth), ("segmentation", segmentation)):
        record = evidence[role]
        if str(array.dtype) != record["dtype"] or list(array.shape) != record["shape"]:
            raise AppearanceAuditError(f"audit evidence shape or dtype mismatch: {role}")
        if logical_array_hash(array) != evidence[role]["logical_sha256"]:
            raise AppearanceAuditError(f"audit evidence logical hash mismatch: {role}")
    return rgb, depth, segmentation


def _frame_metrics(
    rgb: Any,
    segmentation: Any,
    control_rgb: Any,
    profile: AppearanceProfile,
) -> dict[str, Any]:
    controlled = segmentation > 0
    controlled_count = int(np.count_nonzero(controlled))
    if controlled_count == 0:
        raise AppearanceAuditError("controlled surface exposure is empty")
    changed = np.any(rgb != control_rgb, axis=2) & controlled
    fraction = float(np.count_nonzero(changed) / controlled_count)
    mad = float(
        np.mean(np.abs(rgb[controlled].astype(np.int16) - control_rgb[controlled].astype(np.int16)))
        / 255.0
    )
    surface_metrics: list[dict[str, Any]] = []
    exposure_pass = True
    texture_pass = True
    for label in sorted(int(value) for value in np.unique(segmentation) if int(value) > 0):
        mask = segmentation == label
        pixels = rgb[mask].astype(np.float64) / 255.0
        luminance = 0.2126 * pixels[:, 0] + 0.7152 * pixels[:, 1] + 0.0722 * pixels[:, 2]
        mean = float(np.mean(luminance))
        standard_deviation = float(np.std(luminance))
        count = int(luminance.size)
        exposed = (
            profile.non_degeneracy_thresholds.visible_surface_mean_luminance_minimum
            <= mean
            <= profile.non_degeneracy_thresholds.visible_surface_mean_luminance_maximum
        )
        textured = True
        if profile.texture.family != TextureFamily.SOLID and count >= 100:
            textured = (
                standard_deviation
                >= profile.non_degeneracy_thresholds.textured_surface_luminance_std_minimum
            )
        exposure_pass &= exposed
        texture_pass &= textured
        covariance = np.cov(pixels, rowvar=False, bias=True)
        maximum = np.max(pixels, axis=1)
        saturation = np.divide(
            maximum - np.min(pixels, axis=1),
            maximum,
            out=np.zeros_like(maximum),
            where=maximum > 0.0,
        )
        surface_metrics.append(
            {
                "opaque_surface_label": label,
                "pixel_count": count,
                "rgb_mean": [float(value) for value in np.mean(pixels, axis=0)],
                "rgb_covariance": covariance.tolist(),
                "luminance_mean": mean,
                "luminance_standard_deviation": standard_deviation,
                "luminance_percentiles": [
                    float(value) for value in np.percentile(luminance, [1, 5, 50, 95, 99])
                ],
                "clipped_fraction": float(
                    np.mean(np.any((pixels <= 0.0) | (pixels >= 1.0), axis=1))
                ),
                "mean_saturation": float(np.mean(saturation)),
                "high_saturation_fraction": float(np.mean(saturation >= 0.95)),
            }
        )
    boundary_deltas: list[Any] = []
    horizontal = (
        (segmentation[:, 1:] > 0)
        & (segmentation[:, :-1] > 0)
        & (segmentation[:, 1:] != segmentation[:, :-1])
    )
    vertical = (
        (segmentation[1:, :] > 0)
        & (segmentation[:-1, :] > 0)
        & (segmentation[1:, :] != segmentation[:-1, :])
    )
    if np.any(horizontal):
        boundary_deltas.append(
            np.abs(rgb[:, 1:, :].astype(np.int16) - rgb[:, :-1, :].astype(np.int16))[horizontal]
        )
    if np.any(vertical):
        boundary_deltas.append(
            np.abs(rgb[1:, :, :].astype(np.int16) - rgb[:-1, :, :].astype(np.int16))[vertical]
        )
    boundary_contrast = (
        float(np.mean(np.concatenate(boundary_deltas, axis=0)) / 255.0) if boundary_deltas else 0.0
    )
    return {
        "controlled_pixel_count": controlled_count,
        "changed_controlled_pixel_fraction": fraction,
        "normalized_controlled_rgb_mad": mad,
        "material_change_pass": (
            fraction >= profile.non_degeneracy_thresholds.changed_controlled_pixel_fraction_minimum
            and mad >= profile.non_degeneracy_thresholds.normalized_controlled_rgb_mad_minimum
        ),
        "controlled_surface_exposure_pass": exposure_pass,
        "textured_surface_variation_pass": texture_pass,
        "controlled_boundary_normalized_rgb_contrast": boundary_contrast,
        "surface_diagnostics": surface_metrics,
    }


def _source_texture_diagnostics(
    profile: AppearanceProfile,
    scene_family: str,
    appearance_instance: Any,
) -> list[dict[str, Any]]:
    slots = (
        profile.palette.single_occluder_slots
        if scene_family == "single_occluder"
        else profile.palette.corridor_slots
    )
    slots_by_id = {slot.slot_id: slot for slot in slots}
    diagnostics: list[dict[str, Any]] = []
    for record in appearance_instance.textures:
        if record.source_texture_logical_sha256 is None:
            continue
        texture = generate_texture(profile, slots_by_id[record.style_slot_id], record.phase_offset)
        pixels = texture.astype(np.float64) / 255.0
        luminance = 0.2126 * pixels[:, :, 0] + 0.7152 * pixels[:, :, 1] + 0.0722 * pixels[:, :, 2]
        spectrum = np.abs(np.fft.rfft2(luminance - np.mean(luminance))) ** 2
        spectrum[0, 0] = 0.0
        dominant = np.unravel_index(int(np.argmax(spectrum)), spectrum.shape)
        row_frequency = np.fft.fftfreq(luminance.shape[0])[:, None]
        column_frequency = np.fft.rfftfreq(luminance.shape[1])[None, :]
        radius = np.sqrt(row_frequency**2 + column_frequency**2)
        total_power = float(np.sum(spectrum))
        high_frequency_power_fraction = (
            float(np.sum(spectrum[radius >= 0.25]) / total_power) if total_power > 0.0 else 0.0
        )
        diagnostics.append(
            {
                "semantic_surface_name": record.semantic_surface_name,
                "source_texture_logical_sha256": record.source_texture_logical_sha256,
                "source_luminance_standard_deviation": float(np.std(luminance)),
                "dominant_spectrum_index": [int(dominant[0]), int(dominant[1])],
                "high_frequency_power_fraction": high_frequency_power_fraction,
            }
        )
    return diagnostics


def _structural_domain(cell: dict[str, Any]) -> dict[str, Any]:
    return {
        key: cell[key]
        for key in (
            "scene_content_sha256",
            "analytic_transport_sha256",
            "oriented_boundary_sha256",
            "visibility_event_sha256",
            "occlusion_sha256",
            "action_sha256",
            "camera_trajectory_sha256",
            "geometry_sha256",
            "opaque_remapping_sha256",
        )
    }


def _portable_analytic_identity_domain(cell: dict[str, Any]) -> dict[str, Any]:
    return {
        key: cell[key]
        for key in (
            "scene_content_sha256",
            "analytic_transport_sha256",
            "oriented_boundary_sha256",
            "visibility_event_sha256",
            "occlusion_sha256",
            "action_sha256",
        )
    }


def _admission_evidence_domain(cell: dict[str, Any]) -> dict[str, Any]:
    return {
        "frame_metrics": cell.get("frame_metrics"),
        "admission_checks": cell.get("admission_checks"),
        "admission_status": cell.get("admission_status"),
        "rejection_reasons": cell.get("rejection_reasons"),
        "matched_control_failure_type": cell.get("matched_control_failure_type"),
        "matched_control_failure_message": cell.get("matched_control_failure_message"),
    }


def _evaluate_cell(
    packet_root: Path,
    cell: dict[str, Any],
    control: dict[str, Any],
    profile: AppearanceProfile,
    artifact_registry: _PacketArtifactRegistry | None = None,
) -> None:
    if cell["generation_status"] != "success":
        cell["frame_metrics"] = {}
        cell["admission_checks"] = {
            "structural_invariance": False,
            "portable_analytic_identity_equality": False,
            "ecological_label_equality": False,
            "depth_segmentation_invariance": False,
            "determinism": False,
            "material_rgb_change": False,
            "controlled_surface_exposure": False,
            "textured_surface_variation": False,
        }
        cell["admission_status"] = "rejected"
        cell["rejection_reasons"] = ["generation_or_validation_failed"]
        return
    if control["generation_status"] != "success":
        cell["frame_metrics"] = {}
        cell["admission_checks"] = {
            "structural_invariance": False,
            "portable_analytic_identity_equality": False,
            "ecological_label_equality": False,
            "depth_segmentation_invariance": False,
            "determinism": bool(cell["determinism_pass"]),
            "material_rgb_change": False,
            "controlled_surface_exposure": False,
            "textured_surface_variation": False,
        }
        cell["admission_status"] = "rejected"
        cell["rejection_reasons"] = ["matched_control_generation_or_validation_failed"]
        cell["matched_control_failure_type"] = control.get("failure_type")
        cell["matched_control_failure_message"] = control.get("failure_message")
        return
    structural_pass = _structural_domain(cell) == _structural_domain(control)
    portable_analytic_pass = _portable_analytic_identity_domain(
        cell
    ) == _portable_analytic_identity_domain(control)
    ecological_label_pass = cell["ecological_label_sha256"] == control["ecological_label_sha256"]
    depth_segmentation_pass = True
    frames: dict[str, Any] = {}
    for frame_name in ("before", "after"):
        rgb, depth, segmentation = _load_frame(packet_root, cell, frame_name, artifact_registry)
        control_rgb, control_depth, control_segmentation = _load_frame(
            packet_root, control, frame_name, artifact_registry
        )
        depth_segmentation_pass &= np.array_equal(depth, control_depth) and np.array_equal(
            segmentation, control_segmentation
        )
        frame_metrics = _frame_metrics(rgb, segmentation, control_rgb, profile)
        semantic_by_label = {label: name for name, label in cell["semantic_surface_labels"].items()}
        source_by_semantic = {
            item["semantic_surface_name"]: item for item in cell["source_texture_diagnostics"]
        }
        for surface in frame_metrics["surface_diagnostics"]:
            semantic_name = semantic_by_label[surface["opaque_surface_label"]]
            source = source_by_semantic.get(semantic_name)
            if source is not None:
                source_standard_deviation = source["source_luminance_standard_deviation"]
                surface["rendered_to_source_luminance_std_ratio"] = (
                    surface["luminance_standard_deviation"] / source_standard_deviation
                    if source_standard_deviation > 0.0
                    else 0.0
                )
        frames[frame_name] = frame_metrics
    material_required = profile.candidate_class != CandidateClass.LEGACY_REGRESSION_CONTROL
    material_pass = all(frame["material_change_pass"] for frame in frames.values())
    exposure_pass = all(frame["controlled_surface_exposure_pass"] for frame in frames.values())
    texture_pass = all(frame["textured_surface_variation_pass"] for frame in frames.values())
    checks = {
        "structural_invariance": structural_pass,
        "portable_analytic_identity_equality": portable_analytic_pass,
        "ecological_label_equality": ecological_label_pass,
        "depth_segmentation_invariance": depth_segmentation_pass,
        "determinism": bool(cell["determinism_pass"]),
        "material_rgb_change": material_pass if material_required else True,
        "controlled_surface_exposure": exposure_pass,
        "textured_surface_variation": texture_pass,
    }
    reasons = [name for name, passed in checks.items() if not passed]
    cell["frame_metrics"] = frames
    cell["admission_checks"] = checks
    cell["admission_status"] = "admitted" if not reasons else "rejected"
    cell["rejection_reasons"] = reasons


def _dataset_cell(
    dataset: Path,
    repeat: Path,
    packet_root: Path,
    config: BenchmarkConfig,
    profile: AppearanceProfile,
    seed_index: int,
    candidate_seed: int,
    registry: AppearanceRegistry,
    seeds: EvaluationSeedRegistry,
) -> dict[str, Any]:
    scene = config.scene_family.value
    cell_id = _cell_id(scene, profile.profile_id, seed_index)
    cell: dict[str, Any] = {
        "cell_id": cell_id,
        "scene_family": scene,
        "seed_index": seed_index,
        "candidate_seed": candidate_seed,
        "profile_id": profile.profile_id,
        "matched_control_profile_id": profile.matched_control_profile_id,
        "generation_status": "failed",
        "admission_status": "rejected",
        "rejection_reasons": [],
    }
    try:
        selected = _profile_config(config, profile, candidate_seed)
        manifest = generate_dataset(
            selected,
            1,
            dataset,
            appearance_registry=registry,
            seed_registry=seeds,
        )
        validate_dataset(dataset)
        repeat_manifest = generate_dataset(
            selected,
            1,
            repeat,
            appearance_registry=registry,
            seed_registry=seeds,
        )
        validate_dataset(repeat)
        episode = manifest.episodes[0]
        repeat_episode = repeat_manifest.episodes[0]
        transition = TransitionRecord.model_validate_json(
            (dataset / episode.transition.path).read_bytes()
        )
        instrumentation = parse_privileged_instrumentation_json(
            (dataset / episode.privileged_instrumentation.path).read_bytes()
        )
        repeat_instrumentation = parse_privileged_instrumentation_json(
            (repeat / repeat_episode.privileged_instrumentation.path).read_bytes()
        )
        camera_records = []
        for frame in (transition.before, transition.after):
            camera_records.append(_read_json(dataset / frame.camera_world_transform.path))
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
        evidence_directory = packet_root / "metric_evidence" / cell_id
        evidence = _copy_evidence(dataset, manifest, evidence_directory)
        for frame in evidence.values():
            for artifact in frame.values():
                artifact["path"] = Path(artifact["path"]).relative_to(packet_root).as_posix()
        deterministic_rgb = all(
            (dataset / first.path).read_bytes() == (repeat / second.path).read_bytes()
            for first, second in (
                (
                    transition.before.rgb,
                    TransitionRecord.model_validate_json(
                        (repeat / repeat_episode.transition.path).read_bytes()
                    ).before.rgb,
                ),
                (
                    transition.after.rgb,
                    TransitionRecord.model_validate_json(
                        (repeat / repeat_episode.transition.path).read_bytes()
                    ).after.rgb,
                ),
            )
        )
        deterministic_records = (
            instrumentation.appearance == repeat_instrumentation.appearance
            and manifest.resolved_config.logical_sha256
            == repeat_manifest.resolved_config.logical_sha256
        )
        opaque_labels = {
            surface.surface_id: surface.segmentation_label for surface in transition.surfaces
        }
        semantic_surface_labels = {
            semantic_name: opaque_labels[
                instrumentation.raw_to_opaque_surface_ids[str(raw_geom_id)]
            ]
            for semantic_name, raw_geom_id in instrumentation.raw_geom_ids.items()
        }
        source_texture_diagnostics = _source_texture_diagnostics(
            profile, scene, instrumentation.appearance
        )
        cell.update(
            {
                "generation_status": "success",
                "episode_seed": episode.episode_seed,
                "appearance_profile_sha256": manifest.appearance_profile_sha256,
                "appearance_instance_sha256": episode.appearance_instance_sha256,
                "appearance_instance": instrumentation.appearance.model_dump(mode="json"),
                "evaluation_seed_registry_sha256": (manifest.evaluation_seed_registry_sha256),
                "appearance_assignment_schedule_source": (
                    manifest.appearance_assignment_schedule_source
                ),
                "renderer_provenance": manifest.renderer_provenance.model_dump(mode="json"),
                "rgb_logical_sha256": list(episode.rgb_logical_sha256),
                "scene_content_sha256": episode.scene_content_sha256,
                "analytic_transport_sha256": episode.analytic_transport_sha256,
                "oriented_boundary_sha256": episode.oriented_boundary_sha256,
                "visibility_event_sha256": episode.visibility_event_sha256,
                "ecological_label_sha256": episode.ecological_label_sha256,
                "occlusion_sha256": _hash_json(transition.occlusion),
                "action_sha256": _hash_json(transition.action),
                "camera_trajectory_sha256": _hash_json(camera_records),
                "geometry_sha256": _hash_json(geometry_domain),
                "opaque_remapping_sha256": _hash_json(instrumentation.raw_to_opaque_surface_ids),
                "depth_logical_sha256": [
                    transition.before.depth.logical_sha256,
                    transition.after.depth.logical_sha256,
                ],
                "segmentation_logical_sha256": [
                    transition.before.segmentation.logical_sha256,
                    transition.after.segmentation.logical_sha256,
                ],
                "determinism_pass": deterministic_rgb and deterministic_records,
                "semantic_surface_labels": semantic_surface_labels,
                "source_texture_diagnostics": source_texture_diagnostics,
                "evidence": evidence,
            }
        )
    except Exception as error:
        cell["failure_type"] = type(error).__name__
        cell["failure_message"] = str(error)
        cell["rejection_reasons"] = ["generation_or_validation_failed"]
    finally:
        shutil.rmtree(dataset, ignore_errors=True)
        shutil.rmtree(repeat, ignore_errors=True)
    return cell


def _contact_sheet_image(
    packet_root: Path,
    cells: list[dict[str, Any]],
    scene: str,
    artifact_registry: _PacketArtifactRegistry | None = None,
) -> Image.Image:
    representatives = [
        cell
        for cell in cells
        if cell["scene_family"] == scene
        and cell["seed_index"] == 0
        and cell["generation_status"] == "success"
    ]
    tile_size = 256
    row_height = 304
    sheet = Image.new("RGB", (3 * tile_size, max(1, len(representatives)) * row_height), "white")
    draw = ImageDraw.Draw(sheet)
    for row, cell in enumerate(representatives):
        y = row * row_height
        before, _, segmentation = _load_frame(packet_root, cell, "before", artifact_registry)
        after, _, _ = _load_frame(packet_root, cell, "after", artifact_registry)
        labels = segmentation.astype(np.int64)
        reference = np.stack(
            ((labels * 67) % 255, (labels * 131) % 255, (labels * 197) % 255), axis=2
        ).astype(np.uint8)
        before_image = Image.fromarray(before).resize(
            (tile_size, tile_size), Image.Resampling.NEAREST
        )
        after_image = Image.fromarray(after).resize(
            (tile_size, tile_size), Image.Resampling.NEAREST
        )
        reference_image = Image.fromarray(reference).resize(
            (tile_size, tile_size), Image.Resampling.NEAREST
        )
        sheet.paste(before_image, (0, y + 48))
        sheet.paste(after_image, (tile_size, y + 48))
        sheet.paste(reference_image, (2 * tile_size, y + 48))
        metrics = cell.get("frame_metrics", {}).get("before", {})
        axes = ",".join(cell["appearance_instance"]["profile"]["axis_tags"])
        draw.text(
            (2, y + 2),
            f"{scene} | {cell['profile_id']} | {cell['admission_status']} | axes={axes}",
            fill="black",
        )
        draw.text(
            (2, y + 18),
            f"changed={metrics.get('changed_controlled_pixel_fraction', 0.0):.3f} "
            f"mad={metrics.get('normalized_controlled_rgb_mad', 0.0):.3f} | "
            "before RGB | after RGB | controlled segmentation",
            fill="black",
        )
        ecological_equal = cell.get("admission_checks", {}).get("ecological_label_equality", False)
        draw.text(
            (2, y + 34),
            f"renderer-local ecological label equal={str(bool(ecological_equal)).lower()}",
            fill="black",
        )
    return sheet


def _contact_sheets(
    packet_root: Path,
    cells: list[dict[str, Any]],
) -> dict[str, Any]:
    output = packet_root / "representative_contact_sheets"
    output.mkdir(exist_ok=True)
    records: list[dict[str, Any]] = []
    for scene in SCENE_FAMILIES:
        sheet = _contact_sheet_image(packet_root, cells, scene)
        relative_path = f"representative_contact_sheets/{scene}_representative_seed_0.png"
        path = packet_root / relative_path
        sheet.save(path, format="PNG")
        pixels = np.asarray(sheet, dtype=np.uint8)
        records.append(
            {
                "scene_family": scene,
                "seed_index": 0,
                "path": relative_path,
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
    return {
        "schema_version": CONTACT_SHEET_MANIFEST_VERSION,
        "sheets": records,
    }


def _validate_contact_sheet_manifest(
    packet_root: Path,
    cells: list[dict[str, Any]],
    manifest: Any,
    artifact_registry: _PacketArtifactRegistry,
) -> None:
    if not isinstance(manifest, dict) or set(manifest) != {"schema_version", "sheets"}:
        raise AppearanceAuditError("contact-sheet manifest is not strict")
    if manifest["schema_version"] != CONTACT_SHEET_MANIFEST_VERSION:
        raise AppearanceAuditError("contact-sheet manifest version is unsupported")
    sheets = manifest["sheets"]
    if not isinstance(sheets, list) or len(sheets) != len(SCENE_FAMILIES):
        raise AppearanceAuditError("contact-sheet manifest is incomplete")
    directory = artifact_registry.root / "representative_contact_sheets"
    try:
        directory_stat = directory.lstat()
    except OSError as error:
        raise AppearanceAuditError("representative contact-sheet directory is missing") from error
    if directory.is_symlink() or not stat.S_ISDIR(directory_stat.st_mode):
        raise AppearanceAuditError("contact-sheet directory must be a non-link directory")
    expected_names = {f"{scene}_representative_seed_0.png" for scene in SCENE_FAMILIES}
    if {entry.name for entry in directory.iterdir()} != expected_names:
        raise AppearanceAuditError("contact-sheet directory contains missing or additional entries")

    expected_fields = {
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
    for scene, record in zip(SCENE_FAMILIES, sheets, strict=True):
        if not isinstance(record, dict) or set(record) != expected_fields:
            raise AppearanceAuditError("contact-sheet record is not strict")
        expected_path = f"representative_contact_sheets/{scene}_representative_seed_0.png"
        if (
            record["scene_family"] != scene
            or record["seed_index"] != 0
            or record["path"] != expected_path
            or record["media_type"] != "image/png"
            or record["mode"] != "RGB"
        ):
            raise AppearanceAuditError("contact-sheet canonical metadata is inconsistent")
        path = artifact_registry.claim(
            expected_path,
            f"contact-sheet:{scene}",
            expected_file_sha256=record["file_sha256"],
            expected_byte_count=record["byte_count"],
        )
        try:
            with Image.open(path) as image:
                if image.mode != "RGB":
                    raise AppearanceAuditError("contact sheet must use RGB mode")
                actual = np.asarray(image, dtype=np.uint8).copy()
                dimensions = list(image.size)
        except OSError as error:
            raise AppearanceAuditError("contact sheet cannot be decoded") from error
        expected = np.asarray(
            _contact_sheet_image(packet_root, cells, scene, artifact_registry),
            dtype=np.uint8,
        )
        if (
            dimensions != record["dimensions"]
            or str(actual.dtype) != record["dtype"]
            or list(actual.shape) != record["shape"]
            or logical_array_hash(actual) != record["logical_sha256"]
            or not np.array_equal(actual, expected)
        ):
            raise AppearanceAuditError("contact sheet differs from independent reconstruction")


def _roots(
    registry: AppearanceRegistry,
    seeds: EvaluationSeedRegistry,
    cells: list[dict[str, Any]],
) -> dict[str, str]:
    successful = [cell for cell in cells if cell["generation_status"] == "success"]
    procedural = [
        {
            "cell_id": cell["cell_id"],
            "profile_hash": cell["appearance_profile_sha256"],
            "textures": cell["appearance_instance"]["textures"],
        }
        for cell in successful
    ]
    assignment = [
        {
            "cell_id": cell["cell_id"],
            "appearance_instance_sha256": cell["appearance_instance_sha256"],
            "evaluation_seed_registry_sha256": cell["evaluation_seed_registry_sha256"],
            "appearance_assignment_schedule_source": cell["appearance_assignment_schedule_source"],
            "candidate_schedule_index": cell["appearance_instance"]["seeds"][
                "candidate_schedule_index"
            ],
            "assignment_schedule_posture": cell["appearance_instance"][
                "assignment_schedule_posture"
            ],
            "style_assignment": cell["appearance_instance"]["style_assignment"],
        }
        for cell in successful
    ]
    portable_analytic = [
        {"cell_id": cell["cell_id"], **_portable_analytic_identity_domain(cell)}
        for cell in successful
    ]
    outcomes = [
        {
            "cell_id": cell["cell_id"],
            **_portable_analytic_identity_domain(cell),
            "structural_invariance": cell["admission_checks"]["structural_invariance"],
            "ecological_label_equality": cell["admission_checks"]["ecological_label_equality"],
            "depth_segmentation_invariance": cell["admission_checks"][
                "depth_segmentation_invariance"
            ],
            "determinism": cell["admission_checks"]["determinism"],
        }
        for cell in successful
    ]
    renderer_local_labels = [
        {
            "cell_id": cell["cell_id"],
            "ecological_label_sha256": cell["ecological_label_sha256"],
        }
        for cell in successful
    ]
    renderer = [
        {
            "cell_id": cell["cell_id"],
            "scene_family": cell["scene_family"],
            "profile_id": cell["profile_id"],
            "seed_index": cell["seed_index"],
            "candidate_seed": cell["candidate_seed"],
            "renderer_provenance": cell["renderer_provenance"],
            "rgb_logical_sha256": cell["rgb_logical_sha256"],
            "depth_logical_sha256": cell["depth_logical_sha256"],
            "segmentation_logical_sha256": cell["segmentation_logical_sha256"],
            "retained_evidence": cell["evidence"],
            "metrics": cell.get("frame_metrics"),
            "determinism_pass": cell["determinism_pass"],
            "admission_status": cell["admission_status"],
        }
        for cell in successful
    ]
    return {
        "appearance_registry_sha256": appearance_registry_hash(registry),
        "seed_registry_sha256": seed_registry_hash(seeds),
        "procedural_asset_root_sha256": _hash_json(procedural),
        "appearance_assignment_root_sha256": _hash_json(assignment),
        "portable_analytic_identity_root_sha256": _hash_json(portable_analytic),
        "appearance_invariance_outcome_root_sha256": _hash_json(outcomes),
        "renderer_local_ecological_label_root_sha256": _hash_json(renderer_local_labels),
        "renderer_specific_audit_root_sha256": _hash_json(renderer),
    }


def _write_packet_reports(
    packet_root: Path,
    registry: AppearanceRegistry,
    seeds: EvaluationSeedRegistry,
    cells: list[dict[str, Any]],
    contact_sheet_manifest: dict[str, Any],
) -> dict[str, Any]:
    counts = Counter(cell["admission_status"] for cell in cells)
    summaries = []
    for profile in registry.profiles:
        selected = [cell for cell in cells if cell["profile_id"] == profile.profile_id]
        summaries.append(
            {
                "profile_id": profile.profile_id,
                "profile_sha256": appearance_profile_hash(profile),
                "candidate_class": profile.candidate_class,
                "freeze_eligible": profile.freeze_eligible,
                "axis_tags": profile.axis_tags,
                "cell_counts": dict(Counter(cell["admission_status"] for cell in selected)),
                "profile_admission_status": (
                    "admitted"
                    if all(cell["admission_status"] == "admitted" for cell in selected)
                    else "rejected"
                ),
            }
        )
    negative = [
        {
            "cell_id": cell["cell_id"],
            "profile_id": cell["profile_id"],
            "scene_family": cell["scene_family"],
            "seed_index": cell["seed_index"],
            "rejection_reasons": cell["rejection_reasons"],
            "failure_type": cell.get("failure_type"),
            "failure_message": cell.get("failure_message"),
            "matched_control_failure_type": cell.get("matched_control_failure_type"),
            "matched_control_failure_message": cell.get("matched_control_failure_message"),
        }
        for cell in cells
        if cell["admission_status"] == "rejected"
    ]
    profile_summary = {"schema_version": AUDIT_SCHEMA_VERSION, "profiles": summaries}
    seed_matrix = {"schema_version": AUDIT_SCHEMA_VERSION, "cells": cells}
    negative_evidence = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "retention_rule": "all_rejected_candidates_and_failing_seeds_retained_v1",
        "rejected_cells": negative,
    }
    write_canonical_json(packet_root / "profile_summary.json", profile_summary)
    write_canonical_json(packet_root / "seed_matrix.json", seed_matrix)
    write_canonical_json(packet_root / "negative_evidence.json", negative_evidence)
    roots = _roots(registry, seeds, cells)
    report_hashes = {
        name: sha256_file(packet_root / name)
        for name in ("profile_summary.json", "seed_matrix.json", "negative_evidence.json")
    }
    source = collect_source_provenance(Path.cwd()).model_dump(mode="json")
    governing_hashes = {path: sha256_file(Path(path)) for path in GOVERNING_DOCUMENTS}
    logical_domain = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "root_schema_version": ROOT_SCHEMA_VERSION,
        "freeze_status": PACKET_FREEZE_STATUS,
        "source_provenance": source,
        "governing_document_hashes": governing_hashes,
        "roots": roots,
        "report_file_sha256": report_hashes,
        "contact_sheet_manifest": contact_sheet_manifest,
        "matrix_counts": dict(counts),
        "profile_count": len(registry.profiles),
        "candidate_seed_count": len(seeds.candidate_episode_seeds),
        "scene_families": list(SCENE_FAMILIES),
        "final_split": None,
        "final_evaluation_seeds": None,
    }
    packet = {
        **logical_domain,
        "packet_logical_root_sha256": _hash_json(logical_domain),
    }
    write_canonical_json(packet_root / "candidate_packet.json", packet)
    return packet


def validate_appearance_audit(packet_root: Path) -> dict[str, Any]:
    """Independently recompute matrix, metrics, balance, and packet roots."""

    artifact_registry = _PacketArtifactRegistry(packet_root)
    packet_path = artifact_registry.claim("candidate_packet.json", "candidate-packet")
    packet = _read_json(packet_path)
    expected_packet_fields = set(PACKET_LOGICAL_FIELDS) | {"packet_logical_root_sha256"}
    if not isinstance(packet, dict) or set(packet) != expected_packet_fields:
        raise AppearanceAuditError("candidate packet schema is not strict")
    if packet["schema_version"] != AUDIT_SCHEMA_VERSION:
        raise AppearanceAuditError("candidate packet audit schema version is unsupported")
    if packet["root_schema_version"] != ROOT_SCHEMA_VERSION:
        raise AppearanceAuditError("candidate packet root schema version is unsupported")
    if packet["freeze_status"] != PACKET_FREEZE_STATUS:
        raise AppearanceAuditError("candidate packet must not claim a benchmark freeze")
    if packet["final_split"] is not None or packet["final_evaluation_seeds"] is not None:
        raise AppearanceAuditError("candidate packet must not select final split or seeds")

    report_names = ("profile_summary.json", "seed_matrix.json", "negative_evidence.json")
    report_hashes = packet["report_file_sha256"]
    if (
        not isinstance(report_hashes, dict)
        or set(report_hashes) != set(report_names)
        or any(not isinstance(value, str) for value in report_hashes.values())
    ):
        raise AppearanceAuditError("candidate packet report-hash schema is not strict")
    snapshot_paths = {
        "appearance": artifact_registry.claim(
            "appearance_registry_snapshot.json", "appearance-registry-snapshot"
        ),
        "seeds": artifact_registry.claim("seed_registry_snapshot.json", "seed-registry-snapshot"),
    }
    claimed_reports = {
        name: artifact_registry.claim(
            name,
            f"report:{name}",
            expected_file_sha256=report_hashes[name],
        )
        for name in report_names
    }
    try:
        registry = AppearanceRegistry.model_validate_json(snapshot_paths["appearance"].read_bytes())
        seeds = EvaluationSeedRegistry.model_validate_json(snapshot_paths["seeds"].read_bytes())
    except Exception as error:
        raise AppearanceAuditError("candidate packet registry snapshot is invalid") from error
    validate_axis_isolation(registry)
    matrix = _read_json(claimed_reports["seed_matrix.json"])
    if (
        not isinstance(matrix, dict)
        or set(matrix) != {"schema_version", "cells"}
        or matrix["schema_version"] != AUDIT_SCHEMA_VERSION
    ):
        raise AppearanceAuditError("seed matrix audit schema version is unsupported")
    cells = matrix["cells"]
    if not isinstance(cells, list):
        raise AppearanceAuditError("seed matrix cells must be a list")
    expected_cells = [
        (scene, profile, seed_index, candidate_seed)
        for scene in SCENE_FAMILIES
        for profile in registry.profiles
        for seed_index, candidate_seed in zip(
            seeds.indices, seeds.candidate_episode_seeds, strict=True
        )
    ]
    if len(cells) != len(expected_cells):
        raise AppearanceAuditError("candidate audit matrix has the wrong cell count")
    for cell, (scene, profile, seed_index, candidate_seed) in zip(
        cells, expected_cells, strict=True
    ):
        if not isinstance(cell, dict):
            raise AppearanceAuditError("candidate audit cell must be an object")
        expected_identity = {
            "cell_id": _cell_id(scene, profile.profile_id, seed_index),
            "scene_family": scene,
            "profile_id": profile.profile_id,
            "matched_control_profile_id": profile.matched_control_profile_id,
            "seed_index": seed_index,
            "candidate_seed": candidate_seed,
        }
        if any(cell.get(key) != value for key, value in expected_identity.items()):
            raise AppearanceAuditError("candidate audit cell identity is not canonical")
        if cell.get("generation_status") not in {"success", "failed"}:
            raise AppearanceAuditError("candidate audit generation status is invalid")
        if cell["generation_status"] == "failed" and "evidence" in cell:
            raise AppearanceAuditError("failed candidate cell contains unclaimed evidence")

    by_key = {
        (cell["scene_family"], cell["profile_id"], cell["seed_index"]): cell for cell in cells
    }
    profiles = {profile.profile_id: profile for profile in registry.profiles}
    for cell in cells:
        if cell["generation_status"] == "success":
            _claim_cell_evidence(artifact_registry, cell)

    for cell in cells:
        profile = profiles[cell["profile_id"]]
        if cell["generation_status"] == "success":
            if cell["episode_seed"] != derive_seed(cell["candidate_seed"], "episode:0"):
                raise AppearanceAuditError("episode seed differs from ordinary derivation")
            if cell["evaluation_seed_registry_sha256"] != seed_registry_hash(seeds):
                raise AppearanceAuditError("cell seed-registry identity is inconsistent")
            if (
                cell["appearance_assignment_schedule_source"]
                != "snapshotted_evaluation_seed_registry_v1"
            ):
                raise AppearanceAuditError("cell assignment schedule source is inconsistent")
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
            expected_appearance = resolve_appearance(
                registry,
                profile.profile_id,
                cell["scene_family"],
                surface_names,
                cell["episode_seed"],
                cell["candidate_seed"],
                seeds,
            )
            if (
                expected_appearance.record.model_dump(mode="json") != cell["appearance_instance"]
                or cell["appearance_profile_sha256"]
                != expected_appearance.record.appearance_profile_sha256
                or cell["appearance_instance_sha256"]
                != expected_appearance.record.appearance_instance_sha256
            ):
                raise AppearanceAuditError("appearance instance failed independent recomputation")
            if cell["source_texture_diagnostics"] != _source_texture_diagnostics(
                profile, cell["scene_family"], expected_appearance.record
            ):
                raise AppearanceAuditError("source-texture diagnostics failed recomputation")
            declared_logical_hashes = {
                "rgb": cell.get("rgb_logical_sha256"),
                "depth": cell.get("depth_logical_sha256"),
                "segmentation": cell.get("segmentation_logical_sha256"),
            }
            if any(
                not isinstance(values, list)
                or len(values) != 2
                or any(not isinstance(value, str) for value in values)
                for values in declared_logical_hashes.values()
            ):
                raise AppearanceAuditError("cell logical evidence identities are malformed")
            for frame_index, frame_name in enumerate(("before", "after")):
                arrays = _load_frame(packet_root, cell, frame_name, artifact_registry)
                for role, array in zip(("rgb", "depth", "segmentation"), arrays, strict=True):
                    if logical_array_hash(array) != declared_logical_hashes[role][frame_index]:
                        raise AppearanceAuditError(
                            f"retained evidence differs from declared {role} identity: "
                            f"{cell['cell_id']}:{frame_name}"
                        )
        control = by_key[
            (cell["scene_family"], profile.matched_control_profile_id, cell["seed_index"])
        ]
        stored = _admission_evidence_domain(cell)
        recomputed = json.loads(json.dumps(cell))
        for field in (
            "frame_metrics",
            "admission_checks",
            "admission_status",
            "rejection_reasons",
            "matched_control_failure_type",
            "matched_control_failure_message",
        ):
            recomputed.pop(field, None)
        _evaluate_cell(packet_root, recomputed, control, profile, artifact_registry)
        if canonical_json_bytes(stored) != canonical_json_bytes(
            _admission_evidence_domain(recomputed)
        ):
            raise AppearanceAuditError(f"altered admission evidence: {cell['cell_id']}")
    for profile in registry.profiles:
        if profile.freeze_eligible:
            names = (
                ("support_surface", "occluding_surface", "background_surface")
                if len(profile.palette.single_occluder_slots) == 3
                else ()
            )
            if not names:
                raise AppearanceAuditError("single-occluder slot domain is malformed")
            single_counts = assignment_balance(profile, names, seeds.candidate_episode_seeds, seeds)
            if any(
                max(counts.values()) - min(counts.values()) > 1 for counts in single_counts.values()
            ):
                raise AppearanceAuditError("single-occluder assignment schedule is unbalanced")
            corridor_counts = assignment_balance(
                profile,
                (
                    "corridor_floor",
                    "corridor_left_surface",
                    "corridor_right_surface",
                    "corridor_end_surface",
                ),
                seeds.candidate_episode_seeds,
                seeds,
            )
            if any(set(counts.values()) != {2} for counts in corridor_counts.values()):
                raise AppearanceAuditError("corridor assignment schedule is not exactly balanced")
    expected_summaries = []
    for profile in registry.profiles:
        selected = [cell for cell in cells if cell["profile_id"] == profile.profile_id]
        expected_summaries.append(
            {
                "profile_id": profile.profile_id,
                "profile_sha256": appearance_profile_hash(profile),
                "candidate_class": profile.candidate_class,
                "freeze_eligible": profile.freeze_eligible,
                "axis_tags": profile.axis_tags,
                "cell_counts": dict(Counter(cell["admission_status"] for cell in selected)),
                "profile_admission_status": (
                    "admitted"
                    if all(cell["admission_status"] == "admitted" for cell in selected)
                    else "rejected"
                ),
            }
        )
    expected_profile_summary = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "profiles": expected_summaries,
    }
    if canonical_json_bytes(expected_profile_summary) != canonical_json_bytes(
        _read_json(claimed_reports["profile_summary.json"])
    ):
        raise AppearanceAuditError("profile summary differs from the seed matrix")
    expected_negative = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "retention_rule": "all_rejected_candidates_and_failing_seeds_retained_v1",
        "rejected_cells": [
            {
                "cell_id": cell["cell_id"],
                "profile_id": cell["profile_id"],
                "scene_family": cell["scene_family"],
                "seed_index": cell["seed_index"],
                "rejection_reasons": cell["rejection_reasons"],
                "failure_type": cell.get("failure_type"),
                "failure_message": cell.get("failure_message"),
                "matched_control_failure_type": cell.get("matched_control_failure_type"),
                "matched_control_failure_message": cell.get("matched_control_failure_message"),
            }
            for cell in cells
            if cell["admission_status"] == "rejected"
        ],
    }
    if canonical_json_bytes(expected_negative) != canonical_json_bytes(
        _read_json(claimed_reports["negative_evidence.json"])
    ):
        raise AppearanceAuditError("negative evidence does not retain every rejected cell")

    expected_counts = dict(Counter(cell["admission_status"] for cell in cells))
    if packet["matrix_counts"] != expected_counts:
        raise AppearanceAuditError("candidate packet matrix counts are inconsistent")
    if packet["profile_count"] != len(registry.profiles):
        raise AppearanceAuditError("candidate packet profile count is inconsistent")
    if packet["candidate_seed_count"] != len(seeds.candidate_episode_seeds):
        raise AppearanceAuditError("candidate packet seed count is inconsistent")
    if packet["scene_families"] != list(SCENE_FAMILIES):
        raise AppearanceAuditError("candidate packet scene-family domain is inconsistent")

    roots = _roots(registry, seeds, cells)
    if roots != packet["roots"]:
        raise AppearanceAuditError("audit shared or renderer-specific root mismatch")
    _validate_contact_sheet_manifest(
        packet_root,
        cells,
        packet["contact_sheet_manifest"],
        artifact_registry,
    )
    if _hash_json(_packet_logical_domain(packet)) != packet["packet_logical_root_sha256"]:
        raise AppearanceAuditError("candidate packet logical root mismatch")
    current_governing_hashes = {path: sha256_file(Path(path)) for path in GOVERNING_DOCUMENTS}
    if current_governing_hashes != packet["governing_document_hashes"]:
        raise AppearanceAuditError("governing-document hash mismatch")
    if collect_source_provenance(Path.cwd()).model_dump(mode="json") != packet["source_provenance"]:
        raise AppearanceAuditError("packet source provenance is not truthful for this source tree")
    return packet


def create_appearance_audit(
    registry_path: Path,
    seeds_path: Path,
    single_config_path: Path,
    corridor_config_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Generate, validate, and atomically publish the complete candidate packet."""

    if output.is_symlink():
        raise FileExistsError(f"audit output cannot be a symbolic link: {output}")
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError(f"audit output is not an empty directory: {output}")
        output.rmdir()
    registry = load_appearance_registry(registry_path)
    seeds = load_evaluation_seed_registry(seeds_path)
    validate_axis_isolation(registry)
    configs = (load_config(single_config_path), load_config(corridor_config_path))
    if tuple(config.scene_family.value for config in configs) != SCENE_FAMILIES:
        raise AppearanceAuditError("audit configurations must be single_occluder then corridor")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    started = time.monotonic()
    try:
        write_canonical_json(staging / "appearance_registry_snapshot.json", registry)
        write_canonical_json(staging / "seed_registry_snapshot.json", seeds)
        work = staging / "_temporary_datasets"
        work.mkdir()
        cells: list[dict[str, Any]] = []
        for config in configs:
            for profile in registry.profiles:
                for seed_index, candidate_seed in zip(
                    seeds.indices, seeds.candidate_episode_seeds, strict=True
                ):
                    cell_id = _cell_id(config.scene_family.value, profile.profile_id, seed_index)
                    cells.append(
                        _dataset_cell(
                            work / cell_id,
                            work / f"{cell_id}--repeat",
                            staging,
                            config,
                            profile,
                            seed_index,
                            candidate_seed,
                            registry,
                            seeds,
                        )
                    )
        by_key = {
            (cell["scene_family"], cell["profile_id"], cell["seed_index"]): cell for cell in cells
        }
        for cell in cells:
            profile = next(
                item for item in registry.profiles if item.profile_id == cell["profile_id"]
            )
            control = by_key[
                (cell["scene_family"], profile.matched_control_profile_id, cell["seed_index"])
            ]
            _evaluate_cell(staging, cell, control, profile)
        shutil.rmtree(work)
        contact_sheet_manifest = _contact_sheets(staging, cells)
        _write_packet_reports(
            staging,
            registry,
            seeds,
            cells,
            contact_sheet_manifest,
        )
        volatile = {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "hostname": socket.gethostname(),
            "operating_system": platform.platform(),
            "wall_clock_seconds": time.monotonic() - started,
        }
        (staging / "run.json").write_text(
            json.dumps(volatile, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        packet = validate_appearance_audit(staging)
        _atomic_no_replace_directory(staging, output)
        return packet
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
