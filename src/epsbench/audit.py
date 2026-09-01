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
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image, ImageDraw

from epsbench.appearance import (
    APPEARANCE_REGISTRY_VERSION,
    AppearanceInstanceRecord,
    AppearanceProfile,
    AppearanceRegistry,
    AppearanceRegistryType,
    CandidateClass,
    EvaluationSeedRegistry,
    SeedRegistryType,
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
from epsbench.data.paths import (
    OwnedRegularFile,
    UnsafeOwnedFileError,
    open_owned_regular_file,
    sha256_open_file,
    validate_exact_owned_file_tree,
)
from epsbench.data.provenance import collect_source_provenance
from epsbench.data.validate import validate_dataset
from epsbench.schema import (
    DatasetManifest,
    RendererProvenance,
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


@dataclass(frozen=True, slots=True)
class _ValidatedAppearanceAudit:
    """Immutable packet evidence retained from one complete validation snapshot."""

    packet_payload: bytes
    seed_matrix_payload: bytes

    def packet(self) -> dict[str, Any]:
        payload = json.loads(self.packet_payload)
        if not isinstance(payload, dict):
            raise AppearanceAuditError("validated candidate packet is not an object")
        return payload

    def seed_matrix(self) -> dict[str, Any]:
        payload = json.loads(self.seed_matrix_payload)
        if not isinstance(payload, dict):
            raise AppearanceAuditError("validated seed matrix is not an object")
        return payload


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

_CELL_IDENTITY_FIELDS = {
    "cell_id",
    "scene_family",
    "seed_index",
    "candidate_seed",
    "profile_id",
    "matched_control_profile_id",
    "generation_status",
}
_CELL_ADMISSION_FIELDS = {
    "frame_metrics",
    "admission_checks",
    "admission_status",
    "rejection_reasons",
}
_FAILED_CELL_FIELDS = (
    _CELL_IDENTITY_FIELDS
    | _CELL_ADMISSION_FIELDS
    | {
        "failure_type",
        "failure_message",
    }
)
_SUCCESS_CELL_FIELDS = (
    _CELL_IDENTITY_FIELDS
    | _CELL_ADMISSION_FIELDS
    | {
        "episode_seed",
        "appearance_profile_sha256",
        "appearance_instance_sha256",
        "appearance_instance",
        "evaluation_seed_registry_sha256",
        "appearance_assignment_schedule_source",
        "renderer_provenance",
        "rgb_logical_sha256",
        "scene_content_sha256",
        "analytic_transport_sha256",
        "oriented_boundary_sha256",
        "visibility_event_sha256",
        "ecological_label_sha256",
        "occlusion_sha256",
        "action_sha256",
        "camera_trajectory_sha256",
        "geometry_sha256",
        "opaque_remapping_sha256",
        "depth_logical_sha256",
        "segmentation_logical_sha256",
        "determinism_pass",
        "semantic_surface_labels",
        "source_texture_diagnostics",
        "evidence",
    }
)
_MATCHED_CONTROL_FAILURE_FIELDS = {
    "matched_control_failure_type",
    "matched_control_failure_message",
}
_ADMISSION_CHECK_FIELDS = {
    "structural_invariance",
    "portable_analytic_identity_equality",
    "ecological_label_equality",
    "depth_segmentation_invariance",
    "determinism",
    "material_rgb_change",
    "controlled_surface_exposure",
    "textured_surface_variation",
}
_FRAME_METRIC_FIELDS = {
    "controlled_pixel_count",
    "changed_controlled_pixel_fraction",
    "normalized_controlled_rgb_mad",
    "material_change_pass",
    "controlled_surface_exposure_pass",
    "textured_surface_variation_pass",
    "controlled_boundary_normalized_rgb_contrast",
    "surface_diagnostics",
}
_SURFACE_DIAGNOSTIC_FIELDS = {
    "opaque_surface_label",
    "pixel_count",
    "rgb_mean",
    "rgb_covariance",
    "luminance_mean",
    "luminance_standard_deviation",
    "luminance_percentiles",
    "clipped_fraction",
    "mean_saturation",
    "high_saturation_fraction",
}
_SOURCE_TEXTURE_DIAGNOSTIC_FIELDS = {
    "semantic_surface_name",
    "source_texture_logical_sha256",
    "source_luminance_standard_deviation",
    "dominant_spectrum_index",
    "high_frequency_power_fraction",
}

_PORTABLE_METRIC_DECIMAL_PLACES = 12


def _portable_metric_value(value: Any) -> Any:
    """Canonicalize derived floats beyond the precision used by benchmark thresholds."""

    if type(value) is float:
        return float(round(value, _PORTABLE_METRIC_DECIMAL_PLACES))
    if type(value) is list:
        return [_portable_metric_value(item) for item in value]
    if type(value) is dict:
        return {key: _portable_metric_value(item) for key, item in value.items()}
    return value


def _canonical_dominant_spectrum_index(spectrum: Any, row_count: int) -> list[int]:
    """Choose one sign-normalized index from numerically tied real-FFT peaks."""

    maximum = float(np.max(spectrum))
    tolerance = max(abs(maximum) * 1e-12, 1e-24)
    candidates = np.argwhere(np.abs(spectrum - maximum) <= tolerance)
    if candidates.size == 0:
        raise AppearanceAuditError("source texture spectrum has no dominant index")
    row, column = min(
        (min(int(index[0]), row_count - int(index[0])), int(index[1])) for index in candidates
    )
    return [row, column]


class _PacketArtifactRegistry:
    """Require every declared packet artifact to be one uniquely owned regular file."""

    def __init__(self, root: Path) -> None:
        try:
            root_stat = root.lstat()
            self.root = root.resolve(strict=True)
        except OSError as error:
            raise AppearanceAuditError("candidate packet root does not exist") from error
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        if (
            stat.S_ISLNK(root_stat.st_mode)
            or bool(getattr(root_stat, "st_file_attributes", 0) & reparse_flag)
            or not stat.S_ISDIR(root_stat.st_mode)
        ):
            raise AppearanceAuditError("candidate packet root must be a non-link directory")
        self.roles: set[str] = set()
        self.paths: set[str] = set()
        self.resolved_paths: set[Path] = set()
        self.file_identities: set[tuple[int, int]] = set()

    @contextmanager
    def claim(
        self,
        relative_path: str,
        role: str,
        *,
        expected_file_sha256: str | None = None,
        expected_byte_count: int | None = None,
    ) -> Iterator[OwnedRegularFile]:
        if role in self.roles:
            raise AppearanceAuditError(f"duplicate packet artifact role: {role}")
        if relative_path in self.paths:
            raise AppearanceAuditError(f"duplicate packet artifact path: {relative_path}")
        try:
            owned_context = open_owned_regular_file(self.root, relative_path)
            with owned_context as owned:
                identity = (owned.device, owned.inode)
                if owned.path in self.resolved_paths or identity in self.file_identities:
                    raise AppearanceAuditError(
                        f"packet artifact aliases another role: {relative_path}"
                    )
                if expected_byte_count is not None and owned.byte_count != expected_byte_count:
                    raise AppearanceAuditError(
                        f"packet artifact byte count mismatch: {relative_path}"
                    )
                if (
                    expected_file_sha256 is not None
                    and sha256_open_file(owned) != expected_file_sha256
                ):
                    raise AppearanceAuditError(f"packet artifact hash mismatch: {relative_path}")
                self.roles.add(role)
                self.paths.add(relative_path)
                self.resolved_paths.add(owned.path)
                self.file_identities.add(identity)
                yield owned
        except UnsafeOwnedFileError as error:
            raise AppearanceAuditError(str(error)) from error

    def assert_exact_tree(self) -> None:
        """Reject every unclaimed path, link, special file, and aliased identity."""

        try:
            validate_exact_owned_file_tree(self.root, self.paths)
        except UnsafeOwnedFileError as error:
            raise AppearanceAuditError(str(error)) from error


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


def _require_json_type(value: Any, expected: type[Any], field: str) -> None:
    if type(value) is not expected:
        raise AppearanceAuditError(f"candidate audit cell field has noncanonical type: {field}")


def _require_string_list(value: Any, field: str, length: int | None = None) -> None:
    if type(value) is not list or (length is not None and len(value) != length):
        raise AppearanceAuditError(f"candidate audit cell field is malformed: {field}")
    if any(type(item) is not str for item in value):
        raise AppearanceAuditError(f"candidate audit cell field has noncanonical type: {field}")


def _require_float_list(value: Any, field: str, length: int) -> None:
    if (
        type(value) is not list
        or len(value) != length
        or any(type(item) is not float for item in value)
    ):
        raise AppearanceAuditError(f"candidate audit cell field has noncanonical type: {field}")


def _validate_admission_checks(value: Any) -> None:
    if type(value) is not dict or set(value) != _ADMISSION_CHECK_FIELDS:
        raise AppearanceAuditError("candidate audit admission-check schema is not strict")
    if any(type(item) is not bool for item in value.values()):
        raise AppearanceAuditError("candidate audit admission-check types are not canonical")


def _validate_frame_metrics(value: Any, *, empty: bool) -> None:
    if empty:
        if type(value) is not dict or value:
            raise AppearanceAuditError("candidate audit empty frame-metric schema is not strict")
        return
    if type(value) is not dict or set(value) != {"before", "after"}:
        raise AppearanceAuditError("candidate audit frame-metric schema is not strict")
    for frame_name in ("before", "after"):
        metrics = value[frame_name]
        if type(metrics) is not dict or set(metrics) != _FRAME_METRIC_FIELDS:
            raise AppearanceAuditError("candidate audit per-frame metric schema is not strict")
        _require_json_type(
            metrics["controlled_pixel_count"],
            int,
            f"frame_metrics.{frame_name}.controlled_pixel_count",
        )
        for field in (
            "changed_controlled_pixel_fraction",
            "normalized_controlled_rgb_mad",
            "controlled_boundary_normalized_rgb_contrast",
        ):
            _require_json_type(metrics[field], float, f"frame_metrics.{frame_name}.{field}")
        for field in (
            "material_change_pass",
            "controlled_surface_exposure_pass",
            "textured_surface_variation_pass",
        ):
            _require_json_type(metrics[field], bool, f"frame_metrics.{frame_name}.{field}")
        surfaces = metrics["surface_diagnostics"]
        if type(surfaces) is not list:
            raise AppearanceAuditError("candidate audit surface diagnostics must be a list")
        for surface in surfaces:
            if type(surface) is not dict:
                raise AppearanceAuditError("candidate audit surface diagnostic must be an object")
            fields = frozenset(surface)
            optional_ratio = "rendered_to_source_luminance_std_ratio"
            if fields not in {
                frozenset(_SURFACE_DIAGNOSTIC_FIELDS),
                frozenset(_SURFACE_DIAGNOSTIC_FIELDS | {optional_ratio}),
            }:
                raise AppearanceAuditError(
                    "candidate audit surface-diagnostic schema is not strict"
                )
            _require_json_type(surface["opaque_surface_label"], int, "opaque_surface_label")
            _require_json_type(surface["pixel_count"], int, "pixel_count")
            _require_float_list(surface["rgb_mean"], "rgb_mean", 3)
            covariance = surface["rgb_covariance"]
            if type(covariance) is not list or len(covariance) != 3:
                raise AppearanceAuditError("candidate audit RGB covariance is malformed")
            for row in covariance:
                _require_float_list(row, "rgb_covariance", 3)
            _require_float_list(surface["luminance_percentiles"], "luminance_percentiles", 5)
            for field in (
                "luminance_mean",
                "luminance_standard_deviation",
                "clipped_fraction",
                "mean_saturation",
                "high_saturation_fraction",
            ):
                _require_json_type(surface[field], float, field)
            if optional_ratio in surface:
                _require_json_type(surface[optional_ratio], float, optional_ratio)


def _validate_source_texture_diagnostics(value: Any) -> None:
    if type(value) is not list:
        raise AppearanceAuditError("candidate audit source-texture diagnostics must be a list")
    for item in value:
        if type(item) is not dict or set(item) != _SOURCE_TEXTURE_DIAGNOSTIC_FIELDS:
            raise AppearanceAuditError(
                "candidate audit source-texture diagnostic schema is not strict"
            )
        _require_json_type(item["semantic_surface_name"], str, "semantic_surface_name")
        _require_json_type(
            item["source_texture_logical_sha256"],
            str,
            "source_texture_logical_sha256",
        )
        _require_json_type(
            item["source_luminance_standard_deviation"],
            float,
            "source_luminance_standard_deviation",
        )
        dominant = item["dominant_spectrum_index"]
        if (
            type(dominant) is not list
            or len(dominant) != 2
            or any(type(index) is not int for index in dominant)
        ):
            raise AppearanceAuditError("candidate audit dominant-spectrum index is malformed")
        _require_json_type(
            item["high_frequency_power_fraction"],
            float,
            "high_frequency_power_fraction",
        )


def _validate_evidence_schema(cell: dict[str, Any]) -> None:
    evidence = cell["evidence"]
    if type(evidence) is not dict or set(evidence) != {"before", "after"}:
        raise AppearanceAuditError(f"audit evidence frames are not strict: {cell['cell_id']}")
    for frame in ("before", "after"):
        if type(evidence[frame]) is not dict or set(evidence[frame]) != {
            "rgb",
            "depth",
            "segmentation",
        }:
            raise AppearanceAuditError(f"audit evidence roles are not strict: {cell['cell_id']}")
        for role in ("rgb", "depth", "segmentation"):
            record = _evidence_record(cell, frame, role)
            for field in (
                "path",
                "media_type",
                "dtype",
                "file_sha256",
                "logical_sha256",
            ):
                _require_json_type(record[field], str, f"evidence.{frame}.{role}.{field}")
            _require_json_type(record["byte_count"], int, f"evidence.{frame}.{role}.byte_count")
            shape = record["shape"]
            if type(shape) is not list or any(type(dimension) is not int for dimension in shape):
                raise AppearanceAuditError("candidate audit evidence shape is not canonical")


def _validate_cell_schema(cell: dict[str, Any]) -> None:
    status = cell.get("generation_status")
    if type(status) is not str or status not in {"success", "failed"}:
        raise AppearanceAuditError("candidate audit generation status is invalid")
    matched_fields = set(cell) & _MATCHED_CONTROL_FAILURE_FIELDS
    if matched_fields and matched_fields != _MATCHED_CONTROL_FAILURE_FIELDS:
        raise AppearanceAuditError("candidate audit matched-control failure schema is not strict")
    expected_fields = _FAILED_CELL_FIELDS if status == "failed" else _SUCCESS_CELL_FIELDS
    if matched_fields:
        if status != "success":
            raise AppearanceAuditError("failed candidate cell has matched-control failure fields")
        expected_fields = expected_fields | _MATCHED_CONTROL_FAILURE_FIELDS
    if set(cell) != expected_fields:
        raise AppearanceAuditError("candidate audit cell schema is not strict")

    for field in (
        "cell_id",
        "scene_family",
        "profile_id",
        "matched_control_profile_id",
        "generation_status",
        "admission_status",
    ):
        _require_json_type(cell[field], str, field)
    _require_json_type(cell["seed_index"], int, "seed_index")
    _require_json_type(cell["candidate_seed"], int, "candidate_seed")
    _require_string_list(cell["rejection_reasons"], "rejection_reasons")
    if cell["admission_status"] not in {"admitted", "rejected"}:
        raise AppearanceAuditError("candidate audit admission status is invalid")
    _validate_admission_checks(cell["admission_checks"])

    if status == "failed":
        _require_json_type(cell["failure_type"], str, "failure_type")
        _require_json_type(cell["failure_message"], str, "failure_message")
        _validate_frame_metrics(cell["frame_metrics"], empty=True)
        return

    _require_json_type(cell["episode_seed"], int, "episode_seed")
    for field in (
        "appearance_profile_sha256",
        "appearance_instance_sha256",
        "evaluation_seed_registry_sha256",
        "appearance_assignment_schedule_source",
        "scene_content_sha256",
        "analytic_transport_sha256",
        "oriented_boundary_sha256",
        "visibility_event_sha256",
        "ecological_label_sha256",
        "occlusion_sha256",
        "action_sha256",
        "camera_trajectory_sha256",
        "geometry_sha256",
        "opaque_remapping_sha256",
    ):
        _require_json_type(cell[field], str, field)
    for field in (
        "rgb_logical_sha256",
        "depth_logical_sha256",
        "segmentation_logical_sha256",
    ):
        _require_string_list(cell[field], field, 2)
    _require_json_type(cell["determinism_pass"], bool, "determinism_pass")
    try:
        AppearanceInstanceRecord.model_validate_json(
            canonical_json_bytes(cell["appearance_instance"])
        )
        RendererProvenance.model_validate_json(canonical_json_bytes(cell["renderer_provenance"]))
    except Exception as error:
        raise AppearanceAuditError("candidate audit typed nested record is invalid") from error
    surface_names = (
        {"support_surface", "occluding_surface", "background_surface"}
        if cell["scene_family"] == "single_occluder"
        else {
            "corridor_floor",
            "corridor_left_surface",
            "corridor_right_surface",
            "corridor_end_surface",
        }
    )
    labels = cell["semantic_surface_labels"]
    if (
        type(labels) is not dict
        or set(labels) != surface_names
        or any(type(name) is not str or type(label) is not int for name, label in labels.items())
    ):
        raise AppearanceAuditError("candidate audit semantic-surface labels are not canonical")
    _validate_source_texture_diagnostics(cell["source_texture_diagnostics"])
    _validate_evidence_schema(cell)
    _validate_frame_metrics(cell["frame_metrics"], empty=bool(matched_fields))
    if matched_fields:
        for field in _MATCHED_CONTROL_FAILURE_FIELDS:
            _require_json_type(cell[field], str, field)


def _profile_config(
    config: BenchmarkConfig,
    profile: AppearanceProfile,
    candidate_seed: int,
    registry_version: str = APPEARANCE_REGISTRY_VERSION,
) -> BenchmarkConfig:
    return type(config).model_validate(
        {
            **config.model_dump(mode="python"),
            "schema_version": (
                "0.1.0-dev.5"
                if registry_version == "appearance_candidate_registry_v2"
                else "0.1.0-dev.4"
            ),
            "seed": candidate_seed,
            "appearance": {
                "registry_version": registry_version,
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


def _retain_source_dataset(
    dataset: Path,
    manifest: DatasetManifest,
    packet_root: Path,
    cell_id: str,
) -> dict[str, Any]:
    """Retain the validated generator output needed for independent identity reconstruction."""

    destination = packet_root / "source_evidence" / cell_id
    shutil.copytree(dataset, destination)
    files = sorted(path for path in destination.rglob("*") if path.is_file())
    return {
        "schema_version": "appearance_benchmark_source_dataset_evidence_v1",
        "dataset_path": destination.relative_to(packet_root).as_posix(),
        "manifest_file_sha256": sha256_file(destination / "manifest.json"),
        "dataset_logical_sha256": manifest.dataset_logical_sha256,
        "file_count": len(files),
        "file_manifest": [
            {
                "path": path.relative_to(destination).as_posix(),
                "file_sha256": sha256_file(path),
                "byte_count": path.stat().st_size,
            }
            for path in files
        ],
    }


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


@contextmanager
def _open_packet_artifact(
    packet_root: Path,
    relative_path: str,
    role: str,
    expected_file_sha256: str,
    expected_byte_count: int,
    artifact_registry: _PacketArtifactRegistry | None,
) -> Iterator[OwnedRegularFile]:
    if artifact_registry is not None:
        with artifact_registry.claim(
            relative_path,
            role,
            expected_file_sha256=expected_file_sha256,
            expected_byte_count=expected_byte_count,
        ) as owned:
            yield owned
        return
    try:
        with open_owned_regular_file(packet_root, relative_path) as owned:
            if owned.byte_count != expected_byte_count:
                raise AppearanceAuditError(f"packet artifact byte count mismatch: {relative_path}")
            if sha256_open_file(owned) != expected_file_sha256:
                raise AppearanceAuditError(f"packet artifact hash mismatch: {relative_path}")
            yield owned
    except UnsafeOwnedFileError as error:
        raise AppearanceAuditError(str(error)) from error


def _load_frame(
    packet_root: Path,
    cell: dict[str, Any],
    frame: str,
    artifact_registry: _PacketArtifactRegistry | None = None,
    frame_cache: dict[tuple[str, str], tuple[Any, Any, Any]] | None = None,
) -> tuple[Any, Any, Any]:
    cache_key = (cell["cell_id"], frame)
    if frame_cache is not None and cache_key in frame_cache:
        return frame_cache[cache_key]
    arrays: dict[str, Any] = {}
    for role in ("rgb", "depth", "segmentation"):
        record = _evidence_record(cell, frame, role)
        expected_media_type = "image/png" if role == "rgb" else "application/x-npy"
        if record["media_type"] != expected_media_type:
            raise AppearanceAuditError(f"audit evidence media type mismatch: {role}")
        with _open_packet_artifact(
            packet_root,
            record["path"],
            f"evidence:{cell['cell_id']}:{frame}:{role}",
            record["file_sha256"],
            record["byte_count"],
            artifact_registry,
        ) as owned:
            try:
                if role == "rgb":
                    with Image.open(BytesIO(owned.payload)) as image:
                        if image.mode != "RGB":
                            raise AppearanceAuditError("audit RGB evidence must use RGB mode")
                        array = np.asarray(image, dtype=np.uint8).copy()
                else:
                    array = np.load(BytesIO(owned.payload), allow_pickle=False)
            except (OSError, ValueError) as error:
                raise AppearanceAuditError("audit evidence cannot be decoded") from error
            if str(array.dtype) != record["dtype"] or list(array.shape) != record["shape"]:
                raise AppearanceAuditError(f"audit evidence shape or dtype mismatch: {role}")
            if logical_array_hash(array) != record["logical_sha256"]:
                raise AppearanceAuditError(f"audit evidence logical hash mismatch: {role}")
            arrays[role] = array
    result = arrays["rgb"], arrays["depth"], arrays["segmentation"]
    if frame_cache is not None:
        frame_cache[cache_key] = result
    return result


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
    return cast(
        dict[str, Any],
        _portable_metric_value(
            {
                "controlled_pixel_count": controlled_count,
                "changed_controlled_pixel_fraction": fraction,
                "normalized_controlled_rgb_mad": mad,
                "material_change_pass": (
                    fraction
                    >= profile.non_degeneracy_thresholds.changed_controlled_pixel_fraction_minimum
                    and mad
                    >= profile.non_degeneracy_thresholds.normalized_controlled_rgb_mad_minimum
                ),
                "controlled_surface_exposure_pass": exposure_pass,
                "textured_surface_variation_pass": texture_pass,
                "controlled_boundary_normalized_rgb_contrast": boundary_contrast,
                "surface_diagnostics": surface_metrics,
            }
        ),
    )


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
        dominant = _canonical_dominant_spectrum_index(spectrum, luminance.shape[0])
        row_frequency = np.fft.fftfreq(luminance.shape[0])[:, None]
        column_frequency = np.fft.rfftfreq(luminance.shape[1])[None, :]
        radius = np.sqrt(row_frequency**2 + column_frequency**2)
        total_power = float(np.sum(spectrum))
        high_frequency_power_fraction = (
            float(np.sum(spectrum[radius >= 0.25]) / total_power) if total_power > 0.0 else 0.0
        )
        diagnostics.append(
            _portable_metric_value(
                {
                    "semantic_surface_name": record.semantic_surface_name,
                    "source_texture_logical_sha256": record.source_texture_logical_sha256,
                    "source_luminance_standard_deviation": float(np.std(luminance)),
                    "dominant_spectrum_index": dominant,
                    "high_frequency_power_fraction": high_frequency_power_fraction,
                }
            )
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
    frame_cache: dict[tuple[str, str], tuple[Any, Any, Any]] | None = None,
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
        rgb, depth, segmentation = _load_frame(
            packet_root,
            cell,
            frame_name,
            artifact_registry,
            frame_cache,
        )
        control_rgb, control_depth, control_segmentation = _load_frame(
            packet_root,
            control,
            frame_name,
            artifact_registry,
            frame_cache,
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
    registry: AppearanceRegistryType,
    seeds: SeedRegistryType,
    *,
    cell_id_prefix: str | None = None,
    retain_source_evidence: bool = False,
) -> dict[str, Any]:
    scene = config.scene_family.value
    base_cell_id = _cell_id(scene, profile.profile_id, seed_index)
    cell_id = f"{cell_id_prefix}--{base_cell_id}" if cell_id_prefix else base_cell_id
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
        selected = _profile_config(config, profile, candidate_seed, registry.registry_version)
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
        source_evidence = (
            _retain_source_dataset(dataset, manifest, packet_root, cell_id)
            if retain_source_evidence
            else None
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
        if source_evidence is not None:
            cell["source_evidence"] = source_evidence
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
    frame_cache: dict[tuple[str, str], tuple[Any, Any, Any]] | None = None,
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
        before, _, segmentation = _load_frame(
            packet_root,
            cell,
            "before",
            artifact_registry,
            frame_cache,
        )
        after, _, _ = _load_frame(
            packet_root,
            cell,
            "after",
            artifact_registry,
            frame_cache,
        )
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
    frame_cache: dict[tuple[str, str], tuple[Any, Any, Any]],
) -> None:
    if type(manifest) is not dict or set(manifest) != {"schema_version", "sheets"}:
        raise AppearanceAuditError("contact-sheet manifest is not strict")
    if (
        type(manifest["schema_version"]) is not str
        or manifest["schema_version"] != CONTACT_SHEET_MANIFEST_VERSION
    ):
        raise AppearanceAuditError("contact-sheet manifest version is unsupported")
    sheets = manifest["sheets"]
    if type(sheets) is not list or len(sheets) != len(SCENE_FAMILIES):
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
        if type(record) is not dict or set(record) != expected_fields:
            raise AppearanceAuditError("contact-sheet record is not strict")
        string_fields = (
            "scene_family",
            "path",
            "media_type",
            "mode",
            "dtype",
            "logical_sha256",
            "file_sha256",
        )
        dimensions = record["dimensions"]
        shape = record["shape"]
        if (
            any(type(record[field]) is not str for field in string_fields)
            or type(record["seed_index"]) is not int
            or type(record["byte_count"]) is not int
            or record["byte_count"] <= 0
            or type(dimensions) is not list
            or len(dimensions) != 2
            or any(type(value) is not int or value <= 0 for value in dimensions)
            or type(shape) is not list
            or len(shape) != 3
            or any(type(value) is not int or value <= 0 for value in shape)
        ):
            raise AppearanceAuditError("contact-sheet record types are not canonical")
        expected_path = f"representative_contact_sheets/{scene}_representative_seed_0.png"
        if (
            record["scene_family"] != scene
            or record["seed_index"] != 0
            or record["path"] != expected_path
            or record["media_type"] != "image/png"
            or record["mode"] != "RGB"
        ):
            raise AppearanceAuditError("contact-sheet canonical metadata is inconsistent")
        with artifact_registry.claim(
            expected_path,
            f"contact-sheet:{scene}",
            expected_file_sha256=record["file_sha256"],
            expected_byte_count=record["byte_count"],
        ) as owned:
            try:
                with Image.open(BytesIO(owned.payload)) as image:
                    if image.mode != "RGB":
                        raise AppearanceAuditError("contact sheet must use RGB mode")
                    actual = np.asarray(image, dtype=np.uint8).copy()
                    dimensions = list(image.size)
            except OSError as error:
                raise AppearanceAuditError("contact sheet cannot be decoded") from error
        expected = np.asarray(
            _contact_sheet_image(
                packet_root,
                cells,
                scene,
                artifact_registry,
                frame_cache,
            ),
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
    registry: AppearanceRegistryType,
    seeds: SeedRegistryType,
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


def _validate_appearance_audit_evidence(
    packet_root: Path,
    *,
    verify_current_source_provenance: bool = True,
) -> _ValidatedAppearanceAudit:
    """Validate one packet snapshot and retain its exact seed-matrix bytes."""

    artifact_registry = _PacketArtifactRegistry(packet_root)
    with artifact_registry.claim("candidate_packet.json", "candidate-packet") as owned:
        packet_payload = owned.payload
        try:
            packet = json.loads(packet_payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise AppearanceAuditError("candidate packet JSON is invalid") from error
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
    try:
        with artifact_registry.claim(
            "appearance_registry_snapshot.json",
            "appearance-registry-snapshot",
        ) as owned:
            registry = AppearanceRegistry.model_validate_json(owned.payload)
        with artifact_registry.claim(
            "seed_registry_snapshot.json",
            "seed-registry-snapshot",
        ) as owned:
            seeds = EvaluationSeedRegistry.model_validate_json(owned.payload)
    except Exception as error:
        raise AppearanceAuditError("candidate packet registry snapshot is invalid") from error
    report_payloads: dict[str, dict[str, Any]] = {}
    report_bytes: dict[str, bytes] = {}
    for name in report_names:
        with artifact_registry.claim(
            name,
            f"report:{name}",
            expected_file_sha256=report_hashes[name],
        ) as owned:
            raw = owned.payload
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise AppearanceAuditError(f"candidate packet report is invalid: {name}") from error
            if not isinstance(payload, dict):
                raise AppearanceAuditError(f"candidate packet report must be an object: {name}")
            report_bytes[name] = raw
            report_payloads[name] = payload
    validate_axis_isolation(registry)
    matrix = report_payloads["seed_matrix.json"]
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
        _validate_cell_schema(cell)
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

    by_key = {
        (cell["scene_family"], cell["profile_id"], cell["seed_index"]): cell for cell in cells
    }
    profiles = {profile.profile_id: profile for profile in registry.profiles}
    frame_cache: dict[tuple[str, str], tuple[Any, Any, Any]] = {}
    for cell in cells:
        if cell["generation_status"] == "success":
            for frame_name in ("before", "after"):
                _load_frame(
                    packet_root,
                    cell,
                    frame_name,
                    artifact_registry,
                    frame_cache,
                )

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
                arrays = _load_frame(
                    packet_root,
                    cell,
                    frame_name,
                    frame_cache=frame_cache,
                )
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
        _evaluate_cell(
            packet_root,
            recomputed,
            control,
            profile,
            frame_cache=frame_cache,
        )
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
        report_payloads["profile_summary.json"]
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
        report_payloads["negative_evidence.json"]
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
        frame_cache,
    )
    if _hash_json(_packet_logical_domain(packet)) != packet["packet_logical_root_sha256"]:
        raise AppearanceAuditError("candidate packet logical root mismatch")
    current_governing_hashes = {path: sha256_file(Path(path)) for path in GOVERNING_DOCUMENTS}
    if current_governing_hashes != packet["governing_document_hashes"]:
        raise AppearanceAuditError("governing-document hash mismatch")
    if (
        verify_current_source_provenance
        and collect_source_provenance(Path.cwd()).model_dump(mode="json")
        != packet["source_provenance"]
    ):
        raise AppearanceAuditError("packet source provenance is not truthful for this source tree")
    return _ValidatedAppearanceAudit(
        packet_payload=packet_payload,
        seed_matrix_payload=report_bytes["seed_matrix.json"],
    )


def validate_appearance_audit(
    packet_root: Path,
    *,
    verify_current_source_provenance: bool = True,
) -> dict[str, Any]:
    """Independently recompute matrix, metrics, balance, and packet roots."""

    evidence = _validate_appearance_audit_evidence(
        packet_root,
        verify_current_source_provenance=verify_current_source_provenance,
    )
    return evidence.packet()


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
    if not isinstance(registry, AppearanceRegistry) or not isinstance(
        seeds, EvaluationSeedRegistry
    ):
        raise AppearanceAuditError("canonical audit requires the v1 registry and design seeds")
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
