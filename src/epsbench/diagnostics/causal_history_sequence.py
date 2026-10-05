"""Bounded native-free causal sequence artifacts and permissioned reads.

Control-plane envelopes never enter a CausalState. No filesystem or native SDK
is needed: a later reviewed controller supplies the bounded artifact source/sink.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from epsbench.diagnostics.causal_history_core import (
    Array,
    Command,
    CompletedFlow,
    OpticalFrame,
    TemporalProjection,
    owned,
)
from epsbench.schema import Modality, ModalityPermissionSet

VERSION: Literal["causal-history-sequence-v1"] = "causal-history-sequence-v1"
CONFIG_SHA256 = "478fe170a50110118f44158e4a76dd12871c3adc801a8847f3dae2f3044fe7aa"
HEIGHT, WIDTH = 120, 160
# Admission ceiling: 2048 sparse boundary records, 512 bytes each. Full native diagnostics,
# manifests and partial-failure/forecast/audit reserve fit this logical envelope.
MAX_BOUNDARY_RECORDS = 2048
MAX_BOUNDARY_BYTES = MAX_BOUNDARY_RECORDS * 512
MAX_ARTIFACT_BYTES = 3 * 1024 * 1024
MAX_SEQUENCE_BYTES = 24 * 1024 * 1024
MAX_SIX_BYTES = 6 * MAX_SEQUENCE_BYTES
LOGICAL_OUTPUT_BYTES = 1024**3


def retention_estimate() -> dict[str, int]:
    """Prospective conservative logical arithmetic, not execution admission."""
    mib = 1024 * 1024
    raw = MAX_SIX_BYTES
    forecasts_audits = 32 * mib
    partial_failure = 64 * mib
    staging = raw + forecasts_audits + partial_failure
    # RetainedArchive uses 64 KiB chunks, base64 JSON and four-byte framing.
    # 16 MiB covers >1 KiB/chunk plus per-artifact records/manifests/journals.
    archive = 4 * ((staging + 2) // 3) + 16 * mib + 8 * mib
    return {
        "raw": raw,
        "forecasts_audits": forecasts_audits,
        "partial_failure": partial_failure,
        "staging": staging,
        "archive_with_framing_terminal": archive,
        "shared": mib,
        "total": staging + archive + mib,
    }


MEMBERS = {
    "chf-v1-p1-a": (1, 0, 202610061001, 4, 2, 3),
    "chf-v1-p1-b": (1, 1, 202610061002, 4, 2, 3),
    "chf-v1-p2-a": (2, 0, 202610061003, 5, 3, 4),
    "chf-v1-p2-b": (2, 1, 202610061004, 5, 3, 4),
    "chf-v1-p3-a": (3, 0, 202610061005, 3, 1, 2),
    "chf-v1-p3-b": (3, 1, 202610061006, 3, 1, 2),
}


def canonical(value: Any) -> bytes:
    # Preflight structured evidence before constructing its encoded payload.
    nodes = 0
    text = 0

    def visit(item: Any, depth: int) -> None:
        nonlocal nodes, text
        nodes += 1
        if nodes > 100000 or depth > 64:
            raise ValueError("JSON structure ceiling exceeded")
        if isinstance(item, str):
            if len(item) > 65536:
                raise ValueError("JSON string ceiling exceeded")
            text += len(item.encode())
            if text > MAX_ARTIFACT_BYTES:
                raise ValueError("JSON text ceiling exceeded")
        elif isinstance(item, dict):
            if len(item) > 4096:
                raise ValueError("JSON mapping ceiling exceeded")
            for key, child in item.items():
                visit(key, depth + 1)
                visit(child, depth + 1)
        elif isinstance(item, (tuple, list)):
            if len(item) > 65536:
                raise ValueError("JSON list ceiling exceeded")
            for child in item:
                visit(child, depth + 1)

    visit(value, 0)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(payload) > MAX_ARTIFACT_BYTES:
        raise ValueError("JSON byte ceiling exceeded")
    return payload


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def parse(payload: bytes, ceiling: int = MAX_ARTIFACT_BYTES) -> Any:
    if type(payload) is not bytes or len(payload) > ceiling:
        raise ValueError("artifact byte ceiling exceeded")
    return json.loads(payload, object_pairs_hook=_pairs, parse_constant=_invalid_constant)


def _invalid_constant(value: str) -> None:
    raise ValueError("nonfinite JSON number: " + value)


def candidate(payload: bytes, member: str) -> tuple[dict[str, Any], dict[str, Any], int]:
    config = parse(payload, 16384)
    if digest(canonical(config)) != CONFIG_SHA256 or member not in MEMBERS:
        raise ValueError("exact fixed candidate/member required")
    pair_no, ordinal, seed, count, decision, target = MEMBERS[member]
    pair = config["pairs"][pair_no - 1]
    if (
        pair["pair"],
        pair["identities"][ordinal],
        pair["seeds"][ordinal],
        len(pair["poses"]),
        pair["decision"],
        pair["target"],
    ) != (pair_no, member, seed, count, decision, target):
        raise ValueError("fixed membership/order/timing differs")
    return config, pair, ordinal


def commands(config: dict[str, Any], member: str) -> tuple[Command, ...]:
    _, pair, _ = candidate(canonical(config), member)
    # JSON decimal interpretation is explicit, never arbitrary float display.
    positions = [Fraction(str(v)) for v in pair["poses"]]
    return tuple(Command(b - a) for a, b in pairwise(positions))


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class ExactCommand(Record):
    lateral: tuple[int, int]
    forward: tuple[int, int]
    yaw: tuple[int, int]

    def command(self) -> Command:
        values = []
        for numerator, denominator in (self.lateral, self.forward, self.yaw):
            if denominator <= 0:
                raise ValueError("positive canonical denominator required")
            value = Fraction(numerator, denominator)
            if (value.numerator, value.denominator) != (numerator, denominator):
                raise ValueError("reduced exact command required")
            values.append(value)
        return Command(*values)

    @classmethod
    def from_command(cls, c: Command) -> ExactCommand:
        return cls(
            **{
                k: (getattr(c, k).numerator, getattr(c, k).denominator)
                for k in ("lateral", "forward", "yaw")
            }
        )


class Artifact(Record):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int = Field(ge=1, le=MAX_ARTIFACT_BYTES)
    kind: Literal["json", "array"]
    dtype: Literal["|u1", "|b1", "<i4", "<f4"] | None
    shape: tuple[int, ...]
    modality: Modality
    end: int = Field(ge=0, le=4)


class FrameRef(Record):
    index: int
    identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    optical: str
    segmentation: str
    rgb: str
    privileged: str
    arrays: tuple[str, ...]


class FlowRef(Record):
    index: int
    identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_frame: str
    target_frame: str
    command: ExactCommand
    admitted: bool
    vectors: str
    validity: str
    reasons: str
    privileged: str
    arrays: tuple[str, ...]


class Envelope(Record):
    version: Literal["causal-history-sequence-v1"]
    member: str
    config: dict[str, Any]
    source_head: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_tree: str = Field(pattern=r"^[0-9a-f]{40}$")
    image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    renderer_identity: dict[str, Any]
    mapping_commitment: str = Field(pattern=r"^[0-9a-f]{64}$")
    sequence_privileged: str
    frames: tuple[FrameRef, ...]
    flows: tuple[FlowRef, ...]
    artifacts: tuple[Artifact, ...]
    logical_identity: str = Field(pattern=r"^[0-9a-f]{64}$")


def _identity(record: Record, artifacts: Mapping[str, Artifact]) -> str:
    obj = record.model_dump(mode="json", exclude={"identity"})
    paths = [
        v
        for k, v in obj.items()
        if k in ("optical", "segmentation", "rgb", "privileged", "vectors", "validity", "reasons")
    ]
    paths.extend(obj["arrays"])
    return digest(
        canonical({"record": obj, "content": [artifacts[p].model_dump(mode="json") for p in paths]})
    )


def logical_identity(envelope: Envelope) -> str:
    return digest(canonical(envelope.model_dump(mode="json", exclude={"logical_identity"})))


def validate_manifest(payload: bytes, paths: tuple[str, ...]) -> Envelope:
    parse(payload, 1024 * 1024)
    env = Envelope.model_validate_json(payload)
    config, pair, _ = candidate(canonical(env.config), env.member)
    count = len(pair["poses"])
    if len(env.frames) != count or len(env.flows) != count - 1 or not env.renderer_identity:
        raise ValueError("missing frames/transitions/renderer identity")
    if len(env.artifacts) > 128 or len(paths) != len(set(paths)):
        raise ValueError("duplicate paths or artifact count ceiling")
    artifacts = {a.path: a for a in env.artifacts}
    if len(artifacts) != len(env.artifacts) or set(paths) != set(artifacts):
        raise ValueError("extra/duplicate/missing artifacts")
    if sum(a.size for a in env.artifacts) + len(payload) > MAX_SEQUENCE_BYTES:
        raise ValueError("sequence byte ceiling exceeded")
    for a in env.artifacts:
        if re.fullmatch(r"[a-z0-9_-]+/[a-z0-9_-]+\.(json|bin)", a.path) is None:
            raise ValueError("unsafe artifact path")
        if a.kind == "json":
            if a.dtype is not None or a.shape:
                raise ValueError("JSON array metadata forbidden")
        elif (
            a.dtype is None
            or a.shape not in ((HEIGHT, WIDTH), (HEIGHT, WIDTH, 2), (HEIGHT, WIDTH, 3))
            or a.size != int(np.prod(a.shape)) * np.dtype(a.dtype).itemsize
        ):
            raise ValueError("array shape/dtype/byte contract differs")
    expected: set[str] = {env.sequence_privileged}

    def require(
        path: str, modality: Modality, end: int, dtype: str | None, shape: tuple[int, ...] = ()
    ) -> None:
        a = artifacts[path]
        if (a.modality, a.end, a.dtype, a.shape, a.kind) != (
            modality,
            end,
            dtype,
            shape,
            "json" if dtype is None else "array",
        ):
            raise ValueError("typed artifact binding differs")
        expected.add(path)

    require(env.sequence_privileged, Modality.PRIVILEGED_GENERATION_RECORDS, 0, None)
    for i, f in enumerate(env.frames):
        if f.index != i or f.identity != _identity(f, artifacts):
            raise ValueError("frame chronology/content identity differs")
        require(f.optical, Modality.ORIENTED_BOUNDARY_OWNERSHIP, i, None)
        require(f.segmentation, Modality.SURFACE_REGIONS, i, "<i4", (HEIGHT, WIDTH))
        require(f.rgb, Modality.RGB, i, "|u1", (HEIGHT, WIDTH, 3))
        require(f.privileged, Modality.PRIVILEGED_GENERATION_RECORDS, i, None)
        if len(f.arrays) != 7 or len(set(f.arrays)) != 7:
            raise ValueError("complete paired/counterfactual arrays required")
        for p, dtype, shape in zip(
            f.arrays,
            ("<i4", "<f4", "|u1", "<f4", "<i4", "<i4", "<i4"),
            (
                (HEIGHT, WIDTH),
                (HEIGHT, WIDTH),
                (HEIGHT, WIDTH, 3),
                (HEIGHT, WIDTH),
                (HEIGHT, WIDTH),
                (HEIGHT, WIDTH),
                (HEIGHT, WIDTH),
            ),
            strict=True,
        ):
            require(p, Modality.PRIVILEGED_GENERATION_RECORDS, i, dtype, shape)
    for i, (transition, c) in enumerate(zip(env.flows, commands(config, env.member), strict=True)):
        if (
            transition.index != i
            or transition.command.command() != c
            or transition.source_frame != env.frames[i].identity
            or transition.target_frame != env.frames[i + 1].identity
            or transition.identity != _identity(transition, artifacts)
        ):
            raise ValueError("shared frame join/action identity differs")
        for path, dtype, shape in (
            (transition.vectors, "<i4", (HEIGHT, WIDTH, 2)),
            (transition.validity, "|u1", (HEIGHT, WIDTH)),
            (transition.reasons, "|u1", (HEIGHT, WIDTH)),
        ):
            require(path, Modality.ANALYTIC_OPTICAL_TRANSPORT, i + 1, dtype, shape)
        require(transition.privileged, Modality.PRIVILEGED_GENERATION_RECORDS, i + 1, None)
        if len(transition.arrays) != 12 or len(set(transition.arrays)) != 12:
            raise ValueError("complete analytic arrays required")
        for p, dtype, shape in zip(
            transition.arrays,
            ("<i4", "<i4", "|b1", "|b1", "<i4", "<i4", "|u1", "|u1", "<i4", "<i4", "|b1", "|b1"),
            (
                (HEIGHT, WIDTH),
                (HEIGHT, WIDTH),
                (HEIGHT, WIDTH),
                (HEIGHT, WIDTH),
                (HEIGHT, WIDTH),
                (HEIGHT, WIDTH, 2),
                (HEIGHT, WIDTH),
                (HEIGHT, WIDTH),
                (HEIGHT, WIDTH),
                (HEIGHT, WIDTH, 2),
                (HEIGHT, WIDTH),
                (HEIGHT, WIDTH),
            ),
            strict=True,
        ):
            require(p, Modality.PRIVILEGED_GENERATION_RECORDS, i + 1, dtype, shape)
    if expected != set(artifacts) or logical_identity(env) != env.logical_identity:
        raise ValueError("unreferenced evidence or logical identity differs")
    return env


class OpticalPayload(Record):
    identities: tuple[tuple[int, str], ...]
    boundaries: tuple[dict[str, Any], ...]


class CameraEvidence(Record):
    world_position: tuple[float, float, float]
    world_rotation_row_major: tuple[float, float, float, float, float, float, float, float, float]
    vertical_field_of_view_degrees: float


class StatisticsEvidence(Record):
    meanmass: float
    meaninertia: float
    meansize: float
    extent: float
    center: tuple[float, float, float]


class VisualEvidence(Record):
    znear: float
    zfar: float
    offsamples: int


class CompiledEvidence(Record):
    raw_geom_ids: dict[str, int]
    raw_geom_world_positions: dict[str, tuple[float, float, float]]
    raw_geom_compiled_sizes: dict[str, tuple[float, float, float]]
    raw_geom_types: dict[str, str]
    raw_geom_world_rotations_row_major: dict[str, tuple[float, ...]]
    camera_field_of_view_degrees: float
    camera_world_position: tuple[float, float, float]
    camera_world_rotation_row_major: tuple[float, ...]
    statistics: StatisticsEvidence
    visual: VisualEvidence


class FrameEvidence(Record):
    camera: dict[str, Any]
    compiled: dict[str, Any]
    paired_stable: dict[str, Any]
    scene_map: tuple[dict[str, Any], ...]
    near: float
    far: float
    orientation: str
    raw_boundaries: tuple[dict[str, Any], ...]
    attachment: dict[str, Any]

    @model_validator(mode="after")
    def complete(self) -> FrameEvidence:
        if (
            not all(
                (
                    self.camera,
                    self.compiled,
                    self.paired_stable,
                    self.scene_map,
                    self.attachment,
                    self.orientation,
                )
            )
            or not np.isfinite((self.near, self.far)).all()
            or not 0 < self.near < self.far
        ):
            raise ValueError("complete finite paired/camera/attachment evidence required")
        CameraEvidence.model_validate_json(canonical(self.camera))
        CompiledEvidence.model_validate_json(canonical(self.compiled))
        return self


class FlowEvidence(Record):
    admitted: bool
    source_mismatch_count: int
    target_mismatch_count: int
    reason_counts: tuple[int, ...]

    @model_validator(mode="after")
    def admission(self) -> FlowEvidence:
        if (
            self.source_mismatch_count < 0
            or self.target_mismatch_count < 0
            or self.admitted != (self.source_mismatch_count == self.target_mismatch_count == 0)
            or len(self.reason_counts) != 5
            or any(v < 0 for v in self.reason_counts)
            or sum(self.reason_counts) != HEIGHT * WIDTH
        ):
            raise ValueError("admission counts/domain differs")
        return self


def admit_flow(
    validity: Array,
    reasons: Array,
    analytic_source: Array,
    actual_source: Array,
    actual_target: Array,
    projected_samples: Array,
) -> tuple[FlowEvidence, Array, Array]:
    """Original validity is unchanged; mismatches deny ecological flow admission."""
    if any(
        not isinstance(a, np.ndarray)
        for a in (
            validity,
            reasons,
            analytic_source,
            actual_source,
            actual_target,
            projected_samples,
        )
    ):
        raise ValueError("native/analytic admission requires typed arrays")
    shape = (HEIGHT, WIDTH)
    if (
        any(
            a.shape != shape or a.dtype != np.int32
            for a in (analytic_source, actual_source, actual_target)
        )
        or projected_samples.shape != (*shape, 2)
        or projected_samples.dtype != np.int32
        or validity.shape != shape
        or validity.dtype != np.uint8
        or reasons.shape != shape
        or reasons.dtype != np.uint8
        or np.any(validity > 1)
        or np.any(reasons > 4)
        or not np.array_equal(validity == 1, reasons == 0)
    ):
        raise ValueError("native/analytic admission array contract differs")
    valid = validity == 1
    source_mismatch = valid & ((analytic_source < 0) | (analytic_source != actual_source))
    rows, columns = projected_samples[..., 1], projected_samples[..., 0]
    inside = (rows >= 0) & (rows < HEIGHT) & (columns >= 0) & (columns < WIDTH)
    sampled = actual_target[np.clip(rows, 0, HEIGHT - 1), np.clip(columns, 0, WIDTH - 1)]
    target_mismatch = valid & (~inside | (sampled != analytic_source))
    source_count, target_count = int(source_mismatch.sum()), int(target_mismatch.sum())
    evidence = FlowEvidence(
        admitted=source_count == target_count == 0,
        source_mismatch_count=source_count,
        target_mismatch_count=target_count,
        reason_counts=tuple(int(np.count_nonzero(reasons == i)) for i in range(5)),
    )
    return evidence, source_mismatch, target_mismatch


class SequenceEvidence(Record):
    mapping: tuple[tuple[int, int, str], ...]
    compiled: dict[str, Any]
    xml_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def complete(self) -> SequenceEvidence:
        if (
            len(self.mapping) != 3
            or not self.compiled
            or len({r for r, _, _ in self.mapping}) != 3
            or any(r < 0 for r, _, _ in self.mapping)
            or {label for _, label, _ in self.mapping} != {1, 2, 3}
            or len({t for _, _, t in self.mapping}) != 3
            or any(re.fullmatch(r"surface-[0-9a-f]{16}", t) is None for _, _, t in self.mapping)
        ):
            raise ValueError("complete bijective privileged mapping/inventory required")
        CompiledEvidence.model_validate_json(canonical(self.compiled))
        return self


@dataclass(frozen=True)
class RetainedFrame:
    optical: OpticalFrame
    rgb: Array
    evidence: FrameEvidence
    arrays: tuple[Array, ...]


@dataclass(frozen=True)
class RetainedFlow:
    optical: CompletedFlow
    evidence: FlowEvidence
    arrays: tuple[Array, ...]


def encode_sequence(
    *,
    config: dict[str, Any],
    member: str,
    source_head: str,
    source_tree: str,
    image_digest: str,
    renderer_identity: dict[str, Any],
    sequence: SequenceEvidence,
    frames: tuple[RetainedFrame, ...],
    flows: tuple[RetainedFlow, ...],
) -> tuple[bytes, dict[str, bytes]]:
    """Bound sizes before copying/serializing; returns only deterministic artifact bytes."""
    _, pair, _ = candidate(canonical(config), member)
    if len(frames) != len(pair["poses"]) or len(flows) != len(frames) - 1:
        raise ValueError("exact sequence count required before encoding")
    artifacts: dict[str, bytes] = {}
    specs: dict[str, Artifact] = {}

    def add(path: str, value: Any, modality: Modality, end: int) -> str:
        dtype = None
        shape: tuple[int, ...] = ()
        if isinstance(value, np.ndarray):
            if value.dtype.str not in ("|u1", "|b1", "<i4", "<f4") or value.shape not in (
                (HEIGHT, WIDTH),
                (HEIGHT, WIDTH, 2),
                (HEIGHT, WIDTH, 3),
            ):
                raise ValueError("bounded nonobject raster required before copy")
            dtype, shape = value.dtype.str, value.shape
            payload = value.tobytes(order="C")
        else:
            payload = canonical(value)
        if path in artifacts or len(payload) > MAX_ARTIFACT_BYTES:
            raise ValueError("duplicate artifact or byte ceiling")
        if sum(map(len, artifacts.values())) + len(payload) > MAX_SEQUENCE_BYTES - 1024 * 1024:
            raise ValueError("sequence byte ceiling before append")
        artifacts[path] = payload
        specs[path] = Artifact(
            path=path,
            sha256=digest(payload),
            size=len(payload),
            kind="array" if dtype else "json",
            dtype=dtype,
            shape=shape,
            modality=modality,
            end=end,
        )
        return path

    priv = Modality.PRIVILEGED_GENERATION_RECORDS
    sequence_path = add("sequence/instrumentation.json", sequence.model_dump(mode="json"), priv, 0)
    frame_refs = []
    for i, f in enumerate(frames):
        if f.optical.sequence_index != i or len(f.arrays) != 7:
            raise ValueError("frame evidence/index differs")
        if len(f.optical.boundaries) > MAX_BOUNDARY_RECORDS or any(
            len(b) > 512 for b in f.optical.boundaries
        ):
            raise ValueError("boundary count/record ceiling before encoding")
        if len(f.evidence.raw_boundaries) > MAX_BOUNDARY_RECORDS or any(
            len(canonical(b)) > 1024 for b in f.evidence.raw_boundaries
        ):
            raise ValueError("raw boundary count/record ceiling before encoding")
        raw = f.arrays[0]
        if raw.dtype != np.int32 or raw.shape != (HEIGHT, WIDTH):
            raise ValueError("raw raster shape/dtype differs before remapping")
        remapped = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
        lookup = {r: (label, t) for r, label, t in sequence.mapping}
        if set(int(v) for v in np.unique(raw)) - {-1} - set(lookup):
            raise ValueError("raw raster outside committed inventory")
        for r, (label, _) in lookup.items():
            remapped[raw == r] = label
        identities = tuple(
            sorted((label, t) for r, (label, t) in lookup.items() if np.any(raw == r))
        )
        if (
            not np.array_equal(remapped, f.optical.segmentation)
            or identities != f.optical.identities
        ):
            raise ValueError("privileged mapping/visible-only projection differs")
        prefix = f"frame-{i}/"
        payload = OpticalPayload(
            identities=f.optical.identities,
            boundaries=tuple(parse(b, 512) for b in f.optical.boundaries),
        )
        record = FrameRef(
            index=i,
            identity="0" * 64,
            optical=add(
                prefix + "optical.json",
                payload.model_dump(mode="json"),
                Modality.ORIENTED_BOUNDARY_OWNERSHIP,
                i,
            ),
            segmentation=add(
                prefix + "segmentation.bin", f.optical.segmentation, Modality.SURFACE_REGIONS, i
            ),
            rgb=add(prefix + "rgb.bin", f.rgb, Modality.RGB, i),
            privileged=add(
                prefix + "instrumentation.json", f.evidence.model_dump(mode="json"), priv, i
            ),
            arrays=tuple(
                add(prefix + f"instrument-{j}.bin", a, priv, i) for j, a in enumerate(f.arrays)
            ),
        )
        frame_refs.append(record.model_copy(update={"identity": _identity(record, specs)}))
    flow_refs = []
    for i, retained_flow in enumerate(flows):
        if retained_flow.optical.source_index != i or len(retained_flow.arrays) != 12:
            raise ValueError("flow evidence/index differs")
        flow = retained_flow.optical
        # Validate every retained diagnostic before interpreting admission samples.
        arrays = retained_flow.arrays
        for array, dtype, shape in zip(
            arrays,
            (
                np.int32,
                np.int32,
                np.bool_,
                np.bool_,
                np.int32,
                np.int32,
                np.uint8,
                np.uint8,
                np.int32,
                np.int32,
                np.bool_,
                np.bool_,
            ),
            (
                (HEIGHT, WIDTH),
                (HEIGHT, WIDTH),
                (HEIGHT, WIDTH),
                (HEIGHT, WIDTH),
                (HEIGHT, WIDTH),
                (HEIGHT, WIDTH, 2),
                (HEIGHT, WIDTH),
                (HEIGHT, WIDTH),
                (HEIGHT, WIDTH),
                (HEIGHT, WIDTH, 2),
                (HEIGHT, WIDTH),
                (HEIGHT, WIDTH),
            ),
            strict=True,
        ):
            if not isinstance(array, np.ndarray) or array.dtype != dtype or array.shape != shape:
                raise ValueError("retained admission diagnostic shape/dtype differs")
        # The paired rasters below are the already encoded immutable frame bytes,
        # so admission binds the same endpoint content as the eventual manifest.
        actual_source = np.frombuffer(artifacts[frame_refs[i].arrays[0]], dtype=np.int32).reshape(
            HEIGHT, WIDTH
        )
        actual_target = np.frombuffer(
            artifacts[frame_refs[i + 1].arrays[0]], dtype=np.int32
        ).reshape(HEIGHT, WIDTH)
        recomputed, source_mismatch, target_mismatch = admit_flow(
            flow.validity, flow.reasons, arrays[0], actual_source, actual_target, arrays[9]
        )
        if (
            recomputed != retained_flow.evidence
            or not np.array_equal(source_mismatch, arrays[10])
            or not np.array_equal(target_mismatch, arrays[11])
        ):
            raise ValueError("retained admission evidence contradicts paired/analytic arrays")
        prefix = f"flow-{i}/"
        flow_record = FlowRef(
            index=i,
            identity="0" * 64,
            source_frame=frame_refs[i].identity,
            target_frame=frame_refs[i + 1].identity,
            command=ExactCommand.from_command(flow.command),
            admitted=recomputed.admitted,
            vectors=add(
                prefix + "vectors.bin", flow.vectors, Modality.ANALYTIC_OPTICAL_TRANSPORT, i + 1
            ),
            validity=add(
                prefix + "validity.bin", flow.validity, Modality.ANALYTIC_OPTICAL_TRANSPORT, i + 1
            ),
            reasons=add(
                prefix + "reasons.bin", flow.reasons, Modality.ANALYTIC_OPTICAL_TRANSPORT, i + 1
            ),
            privileged=add(
                prefix + "instrumentation.json",
                retained_flow.evidence.model_dump(mode="json"),
                priv,
                i + 1,
            ),
            arrays=tuple(
                add(prefix + f"instrument-{j}.bin", a, priv, i + 1)
                for j, a in enumerate(retained_flow.arrays)
            ),
        )
        flow_refs.append(flow_record.model_copy(update={"identity": _identity(flow_record, specs)}))
    env = Envelope(
        version=VERSION,
        member=member,
        config=config,
        source_head=source_head,
        source_tree=source_tree,
        image_digest=image_digest,
        renderer_identity=renderer_identity,
        mapping_commitment=digest(canonical(sequence.mapping)),
        sequence_privileged=sequence_path,
        frames=tuple(frame_refs),
        flows=tuple(flow_refs),
        artifacts=tuple(specs.values()),
        logical_identity="0" * 64,
    )
    env = env.model_copy(update={"logical_identity": logical_identity(env)})
    manifest = canonical(env.model_dump(mode="json"))
    validate_manifest(manifest, tuple(artifacts))
    return manifest, artifacts


class ArtifactProvider:
    """Trusted provider with temporal/modality denial before sink access.

    Construction validates manifest only, never target arrays. Evaluator target
    reads require a separate provider with explicit permissions and later time;
    this API does not implement or authorize sealed target release.
    """

    def __init__(
        self,
        manifest: bytes,
        paths: tuple[str, ...],
        read: Callable[[str], bytes],
        permissions: ModalityPermissionSet,
        decision: int,
    ):
        if not isinstance(permissions, ModalityPermissionSet) or type(decision) is not int:
            raise ValueError("typed permissions/time required")
        self._envelope = validate_manifest(manifest, paths)
        if not 0 <= decision < len(self._envelope.frames):
            raise ValueError("decision outside sequence")
        self._permissions, self._decision, self._read = permissions, decision, read
        self._artifacts = {a.path: a for a in self._envelope.artifacts}
        self._labels: dict[int, str] = {}
        self._tokens: dict[str, int] = {}

    @property
    def envelope(self) -> Envelope:
        return self._envelope.model_copy(deep=True)

    def _require(self, end: int, *modalities: Modality) -> None:
        if type(end) is not int or end < 0:
            raise ValueError("nonnegative typed index required")
        if end > self._decision or any(not self._permissions.permits(m) for m in modalities):
            raise PermissionError("time/modality denied before artifact access")

    def _load(self, path: str) -> Any:
        spec = self._artifacts[path]
        self._require(spec.end, spec.modality)
        payload = self._read(path)
        if (
            type(payload) is not bytes
            or len(payload) != spec.size
            or digest(payload) != spec.sha256
        ):
            raise ValueError("artifact size/hash differs")
        if spec.kind == "json":
            return parse(payload)
        return np.frombuffer(payload, dtype=spec.dtype).reshape(spec.shape)

    def frame(self, sequence_index: int) -> OpticalFrame:
        self._require(
            sequence_index, Modality.SURFACE_REGIONS, Modality.ORIENTED_BOUNDARY_OWNERSHIP
        )
        f = self._envelope.frames[sequence_index]
        p = OpticalPayload.model_validate_json(canonical(self._load(f.optical)))
        if (
            len(p.boundaries) > MAX_BOUNDARY_RECORDS
            or any(len(canonical(b)) > 512 for b in p.boundaries)
            or any(label not in (1, 2, 3) for label, _ in p.identities)
        ):
            raise ValueError("optical payload count/record/label ceiling differs")
        result = OpticalFrame(
            sequence_index,
            self._load(f.segmentation),
            p.identities,
            tuple(canonical(b) for b in p.boundaries),
        )
        for label, token in result.identities:
            if self._labels.get(label, token) != token or self._tokens.get(token, label) != label:
                raise ValueError("opaque mapping changed across frames")
            self._labels[label], self._tokens[token] = token, label
        return result

    def flow(self, source_index: int) -> CompletedFlow:
        if type(source_index) is not int or source_index < 0:
            raise ValueError("typed source index required")
        self._require(
            source_index + 1, Modality.ANALYTIC_OPTICAL_TRANSPORT, Modality.EXECUTED_ACTION
        )
        f = self._envelope.flows[source_index]
        if not f.admitted:
            raise ValueError("native/analytic flow admission failed; original evidence retained")
        return CompletedFlow(
            source_index,
            f.command.command(),
            self._load(f.vectors),
            self._load(f.validity),
            self._load(f.reasons),
        )

    def rgb(self, sequence_index: int) -> Array:
        self._require(sequence_index, Modality.RGB)
        return owned(self._load(self._envelope.frames[sequence_index].rgb))

    def instrumentation(self, end: int) -> tuple[Any, ...]:
        self._require(end, Modality.PRIVILEGED_GENERATION_RECORDS)
        f = self._envelope.frames[end]
        p = FrameEvidence.model_validate_json(canonical(self._load(f.privileged)))
        return (p, *(self._load(path) for path in f.arrays))

    def sequence_instrumentation(self) -> SequenceEvidence:
        self._require(0, Modality.PRIVILEGED_GENERATION_RECORDS)
        result = SequenceEvidence.model_validate_json(
            canonical(self._load(self._envelope.sequence_privileged))
        )
        if digest(canonical(result.mapping)) != self._envelope.mapping_commitment:
            raise ValueError("privileged mapping commitment differs")
        return result

    def completed_instrumentation(self, source_index: int) -> tuple[Any, ...]:
        if type(source_index) is not int or source_index < 0:
            raise ValueError("typed source index required")
        self._require(source_index + 1, Modality.PRIVILEGED_GENERATION_RECORDS)
        f = self._envelope.flows[source_index]
        result = FlowEvidence.model_validate_json(canonical(self._load(f.privileged)))
        if result.admitted != f.admitted:
            raise ValueError("flow admission evidence differs")
        return (result, *(self._load(path) for path in f.arrays))

    def projection(self) -> TemporalProjection:
        return TemporalProjection(self, self._permissions, self._decision)
