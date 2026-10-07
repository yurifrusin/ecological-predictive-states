"""Disabled-default native adapter and fail-closed privileged evidence/projection."""

from __future__ import annotations

import base64
import json
import re
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, TypeAlias

import numpy as np
import numpy.typing as npt

from epsbench.diagnostics.boundary_observation import VisibleRaster
from epsbench.diagnostics.corridor_aperture import (
    HEIGHT,
    NAMES,
    POSES,
    VERSION,
    WIDTH,
    config_root,
    digest,
    encode,
    index,
    scene_xml,
    validate_mapping,
)
from epsbench.diagnostics.corridor_aperture_reference import (
    CompiledBox,
    DrawDomain,
    SceneCamera,
    clip_planes,
    domain_boxes,
    first_hit,
    sample_ray,
    target_cause,
)
from epsbench.schema import Modality, ModalityPermissionSet

Array: TypeAlias = npt.NDArray[Any]
PRIVILEGED = frozenset(
    {
        Modality.DEPTH,
        Modality.CAMERA_WORLD_TRANSFORM,
        Modality.MUJOCO_GEOM_IDS,
        Modality.RAW_SIMULATOR_COORDINATES,
        Modality.SAMPLED_SCENE_GEOMETRY,
        Modality.PRIVILEGED_GENERATION_RECORDS,
    }
)


def require(permissions: ModalityPermissionSet, required: frozenset[Modality]) -> None:
    if type(permissions) is not ModalityPermissionSet or not required <= permissions.allowed:
        raise PermissionError("typed aperture permissions denied before access")


class EvidenceKind(StrEnum):
    NATIVE = "NATIVE_CANONICAL_PAIRED"
    SYNTHETIC = "SYNTHETIC_SOURCE_ONLY"


@dataclass(frozen=True)
class SourceBinding:
    head: str
    tree: str
    configuration_root: str

    def __post_init__(self) -> None:
        if (
            any(
                type(v) is not str or re.fullmatch(r"[0-9a-f]{40}", v) is None
                for v in (self.head, self.tree)
            )
            or self.configuration_root != config_root()
        ):
            raise ValueError("exact source/configuration binding required")


@dataclass(frozen=True)
class Frame:
    kind: EvidenceKind
    domain: DrawDomain
    raw_labels: Array
    depth: Array
    native_rgb: Array
    native_depth: Array
    paired_state: bytes
    operational: bytes
    configuration_root: str
    source: SourceBinding

    def __post_init__(self) -> None:
        if (
            type(self.kind) is not EvidenceKind
            or self.configuration_root != config_root()
            or (type(self.source) is not SourceBinding)
        ):
            raise ValueError("typed kind and exact apparatus identity required")
        domain_boxes(self.domain)
        for name, shape, dtype in (
            ("raw_labels", (HEIGHT, WIDTH), np.int32),
            ("depth", (HEIGHT, WIDTH), np.float32),
            ("native_rgb", (HEIGHT, WIDTH, 3), np.uint8),
            ("native_depth", (HEIGHT, WIDTH), np.float32),
        ):
            array = getattr(self, name)
            if type(array) is not np.ndarray or array.shape != shape or array.dtype != dtype:
                raise ValueError("exact captured array shape/dtype required")
            if name != "native_rgb" and not np.isfinite(array).all():
                raise ValueError("nonfinite captured array")
            object.__setattr__(
                self, name, np.frombuffer(array.tobytes(), dtype=dtype).reshape(shape)
            )
        ids = {b.raw_id for b in self.domain.boxes}
        if not set(map(int, np.unique(self.raw_labels))) <= ids | {-1}:
            raise ValueError("undeclared native box identity")
        for value in (self.paired_state, self.operational):
            if type(value) is not bytes or encode(json.loads(value)) != value:
                raise ValueError("canonical retained JSON required")

    def scientific_bytes(self) -> bytes:
        return encode(
            {
                "version": VERSION,
                "kind": self.kind.value,
                "configuration_root": config_root(),
                "source": asdict(self.source),
                "domain": asdict(self.domain),
                "paired_state": json.loads(self.paired_state),
                "arrays": {
                    name: base64.b64encode(getattr(self, name).tobytes()).decode()
                    for name in ("raw_labels", "depth", "native_rgb", "native_depth")
                },
            }
        )  # Operational host/process/time records deliberately outside this identity.


class Provider(Protocol):
    def frame(self, sequence_index: int) -> Frame: ...


def privileged_frame(provider: Provider, permissions: ModalityPermissionSet, i: int) -> Frame:
    require(permissions, PRIVILEGED)
    index(i)
    frame = provider.frame(i)
    if type(frame) is not Frame or frame.domain.sequence_index != i:
        raise ValueError("exact typed frame chronology required")
    return frame


@dataclass(frozen=True)
class ObservedState:
    observation: VisibleRaster
    observed_ever: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            type(self.observation) is not VisibleRaster
            or self.observation.segmentation.shape != (HEIGHT, WIDTH)
            or type(self.observed_ever) is not tuple
            or self.observed_ever != tuple(sorted(set(self.observed_ever)))
            or any(
                type(t) is not str or re.fullmatch(r"surface-[0-9a-f]{16}", t) is None
                for t in self.observed_ever
            )
            or not {t for _, t in self.observation.identities} <= set(self.observed_ever)
        ):
            raise ValueError("strict observed-only state required")

    def canonical_bytes(self) -> bytes:
        # Canonical token spelling order conveys no raw role; maps only visible masks.
        return encode(
            {
                "version": VERSION + "/observed",
                "index": self.observation.sequence_index,
                "shape": [HEIGHT, WIDTH],
                "inventory": list(self.observed_ever),
                "segmentation": self.observation.segmentation.tolist(),
                "visible_association": [list(v) for v in self.observation.identities],
            }
        )


class Projection:
    """Trusted evaluator-side projection: never a privileged learner loader."""

    def __init__(
        self, mapping: tuple[tuple[int, str], ...], permissions: ModalityPermissionSet
    ) -> None:
        require(permissions, PRIVILEGED)
        validate_mapping(mapping)
        self._mapping = dict(mapping)
        self._seen: set[str] = set()
        self._next = 0
        self._membership: tuple[tuple[str, int], ...] | None = None
        self._source: SourceBinding | None = None

    def observe(self, frame: Frame, permissions: ModalityPermissionSet) -> ObservedState:
        require(
            permissions,
            PRIVILEGED
            | frozenset(
                {
                    Modality.SURFACE_REGIONS,
                    Modality.REGION_CORRESPONDENCE,
                }
            ),
        )
        if type(frame) is not Frame or frame.domain.sequence_index != self._next:
            raise ValueError("consecutive observation required; future/repeated frames denied")
        if set(self._mapping) != {b.raw_id for b in frame.domain.boxes}:
            raise ValueError("privileged association differs from captured membership")
        membership = tuple((b.name, b.raw_id) for b in frame.domain.boxes)
        if self._membership is not None and (
            membership != self._membership or (frame.source != self._source)
        ):
            raise ValueError("cross-frame named-box/opaque/source association changed")
        self._membership, self._source = membership, frame.source
        # Sort by occupied raster position, never raw role, ID or name; labels are local.
        ordered = sorted(
            (int(np.flatnonzero(frame.raw_labels == raw)[0]), raw)
            for raw in self._mapping
            if np.any(frame.raw_labels == raw)
        )
        labels: Array = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
        pairs = []
        for label, (_, raw) in enumerate(ordered, 1):
            labels[frame.raw_labels == raw] = label
            token = self._mapping[raw]
            pairs.append((label, token))
            self._seen.add(token)
        observation = VisibleRaster(self._next, labels, tuple(pairs))
        self._next += 1
        return ObservedState(observation, tuple(sorted(self._seen)))


class ObservationProvider(Protocol):
    def observation(self, sequence_index: int) -> ObservedState: ...


def ecological_observation(
    provider: ObservationProvider,
    permissions: ModalityPermissionSet,
    sequence_index: int,
    decision_index: int,
) -> ObservedState:
    require(permissions, frozenset({Modality.SURFACE_REGIONS, Modality.REGION_CORRESPONDENCE}))
    index(sequence_index)
    index(decision_index)
    if sequence_index > decision_index:
        raise PermissionError("future observation denied before provider access")
    state = provider.observation(sequence_index)
    if type(state) is not ObservedState or state.observation.sequence_index != sequence_index:
        raise ValueError("typed ecological chronology differs")
    return ObservedState(state.observation, state.observed_ever)


def verify_pair(frame: Frame, permissions: ModalityPermissionSet) -> None:
    require(permissions, PRIVILEGED)
    if frame.kind is not EvidenceKind.NATIVE:
        raise PermissionError("synthetic frames cannot certify actual capability")
    # SDK module is imported only beyond the typed boundary, never by source checks.
    from epsbench.sim.canonical_paired import (
        SceneMapEntry,
        convert_native_depth,
        decode_id_colors,
        validate_saved_state,
    )

    state = json.loads(frame.paired_state)
    validate_saved_state(state, WIDTH, HEIGHT)
    if (
        tuple(state["projection_matrix_float32"]) != frame.domain.projection
        or (tuple(state["modelview_matrix_float32"]) != frame.domain.modelview)
        or any(state[name] != getattr(frame.domain, name) for name in ("near", "far", "extent"))
    ):
        raise ValueError("retained draw/domain bindings differ")
    if state["scene_cameras"] != [
        asdict(c) | {"pos": list(c.pos), "forward": list(c.forward), "up": list(c.up)}
        for c in frame.domain.scene_cameras
    ]:
        raise ValueError("retained scene cameras/domain differ")
    scene_ids = {g["objid"] for g in state["scene_geometry"]}
    if scene_ids != {b.raw_id for b in frame.domain.boxes} or len(state["scene_geometry"]) != 9:
        raise ValueError("retained draw membership differs")
    lookup = {b.raw_id: b for b in frame.domain.draw_boxes}
    for g in state["scene_geometry"]:
        b = lookup[g["objid"]]
        if (
            g["type"] != 6
            or tuple(g["pos"]) != b.position
            or (tuple(g["size"]) != b.half_size or tuple(g["mat"]) != b.rotation)
        ):
            raise ValueError("compiled and drawn solids differ")
    mapping = tuple(SceneMapEntry(**v) for v in state["scene_map"])
    if not np.array_equal(decode_id_colors(frame.native_rgb, mapping), frame.raw_labels) or (
        not np.array_equal(
            convert_native_depth(frame.native_depth, frame.domain.near, frame.domain.far),
            frame.depth,
        )
    ):
        raise ValueError("shared draw arrays do not reconstruct")


def disposition(counts: dict[str, int], wanted_visible: bool) -> str:
    """Classify retained facts; valid counterexamples survive unrelated uncertainty."""
    if counts["target_mismatch"] or (not wanted_visible and counts["target_support"]):
        return "FAIL"
    if counts["target_unknown"] or counts["other_mismatch"]:
        return "INCONCLUSIVE"
    if wanted_visible and not counts["target_support"]:
        return "FAIL"
    return "PASS"


def evaluate_frame(frame: Frame, permissions: ModalityPermissionSet) -> dict[str, Any]:
    """Later-use full raster audit; source checks never execute this enumeration."""
    require(permissions, PRIVILEGED)
    verify_pair(frame, permissions)
    boxes = domain_boxes(frame.domain)
    cause = target_cause(frame.domain)
    near, far = clip_planes(frame.domain)
    expected_ids = {b.name: b.raw_id for b in frame.domain.boxes}
    target_id = expected_ids["target"]
    target = tuple(b for b in boxes if b.name == "target")
    counts = {
        "non_target_unknown": 0,
        "target_unknown": 0,
        "target_mismatch": 0,
        "other_mismatch": 0,
        "target_support": 0,
    }
    for row in range(HEIGHT):
        for col in range(WIDTH):
            origin, direction = sample_ray(frame.domain, row, col)
            solo = first_hit(origin, direction, target, near, far)
            hit = first_hit(origin, direction, boxes, near, far)
            actual = int(frame.raw_labels[row, col])
            target_directed = solo.status != "CLEAR" or actual == target_id
            if hit.status not in ("HIT", "CLEAR") or solo.status not in ("HIT", "CLEAR"):
                counts["target_unknown" if target_directed else "non_target_unknown"] += 1
                continue
            expected = expected_ids[hit.names[0]] if hit.status == "HIT" else -1
            if (actual == target_id) != (expected == target_id):
                counts["target_mismatch"] += 1
            elif actual != expected:
                counts["other_mismatch"] += 1
            if expected == target_id:
                counts["target_support"] += 1
    wanted_visible = frame.domain.sequence_index != 1
    result = disposition(counts, wanted_visible)
    return {
        "version": VERSION,
        "index": frame.domain.sequence_index,
        "result": result,
        "cause": cause,
        "counts": counts,
        "frame_root": digest(frame.scientific_bytes()),
        "claim": "box_region_capability_only_not_same_face_persistence",
    }


def collect() -> None:
    """There is deliberately no automatic admitted native launcher."""
    raise PermissionError("aperture qualification HELD; source acceptance grants no launch")


def retain_three(
    provider: Provider, permissions: ModalityPermissionSet, retain: Callable[[str, bytes], None]
) -> tuple[str, ...]:
    """Finite adapter; external admission/resource supervisor is deliberately deferred."""
    require(permissions, PRIVILEGED)
    roots = []
    binding: tuple[SourceBinding, tuple[CompiledBox, ...]] | None = None
    try:
        for i in range(3):
            frame = privileged_frame(provider, permissions, i)
            current = (frame.source, frame.domain.boxes)
            if binding is not None and current != binding:
                raise ValueError("cross-frame source/compiled membership changed")
            binding = current
            payload = frame.scientific_bytes()
            retain(f"view-{i}.json", payload)
            retain(f"view-{i}-operational.json", frame.operational)
            roots.append(digest(payload))
    except Exception as error:
        # Retain prior originals; surface both initiating and retention errors honestly.
        try:
            retain(
                "failure.json",
                encode(
                    {
                        "completed": len(roots),
                        "initiating_type": type(error).__name__,
                        "initiating": str(error),
                    }
                ),
            )
        except Exception as retention_error:
            raise RuntimeError("aperture initiating and failure-retention errors") from (
                ExceptionGroup("both failures", [error, retention_error])
            )
        raise
    return tuple(roots)


class RetentionFailure(RuntimeError):
    """A supplied sink failed; the original exception remains the cause."""


def retained_capture(
    renderer: Any,
    capture: Callable[[Callable[[str, Any], None]], Any],
    build: Callable[[Any], Frame],
    permissions: ModalityPermissionSet,
    retain: Callable[[str, bytes], None],
    sequence_index: int,
    source: SourceBinding,
) -> Frame:
    """Bounded privileged lifecycle hook; fake interfaces suffice for source checks."""
    require(permissions, PRIVILEGED)
    index(sequence_index)
    stages = (
        "draw_input",
        "draw_attempt",
        "draw_complete",
        "draw_output",
        "read_input",
        "read_attempt",
        "read_complete",
        "read_output",
    )
    stage = "capture_start"
    next_stage = 0

    def save(label: str, payload: dict[str, Any]) -> None:
        encoded = encode(
            {"index": sequence_index, "source": asdict(source), "certified_frame": False, **payload}
        )
        if len(encoded) > 2 * 1024 * 1024:
            raise ValueError("current-view retention record exceeds fixed bound")
        try:
            retain(f"view-{sequence_index}-{label}.json", encoded)
        except Exception as error:
            raise RetentionFailure("privileged sink failed at " + label) from error

    def progress(label: str, value: Any) -> None:
        nonlocal stage, next_stage
        if next_stage == len(stages) or label != stages[next_stage]:
            raise ValueError("paired progress order/bound differs")
        stage = label
        next_stage += 1
        record: dict[str, Any] = {"stage": label, "validity": "unvalidated_progress"}
        if label == "read_complete":
            if (
                type(value) is not tuple
                or len(value) != 2
                or (
                    any(type(v) is not bytes for v in value)
                    or len(value[0]) != HEIGHT * WIDTH * 3
                    or len(value[1]) != HEIGHT * WIDTH * 4
                )
            ):
                raise ValueError("native readback bytes/shape differ")
            record |= {
                "orientation": "native_bottom_up",
                "rgb_dtype": "uint8",
                "depth_dtype": "float32",
                "shape": [HEIGHT, WIDTH],
                "raw_rgb": base64.b64encode(value[0]).decode(),
                "raw_depth": base64.b64encode(value[1]).decode(),
            }
        elif label in ("draw_input", "draw_output", "read_input", "read_output"):
            if type(value) is not bytes or len(value) > 256 * 1024:
                raise ValueError("paired snapshot bytes/bound differ")
            record["snapshot_bytes"] = base64.b64encode(value).decode()
        elif value is not None:
            raise ValueError("paired marker payload differs")
        save(label, record)

    errors: list[Exception] = []
    roles: list[str] = []
    frame: Frame | None = None
    try:
        pair = capture(progress)
        stage = "pair_complete"
        arrays = {}
        for name in ("raw_geom_segmentation", "depth", "native_id_rgb", "native_depth_pre_metric"):
            a = getattr(pair, name)
            if not isinstance(a, np.ndarray) or a.nbytes > HEIGHT * WIDTH * 4:
                raise ValueError("completed pair array/bound differs")
            arrays[name] = {
                "dtype": str(a.dtype),
                "shape": list(a.shape),
                "bytes": base64.b64encode(a.tobytes(order="C")).decode(),
            }
        save(
            stage,
            {
                "stage": stage,
                "validity": "paired_capture_complete_domain_unvalidated",
                "orientation": "oriented_top_down",
                "arrays": arrays,
                "stable_state": pair.stable_state,
                "operational_state": pair.operational_state,
                "near": pair.near,
                "far": pair.far,
            },
        )
        stage = "domain_frame_verification"
        frame = build(pair)
        save(
            "verified_frame",
            {
                "stage": "verified_frame",
                "validity": "verified_before_close",
                "frame_bytes": base64.b64encode(frame.scientific_bytes()).decode(),
            },
        )
    except Exception as error:
        errors.append(error)
        roles.append("retention" if isinstance(error, RetentionFailure) else "initiating")
    try:
        renderer.close()
    except Exception as cleanup:
        errors.append(cleanup)
        roles.append("renderer_close")
    if errors:
        try:
            save(
                "native_failure",
                {
                    "stage": stage,
                    "validity": "failed_not_certified",
                    "errors": [
                        {
                            "role": role,
                            "type": type(e).__name__,
                            "message": str(e),
                            "cause_type": type(e.__cause__).__name__ if e.__cause__ else None,
                            "cause": str(e.__cause__) if e.__cause__ else None,
                        }
                        for role, e in zip(roles, errors, strict=True)
                    ],
                },
            )
        except Exception as retention_error:
            errors.append(retention_error)
        if len(errors) == 1:
            raise errors[0]
        raise ExceptionGroup("capture/retention/close failures", errors)
    assert frame is not None
    return frame


class NativeAdapter:
    """Finite future-use OSMesa adapter. Explicit enablement is not owner authority."""

    def __init__(
        self,
        permissions: ModalityPermissionSet,
        *,
        enabled: bool = False,
        expected_source: SourceBinding | None = None,
        retain: Callable[[str, bytes], None] | None = None,
    ) -> None:
        require(permissions, PRIVILEGED)
        if type(enabled) is not bool or not enabled:
            raise PermissionError("native aperture adapter disabled by default")
        if type(expected_source) is not SourceBinding:
            raise PermissionError("separately reviewed exact source/configuration required")
        if not callable(retain):
            raise PermissionError("privileged current-view retention sink required before capture")
        self._source = expected_source
        self._permissions = permissions.model_copy(deep=True)
        self._next = 0
        self._retain = retain
        self._failed = False

    def frame(self, sequence_index: int) -> Frame:
        require(self._permissions, PRIVILEGED)
        index(sequence_index)
        if self._failed or sequence_index != self._next:
            raise PermissionError("native repeat/future view denied before SDK access")
        self._next += 1  # Claim consumed even on failure; no retry through this adapter.
        root = Path(__file__).resolve().parents[3]

        def git(*args: str) -> str:
            return subprocess.run(
                ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
            ).stdout.strip()

        if git("rev-parse", "HEAD", "HEAD^{tree}").splitlines() != [
            self._source.head,
            self._source.tree,
        ] or git("status", "--porcelain", "--untracked-files=normal"):
            raise PermissionError("native source identity/cleanliness changed")
        import mujoco

        from epsbench.sim.canonical_paired import CanonicalPairedRenderer

        model = mujoco.MjModel.from_xml_string(scene_xml())
        if int(model.ngeom) != 9 or int(model.ncam) != 1:
            raise ValueError("compiled nine-box/one-camera membership differs")
        ids = tuple(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in NAMES)
        camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "aperture_camera")
        if any(v < 0 for v in ids) or len(set(ids)) != 9 or camera < 0:
            raise ValueError("compiled member absent before indexing")
        model.cam_pos[camera, 1] = POSES[sequence_index]
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

        def build(pair: Any) -> Frame:
            boxes = tuple(
                CompiledBox(
                    name,
                    int(raw),
                    "box"
                    if int(model.geom_type[raw]) == int(mujoco.mjtGeom.mjGEOM_BOX)
                    else "unsupported",
                    tuple(map(float, data.geom_xpos[raw])),
                    tuple(map(float, model.geom_size[raw])),
                    tuple(map(float, data.geom_xmat[raw].flat)),
                )
                for name, raw in zip(NAMES, ids, strict=True)
            )
            state = pair.stable_state
            drawn = {g["objid"]: g for g in state["scene_geometry"]}
            draw_boxes = tuple(
                CompiledBox(
                    name,
                    int(raw),
                    "box" if drawn[raw]["type"] == 6 else "unsupported",
                    tuple(map(float, drawn[raw]["pos"])),
                    tuple(map(float, drawn[raw]["size"])),
                    tuple(map(float, drawn[raw]["mat"])),
                )
                for name, raw in zip(NAMES, ids, strict=True)
            )
            domain = DrawDomain(
                sequence_index,
                boxes,
                tuple(map(float, data.cam_xpos[camera])),
                tuple(map(float, data.cam_xmat[camera].flat)),
                float(model.cam_fovy[camera]),
                tuple(map(float, state["projection_matrix_float32"])),
                tuple(map(float, state["modelview_matrix_float32"])),
                pair.near,
                pair.far,
                float(state["extent"]),
                float(model.vis.map.znear),
                float(model.vis.map.zfar),
                tuple(
                    SceneCamera(
                        **(
                            c
                            | {
                                "pos": tuple(map(float, c["pos"])),
                                "forward": tuple(map(float, c["forward"])),
                                "up": tuple(map(float, c["up"])),
                            }
                        )
                    )
                    for c in state["scene_cameras"]
                ),
                draw_boxes,
            )
            frame = Frame(
                EvidenceKind.NATIVE,
                domain,
                pair.raw_geom_segmentation,
                pair.depth,
                pair.native_id_rgb,
                pair.native_depth_pre_metric,
                encode(state),
                encode(pair.operational_state),
                config_root(),
                self._source,
            )
            verify_pair(frame, self._permissions)
            return frame

        def capture(progress: Callable[[str, Any], None]) -> Any:
            renderer.update_scene(data, camera=camera)
            return CanonicalPairedRenderer(renderer, progress_observer=progress).capture()

        try:
            return retained_capture(
                renderer,
                capture,
                build,
                self._permissions,
                self._retain,
                sequence_index,
                self._source,
            )
        except Exception:
            self._failed = True
            raise
