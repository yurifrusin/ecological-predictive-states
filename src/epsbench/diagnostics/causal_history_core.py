"""Native-free sequence-local causal projections and the specified lossy state."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from epsbench.schema import BoundaryAxis, Modality, ModalityPermissionSet, OrientedBoundaryElement

Array = npt.NDArray[Any]
VERSION = "causal-history-core-v1"
FLOW_SCALE = 1024
VALID_TRANSPORT = 0


def owned(array: Array) -> Array:
    """Owned bytes-backed snapshots cannot regain writeability through their base."""
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def index(value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("nonnegative integer sequence index required")


@dataclass(frozen=True)
class Command:
    lateral: Fraction
    forward: Fraction = Fraction(0)
    yaw: Fraction = Fraction(0)

    def __post_init__(self) -> None:
        if any(type(v) is not Fraction for v in (self.lateral, self.forward, self.yaw)):
            raise ValueError("canonical exact Fraction commands required")

    @property
    def supported(self) -> bool:
        return self.forward == 0 and self.yaw == 0


@dataclass(frozen=True)
class OpticalFrame:
    sequence_index: int
    segmentation: Array
    identities: tuple[tuple[int, str], ...]
    boundaries: tuple[bytes, ...] = ()

    def __post_init__(self) -> None:
        index(self.sequence_index)
        a = self.segmentation
        if a.dtype != np.int32 or a.ndim != 2 or min(a.shape) < 1 or np.any(a < 0):
            raise ValueError("nonnegative int32 raster required")
        identities = tuple((label, token) for label, token in self.identities)
        if any(type(label) is not int or type(token) is not str for label, token in identities):
            raise ValueError("typed opaque label/token pairs required")
        labels = {int(v) for v in np.unique(a) if v != 0}
        if {v for v, _ in identities} != labels or len(identities) != len(labels):
            raise ValueError(
                "UNKNOWN_IDENTITY: identities must cover exactly visible nonzero labels"
            )
        tokens = [token for _, token in identities]
        if len(set(tokens)) != len(tokens) or any(
            re.fullmatch(r"surface-[0-9a-f]{16}", t) is None for t in tokens
        ):
            raise ValueError("bijective opaque visible identities required")
        boundaries = tuple(self.boundaries)
        canonical = []
        positions = set()
        for payload in boundaries:
            if type(payload) is not bytes:
                raise ValueError("immutable boundary payload required")
            b = OrientedBoundaryElement.model_validate_json(payload)
            if b.frame_index != 0:
                raise ValueError("sequence-local boundary payload requires frame_index=0")
            if any(
                t is not None and t not in tokens
                for t in (b.negative_surface_id, b.positive_surface_id, b.owner_surface_id)
            ):
                raise ValueError("boundary references an invisible identity")
            r, c = b.row, b.column
            h, w = a.shape
            if b.axis == BoundaryAxis.HORIZONTAL:
                fits = r < h and c < w - 1
                neighbours = (int(a[r, c]), int(a[r, c + 1])) if fits else (0, 0)
            else:
                fits = r < h - 1 and c < w
                neighbours = (int(a[r, c]), int(a[r + 1, c])) if fits else (0, 0)
            lookup = dict(identities)
            if not fits or (lookup.get(neighbours[0]), lookup.get(neighbours[1])) != (
                b.negative_surface_id,
                b.positive_surface_id,
            ):
                raise ValueError("boundary lattice/segmentation mapping differs")
            position = (b.axis, r, c)
            if position in positions:
                raise ValueError("duplicate boundary lattice location")
            positions.add(position)
            canonical.append(b.model_dump_json().encode())
        object.__setattr__(self, "segmentation", owned(a))
        object.__setattr__(self, "identities", tuple(sorted(identities)))
        object.__setattr__(self, "boundaries", tuple(sorted(canonical)))

    def mask(self, token: str) -> Array:
        label = next((v for v, t in self.identities if t == token), None)
        return (
            owned(self.segmentation == label)
            if label is not None
            else owned(np.zeros(self.segmentation.shape, dtype=np.bool_))
        )


@dataclass(frozen=True)
class CompletedFlow:
    source_index: int
    command: Command
    vectors: Array
    validity: Array
    reasons: Array

    def __post_init__(self) -> None:
        index(self.source_index)
        if not isinstance(self.command, Command):
            raise ValueError("typed command required")
        if (
            self.vectors.dtype != np.int32
            or self.vectors.ndim != 3
            or self.vectors.shape[-1] != 2
            or self.validity.dtype != np.uint8
            or self.reasons.dtype != np.uint8
            or self.validity.ndim != 2
            or self.validity.shape != self.vectors.shape[:2]
            or self.reasons.shape != self.validity.shape
            or min(self.validity.shape) < 1
            or np.any(self.validity > 1)
            or np.any(self.reasons > 4)
            or not np.array_equal(self.validity == 1, self.reasons == VALID_TRANSPORT)
        ):
            raise ValueError("completed flow shape/validity/reason contract differs")
        for name in ("vectors", "validity", "reasons"):
            object.__setattr__(self, name, owned(getattr(self, name)))


class SequenceProvider(Protocol):
    def frame(self, sequence_index: int) -> OpticalFrame: ...
    def flow(self, source_index: int) -> CompletedFlow: ...
    def rgb(self, sequence_index: int) -> Array: ...


@dataclass(frozen=True)
class MemorySequence:
    frames: tuple[OpticalFrame, ...]
    flows: tuple[CompletedFlow, ...]
    rgbs: tuple[Array, ...]

    def frame(self, sequence_index: int) -> OpticalFrame:
        return self.frames[sequence_index]

    def flow(self, source_index: int) -> CompletedFlow:
        return self.flows[source_index]

    def rgb(self, sequence_index: int) -> Array:
        return self.rgbs[sequence_index]


@dataclass(frozen=True)
class TemporalProjection:
    provider: SequenceProvider
    permissions: ModalityPermissionSet
    decision: int

    def __post_init__(self) -> None:
        index(self.decision)
        if not isinstance(self.permissions, ModalityPermissionSet):
            raise ValueError("typed permissions required")

    def require(self, end: int, *modalities: Modality) -> None:
        index(end)
        if end > self.decision:
            raise PermissionError("future evidence denied before provider access")
        if any(not self.permissions.permits(m) for m in modalities):
            raise PermissionError("modality denied before provider access")

    def frame(self, i: int) -> OpticalFrame:
        self.require(i, Modality.SURFACE_REGIONS, Modality.ORIENTED_BOUNDARY_OWNERSHIP)
        f = self.provider.frame(i)
        if not isinstance(f, OpticalFrame) or f.sequence_index != i:
            raise ValueError("frame chronology differs")
        return OpticalFrame(i, f.segmentation, f.identities, f.boundaries)

    def flow(self, i: int) -> CompletedFlow:
        index(i)
        self.require(i + 1, Modality.ANALYTIC_OPTICAL_TRANSPORT, Modality.EXECUTED_ACTION)
        f = self.provider.flow(i)
        if not isinstance(f, CompletedFlow) or f.source_index != i:
            raise ValueError("transition chronology differs")
        return CompletedFlow(i, f.command, f.vectors, f.validity, f.reasons)

    def rgb(self, i: int) -> Array:
        self.require(i, Modality.RGB)
        a = self.provider.rgb(i)
        if a.dtype != np.uint8 or a.ndim != 3 or a.shape[-1] != 3:
            raise ValueError("uint8 RGB required")
        return owned(a)


@dataclass(frozen=True)
class Motion:
    sums: tuple[int, int]
    count: int
    source_index: int
    reference: Command
    reason_counts: tuple[tuple[int, int, int], ...]

    def __post_init__(self) -> None:
        index(self.source_index)
        if (
            type(self.count) is not int
            or self.count < 1
            or not isinstance(self.reference, Command)
            or not self.reference.supported
            or self.reference.lateral == 0
            or len(self.sums) != 2
            or any(type(v) is not int for v in self.sums)
            or len(self.reason_counts) != 10
            or tuple((v, r) for v, r, _ in self.reason_counts)
            != tuple((v, r) for v in range(2) for r in range(5))
            or any(type(c) is not int or c < 0 for _, _, c in self.reason_counts)
            or self.reason_counts[5][2] != self.count
        ):
            raise ValueError("invalid usable motion summary")
        object.__setattr__(self, "sums", tuple(self.sums))
        object.__setattr__(self, "reason_counts", tuple(tuple(v) for v in self.reason_counts))


@dataclass(frozen=True)
class TokenState:
    token: str
    visible: bool
    last_mask: Array
    last_index: int
    motion: Motion | None

    def __post_init__(self) -> None:
        index(self.last_index)
        if (
            re.fullmatch(r"surface-[0-9a-f]{16}", self.token) is None
            or type(self.visible) is not bool
            or self.last_mask.dtype != np.bool_
            or self.last_mask.ndim != 2
            or not np.any(self.last_mask)
            or (self.motion is not None and not isinstance(self.motion, Motion))
        ):
            raise ValueError("observed identity and nonempty last mask required")
        object.__setattr__(self, "last_mask", owned(self.last_mask))


@dataclass(frozen=True)
class CausalState:
    current: OpticalFrame
    tokens: tuple[TokenState, ...]
    executed: tuple[Command, ...]
    announced: Command
    version: str = VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.current, OpticalFrame)
            or self.version != VERSION
            or not isinstance(self.announced, Command)
            or len(self.executed) != self.current.sequence_index
            or any(not isinstance(c, Command) for c in self.executed)
            or any(not isinstance(t, TokenState) for t in self.tokens)
        ):
            raise ValueError("typed chronological state required")
        tokens = tuple(self.tokens)
        if len({t.token for t in tokens}) != len(tokens):
            raise ValueError("duplicate retained identity")
        visible = {t for _, t in self.current.identities}
        if visible != {t.token for t in tokens if t.visible}:
            raise ValueError("visible state identity differs")
        for t in tokens:
            if (
                t.last_mask.shape != self.current.segmentation.shape
                or t.last_index > self.current.sequence_index
                or (
                    t.visible
                    and (
                        t.last_index != self.current.sequence_index
                        or not np.array_equal(t.last_mask, self.current.mask(t.token))
                    )
                )
                or (
                    t.motion is not None
                    and (
                        t.motion.source_index >= self.current.sequence_index
                        or t.motion.reference != self.executed[t.motion.source_index]
                    )
                )
            ):
                raise ValueError("retained mask/motion chronology differs")
        object.__setattr__(self, "tokens", tuple(sorted(tokens, key=lambda t: t.token)))
        object.__setattr__(self, "executed", tuple(self.executed))


def build_state(view: TemporalProjection, announced: Command) -> CausalState:
    """Read only completed optical evidence; overwrite mask and motion independently."""
    if not isinstance(announced, Command):
        raise ValueError("typed announced command required")
    retained: dict[str, TokenState] = {}
    label_identity: dict[int, str] = {}
    identity_label: dict[str, int] = {}
    commands = []
    previous = None
    shape = None
    for i in range(view.decision + 1):
        current = view.frame(i)
        if shape is not None and shape != current.segmentation.shape:
            raise ValueError("sequence raster changed")
        shape = current.segmentation.shape
        for label, token in current.identities:
            if (label in label_identity and label_identity[label] != token) or (
                token in identity_label and identity_label[token] != label
            ):
                raise ValueError("UNKNOWN_IDENTITY: inconsistent opaque identity continuity")
            label_identity[label] = token
            identity_label[token] = label
        if previous is not None:
            flow = view.flow(i - 1)
            if flow.validity.shape != shape:
                raise ValueError("flow/frame raster differs")
            commands.append(flow.command)
            for label, token in previous.identities:
                source = previous.segmentation == label
                valid = source & (flow.validity == 1)
                count = int(np.count_nonzero(valid))
                if count and flow.command.supported and flow.command.lateral != 0:
                    counts = tuple(
                        (
                            v,
                            r,
                            int(
                                np.count_nonzero(
                                    source & (flow.validity == v) & (flow.reasons == r)
                                )
                            ),
                        )
                        for v in range(2)
                        for r in range(5)
                    )
                    vectors = flow.vectors[valid]
                    sums = tuple(sum(int(v) for v in vectors[:, axis]) for axis in range(2))
                    old = retained[token]
                    retained[token] = TokenState(
                        token,
                        old.visible,
                        old.last_mask,
                        old.last_index,
                        Motion((sums[0], sums[1]), count, i - 1, flow.command, counts),
                    )
        visible = {t for _, t in current.identities}
        for token, old in tuple(retained.items()):
            retained[token] = TokenState(
                token, token in visible, old.last_mask, old.last_index, old.motion
            )
        for token in visible:
            prior = retained.get(token)
            retained[token] = TokenState(
                token, True, current.mask(token), i, prior.motion if prior else None
            )
        previous = current
    return CausalState(
        current, tuple(retained[t] for t in sorted(retained)), tuple(commands), announced
    )


@dataclass(frozen=True)
class Forecast:
    token: str
    status: str
    mask: Array | None
    last_index: int | None = None
    motion: Motion | None = None
    cumulative: Fraction | None = None
    displacement: tuple[Fraction, Fraction] | None = None
    shift: tuple[int, int] | None = None
    clipped_pixels: int | None = None

    def __post_init__(self) -> None:
        if self.status not in {"MASK", "UNKNOWN", "UNKNOWN_MOTION"}:
            raise ValueError("fixed forecast status required")
        if self.status == "MASK":
            if (
                self.mask is None
                or self.mask.dtype != np.bool_
                or self.mask.ndim != 2
                or min(self.mask.shape) < 1
            ):
                raise ValueError("owned binary forecast required")
            object.__setattr__(self, "mask", owned(self.mask))
        elif self.mask is not None:
            raise ValueError("UNKNOWN must not substitute a binary mask")


def persistence(state: CausalState) -> tuple[Forecast, ...]:
    return tuple(Forecast(t.token, "MASK", state.current.mask(t.token)) for t in state.tokens)


def extrapolate(state: CausalState) -> tuple[Forecast, ...]:
    results = []
    for token in state.tokens:
        commands = (*state.executed[token.last_index :], state.announced)
        cumulative = sum((u.lateral for u in commands), Fraction(0))
        if any(not u.supported for u in commands):
            results.append(
                Forecast(token.token, "UNKNOWN", None, token.last_index, token.motion, cumulative)
            )
            continue
        if cumulative == 0:
            displacement = (Fraction(0), Fraction(0))
        elif token.motion is None:
            results.append(
                Forecast(token.token, "UNKNOWN_MOTION", None, token.last_index, None, cumulative)
            )
            continue
        else:
            m = token.motion
            displacement = (
                Fraction(m.sums[0], m.count * FLOW_SCALE) * cumulative / m.reference.lateral,
                Fraction(m.sums[1], m.count * FLOW_SCALE) * cumulative / m.reference.lateral,
            )
        dx, dy = (round(v) for v in displacement)
        rows, cols = np.nonzero(token.last_mask)
        h, w = token.last_mask.shape
        if abs(dx) >= w or abs(dy) >= h:
            results.append(
                Forecast(
                    token.token,
                    "MASK",
                    owned(np.zeros((h, w), dtype=np.bool_)),
                    token.last_index,
                    token.motion,
                    cumulative,
                    displacement,
                    (dx, dy),
                    len(rows),
                )
            )
            continue
        rr, cc = rows + dy, cols + dx
        inside = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        mask = np.zeros((h, w), dtype=np.bool_)
        mask[rr[inside], cc[inside]] = True
        results.append(
            Forecast(
                token.token,
                "MASK",
                owned(mask),
                token.last_index,
                token.motion,
                cumulative,
                displacement,
                (dx, dy),
                int(np.count_nonzero(~inside)),
            )
        )
    return tuple(results)


def boundary_key(payload: bytes, alignment: dict[str, str]) -> str:
    value = json.loads(payload)
    for key in ("negative_surface_id", "positive_surface_id", "owner_surface_id"):
        if value[key] is not None:
            value[key] = alignment[value[key]]
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
