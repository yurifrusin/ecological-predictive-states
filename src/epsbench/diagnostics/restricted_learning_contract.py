"""Observed-only three-step inputs and strict Bernoulli forecasts; no model."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations

import numpy as np

from epsbench.diagnostics.boundary_observation import VisibleRaster
from epsbench.diagnostics.visible_forecast_contract import REQUIRED as REQUIRED
from epsbench.diagnostics.visible_forecast_contract import (
    CausalInput,
    CausalView,
    Command,
    Limits,
    RasterProvider,
    TokenFrame,
    _json,
    _keys,
    command,
    mask_view,
    permissions,
    snapshot,
    token,
)
from epsbench.schema import ModalityPermissionSet
from epsbench.utils.canonical import canonical_json_bytes

VERSION = "restricted-occupancy-learning-v1"
PROPOSAL = "a47df0e39e0f69ce0ca137e371cb31c84de5fcd15aa767ccb66b2dca92624329"
ARCHITECTURE = "7193c6c7f789ceda45b447e75a199657d21f6279ea84be5544034a2c9ce4b186"
LIMITS = Limits(3, 1024, 3)
BIJECTIONS = tuple(permutations(range(3)))
PAIRS = tuple((i, j) for i in range(3) for j in range(3) if i != j)
ZERO = (Fraction(0), Fraction(0), Fraction(0))
CONDITIONS = ("relational", "dense", "action-zero", "memory-reset")
CONTROLS = ("persistence", "absent", "visible", "half", "train-frequency")
INITIALIZATIONS = ("init-0", "init-1", "init-2")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha(value: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("strict SHA256 required")


@dataclass(frozen=True)
class Observation:
    index: int
    handles: tuple[str, ...]
    first_seen: tuple[int, ...]
    masks: tuple[bytes, bytes, bytes]
    present: tuple[int, int, int]
    visible: tuple[int, int, int]
    boundary: tuple[tuple[Fraction, Fraction], ...]
    executed: Command

    def __post_init__(self) -> None:
        if type(self.index) is not int or not 0 <= self.index < 3:
            raise ValueError("three-step index required")
        if (
            any(
                type(v) is not tuple
                for v in (
                    self.handles,
                    self.first_seen,
                    self.masks,
                    self.present,
                    self.visible,
                    self.boundary,
                )
            )
            or any(type(v) is not int or v not in (0, 1) for v in self.present)
            or any(
                type(b) is not tuple or len(b) != 2 or any(type(v) is not Fraction for v in b)
                for b in self.boundary
            )
        ):
            raise ValueError("exact immutable typed observation fields required")
        if len(self.handles) > 3 or len(set(self.handles)) != len(self.handles):
            raise ValueError("unique observed handles required")
        for handle in self.handles:
            token(handle)
        if len(self.first_seen) != len(self.handles) or any(
            type(i) is not int or not 0 <= i <= self.index for i in self.first_seen
        ):
            raise ValueError("causal arrival bookkeeping required")
        if self.present != tuple(int(i < len(self.handles)) for i in range(3)):
            raise ValueError("reserved slots must be observed only")
        if len(self.masks) != 3 or len(self.visible) != 3:
            raise ValueError("fixed three anonymous slots required")
        occupied = np.zeros((32, 32), dtype=np.bool_)
        for i, data in enumerate(self.masks):
            if type(data) is not bytes or len(data) != 1024 or any(v not in (0, 1) for v in data):
                raise ValueError("immutable binary current mask required")
            a = mask_view(data, (32, 32))
            if type(self.visible[i]) is not int or self.visible[i] != int(np.any(a)):
                raise ValueError("visibility must follow observed mask")
            if np.any(occupied & a) or (not self.present[i] and np.any(a)):
                raise ValueError("overlapping/unobserved mask denied")
            occupied |= a
        if self.boundary != boundaries(self.masks):
            raise ValueError("boundary must derive solely from current masks")
        command(self.executed)
        if self.index == 0 and self.executed != ZERO:
            raise ValueError("initial executed command must be zero")


def boundaries(masks: tuple[bytes, bytes, bytes]) -> tuple[tuple[Fraction, Fraction], ...]:
    arrays = tuple(mask_view(b, (32, 32)) for b in masks)
    return tuple(
        (
            Fraction(
                int(
                    np.count_nonzero(
                        (arrays[i][:, :-1] & arrays[j][:, 1:])
                        | (arrays[j][:, :-1] & arrays[i][:, 1:])
                    )
                ),
                992,
            ),
            Fraction(
                int(
                    np.count_nonzero(
                        (arrays[i][:-1, :] & arrays[j][1:, :])
                        | (arrays[j][:-1, :] & arrays[i][1:, :])
                    )
                ),
                992,
            ),
        )
        for i, j in PAIRS
    )


class StreamingView:
    """One pass; retained history is evidence, never a learner readout cache.

    Handles/first_seen are indexing bookkeeping, not numeric learner features.
    The provider capability exposes only prefix rasters; no target/geometry method.
    """

    def __init__(
        self,
        provider: RasterProvider,
        access: ModalityPermissionSet,
        executed: tuple[Command, Command],
        announced: Command,
    ) -> None:
        permissions(access)
        if type(executed) is not tuple or len(executed) != 2:
            raise ValueError("exact two executed commands required before access")
        for c in (*executed, announced):
            command(c)
        self._view = CausalView(provider, access, 2, LIMITS)
        self._executed = executed
        self._announced = announced
        self._next = 0
        self._failed = False
        self._handles: list[str] = []
        self._first: list[int] = []

    @property
    def announced(self) -> Command:
        if self._next != 3 or self._failed:
            raise PermissionError("announced command is readout-only after complete prefix")
        return self._announced

    def next_observation(self) -> Observation:
        if self._failed:
            raise PermissionError("failed prefix terminal; no skip/continue")
        try:
            return self._advance()
        except Exception:
            self._failed = True
            raise

    def _advance(self) -> Observation:
        if self._next >= 3:
            raise PermissionError("three observations exhausted; no replay/target")
        i = self._next
        # Reserve access before fetch; failed fetch cannot retry a changed observation.
        self._next += 1
        raster = self._view.raster(i)
        if raster.segmentation.shape != (32, 32):
            raise ValueError("exact 32x32 calibration required")
        frame = TokenFrame(
            i, (32, 32), tuple((h, raster.segmentation == label) for label, h in raster.identities)
        )
        current = dict(frame.masks)
        arrivals = [h for h in current if h not in self._handles]
        arrivals.sort(key=lambda h: int(np.flatnonzero(current[h])[0]))
        if len(self._handles) + len(arrivals) > 3:
            raise ValueError("observed union capacity exceeded")
        self._handles.extend(arrivals)
        self._first.extend([i] * len(arrivals))
        padded = tuple(
            snapshot(current.get(h, np.zeros((32, 32), dtype=np.bool_)), (32, 32))
            for h in self._handles
        ) + (bytes(1024),) * (3 - len(self._handles))
        masks = (padded[0], padded[1], padded[2])
        return Observation(
            i,
            tuple(self._handles),
            tuple(self._first),
            masks,
            tuple(int(j < len(self._handles)) for j in range(3)),  # type: ignore[arg-type]
            tuple(int(any(b)) for b in masks),  # type: ignore[arg-type]
            boundaries(masks),
            ZERO if i == 0 else self._executed[i - 1],
        )


@dataclass(frozen=True)
class InputEvidence:
    prefix: CausalInput

    def __post_init__(self) -> None:
        if (
            type(self.prefix) is not CausalInput
            or self.prefix.limits != LIMITS
            or (len(self.prefix.frames) != 3 or self.prefix.shape != (32, 32))
        ):
            raise ValueError("complete bounded three-frame evidence required")

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(
            {
                "version": VERSION,
                "architecture": ARCHITECTURE,
                "prefix": _json(self.prefix.canonical_bytes()),
            }
        )

    @property
    def digest(self) -> str:
        return digest(self.canonical_bytes())

    @classmethod
    def from_bytes(cls, data: bytes, access: ModalityPermissionSet) -> InputEvidence:
        permissions(access)
        if len(data) > 65536:
            raise ValueError("bounded input serialization required")
        p = _json(data)
        _keys(p, {"version", "architecture", "prefix"})
        if p["version"] != VERSION or p["architecture"] != ARCHITECTURE:
            raise ValueError("input specification differs")
        return cls(CausalInput.from_bytes(canonical_json_bytes(p["prefix"]), access, LIMITS))

    def streaming(self) -> StreamingView:
        # Evidence owner creates a prefix-only capability; learner gets this view, not evidence.
        frames = self.prefix.frames

        class PrefixProvider:
            def raster(self, sequence_index: int) -> VisibleRaster:
                if type(sequence_index) is not int or not 0 <= sequence_index < 3:
                    raise PermissionError("prefix only")
                f = frames[sequence_index]
                raw = np.zeros((32, 32), dtype=np.int32)
                identities = []
                for label, (h, mask) in enumerate(f.masks, 1):
                    raw[mask] = label
                    identities.append((label, h))
                return VisibleRaster(sequence_index, raw, tuple(identities))

        return StreamingView(
            PrefixProvider(),
            self.prefix.permissions,
            (self.prefix.executed[0], self.prefix.executed[1]),
            self.prefix.announced,
        )


@dataclass(frozen=True)
class Fit:
    budget: int
    initialization: str
    condition: str

    def __post_init__(self) -> None:
        if (
            type(self.budget) is not int
            or self.budget not in (16, 64)
            or (
                self.initialization not in INITIALIZATIONS
                or self.condition not in (*CONDITIONS, *CONTROLS)
            )
        ):
            raise ValueError("fixed fit/control roster required")

    @property
    def key(self) -> str:
        return f"{self.budget}/{self.initialization}/{self.condition}"


ROSTER = tuple(
    Fit(b, i, c) for b in (16, 64) for i in INITIALIZATIONS for c in (*CONDITIONS, *CONTROLS)
)


@dataclass(frozen=True)
class Forecast:
    input_sha256: str
    checkpoint_sha256: str
    run_sha256: str
    fit: Fit
    channels: tuple[tuple[str, float | None], ...]

    def __post_init__(self) -> None:
        for root in (self.input_sha256, self.checkpoint_sha256, self.run_sha256):
            sha(root)
        if type(self.fit) is not Fit or type(self.channels) is not tuple:
            raise ValueError("typed forecast required")
        seen = set()
        for channel in self.channels:
            if type(channel) is not tuple or len(channel) != 2:
                raise ValueError("immutable channel pair required")
            h, p = channel
            token(h)
            if h in seen:
                raise ValueError("duplicate forecast channel")
            seen.add(h)
            if p is not None and (type(p) is not float or not math.isfinite(p) or not 0 <= p <= 1):
                raise ValueError("finite Bernoulli probability or explicit UNKNOWN required")

        object.__setattr__(self, "channels", tuple(sorted(self.channels)))

    def validate_input(self, evidence: InputEvidence) -> None:
        if self.input_sha256 != evidence.digest or {h for h, _ in self.channels} != set(
            evidence.prefix.inventory
        ):
            raise ValueError("complete exact observed inventory/input binding required")

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(
            {
                "version": VERSION,
                "architecture": ARCHITECTURE,
                "input": self.input_sha256,
                "checkpoint": self.checkpoint_sha256,
                "run": self.run_sha256,
                "budget": self.fit.budget,
                "initialization": self.fit.initialization,
                "condition": self.fit.condition,
                "channels": {
                    h: {"probability": p} if p is not None else {"status": "UNKNOWN"}
                    for h, p in self.channels
                },
            }
        )

    @classmethod
    def from_bytes(cls, data: bytes, evidence: InputEvidence) -> Forecast:
        if type(data) is not bytes or len(data) > 4096:
            raise ValueError("bounded forecast bytes required")
        p = _json(data)
        _keys(
            p,
            {
                "version",
                "architecture",
                "input",
                "checkpoint",
                "run",
                "budget",
                "initialization",
                "condition",
                "channels",
            },
        )
        if p["version"] != VERSION or p["architecture"] != ARCHITECTURE:
            raise ValueError("forecast specification differs")
        if type(p["channels"]) is not dict:
            raise ValueError("closed probability map required")
        values = []
        for h, value in p["channels"].items():
            if type(value) is dict and set(value) == {"status"} and value["status"] == "UNKNOWN":
                values.append((h, None))
            else:
                _keys(value, {"probability"})
                values.append((h, value["probability"]))
        result = cls(
            p["input"],
            p["checkpoint"],
            p["run"],
            Fit(p["budget"], p["initialization"], p["condition"]),
            tuple(values),
        )
        result.validate_input(evidence)  # Before any evaluator target capability is touched.
        return result
