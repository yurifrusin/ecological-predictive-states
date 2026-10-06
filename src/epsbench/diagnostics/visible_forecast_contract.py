"""Source-only visible occupancy contract; no native adapter or model."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from epsbench.diagnostics.boundary_observation import VisibleRaster
from epsbench.schema import Modality, ModalityPermissionSet
from epsbench.utils.canonical import canonical_json_bytes

VERSION = "causal-visible-forecast-v1"
REQUIRED = frozenset(
    {Modality.SURFACE_REGIONS, Modality.REGION_CORRESPONDENCE, Modality.EXECUTED_ACTION}
)
Array = npt.NDArray[Any]
Command = tuple[Fraction, Fraction, Fraction]
Masks = tuple[tuple[str, Array], ...]


def integer(value: int, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError("strict integer outside domain")


def permissions(value: ModalityPermissionSet) -> None:
    if type(value) is not ModalityPermissionSet or value.allowed != REQUIRED:
        raise PermissionError("exact typed forecast permissions required")
    if any(type(m) is not Modality for m in value.allowed):
        raise PermissionError("typed modalities required")


def token(value: str) -> None:
    if type(value) is not str or re.fullmatch(r"surface-[0-9a-f]{16}", value) is None:
        raise ValueError("opaque token required")


def command(value: Command) -> None:
    if type(value) is not tuple or len(value) != 3 or any(type(v) is not Fraction for v in value):
        raise ValueError("three exact rational action components required")


def validate_shape(shape: tuple[int, int]) -> None:
    if type(shape) is not tuple or len(shape) != 2:
        raise ValueError("two-dimensional shape required")
    for v in shape:
        integer(v, 1)


def snapshot(array: Array, shape: tuple[int, int]) -> bytes:
    if not isinstance(array, np.ndarray) or array.dtype != np.bool_ or array.shape != shape:
        raise ValueError("fixed-shape Boolean mask required")
    return array.tobytes(order="C")


def mask_view(data: bytes, shape: tuple[int, int]) -> Array:
    # Each access owns its metadata; only immutable bytes are shared with the contract.
    return np.frombuffer(data, dtype=np.bool_).reshape(shape)


@dataclass(frozen=True)
class Limits:
    max_frames: int
    max_pixels: int
    max_tokens: int

    def __post_init__(self) -> None:
        for v in (self.max_frames, self.max_pixels, self.max_tokens):
            integer(v, 1)

    def check(self, frames: int, shape: tuple[int, int], count: int) -> None:
        integer(frames, 1)
        integer(count)
        validate_shape(shape)
        if frames > self.max_frames or shape[0] * shape[1] > self.max_pixels:
            raise ValueError("frame/raster budget exceeded")
        if count > self.max_tokens:
            raise ValueError("token budget exceeded before mask expansion")


@dataclass(frozen=True, init=False)
class TokenFrame:
    index: int
    shape: tuple[int, int]
    _mask_bytes: tuple[tuple[str, bytes], ...] = field(repr=False)

    def __init__(self, index: int, shape: tuple[int, int], masks: Masks) -> None:
        integer(index)
        validate_shape(shape)
        Limits(1, shape[0] * shape[1], max(1, len(masks))).check(1, shape, 0)
        seen = set()
        owned = []
        occupied = np.zeros(shape, dtype=np.bool_)
        for t, a in masks:
            token(t)
            if t in seen:
                raise ValueError("duplicate token")
            seen.add(t)
            data = snapshot(a, shape)
            a = mask_view(data, shape)
            if not np.any(a) or np.any(occupied & a):
                raise ValueError("visible token masks must be nonempty and disjoint")
            occupied |= a
            owned.append((t, data))
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "_mask_bytes", tuple(sorted(owned, key=lambda p: p[0])))

    @property
    def masks(self) -> Masks:
        return tuple((t, mask_view(data, self.shape)) for t, data in self._mask_bytes)

    def payload(self) -> dict[str, Any]:
        return {"index": self.index, "masks": {t: a.tolist() for t, a in self.masks}}


def _project(raster: VisibleRaster) -> TokenFrame:
    return TokenFrame(
        raster.sequence_index,
        raster.segmentation.shape,
        tuple((t, raster.segmentation == label) for label, t in raster.identities),
    )


class RasterProvider(Protocol):
    def raster(self, sequence_index: int) -> VisibleRaster: ...


@dataclass(frozen=True)
class CausalInput:
    frames: tuple[TokenFrame, ...]
    executed: tuple[Command, ...]
    announced: Command
    permissions: ModalityPermissionSet
    limits: Limits

    def __post_init__(self) -> None:
        permissions(self.permissions)
        if type(self.limits) is not Limits or not self.frames:
            raise ValueError("typed limits and complete nonempty prefix required")
        if any(type(f) is not TokenFrame for f in self.frames):
            raise ValueError("typed token frames required")
        if tuple(f.index for f in self.frames) != tuple(range(len(self.frames))):
            raise ValueError("complete consecutive prefix required")
        shape = self.frames[0].shape
        inventory = {t for f in self.frames for t, _ in f.masks}
        self.limits.check(len(self.frames), shape, len(inventory))
        if any(f.shape != shape for f in self.frames) or len(self.executed) != len(self.frames) - 1:
            raise ValueError("fixed shape and exact executed chronology required")
        for c in (*self.executed, self.announced):
            command(c)
        object.__setattr__(
            self, "frames", tuple(TokenFrame(f.index, shape, f.masks) for f in self.frames)
        )
        object.__setattr__(self, "executed", tuple(self.executed))
        object.__setattr__(self, "permissions", self.permissions.model_copy(deep=True))

    @property
    def inventory(self) -> tuple[str, ...]:
        return tuple(sorted({t for f in self.frames for t, _ in f.masks}))

    @property
    def shape(self) -> tuple[int, int]:
        return self.frames[0].shape

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(
            {
                "version": VERSION,
                "permissions": sorted(m.value for m in REQUIRED),
                "limits": [self.limits.max_frames, self.limits.max_pixels, self.limits.max_tokens],
                "shape": list(self.shape),
                "frames": [f.payload() for f in self.frames],
                "executed": [[str(v) for v in c] for c in self.executed],
                "announced": [str(v) for v in self.announced],
            }
        )

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes, access: ModalityPermissionSet, limits: Limits) -> CausalInput:
        permissions(access)  # Denial precedes parsing or mask allocation.
        if type(limits) is not Limits:
            raise ValueError("typed limits required")
        p = _json(data)
        _keys(p, {"version", "permissions", "limits", "shape", "frames", "executed", "announced"})
        if p["permissions"] != sorted(m.value for m in REQUIRED):
            raise PermissionError("serialized permissions differ")
        if type(p["limits"]) is not list or len(p["limits"]) != 3:
            raise ValueError("strict limits list required")
        for v in p["limits"]:
            integer(v, 1)
        if p["version"] != VERSION or p["limits"] != [
            limits.max_frames,
            limits.max_pixels,
            limits.max_tokens,
        ]:
            raise ValueError("version/declared limits differ")
        shape = _shape(p["shape"])
        if type(p["frames"]) is not list or not p["frames"]:
            raise ValueError("complete prefix required")
        inventory = set()
        for f in p["frames"]:
            _keys(f, {"index", "masks"})
            integer(f["index"])
            if type(f["masks"]) is not dict:
                raise ValueError("token-mask map required")
            for t in f["masks"]:
                token(t)
                inventory.add(t)
        limits.check(len(p["frames"]), shape, len(inventory))
        if [f["index"] for f in p["frames"]] != list(range(len(p["frames"]))):
            raise ValueError("complete consecutive prefix required before expansion")
        frames = tuple(
            TokenFrame(
                f["index"], shape, tuple((t, _mask(a, shape)) for t, a in f["masks"].items())
            )
            for f in p["frames"]
        )
        if type(p["executed"]) is not list:
            raise ValueError("executed list required")
        return cls(
            frames,
            tuple(_command(c) for c in p["executed"]),
            _command(p["announced"]),
            access,
            limits,
        )


@dataclass(frozen=True)
class CausalView:
    provider: RasterProvider
    permissions: ModalityPermissionSet
    decision_index: int
    limits: Limits

    def __post_init__(self) -> None:
        permissions(self.permissions)
        integer(self.decision_index)
        if type(self.limits) is not Limits or self.decision_index >= self.limits.max_frames:
            raise ValueError("decision exceeds finite prefix budget")
        object.__setattr__(self, "permissions", self.permissions.model_copy(deep=True))

    def raster(self, index: int) -> VisibleRaster:
        permissions(self.permissions)
        integer(index)
        if index > self.decision_index:
            raise PermissionError("future denied before provider access")
        r = self.provider.raster(index)
        if type(r) is not VisibleRaster or r.sequence_index != index:
            raise ValueError("provider raster chronology differs")
        self.limits.check(index + 1, r.segmentation.shape, len(r.identities))
        return VisibleRaster(index, r.segmentation, r.identities)

    def materialize(self, executed: tuple[Command, ...], announced: Command) -> CausalInput:
        permissions(self.permissions)
        if len(executed) != self.decision_index:
            raise ValueError("exact executed chronology required before fetch")
        for c in (*executed, announced):
            command(c)
        raw = tuple(self.raster(i) for i in range(self.decision_index + 1))
        shape = raw[0].segmentation.shape
        count = len({t for r in raw for _, t in r.identities})
        self.limits.check(len(raw), shape, count)  # Entire inventory checked before expansion.
        if any(r.segmentation.shape != shape for r in raw):
            raise ValueError("provider shape changed")
        return CausalInput(
            tuple(_project(r) for r in raw), executed, announced, self.permissions, self.limits
        )


def _keys(p: Any, wanted: set[str]) -> None:
    if type(p) is not dict or set(p) != wanted:
        raise ValueError("exact serialized fields required")


def _json(data: bytes) -> dict[str, Any]:
    if type(data) is not bytes:
        raise ValueError("bytes required")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for k, v in items:
            if k in d:
                raise ValueError("duplicate serialized key")
            d[k] = v
        return d

    p = json.loads(data, object_pairs_hook=pairs)
    if type(p) is not dict:
        raise ValueError("object required")
    return p


def _shape(value: Any) -> tuple[int, int]:
    if type(value) is not list or len(value) != 2:
        raise ValueError("shape list required")
    for v in value:
        integer(v, 1)
    return (value[0], value[1])


def _mask(value: Any, shape: tuple[int, int]) -> Array:
    if (
        type(value) is not list
        or len(value) != shape[0]
        or any(
            type(row) is not list or len(row) != shape[1] or any(type(v) is not bool for v in row)
            for row in value
        )
    ):
        raise ValueError("strict Boolean raster required")
    return np.array(value, dtype=np.bool_)


def _command(value: Any) -> Command:
    if type(value) is not list or len(value) != 3 or any(type(v) is not str for v in value):
        raise ValueError("exact command strings required")
    result = tuple(Fraction(v) for v in value)
    if [str(v) for v in result] != value:
        raise ValueError("canonical exact rationals required")
    return (result[0], result[1], result[2])


@dataclass(frozen=True, init=False)
class Forecast:
    input_sha256: str
    rule: str
    window: int
    decision_index: int
    shape: tuple[int, int]
    _mask_bytes: tuple[tuple[str, bytes | None], ...] = field(repr=False)

    def __init__(
        self,
        input_sha256: str,
        rule: str,
        window: int,
        decision_index: int,
        shape: tuple[int, int],
        masks: tuple[tuple[str, Array | None], ...],
    ) -> None:
        if type(input_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", input_sha256) is None:
            raise ValueError("input digest required")
        integer(window, 1)
        integer(decision_index)
        validate_shape(shape)
        if (
            rule not in {"current-mask-persistence", "k-frame-agreement"}
            or (rule == "current-mask-persistence" and window != 1)
            or window > decision_index + 1
        ):
            raise ValueError("predeclared control/window required")
        Limits(1, shape[0] * shape[1], max(1, len(masks))).check(1, shape, 0)
        seen = set()
        owned = []
        for t, a in masks:
            token(t)
            if t in seen:
                raise ValueError("unique forecast inventory required")
            seen.add(t)
            owned.append((t, None if a is None else snapshot(a, shape)))
        object.__setattr__(self, "input_sha256", input_sha256)
        object.__setattr__(self, "rule", rule)
        object.__setattr__(self, "window", window)
        object.__setattr__(self, "decision_index", decision_index)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "_mask_bytes", tuple(sorted(owned, key=lambda p: p[0])))

    @property
    def masks(self) -> tuple[tuple[str, Array | None], ...]:
        return tuple(
            (t, None if data is None else mask_view(data, self.shape))
            for t, data in self._mask_bytes
        )

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(
            {
                "version": VERSION + ":forecast",
                "input_sha256": self.input_sha256,
                "rule": self.rule,
                "window": self.window,
                "decision_index": self.decision_index,
                "shape": list(self.shape),
                "masks": {t: None if a is None else a.tolist() for t, a in self.masks},
            }
        )

    @classmethod
    def from_bytes(cls, data: bytes, source: CausalInput) -> Forecast:
        permissions(source.permissions)
        p = _json(data)
        _keys(p, {"version", "input_sha256", "rule", "window", "decision_index", "shape", "masks"})
        if p["version"] != VERSION + ":forecast" or p["input_sha256"] != source.digest:
            raise ValueError("forecast/input binding differs")
        if type(p["masks"]) is not dict or set(p["masks"]) != set(source.inventory):
            raise ValueError("forecast inventory differs")
        if (
            _shape(p["shape"]) != source.shape
            or type(p["decision_index"]) is not int
            or (p["decision_index"] != len(source.frames) - 1)
        ):
            raise ValueError("forecast shape/chronology differs")
        return cls(
            p["input_sha256"],
            p["rule"],
            p["window"],
            p["decision_index"],
            source.shape,
            tuple(
                (t, None if a is None else _mask(a, source.shape)) for t, a in p["masks"].items()
            ),
        )


def forecast(source: CausalInput, rule: str, window: int = 1) -> Forecast:
    permissions(source.permissions)
    integer(window, 1)
    if window > len(source.frames):
        raise ValueError("required K-window unavailable")
    if rule not in {"current-mask-persistence", "k-frame-agreement"} or (
        rule == "current-mask-persistence" and window != 1
    ):
        raise ValueError("fixed control required")
    selected = source.frames[-window:]
    output: list[tuple[str, Array | None]] = []
    for t in source.inventory:
        # All omitted current-frame tokens are remembered, not destroyed.
        values = [dict(f.masks).get(t) for f in selected]
        last = values[-1]

        def equal(a: Array | None, b: Array | None) -> bool:
            return (a is None and b is None) or (
                a is not None and b is not None and np.array_equal(a, b)
            )

        if all(equal(a, last) for a in values):
            output.append((t, np.zeros(source.shape, dtype=np.bool_) if last is None else last))
        else:
            output.append((t, None))
    return Forecast(
        source.digest, rule, window, len(source.frames) - 1, source.shape, tuple(output)
    )


def storage(source: CausalInput, saved_forecast: bytes) -> dict[str, int]:
    """Logical buffers/UTF-8; Python overhead and allocator peaks are not RSS claims."""
    f = Forecast.from_bytes(saved_forecast, source)
    arrays = sum(a.nbytes for frame in source.frames for _, a in frame.masks)
    output = sum(a.nbytes for _, a in f.masks if a is not None)
    window = sum(a.nbytes for frame in source.frames[-f.window :] for _, a in frame.masks)
    return {
        "materialized_input_mask_bytes": arrays,
        "input_serialized_bytes": len(source.canonical_bytes()),
        "forecast_mask_bytes": output,
        "forecast_serialized_bytes": len(saved_forecast),
        "window_referenced_input_mask_bytes": window,
        "additional_retained_window_bytes": 0,
        "inventory_utf8_bytes": sum(len(t.encode()) for t in source.inventory),
        "commands_utf8_bytes": sum(
            len(str(v).encode()) for c in (*source.executed, source.announced) for v in c
        ),
        "snapshot_mask_bytes": arrays + output,
    }


def evaluate(
    source: CausalInput, saved_forecast: bytes, provider: RasterProvider
) -> dict[str, Any]:
    """Validate a saved forecast before target fetch; never invokes a predictor."""
    f = Forecast.from_bytes(saved_forecast, source)
    r = provider.raster(len(source.frames))
    if type(r) is not VisibleRaster or r.sequence_index != len(source.frames):
        raise ValueError("target chronology differs")
    source.limits.check(1, r.segmentation.shape, len(r.identities))
    if r.segmentation.shape != source.shape:
        raise ValueError("target shape differs")
    lookup = {t: label for label, t in r.identities}
    pixels = source.shape[0] * source.shape[1]
    n = len(source.inventory) * pixels
    e = u = 0
    rows = []
    for t, a in f.masks:
        truth = r.segmentation == lookup.get(t, -1)
        error = None if a is None else int(np.count_nonzero(a != truth))
        e += 0 if error is None else error
        u += pixels if a is None else 0
        rows.append(
            {
                "token": t,
                "target_pixels": int(np.count_nonzero(truth)),
                "target_visible": bool(np.any(truth)),
                "error": error,
                "unknown": a is None,
            }
        )
    new = set(lookup) - set(source.inventory)
    new_pixels = sum(int(np.count_nonzero(r.segmentation == lookup[t])) for t in new)
    total_pixels = int(np.count_nonzero(r.segmentation))
    known = n - u
    strata = {
        name: [row for row in rows if row["target_visible"] == visible]
        for name, visible in (("target_visible", True), ("target_absent", False))
    }
    return {
        "status": "NOT_APPLICABLE" if not n else "SCORED",
        "E": e,
        "U": u,
        "C": known,
        "N": n,
        "conditional_error": None if not known else Fraction(e, known),
        "ignorance_interval": None if not n else (Fraction(e, n), Fraction(e + u, n)),
        "pixel_coverage": None if not n else Fraction(known, n),
        "channel_coverage": None
        if not rows
        else Fraction(sum(not row["unknown"] for row in rows), len(rows)),
        "whole_episode_coverage": None if not n else u == 0,
        "inventory_exact": None if not n else u == 0 and e == 0,
        "omitted_new_tokens": len(new),
        "omitted_new_pixels": new_pixels,
        "future_surface_pixels": total_pixels,
        "inventory_future_pixels": total_pixels - new_pixels,
        "omitted_new_pixel_fraction": None
        if not total_pixels
        else Fraction(new_pixels, total_pixels),
        "tokens": rows,
        "strata": strata,
    }
