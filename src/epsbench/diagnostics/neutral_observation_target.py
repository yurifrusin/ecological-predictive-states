"""Neutral known/NEW targets; no predictor, native adapter or optical-cause witness."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from types import MappingProxyType
from typing import Any

import numpy as np

from epsbench.annotations.derive import derive_visibility
from epsbench.diagnostics.boundary_observation import VisibleRaster
from epsbench.diagnostics.visible_forecast_contract import (
    Array,
    CausalInput,
    RasterProvider,
    _json,
    _keys,
    _mask,
    _shape,
    integer,
    mask_view,
    snapshot,
    token,
    validate_shape,
)
from epsbench.schema import SurfaceReference
from epsbench.utils.canonical import canonical_json_bytes

VERSION = "neutral-observation-target-v1"
CAUSE_STATUS = "UNAVAILABLE_UNSUPPORTED"
CAUSE_REASON = "no separately qualified optical-cause witness/site-domain contract admitted"
Masks = tuple[tuple[str, Array | None], ...]


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source(source: CausalInput) -> CausalInput:
    if type(source) is not CausalInput:
        raise ValueError("typed causal input required")
    # Revalidate/snapshot even direct caller data, using unchanged input permissions/limits.
    return CausalInput.from_bytes(source.canonical_bytes(), source.permissions, source.limits)


@dataclass(frozen=True, init=False)
class _Channels:
    shape: tuple[int, int]
    _known: tuple[tuple[str, bytes | None], ...] = field(repr=False)
    _new: bytes | None = field(repr=False)

    def __init__(self, shape: tuple[int, int], known: Masks, new: Array | None) -> None:
        validate_shape(shape)
        seen: set[str] = set()
        owned = []
        occupied = np.zeros(shape, dtype=np.bool_)
        for name, value in (*known, ("NEW", new)):
            if name != "NEW":
                token(name)
            if name in seen or (name == "NEW" and len(seen) != len(known)):
                raise ValueError("unique known channels and reserved NEW channel required")
            seen.add(name)
            data = None if value is None else snapshot(value, shape)
            if data is not None:
                view = mask_view(data, shape)
                if np.any(occupied & view):
                    raise ValueError("asserted channels must not overlap")
                occupied |= view
            owned.append((name, data))
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "_known", tuple(sorted(owned[:-1])))
        object.__setattr__(self, "_new", owned[-1][1])

    @property
    def known(self) -> Masks:
        return tuple(
            (name, None if data is None else mask_view(data, self.shape))
            for name, data in self._known
        )

    @property
    def new(self) -> Array | None:
        return None if self._new is None else mask_view(self._new, self.shape)

    def payload(self) -> dict[str, Any]:
        return {
            "known": {name: None if a is None else a.tolist() for name, a in self.known},
            "new": None if self.new is None else self.new.tolist(),
        }


@dataclass(frozen=True, init=False)
class Candidate:
    input_sha256: str
    target_index: int
    channels: _Channels

    def __init__(self, source: CausalInput, known: Masks, new: Array | None) -> None:
        source = _source(source)
        if tuple(sorted(name for name, _ in known)) != source.inventory:
            raise ValueError("candidate requires exactly prefix-known channels")
        object.__setattr__(self, "input_sha256", source.digest)
        object.__setattr__(self, "target_index", len(source.frames))
        object.__setattr__(self, "channels", _Channels(source.shape, known, new))

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(
            {
                "version": VERSION + ":candidate",
                "input_sha256": self.input_sha256,
                "target_index": self.target_index,
                "shape": list(self.channels.shape),
                **self.channels.payload(),
            }
        )

    @classmethod
    def from_bytes(cls, data: bytes, source: CausalInput) -> Candidate:
        source = _source(source)
        p = _json(data)
        _keys(p, {"version", "input_sha256", "target_index", "shape", "known", "new"})
        integer(p["target_index"], 1)
        if (
            p["version"] != VERSION + ":candidate"
            or p["input_sha256"] != source.digest
            or p["target_index"] != len(source.frames)
            or _shape(p["shape"]) != source.shape
            or type(p["known"]) is not dict
            or set(p["known"]) != set(source.inventory)
        ):
            raise ValueError("candidate version/input/chronology/shape/inventory differs")
        # Exact inventory/budgets precede array expansion; NEW adds one reserved channel.
        return cls(
            source,
            tuple(
                (name, None if value is None else _mask(value, source.shape))
                for name, value in p["known"].items()
            ),
            None if p["new"] is None else _mask(p["new"], source.shape),
        )


@dataclass(frozen=True, init=False)
class SemanticTarget:
    target_index: int
    channels: _Channels

    def __init__(self, target_index: int, shape: tuple[int, int], known: Masks, new: Array) -> None:
        integer(target_index, 1)
        if new is None or any(a is None for _, a in known):
            raise ValueError("semantic target channels cannot be UNKNOWN")
        object.__setattr__(self, "target_index", target_index)
        object.__setattr__(self, "channels", _Channels(shape, known, new))

    def canonical_bytes(self) -> bytes:
        # Candidate/input/provenance hashes and unseen identities/partition are excluded.
        return canonical_json_bytes(
            {
                "version": VERSION + ":semantic",
                "target_index": self.target_index,
                "shape": list(self.channels.shape),
                **self.channels.payload(),
            }
        )

    @property
    def digest(self) -> str:
        return _digest(self.canonical_bytes())


def _target(source: CausalInput, raster: VisibleRaster) -> SemanticTarget:
    if type(raster) is not VisibleRaster or raster.sequence_index != len(source.frames):
        raise ValueError("target chronology differs")
    source.limits.check(1, raster.segmentation.shape, len(raster.identities))
    if raster.segmentation.shape != source.shape:
        raise ValueError("target shape differs")
    raster = VisibleRaster(raster.sequence_index, raster.segmentation, raster.identities)
    lookup = {name: label for label, name in raster.identities}
    known = tuple((name, raster.segmentation == lookup.get(name, -1)) for name in source.inventory)
    new = np.isin(
        raster.segmentation,
        tuple(label for name, label in lookup.items() if name not in source.inventory),
    )
    return SemanticTarget(len(source.frames), source.shape, known, new)


def _replay(source: CausalInput, channels: _Channels) -> dict[str, Any]:
    current = dict(source.frames[-1].masks)
    labels: dict[str, Any] = {}
    for name, after_mask in channels.known:
        if after_mask is None:
            labels[name] = None
            continue
        before_mask = current.get(name, np.zeros(source.shape, dtype=np.bool_))
        states, correspondence, changes = derive_visibility(
            before_mask.astype(np.int32),
            after_mask.astype(np.int32),
            (SurfaceReference(surface_id=name, segmentation_label=1),),
        )
        labels[name] = {
            "visibility": states[0].model_dump(mode="json"),
            "correspondence": correspondence[0].model_dump(mode="json"),
            "changes": [change.model_dump(mode="json") for change in changes],
        }
    return labels


@dataclass(frozen=True)
class Evaluation:
    target: SemanticTarget
    receipt_bytes: bytes
    report: Mapping[str, Any]


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def evaluate(source: CausalInput, saved_candidate: bytes, provider: RasterProvider) -> Evaluation:
    """Validate all saved assertions before one evaluator-only target fetch."""
    source = _source(source)
    candidate = Candidate.from_bytes(saved_candidate, source)
    raster = provider.raster(len(source.frames))
    target = _target(source, raster)
    rows: list[dict[str, Any]] = []
    for (name, prediction), (_, truth) in zip(
        (*candidate.channels.known, ("NEW", candidate.channels.new)),
        (*target.channels.known, ("NEW", target.channels.new)),
        strict=True,
    ):
        assert truth is not None
        rows.append(
            {
                "channel": name,
                "target_pixels": int(np.count_nonzero(truth)),
                "unknown": prediction is None,
                "error": None if prediction is None else int(np.count_nonzero(prediction != truth)),
            }
        )
    pixels = source.shape[0] * source.shape[1]
    n = len(rows) * pixels  # NEW remains applicable even with an empty prefix inventory.
    u = sum(pixels for row in rows if row["unknown"])
    c = n - u
    e = sum(row["error"] or 0 for row in rows)
    occupied = sum(row["target_pixels"] for row in rows)
    report = {
        "N": n,
        "U": u,
        "C": c,
        "E": e,
        "conditional_error": None if not c else Fraction(e, c),
        "ignorance_interval": (Fraction(e, n), Fraction(e + u, n)),
        "channel_pixel_coverage": Fraction(c, n),
        "channel_coverage": Fraction(len(rows) - u // pixels, len(rows)),
        "complete_assertion": u == 0,
        "observation_exact": u == 0 and e == 0,
        "target_occupied_pixels": occupied,
        "first_observed_pixels": rows[-1]["target_pixels"],
        "first_observed_fraction": None
        if not occupied
        else Fraction(rows[-1]["target_pixels"], occupied),
        "channels": rows,
        "neutral_target_labels": _replay(source, target.channels),
        "neutral_candidate_labels": _replay(source, candidate.channels),
        "cause_status": CAUSE_STATUS,
        "cause_reason": CAUSE_REASON,
    }
    receipt = canonical_json_bytes(
        {
            "version": VERSION + ":receipt",
            "input_sha256": source.digest,
            "candidate_sha256": _digest(saved_candidate),
            "semantic_target_sha256": target.digest,
            "target_index": len(source.frames),
            "shape": list(source.shape),
            "observation_domain": "visible-raster-oracle-association",
            "observation_sha256": _digest(
                canonical_json_bytes(
                    {
                        "index": raster.sequence_index,
                        "segmentation": raster.segmentation.tolist(),
                        "identities": raster.identities,
                    }
                )
            ),
        }
    )
    return Evaluation(target, receipt, _freeze(report))
