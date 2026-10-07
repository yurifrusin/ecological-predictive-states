"""Paired scalar extrapolation/persistence; no native adapter or empirical admission."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Protocol

from epsbench.diagnostics.boundary_observation import VisibleRaster
from epsbench.diagnostics.causal_region_lifecycle import (
    IDENTITY,
    REQUIRED,
    State,
    action_bytes,
    integer,
    keys,
)
from epsbench.diagnostics.causal_region_lifecycle import (
    decode as decode_state,
)
from epsbench.schema import Action, Modality, ModalityPermissionSet
from epsbench.utils.canonical import canonical_json_bytes, sha256_bytes

VERSION = "fraction-extrapolation-v1"
REASONS = ("INITIAL_HISTORY", "INCOMPLETE_HISTORY", "EXECUTED_NOT_LATERAL", "ANNOUNCED_NOT_LATERAL")


def permissions(value: ModalityPermissionSet) -> None:
    if (
        type(value) is not ModalityPermissionSet
        or type(value.allowed) is not frozenset
        or any(type(m) is not Modality for m in value.allowed)
        or value.allowed != REQUIRED
    ):
        raise PermissionError("exact lifecycle permissions required before parsing/access")


@dataclass(frozen=True)
class Input:
    """Accepted before-only state and announced command; provenance is not a feature."""

    state: State
    announced: bytes
    episode_key: str
    source_head: str
    history_complete: bool
    permissions: ModalityPermissionSet
    decision_index: int

    def __post_init__(self) -> None:
        permissions(self.permissions)
        if type(self.state) is not State or type(self.announced) is not bytes:
            raise ValueError("exact accepted state and canonical Action bytes required")
        state = decode_state(self.state.canonical_bytes())
        announced = action_bytes(Action.model_validate_json(self.announced))
        if announced != self.announced:
            raise ValueError("canonical announced Action required")
        if type(self.history_complete) is not bool:
            raise ValueError("exact trusted history assertion required")
        for value, pattern in (
            (self.episode_key, r"[0-9a-f]{32}"),
            (self.source_head, r"[0-9a-f]{40}"),
        ):
            if type(value) is not str or re.fullmatch(pattern, value) is None:
                raise ValueError("opaque episode and exact source identity required")
        if state.decision_index > integer(self.decision_index):
            raise PermissionError("future state denied before access")
        object.__setattr__(self, "state", state)


@dataclass(frozen=True)
class Binding:
    episode_key: str
    source_head: str
    lifecycle_root: str
    announced_sha256: str
    observation_index: int
    target_index: int
    shape: tuple[int, int]
    identity_kind: str = IDENTITY


def binding(source: Input) -> Binding:
    s = source.state
    return Binding(
        source.episode_key,
        source.source_head,
        s.digest(),
        sha256_bytes(source.announced),
        s.decision_index,
        s.decision_index + 1,
        s.shape,
    )


@dataclass(frozen=True)
class Prediction:
    token: str
    extrapolation: float | None
    persistence: float | None


@dataclass(frozen=True)
class Forecast:
    binding: Binding
    unknown_reasons: tuple[str, ...]
    predictions: tuple[Prediction, ...]

    def __post_init__(self) -> None:
        if type(self.binding) is not Binding or type(self.predictions) is not tuple:
            raise ValueError("immutable exact forecast records required")
        if type(self.unknown_reasons) is not tuple or self.unknown_reasons != tuple(
            r for r in REASONS if r in self.unknown_reasons
        ):
            raise ValueError("unique ordered eligibility reasons required")
        tokens = []
        for p in self.predictions:
            if (
                type(p) is not Prediction
                or type(p.token) is not str
                or re.fullmatch(r"surface-[0-9a-f]{16}", p.token) is None
            ):
                raise ValueError("exact opaque token prediction required")
            tokens.append(p.token)
            for value in (p.extrapolation, p.persistence):
                if self.unknown_reasons:
                    if value is not None:
                        raise ValueError("unsupported paired context must be UNKNOWN")
                elif type(value) is not float or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("finite binary64 fraction required")
        if tokens != sorted(set(tokens)):
            raise ValueError("unique canonical inventory required")

    def feature_bytes(self) -> bytes:
        return canonical_json_bytes(
            {
                "version": VERSION,
                "unknown_reasons": self.unknown_reasons,
                "predictions": [asdict(p) for p in self.predictions],
            }
        )

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(
            {
                "version": VERSION,
                "binding": asdict(self.binding),
                "unknown_reasons": self.unknown_reasons,
                "predictions": [asdict(p) for p in self.predictions],
            }
        )


def forecast(source: Input) -> Forecast:
    if type(source) is not Input:
        raise ValueError("exact before-only input required")
    permissions(source.permissions)
    s = source.state
    announced = Action.model_validate_json(source.announced)
    reasons = []
    if s.decision_index == 0:
        reasons.append("INITIAL_HISTORY")
    if not source.history_complete:
        reasons.append("INCOMPLETE_HISTORY")
    last = Action.model_validate_json(s.executed_commands[-1]) if s.executed_commands else None
    if last is not None and (
        last.delta_forward != 0 or last.delta_yaw != 0 or last.delta_lateral == 0
    ):
        reasons.append("EXECUTED_NOT_LATERAL")
    if announced.delta_forward != 0 or announced.delta_yaw != 0 or announced.delta_lateral == 0:
        reasons.append("ANNOUNCED_NOT_LATERAL")
    predictions = []
    area = s.shape[0] * s.shape[1]
    for node in s.nodes:
        if reasons:
            predictions.append(Prediction(node.token, None, None))
        else:
            assert last is not None and node.previous_mask is not None
            a = Fraction(int.from_bytes(node.current_mask).bit_count(), area)
            b = Fraction(int.from_bytes(node.previous_mask).bit_count(), area)
            ratio = Fraction(announced.delta_lateral) / Fraction(last.delta_lateral)
            clipped = min(Fraction(1), max(Fraction(0), a + ratio * (a - b)))
            predictions.append(Prediction(node.token, float(clipped), float(a)))
    return Forecast(binding(source), tuple(reasons), tuple(predictions))


def decode(payload: bytes, source: Input) -> Forecast:
    """Validate all saved predictions and context before evaluator-only fetching."""
    if type(source) is not Input:
        raise ValueError("exact before-only input required")
    permissions(source.permissions)
    if type(payload) is not bytes:
        raise ValueError("exact canonical forecast bytes required")
    root = keys(json.loads(payload), {"version", "binding", "unknown_reasons", "predictions"})
    if (
        root["version"] != VERSION
        or type(root["unknown_reasons"]) is not list
        or type(root["predictions"]) is not list
    ):
        raise ValueError("versioned canonical forecast arrays required")
    b = keys(root["binding"], set(asdict(binding(source))))
    if type(b["shape"]) is not list or len(b["shape"]) != 2:
        raise ValueError("exact shape required")
    b["shape"] = tuple(b["shape"])
    predictions = []
    for value in root["predictions"]:
        p = keys(value, {"token", "extrapolation", "persistence"})
        predictions.append(Prediction(p["token"], p["extrapolation"], p["persistence"]))
    result = Forecast(Binding(**b), tuple(root["unknown_reasons"]), tuple(predictions))
    expected = forecast(source)
    if result.canonical_bytes() != payload or expected.canonical_bytes() != payload:
        raise ValueError("saved forecast differs from canonical before-state/action/context/rule")
    return result


@dataclass(frozen=True)
class TargetEvidence:
    binding: Binding
    raster: VisibleRaster | None
    complete_image: bool
    complete_association: bool
    stable_identity: bool
    announced_action_executed: bool


class Provider(Protocol):
    def target(self, index: int) -> TargetEvidence: ...


@dataclass(frozen=True)
class Stratum:
    name: str
    tokens: int
    predicted: int
    extrapolation_mae: Fraction | None
    persistence_mae: Fraction | None
    difference: Fraction | None


@dataclass(frozen=True)
class Report:
    strata: tuple[Stratum, ...]
    targets: tuple[tuple[str, Fraction], ...]
    new_visible_tokens: int
    new_pixels: int
    image_area: int
    changed_known_tokens: int
    mean_absolute_support_change: Fraction | None
    extrapolation_sum: Fraction | None
    persistence_sum: Fraction | None
    extrapolation_sum_gt_one: bool | None


@dataclass(frozen=True)
class Evaluation:
    report: Report | None
    unresolved: str | None


def evaluate(source: Input, saved: bytes, provider: Provider) -> Evaluation:
    f = decode(saved, source)  # Must finish before any future observation call.
    ev = provider.target(f.binding.target_index)
    if type(ev) is not TargetEvidence or ev.binding != f.binding:
        return Evaluation(None, "target does not bind exact episode/state/action/source/index")
    if any(
        v is not True
        for v in (
            ev.complete_image,
            ev.complete_association,
            ev.stable_identity,
            ev.announced_action_executed,
        )
    ):
        return Evaluation(None, "complete same-episode stable association/action unresolved")
    if type(ev.raster) is not VisibleRaster:
        return Evaluation(None, "target raster missing")
    r = ev.raster
    if r.sequence_index != f.binding.target_index or r.segmentation.shape != source.state.shape:
        return Evaluation(None, "target chronology/shape differs")
    # Independent target counts: traverse original raster samples, never forecast/mask counts.
    counts = dict.fromkeys(source.state.inventory, 0)
    lookup = dict(r.identities)
    newly_visible = set()
    new_pixels = 0
    for raw in r.segmentation.flat:
        if int(raw) == 0:
            continue
        token = lookup[int(raw)]
        if token in counts:
            counts[token] += 1
        else:
            newly_visible.add(token)
            new_pixels += 1
    area = source.state.shape[0] * source.state.shape[1]
    targets = {token: Fraction(count, area) for token, count in counts.items()}
    current = {
        n.token: Fraction(int.from_bytes(n.current_mask).bit_count(), area)
        for n in source.state.nodes
    }
    visible = {n.token for n in source.state.nodes if n.status == "VISIBLE"}
    rows = []
    for name, selected in (
        ("KNOWN", set(counts)),
        ("CURRENT_VISIBLE", visible),
        ("REMEMBERED_ABSENT", set(counts) - visible),
    ):
        ps = [p for p in f.predictions if p.token in selected]
        predicted = sum(p.extrapolation is not None for p in ps)
        extra = persist = None
        if ps and predicted == len(ps):
            extra = sum(
                (
                    abs(Fraction(p.extrapolation) - targets[p.token])
                    for p in ps
                    if p.extrapolation is not None
                ),
                Fraction(),
            ) / len(ps)
            persist = sum(
                (
                    abs(Fraction(p.persistence) - targets[p.token])
                    for p in ps
                    if p.persistence is not None
                ),
                Fraction(),
            ) / len(ps)
        rows.append(
            Stratum(
                name,
                len(ps),
                predicted,
                extra,
                persist,
                None if extra is None or persist is None else extra - persist,
            )
        )
    extra_sum = persist_sum = None
    if not f.unknown_reasons:
        extra_sum = sum(
            (Fraction(p.extrapolation) for p in f.predictions if p.extrapolation is not None),
            Fraction(),
        )
        persist_sum = sum(
            (Fraction(p.persistence) for p in f.predictions if p.persistence is not None),
            Fraction(),
        )
    change = sum((abs(targets[t] - current[t]) for t in counts), Fraction())
    report = Report(
        tuple(rows),
        tuple(targets.items()),
        len(newly_visible),
        new_pixels,
        area,
        sum(targets[t] != current[t] for t in counts),
        change / len(counts) if counts else None,
        extra_sum,
        persist_sum,
        None if extra_sum is None else extra_sum > 1,
    )
    return Evaluation(report, None)
