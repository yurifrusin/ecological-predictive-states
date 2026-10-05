"""Trusted evaluator archives and exact finite pair audit, never predictor inputs."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from epsbench.diagnostics.causal_history_core import (
    Array,
    CausalState,
    Command,
    CompletedFlow,
    Forecast,
    MemorySequence,
    OpticalFrame,
    TemporalProjection,
    boundary_key,
    build_state,
    index,
    owned,
)
from epsbench.schema import ModalityPermissionSet


@dataclass(frozen=True)
class AuditHistory:
    frames: tuple[OpticalFrame, ...]
    flows: tuple[CompletedFlow, ...]
    rgbs: tuple[Array, ...]
    state: CausalState

    def __post_init__(self) -> None:
        if not isinstance(self.state, CausalState):
            raise ValueError("typed owned state required")
        t = self.state.current.sequence_index
        frames, flows, rgbs = tuple(self.frames), tuple(self.flows), tuple(self.rgbs)
        shape = self.state.current.segmentation.shape
        if (
            len(frames) != t + 1
            or len(flows) != t
            or len(rgbs) != t + 1
            or any(
                not isinstance(f, OpticalFrame)
                or f.sequence_index != i
                or f.segmentation.shape != shape
                for i, f in enumerate(frames)
            )
            or any(
                not isinstance(f, CompletedFlow) or f.source_index != i or f.validity.shape != shape
                for i, f in enumerate(flows)
            )
            or any(rgb.dtype != np.uint8 or rgb.shape != (*shape, 3) for rgb in rgbs)
        ):
            raise ValueError("typed matched chronological archive required")
        object.__setattr__(self, "frames", frames)
        object.__setattr__(self, "flows", flows)
        object.__setattr__(self, "rgbs", tuple(owned(rgb) for rgb in rgbs))


def archive(view: TemporalProjection, announced: Command) -> AuditHistory:
    """Separate trusted projection requires matched RGB as well as ecological evidence."""
    frames = tuple(view.frame(i) for i in range(view.decision + 1))
    flows = tuple(view.flow(i) for i in range(view.decision))
    rgbs = tuple(view.rgb(i) for i in range(view.decision + 1))
    state = build_state(
        TemporalProjection(MemorySequence(frames, flows, rgbs), view.permissions, view.decision),
        announced,
    )
    if any(rgb.shape[:2] != state.current.segmentation.shape for rgb in rgbs):
        raise ValueError("matched RGB/optical raster differs")
    return AuditHistory(frames, flows, rgbs, state)


@dataclass(frozen=True)
class EvaluatorTarget:
    sequence_index: int
    token: str
    mask: Array

    def __post_init__(self) -> None:
        index(self.sequence_index)
        if self.mask.dtype != np.bool_ or self.mask.ndim != 2:
            raise ValueError("binary evaluator target required")
        object.__setattr__(self, "mask", owned(self.mask))


def optical_equal(a: OpticalFrame, b: OpticalFrame, alignment: dict[str, str]) -> bool:
    if a.sequence_index != b.sequence_index or a.segmentation.shape != b.segmentation.shape:
        return False
    if not np.array_equal(a.segmentation == 0, b.segmentation == 0):
        return False
    if {alignment[t] for _, t in a.identities} != {t for _, t in b.identities}:
        return False
    if any(not np.array_equal(a.mask(t), b.mask(alignment[t])) for _, t in a.identities):
        return False
    identity = {t: t for _, t in b.identities}
    return sorted(boundary_key(p, alignment) for p in a.boundaries) == sorted(
        boundary_key(p, identity) for p in b.boundaries
    )


def state_equal(a: CausalState, b: CausalState, alignment: dict[str, str]) -> bool:
    if (
        a.version != b.version
        or a.executed != b.executed
        or a.announced != b.announced
        or not optical_equal(a.current, b.current, alignment)
        or {alignment[t.token] for t in a.tokens} != {t.token for t in b.tokens}
    ):
        return False
    right = {t.token: t for t in b.tokens}
    for t in a.tokens:
        other = right[alignment[t.token]]
        if (
            t.visible != other.visible
            or t.last_index != other.last_index
            or t.motion != other.motion
            or not np.array_equal(t.last_mask, other.last_mask)
        ):
            return False
    return True


def validate_history(history: AuditHistory) -> None:
    """Rebuild and bind retained state to the entire supplied past archive."""
    t = history.state.current.sequence_index
    if len(history.frames) != t + 1 or len(history.flows) != t or len(history.rgbs) != t + 1:
        raise ValueError("incomplete matched past window")
    view = TemporalProjection(
        MemorySequence(history.frames, history.flows, history.rgbs),
        ModalityPermissionSet.all_modalities(),
        t,
    )
    rebuilt = archive(view, history.state.announced)
    identity = {s.token: s.token for s in rebuilt.state.tokens}
    if not state_equal(rebuilt.state, history.state, identity):
        raise ValueError("state does not bind to supplied history")


@dataclass(frozen=True)
class PairAudit:
    classification: str
    reason: str | None = None
    current_equal: bool | None = None
    oracle_equal: bool | None = None
    state_equal: bool | None = None
    rgb_action_equal: bool | None = None
    target_equal: bool | None = None
    mathematical_classification: str | None = None
    fixture_valid: bool = False


def audit_pair(
    a: AuditHistory,
    b: AuditHistory,
    target_a: EvaluatorTarget,
    target_b: EvaluatorTarget,
    alignment: dict[str, str],
    pair: int,
) -> PairAudit:
    """One supplied bijection for whole histories, state and evaluator designation."""
    try:
        if pair not in (1, 2, 3):
            raise ValueError("exact declared pair required")
        validate_history(a)
        validate_history(b)
        ta = {t.token for t in a.state.tokens}
        tb = {t.token for t in b.state.tokens}
        if set(alignment) != ta or set(alignment.values()) != tb or len(ta) != len(tb):
            raise ValueError("one complete bijective token alignment required")
        t = a.state.current.sequence_index
        shape = a.state.current.segmentation.shape
        if (
            b.state.current.sequence_index != t
            or b.state.current.segmentation.shape != shape
            or target_a.sequence_index != t + 1
            or target_b.sequence_index != t + 1
            or target_a.mask.shape != shape
            or target_b.mask.shape != shape
            or target_a.token not in ta
            or target_b.token not in tb
            or alignment[target_a.token] != target_b.token
        ):
            raise ValueError("matched chronology/raster/observed target designation differs")
        current = optical_equal(a.state.current, b.state.current, alignment)
        actions = a.state.executed == b.state.executed and a.state.announced == b.state.announced
        rgb = actions and all(np.array_equal(x, y) for x, y in zip(a.rgbs, b.rgbs, strict=True))
        oracle = (
            actions
            and all(optical_equal(x, y, alignment) for x, y in zip(a.frames, b.frames, strict=True))
            and all(
                x.source_index == y.source_index
                and x.command == y.command
                and np.array_equal(x.vectors, y.vectors)
                and np.array_equal(x.validity, y.validity)
                and np.array_equal(x.reasons, y.reasons)
                for x, y in zip(a.flows, b.flows, strict=True)
            )
        )
        state = state_equal(a.state, b.state, alignment)
        target = np.array_equal(target_a.mask, target_b.mask)
        values = (current, oracle, state, rgb, target)
        mathematical = (
            "NONCOLLISION"
            if not state
            else "NO_DIFFERENT_TARGET"
            if target
            else "FINITE_E_SPECIFIC_LOSS"
            if not rgb
            else "SHARED_RGB_AMBIGUITY"
        )
        if t != {1: 2, 2: 3, 3: 1}[pair]:
            return PairAudit(
                "INCONCLUSIVE", "declared pair timing failed", *values, mathematical, False
            )
        for history, target_item in ((a, target_a), (b, target_b)):
            if any(
                not c.supported for c in (*history.state.executed, history.state.announced)
            ) or not any(np.any(f.mask(target_item.token)) for f in history.frames[:-1]):
                return PairAudit(
                    "INCONCLUSIVE",
                    "pure lateral/previously observed target condition failed",
                    *values,
                    mathematical,
                    False,
                )
        if pair == 2:
            for history, target_item in ((a, target_a), (b, target_b)):
                if (
                    not np.any(target_item.mask)
                    or any(not np.any(history.frames[i].mask(target_item.token)) for i in (0, 1))
                    or any(np.any(history.frames[i].mask(target_item.token)) for i in (2, 3))
                    or not any(
                        flow.command.supported
                        and flow.command.lateral != 0
                        and np.any(
                            history.frames[flow.source_index].mask(target_item.token)
                            & (flow.validity == 1)
                        )
                        for flow in history.flows[:2]
                    )
                ):
                    return PairAudit(
                        "INCONCLUSIVE",
                        "Pair 2 occlusion/motion condition failed",
                        *values,
                        mathematical,
                        False,
                    )
        if not current or a.state.announced != b.state.announced or target:
            return PairAudit(
                "INCONCLUSIVE",
                "declared current/command/different-target relation failed",
                *values,
                mathematical,
                False,
            )
        if pair == 3:
            classification = "SHARED_AMBIGUITY" if rgb and oracle and state else "INCONCLUSIVE"
            reason = (
                None if classification == "SHARED_AMBIGUITY" else "Pair 3 history equality failed"
            )
        elif rgb:
            classification, reason = "INCONCLUSIVE", "Pair 1/2 RGB histories not distinguishable"
        elif state:
            classification, reason = "FINITE_E_SPECIFIC_LOSS", None
        else:
            classification, reason = "NONCOLLISION", None
        return PairAudit(
            classification, reason, *values, mathematical, classification != "INCONCLUSIVE"
        )
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as error:
        return PairAudit("INCONCLUSIVE", str(error))


@dataclass(frozen=True)
class MaskScore:
    status: str
    exact: bool | None
    error_pixels: int | None
    iou: Fraction | None


def score(forecast: Forecast, target: EvaluatorTarget) -> MaskScore:
    if forecast.token != target.token:
        raise ValueError("score token designation differs")
    if forecast.mask is None:
        return MaskScore(forecast.status, None, None, None)
    if forecast.mask.shape != target.mask.shape:
        raise ValueError("score raster differs")
    error = int(np.count_nonzero(forecast.mask != target.mask))
    union = int(np.count_nonzero(forecast.mask | target.mask))
    intersection = int(np.count_nonzero(forecast.mask & target.mask))
    return MaskScore(
        forecast.status, error == 0, error, Fraction(intersection, union) if union else Fraction(1)
    )
