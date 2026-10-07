"""Handwritten scalar arithmetic and independent public next-raster target cases."""

from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from epsbench.diagnostics.boundary_observation import VisibleRaster
from epsbench.diagnostics.causal_region_lifecycle import (
    REQUIRED,
    State,
    TrustedObservation,
    action_bytes,
)
from epsbench.diagnostics.causal_region_lifecycle import advance as observe
from epsbench.diagnostics.fraction_extrapolation import (
    Input,
    TargetEvidence,
    binding,
    decode,
    evaluate,
    forecast,
)
from epsbench.schema import Action, Modality, ModalityPermissionSet
from epsbench.utils.canonical import canonical_json_bytes, sha256_bytes

A, B, C = ("surface-" + f"{i:016x}" for i in (1, 2, 3))
EPISODE = "a" * 32
HEAD = "b" * 40
PERMISSIONS = ModalityPermissionSet(allowed=REQUIRED)


def lateral(q: float) -> Action:
    return Action(
        name="lateral_right" if q > 0 else "lateral_left",
        delta_forward=0.0,
        delta_lateral=q,
        delta_yaw=0.0,
    )


def raster(i: int, pixels: list[int], tokens: tuple[str, ...] = (A, B, C)) -> VisibleRaster:
    labels = sorted(set(pixels) - {0})
    return VisibleRaster(
        i, np.array([pixels], dtype=np.int32), tuple((k, tokens[k - 1]) for k in labels)
    )


class Observation:
    def __init__(self, value: VisibleRaster) -> None:
        self.value = value

    def observation(self, index: int) -> TrustedObservation:
        return TrustedObservation(self.value, True, True, True)


def state(
    frames: list[list[int]],
    d: float = 0.5,
    executed: Action | None = None,
    tokens: tuple[str, ...] = (A, B, C),
) -> State:
    previous = None
    for i, pixels in enumerate(frames):
        result = observe(
            previous,
            Observation(raster(i, pixels, tokens)),
            PERMISSIONS,
            i,
            i,
            None if i == 0 else executed or lateral(d),
        )
        assert result.state is not None and result.unresolved is None
        previous = result.state
    assert previous is not None
    return previous


def source(s: State, q: float = 0.5, complete: bool = True) -> Input:
    return Input(
        s, action_bytes(lateral(q)), EPISODE, HEAD, complete, PERMISSIONS, s.decision_index
    )


class Target:
    def __init__(self, value: TargetEvidence) -> None:
        self.value = value
        self.calls: list[int] = []

    def target(self, i: int) -> TargetEvidence:
        self.calls.append(i)
        return self.value


def target(value: Input, pixels: list[int]) -> Target:
    return Target(
        TargetEvidence(
            binding(value), raster(value.state.decision_index + 1, pixels), True, True, True, True
        )
    )


@pytest.mark.parametrize(
    "d,q,expected",
    [
        (0.5, 0.5, Fraction(3, 4)),
        (0.5, -0.5, Fraction(1, 4)),
        (-0.5, 0.5, Fraction(1, 4)),
        (-0.5, -0.5, Fraction(3, 4)),
        (0.5, 0.25, Fraction(5, 8)),
        (0.5, 2.0, Fraction(1)),
        (0.5, -2.0, Fraction(0)),
        (float.fromhex("0x0.0000000000001p-1022"), 1e308, Fraction(1)),
    ],
)
def test_direct_rational_examples(d: float, q: float, expected: Fraction) -> None:
    s = state([[1, 0, 0, 0], [1, 1, 0, 0]], d)
    value = source(s, q)
    f = decode(forecast(value).canonical_bytes(), value)
    assert f.predictions[0].extrapolation == float(expected)
    assert f.predictions[0].persistence == 0.5


def test_binary64_round_once_and_score_saved_value() -> None:
    s = state([[0, 0, 0], [1, 0, 0]])
    value = source(s, 0.25)
    f = forecast(value)
    assert f.predictions[0].extrapolation == 0.5
    assert f.predictions[0].persistence == float(Fraction(1, 3))
    result = evaluate(value, f.canonical_bytes(), target(value, [1, 0, 0]))
    assert result.report is not None
    row = result.report.strata[0]
    assert row.extrapolation_mae == Fraction(1, 6)
    assert row.persistence_mae == abs(Fraction(float(Fraction(1, 3))) - Fraction(1, 3))
    assert row.persistence_mae != 0  # Saved float, not concealed exact 1/3.


@pytest.mark.parametrize("case", ["initial", "history", "last_forward", "next_forward"])
def test_same_unknown_eligibility_and_coverage(case: str) -> None:
    forward = Action(name="forward", delta_forward=0.5, delta_lateral=0.0, delta_yaw=0.0)
    s = state(
        [[1, 2, 0]] if case == "initial" else [[1, 2, 0], [1, 0, 0]],
        executed=forward if case == "last_forward" else None,
    )
    value = source(s, complete=case != "history")
    if case == "next_forward":
        value = replace(value, announced=action_bytes(forward))
    f = forecast(value)
    assert f.unknown_reasons and tuple(p.token for p in f.predictions) == s.inventory
    assert all(p.extrapolation is None and p.persistence is None for p in f.predictions)
    result = evaluate(value, f.canonical_bytes(), target(value, [1, 0, 0]))
    assert result.report is not None
    assert result.report.strata[0].tokens == 2 and result.report.strata[0].predicted == 0
    assert all(row.difference is None for row in result.report.strata)


def test_empty_inventory_not_applicable() -> None:
    value = source(state([[0, 0], [0, 0]]))
    f = forecast(value)
    result = evaluate(value, f.canonical_bytes(), target(value, [3, 0]))
    assert result.report is not None and not f.predictions
    assert result.report.strata[0].tokens == 0 and result.report.strata[0].difference is None
    assert result.report.new_visible_tokens == 1 and result.report.new_pixels == 1


def test_remembered_zero_first_seen_and_unphysical_total() -> None:
    # The current A and new B independently grow to 1, retaining total > 1.
    value = source(state([[1, 0, 0, 0], [1, 1, 2, 2]]))
    f = forecast(value)
    assert [p.extrapolation for p in f.predictions] == [0.75, 1.0]
    result = evaluate(value, f.canonical_bytes(), target(value, [1, 2, 3, 0]))
    assert result.report is not None
    assert result.report.extrapolation_sum == Fraction(7, 4)
    assert result.report.extrapolation_sum_gt_one is True
    assert result.report.targets == ((A, Fraction(1, 4)), (B, Fraction(1, 4)))
    assert result.report.new_visible_tokens == 1 and result.report.new_pixels == 1
    assert result.report.changed_known_tokens == 2
    assert result.report.mean_absolute_support_change == Fraction(1, 4)
    remembered = source(state([[1, 2, 0], [1, 0, 0], [1, 0, 0]]))
    f = forecast(remembered)
    assert f.predictions[1].extrapolation == f.predictions[1].persistence == 0.0
    result = evaluate(remembered, f.canonical_bytes(), target(remembered, [1, 2, 0]))
    assert result.report is not None
    absent = result.report.strata[2]
    assert absent.tokens == absent.predicted == 1
    assert absent.extrapolation_mae == absent.persistence_mae == Fraction(1, 3)


@pytest.mark.parametrize(
    "case",
    [
        "episode",
        "source",
        "state",
        "action",
        "target_index",
        "shape",
        "inventory",
        "value",
        "partial_unknown",
        "extra",
        "newline",
        "duplicate_key",
        "integer_float",
    ],
)
def test_saved_binding_and_prediction_validation_before_fetch(case: str) -> None:
    value = source(state([[1, 2, 0], [1, 2, 0]]))
    data: Any = json.loads(forecast(value).canonical_bytes())
    if case in {"episode", "source", "state", "action"}:
        k = {
            "episode": "episode_key",
            "source": "source_head",
            "state": "lifecycle_root",
            "action": "announced_sha256",
        }[case]
        data["binding"][k] = "0" * len(data["binding"][k])
    elif case == "target_index":
        data["binding"]["target_index"] = 3
    elif case == "shape":
        data["binding"]["shape"] = [3, 1]
    elif case == "inventory":
        data["predictions"].pop()
    elif case == "value":
        data["predictions"][0]["extrapolation"] = 0.9
    elif case == "partial_unknown":
        data["predictions"][0]["persistence"] = None
    elif case == "extra":
        data["depth"] = []
    elif case == "integer_float":
        data["predictions"][0]["persistence"] = 0
    payload = canonical_json_bytes(data)
    if case == "newline":
        payload += b"\n"
    elif case == "duplicate_key":
        payload = payload.replace(b"{", b'{"version":"fraction-extrapolation-v1",', 1)
    p = target(value, [1, 2, 0])
    with pytest.raises(ValueError):
        evaluate(value, payload, p)
    assert not p.calls


def test_permissions_and_future_before_parsing() -> None:
    s = state([[1, 0], [1, 0]])
    for allowed in (frozenset(), REQUIRED | {Modality.DEPTH}):
        with pytest.raises(PermissionError):
            Input(s, b"not JSON", EPISODE, HEAD, True, ModalityPermissionSet(allowed=allowed), 1)
    with pytest.raises(PermissionError):
        replace(source(s), decision_index=0)


@pytest.mark.parametrize(
    "case", ["nan_action", "zero_action", "yaw_action", "bad_state", "bad_episode", "bad_history"]
)
def test_malformed_inputs_rejected(case: str) -> None:
    value = source(state([[1, 0], [1, 0]]))
    if case in {"nan_action", "zero_action", "yaw_action"}:
        action = json.loads(value.announced)
        action[
            {
                "nan_action": "delta_lateral",
                "zero_action": "delta_lateral",
                "yaw_action": "delta_yaw",
            }[case]
        ] = float("nan") if case == "nan_action" else 0.0 if case == "zero_action" else 0.25
        with pytest.raises(ValueError):
            replace(value, announced=json.dumps(action).encode())
    elif case == "bad_state":
        with pytest.raises(ValueError):
            state([[1, 0], [1, 0]], d=0.0)
    else:
        with pytest.raises(ValueError):
            replace(
                value,
                **({"episode_key": "raw-id"} if case == "bad_episode" else {"history_complete": 1}),
            )


@pytest.mark.parametrize(
    "case",
    [
        "binding",
        "missing",
        "shape",
        "index",
        "complete_image",
        "complete_association",
        "stable_identity",
        "announced_action_executed",
    ],
)
def test_missing_contradictory_target_unresolved(case: str) -> None:
    value = source(state([[1, 2, 0], [1, 0, 0]]))
    p = target(value, [1, 0, 0])
    if case == "binding":
        p.value = replace(p.value, binding=replace(p.value.binding, episode_key="c" * 32))
    elif case == "missing":
        p.value = replace(p.value, raster=None)
    elif case in {"shape", "index"}:
        p.value = replace(
            p.value,
            raster=raster(3 if case == "index" else 2, [1, 0] if case == "shape" else [1, 0, 0]),
        )
    else:
        change: dict[str, Any] = {case: False}
        p.value = replace(p.value, **change)
    saved = forecast(value).canonical_bytes()
    result = evaluate(value, saved, p)
    assert result.report is None and result.unresolved is not None and p.calls == [2]
    assert forecast(value).canonical_bytes() == saved


def test_independent_targets_strata_renaming_and_smoke() -> None:
    value = source(state([[1, 2, 0, 0], [1, 1, 0, 0]]))
    f = forecast(value)
    saved = f.canonical_bytes()
    p = target(value, [1, 2, 3, 3])
    result = evaluate(value, saved, p)
    assert result.report is not None and p.calls == [2]
    report = result.report
    assert report.targets == ((A, Fraction(1, 4)), (B, Fraction(1, 4)))
    assert report.new_pixels == 2 and report.new_visible_tokens == 1
    assert report.strata[0].extrapolation_mae == Fraction(3, 8)
    assert report.strata[0].persistence_mae == Fraction(1, 4)
    assert report.strata[0].difference == Fraction(1, 8)
    renamed = source(state([[1, 2, 0, 0], [1, 1, 0, 0]], tokens=(B, A, C)))
    rf = forecast(renamed)
    assert {x.token: (x.extrapolation, x.persistence) for x in rf.predictions} == {
        {A: B, B: A}[x.token]: (x.extrapolation, x.persistence) for x in f.predictions
    }
    assert "binding" not in json.loads(f.feature_bytes())
    assert decode(saved, value) == f
    print(
        "Synthetic fraction smoke:",
        json.dumps(
            {
                "sha256": sha256_bytes(saved),
                "forecast_bytes": len(saved),
                "features_bytes": len(f.feature_bytes()),
                "known_tokens": report.strata[0].tokens,
                "coverage": report.strata[0].predicted,
                "paired_difference_exact": str(report.strata[0].difference),
                "new_pixels_omitted": report.new_pixels,
                "empirical_capability": "UNQUALIFIED",
            },
            sort_keys=True,
        ),
    )


@pytest.mark.parametrize("q,expected", [(2.0**-52, 0.5), (3 * 2.0**-52, 0.5 + 2.0**-52)])
def test_binary64_midpoints_ties_to_even(q: float, expected: float) -> None:
    # a=1/2, a-b=1/4, d=1: exact values lie on binary64 midpoints.
    value = source(state([[1, 0, 0, 0], [1, 1, 0, 0]], d=1.0), q)
    assert forecast(value).predictions[0].extrapolation == expected


def test_provider_failure_retains_saved_forecast_and_no_retry() -> None:
    value = source(state([[1, 0], [1, 0]]))
    saved = forecast(value).canonical_bytes()

    class Broken:
        calls = 0

        def target(self, index: int) -> TargetEvidence:
            self.calls += 1
            raise RuntimeError("public target failure")

    provider = Broken()
    with pytest.raises(RuntimeError, match="public target failure"):
        evaluate(value, saved, provider)
    assert provider.calls == 1 and forecast(value).canonical_bytes() == saved
