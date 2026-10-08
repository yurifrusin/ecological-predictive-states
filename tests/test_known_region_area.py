"""Handwritten area forecasts/observations only; no physical scene or outcome reads."""

from __future__ import annotations

import json
from fractions import Fraction as Q
from typing import Any

import numpy as np
import pytest

from epsbench.diagnostics import known_region_area as m
from epsbench.diagnostics.boundary_observation import VisibleRaster
from epsbench.diagnostics.causal_region_lifecycle import TrustedObservation, advance
from epsbench.diagnostics.known_region_events import reconstruct, structured, unstructured
from epsbench.diagnostics.restricted_mask_projection import Projection, Trust, numeric
from epsbench.diagnostics.visible_forecast_contract import REQUIRED, CausalView, Limits
from epsbench.schema import Action, ModalityPermissionSet
from epsbench.utils.canonical import canonical_json_bytes

NAMES = tuple(f"surface-{i:016x}" for i in range(1, 4))
ACCESS = ModalityPermissionSet(allowed=REQUIRED)


def action(value: float, forward: float = 0) -> Action:
    return Action(
        name="lateral_left" if value < 0 else "lateral_right",
        delta_forward=forward,
        delta_lateral=value,
        delta_yaw=0,
    )


class Fake:
    def __init__(
        self, labels: list[list[int]] | None = None, names: tuple[str, ...] = NAMES
    ) -> None:
        self.labels = labels or [[1, 2], [1, 2], [1, 1]]
        self.names = names
        self.calls: list[int] = []
        self.complete = True

    def raster(self, index: int) -> VisibleRaster:
        self.calls.append(index)
        labels = self.labels[index]
        return VisibleRaster(
            index,
            np.array([labels], dtype=np.int32),
            tuple((j, self.names[j - 1]) for j in sorted(set(labels) - {0})),
        )

    def observation(self, index: int) -> TrustedObservation:
        return TrustedObservation(self.raster(index), self.complete, self.complete, self.complete)


def example(
    value: float,
    labels: list[list[int]] | None = None,
    names: tuple[str, ...] = NAMES,
    executed: float = 1,
    forward: bool = False,
) -> tuple[Projection, Fake]:
    f = Fake(labels, names)
    a = (
        Action(name="forward", delta_forward=1, delta_lateral=0, delta_yaw=0)
        if forward
        else action(value)
    )
    e = action(executed)
    source = CausalView(f, ACCESS, 1, Limits(3, 12, 4)).materialize((numeric(e),), numeric(a))
    state = advance(None, f, ACCESS, 0, 1).state
    assert state is not None
    state = advance(state, f, ACCESS, 1, 1, e).state
    assert state is not None
    p = Projection(
        source, state, ACCESS, Trust(True, True, True, True), (True, True), "a" * 32, "b" * 40, 1, a
    )
    f.calls.clear()
    return p, f


def pair() -> tuple[list[Projection], list[Fake]]:
    p, left = example(-1)
    r, right = example(1, [[1, 2], [1, 2], [2, 2]])
    return [p, r], [left, right]


def controls(ps: list[Projection]) -> list[tuple[str, list[bytes]]]:
    return [(name, [m.control(p, name) for p in ps]) for name in m.CONTROLS]


def test_complete_smoke_arithmetic_inspection_and_independent_queries() -> None:
    ps, fs = pair()
    arms = controls(ps)
    arms.append(
        (
            "perfect",
            [
                m.save(ps[0], ((NAMES[0], 1), (NAMES[1], 0))),
                m.save(ps[1], ((NAMES[0], 0), (NAMES[1], 1))),
            ],
        )
    )
    result = m.evaluate(ps, arms, fs)
    report = result.report()
    assert [f.calls for f in fs] == [[2], [2]]
    assert report["arms"]["perfect"]["all"]["selected_action_indices"] == [0, 1]
    assert report["arms"]["perfect"]["all"]["mean_regret"] == "0"
    assert report["arms"]["perfect"]["all"]["mse"] == "0"
    for name in ("persistence", "replay", "linear_trend", "centroid_transport", "history_lookup"):
        assert report["arms"][name]["all"]["mse"] == "1/4"
        assert report["arms"][name]["all"]["mean_regret"] == "1/2"
    assert report["arms"]["zero"]["all"]["mse"] == "1/2"
    assert report["fixed_first_mean_regret"] == report["fixed_last_mean_regret"] == "1/2"
    assert m.inspect(ps, arms, result.receipt_bytes, ACCESS) == result
    assert [f.calls for f in fs] == [[2], [2]]
    changed = result.report()
    changed["arms"].clear()
    assert "perfect" in result.report()["arms"]
    print("Handwritten area smoke/validation/inspection:", json.dumps(report, sort_keys=True))


def test_remembered_recovery_and_new_separate() -> None:
    p, f = example(-1, [[1, 2], [1, 0], [2, 3]])
    result = m.evaluate([p], [("zero", [m.control(p, "zero")])], [f]).report()
    assert result["statuses"] == ["VISIBLE", "REMEMBERED_ABSENT"]
    assert result["target_areas"] == [["0", "1/2"]]
    assert result["new_area"] == ["1/2"]
    assert result["arms"]["zero"]["strata"]["REMEMBERED_ABSENT"]["mse"] == "1/4"
    assert m.validate(p, m.control(p, "centroid_transport")) == (Q(1, 2), Q(0))
    assert m.validate(p, m.control(p, "history_lookup")) == (Q(1, 2), Q(1, 2))


def test_control_transport_rounding_clipping_and_trend_rescaling() -> None:
    # Both current singleton masks moved right; transport can overlap without growth.
    p, _ = example(1, [[1, 0, 2, 0], [0, 1, 0, 2], [0, 0, 0, 0]])
    assert m.validate(p, m.control(p, "centroid_transport")) == (Q(1, 4), Q(0))
    assert m._round(Q(1, 2)) == 1 and m._round(Q(-1, 2)) == -1
    p, _ = example(1, [[1, 2, 0, 0], [1, 1, 2, 2], [0, 0, 0, 0]])
    assert m.validate(p, m.control(p, "linear_trend")) == (Q(1, 2), Q(1, 2))
    assert m._transport(np.array([[True, False]]), np.array([[False, True]]), Q(0)) == Q(1, 2)
    with pytest.raises(ValueError):
        m.control(p, "half")


@pytest.mark.parametrize("value", [True, False, float("nan"), float("inf"), -1, 2, "1/2", None])
def test_numeric_rejection(value: Any) -> None:
    ps, fs = pair()
    with pytest.raises(ValueError):
        m.save(ps[0], ((NAMES[0], value), (NAMES[1], 0)))
    assert [f.calls for f in fs] == [[], []]


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "duplicate",
        "extra",
        "simplex",
        "noncanonical",
        "binding",
        "nonreduced",
        "duplicatekey",
        "armmissing",
        "duplicatename",
    ],
)
def test_all_arm_seal_failure_denies_every_future(case: str) -> None:
    ps, fs = pair()
    arms = controls(ps)
    data = json.loads(arms[-1][1][-1])
    if case == "missing":
        data["rows"].pop()
    elif case == "duplicate":
        data["rows"][1] = data["rows"][0]
    elif case == "extra":
        data["rows"].append([NAMES[2], "0"])
    elif case == "simplex":
        data["rows"] = [[n, "1"] for n in NAMES[:2]]
    elif case == "binding":
        data["binding_sha256"] = "c" * 64
    elif case == "nonreduced":
        data["rows"][0][1] = "2/4"
    raw = canonical_json_bytes(data)
    if case == "noncanonical":
        raw += b"\n"
    elif case == "duplicatekey":
        raw = raw[:-1] + b',"version":"duplicate"}'
    arms[-1][1][-1] = raw
    if case == "armmissing":
        arms[-1][1].pop()
    elif case == "duplicatename":
        arms.append(arms[0])
    with pytest.raises(ValueError):
        m.evaluate(ps, arms, fs)
    assert [f.calls for f in fs] == [[], []]


@pytest.mark.parametrize(
    "case",
    [
        "episode",
        "provenance",
        "history",
        "inventory",
        "shape",
        "action",
        "duplicate",
        "permissions",
        "availability",
    ],
)
def test_common_domain_denial_before_fetch(case: str) -> None:
    ps, fs = pair()
    arms = controls(ps)
    if case == "episode":
        object.__setattr__(ps[1], "episode", "c" * 32)
    elif case == "provenance":
        object.__setattr__(ps[1], "source_head", "c" * 40)
    elif case == "history":
        ps[1], fs[1] = example(1, [[2, 1], [1, 2], [2, 2]])
    elif case == "inventory":
        ps[1], fs[1] = example(1, names=tuple(reversed(NAMES)))
    elif case == "shape":
        ps[1], fs[1] = example(1, [[1, 2, 0], [1, 2, 0], [2, 2, 0]])
    elif case == "action":
        ps[1], fs[1] = example(1, forward=True)
    elif case == "duplicate":
        ps[1] = ps[0]
    elif case == "permissions":
        object.__setattr__(ps[1], "access", ModalityPermissionSet(allowed=frozenset()))
    elif case == "availability":
        object.__setattr__(ps[1], "availability", (False, True))
    with pytest.raises((ValueError, PermissionError)):
        m.evaluate(ps, arms, fs)
    assert [f.calls for f in fs] == [[], []]


@pytest.mark.parametrize(
    "case", ["incomplete", "none", "boolunresolved", "unresolved", "index", "shape"]
)
def test_whole_operation_future_rejection_no_partial_result(case: str) -> None:
    ps, fs = pair()

    class Bad(Fake):
        def observation(self, index: int) -> TrustedObservation:
            result = super().observation(index)
            if case == "none":
                return TrustedObservation(None, True, True, True)
            if case == "boolunresolved":
                return TrustedObservation(result.raster, True, True, True, unresolved_pixels=False)
            if case == "unresolved":
                return TrustedObservation(result.raster, True, True, True, unresolved_labels=1)
            if case == "index":
                return TrustedObservation(self.raster(1), True, True, True)
            return result

    bad = Bad()
    if case == "incomplete":
        bad.complete = False
    if case == "shape":
        bad.labels[2] = [1, 2, 0]
    fs[1] = bad
    with pytest.raises(ValueError):
        m.evaluate(ps, controls(ps), fs)
    assert fs[0].calls == [2] and fs[1].calls[0] == 2


def test_freezes_every_collection_and_metadata_before_callbacks() -> None:
    ps, fs = pair()
    arms = controls(ps)
    original = fs[1]

    class Mutating(Fake):
        def observation(self, index: int) -> TrustedObservation:
            result = super().observation(index)
            arms.clear()
            fs.clear()
            for p in ps:
                object.__setattr__(p, "_features", b"invalid")
                object.__setattr__(p, "_binding", b"invalid")
                object.__setattr__(p, "access", ModalityPermissionSet(allowed=frozenset()))
            ps.clear()
            return result

    first = Mutating()
    fs[0] = first
    result = m.evaluate(ps, arms, fs)
    assert first.calls == original.calls == [2]
    assert len(result.report()["arms"]) == 6


def test_numeric_ties_permutation_token_equivalence_and_binding_replay() -> None:
    ps, fs = pair()
    p, f = example(0.5)
    # Numeric -1, 1/2, 1 order; lexical rational strings would put 1 before 1/2.
    ps.append(p)
    fs.append(f)
    result = m.evaluate(ps, controls(ps), fs)
    assert result.report()["actions"] == [["0", "-1", "0"], ["0", "1/2", "0"], ["0", "1", "0"]]
    reordered = [ps[2], ps[0], ps[1]]
    arms = controls(reordered)
    assert m.inspect(reordered, arms, result.receipt_bytes, ACCESS) == result
    renamed, _ = example(-1, names=(NAMES[1], NAMES[0], NAMES[2]))
    assert structured(renamed, (1, 0)) == structured(ps[0])
    assert reconstruct(unstructured(renamed, (1, 0))) == structured(ps[0])
    assert all(n.encode() not in structured(ps[0]) for n in NAMES)
    with pytest.raises(ValueError):
        m.inspect(ps, controls(ps), result.receipt_bytes + b"\n", ACCESS)
    with pytest.raises(PermissionError):
        m.inspect(ps, controls(ps), b"bad", ModalityPermissionSet(allowed=frozenset()))
    bad = json.loads(result.receipt_bytes)
    bad["report"]["new_area"][0] = "1"
    with pytest.raises(ValueError):
        m.inspect(ps, controls(ps), canonical_json_bytes(bad), ACCESS)


def test_permission_denial_precedes_candidate_parse_and_provider_attribute_access() -> None:
    p, _ = example(-1)
    object.__setattr__(p, "access", ModalityPermissionSet(allowed=frozenset()))

    class ForbiddenProvider:
        @property
        def observation(self) -> Any:
            raise AssertionError("provider attribute accessed before permission denial")

    with pytest.raises(PermissionError):
        m.evaluate([p], [("bad", [b"not JSON"])], [ForbiddenProvider()])
    with pytest.raises(PermissionError):
        m.validate(p, b"not JSON")
    with pytest.raises(PermissionError):
        m.inspect([p], [("bad", [b"not JSON"])], b"bad", ACCESS)


def test_action_names_and_float_exactness_do_not_change_numeric_domain() -> None:
    p, _ = example(-1)
    assert m.validate(p, m.save(p, ((NAMES[0], 0.1), (NAMES[1], Q(1, 3))))) == (Q(0.1), Q(1, 3))
    assert m.save(p, ((NAMES[1], Q(1, 3)), (NAMES[0], 0.1))) == m.save(
        p, ((NAMES[0], 0.1), (NAMES[1], Q(1, 3)))
    )


def test_transport_overlap_is_area_simplex_without_artificial_scaling() -> None:
    p, _ = example(1, [[1, 0, 0, 2], [0, 1, 2, 0], [0, 0, 0, 0]])
    assert m.validate(p, m.control(p, "centroid_transport")) == (Q(1, 4), Q(1, 4))
    # Each moves toward the other; overlapping predictions cannot enlarge each current mask.
    assert sum(m.validate(p, m.control(p, "centroid_transport"))) == Q(1, 2)


def test_caller_projection_limits_are_detached_before_future() -> None:
    ps, fs = pair()
    arms = controls(ps)

    class MutatingLimits(Fake):
        def observation(self, index: int) -> TrustedObservation:
            result = super().observation(index)
            for p in ps:
                object.__setattr__(p.limits, "max_pixels", 1)
                object.__setattr__(p.limits, "max_tokens", 1)
            return result

    fs[0] = MutatingLimits()
    assert m.evaluate(ps, arms, fs).report()["target_areas"] == [["1", "0"], ["0", "1"]]


def test_retained_shape_budget_and_bool_labels_rejected() -> None:
    ps, fs = pair()
    arms = controls(ps)
    receipt = m.evaluate(ps, arms, fs).receipt_bytes
    for rows in ([[1, 1, 1]], [[True, 1]], [[1, 1], [1, 1]]):
        data = json.loads(receipt)
        data["observations"][0]["segmentation"] = rows
        with pytest.raises(ValueError):
            m.inspect(ps, arms, canonical_json_bytes(data), ACCESS)
