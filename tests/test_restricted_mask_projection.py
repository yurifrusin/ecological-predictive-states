"""Handwritten public masks only; no geometry, models or study evidence."""

from __future__ import annotations

import json
from fractions import Fraction as Q
from typing import Any

import numpy as np
import pytest

from epsbench.diagnostics import restricted_mask_projection as m
from epsbench.diagnostics.boundary_observation import VisibleRaster
from epsbench.diagnostics.causal_region_lifecycle import TrustedObservation, advance
from epsbench.diagnostics.visible_forecast_contract import REQUIRED, CausalView, Limits
from epsbench.schema import Action, ModalityPermissionSet
from epsbench.utils.canonical import canonical_json_bytes

NAMES = tuple(f"surface-{i:016x}" for i in range(1, 5))
ACCESS = ModalityPermissionSet(allowed=REQUIRED)
TRUST = m.Trust(True, True, True, True)
CMD = (Q(0), Q(1, 2), Q(0))
ACT = Action(name="lateral_right", delta_forward=0.0, delta_lateral=0.5, delta_yaw=0.0)


class Fake:
    def __init__(self, frames: dict[int, VisibleRaster]) -> None:
        self.frames = frames
        self.calls: list[int] = []

    def raster(self, index: int) -> VisibleRaster:
        self.calls.append(index)
        return self.frames[index]

    def observation(self, index: int) -> TrustedObservation:
        return TrustedObservation(self.raster(index), True, True, True)


def frame(index: int, labels: list[int], names: tuple[str, ...] = NAMES) -> VisibleRaster:
    return VisibleRaster(
        index,
        np.array([labels], dtype=np.int32),
        tuple((label, names[label - 1]) for label in sorted(set(labels) - {0})),
    )


def example(
    names: tuple[str, ...] = NAMES,
    available: tuple[bool, bool] = (True, True),
    empty: bool = False,
    action: Action = ACT,
) -> tuple[m.Projection, Any, Any, Fake]:
    provider = Fake(
        {
            0: frame(0, [1, 2, 4, 0] if not empty else [0] * 4, names),
            1: frame(1, [1, 0, 0, 0] if not empty else [0] * 4, names),
            2: frame(2, [0, 2, 3, 0] if not empty else [0] * 4, names),
            3: frame(3, [1, 0, 0, 0] if not empty else [0] * 4, names),
        }
    )
    cmd = m.numeric(action)
    source = CausalView(provider, ACCESS, 2, Limits(5, 4, 5)).materialize((cmd, cmd), CMD)
    state = None
    for index in range(3):
        state = advance(state, provider, ACCESS, index, 2, None if not index else action).state
    assert state is not None
    projected = m.Projection(source, state, ACCESS, TRUST, available, "a" * 32, "b" * 40, 2, ACT)
    provider.calls.clear()
    return projected, source, state, provider


def test_literal_shared_projection_and_remembered_first_seen() -> None:
    p, _, _, provider = example()
    f = json.loads(p.feature_bytes())
    assert set(f) == {
        "nodes",
        "endpoints",
        "executed_action",
        "announced_action",
        "current_union",
        "previous_union",
    }
    assert p.alignment == NAMES
    assert f["nodes"][3] == {
        "current": [[False] * 4],
        "previous": [[False] * 4],
        "difference": [[0] * 4],
        "status": "REMEMBERED_ABSENT",
        "first_seen_age": 2,
        "last_seen_age": 2,
    }
    assert f["nodes"][2]["first_seen_age"] == 0 and f["nodes"][2]["previous"] == [[False] * 4]
    assert f["nodes"][0]["difference"] == [[-1, 0, 0, 0]]
    assert f["endpoints"] == [
        {"age": 1, "available": True, "pairs": []},
        {"age": 0, "available": True, "pairs": [[1, 2]]},
    ]
    assert f["executed_action"] == f["announced_action"] == ["0", "1/2", "0"]
    assert all(name not in p.feature_bytes().decode() for name in NAMES)
    assert "source_head" not in f and "episode" not in f and "index" not in f
    assert not provider.calls


def test_previous_mask_ablation_exact_fields() -> None:
    p, _, _, _ = example()
    full = json.loads(p.feature_bytes())
    ablated = json.loads(p.feature_bytes(True))
    expected = json.loads(p.feature_bytes())
    del expected["previous_union"]
    for row in expected["nodes"]:
        del row["previous"]
        del row["difference"]
    assert ablated == expected
    assert ablated["endpoints"] == full["endpoints"]
    assert ablated["nodes"][3]["last_seen_age"] == 2
    with pytest.raises(ValueError):
        p.feature_bytes(1)  # type: ignore[arg-type]


@pytest.mark.parametrize("availability", [(False, True), (True, False), (False, False)])
def test_unavailable_not_negative_adjacency(availability: tuple[bool, bool]) -> None:
    p, _, _, _ = example(available=availability)
    endpoints = json.loads(p.feature_bytes())["endpoints"]
    for i, available in enumerate(availability):
        assert endpoints[i]["available"] is available and endpoints[i]["age"] == 1 - i
        if not available:
            assert endpoints[i]["pairs"] is None


def test_denied_access_before_serialization_or_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    p, source, state, provider = example()

    def denied(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("accessed prefix")

    monkeypatch.setattr(type(source), "canonical_bytes", denied)
    with pytest.raises(PermissionError):
        m.Projection(
            source,
            state,
            ModalityPermissionSet(allowed=frozenset()),
            TRUST,
            (True, True),
            "a" * 32,
            "b" * 40,
            2,
            ACT,
        )
    object.__setattr__(p, "access", ModalityPermissionSet(allowed=frozenset()))
    with pytest.raises(PermissionError):
        p.evaluate(b"not JSON", provider)
    assert provider.calls == []


@pytest.mark.parametrize(
    "case",
    [
        "history",
        "association",
        "future",
        "state",
        "action",
        "availability",
        "rational",
        "truncation",
    ],
)
def test_prefix_state_binding_fail_closed(case: str) -> None:
    _, source, state, provider = example()
    trust, cutoff, available, announced = TRUST, 2, (True, True), ACT
    if case == "history":
        trust = m.Trust(False, True, True, True)
    if case == "association":
        trust = m.Trust(True, True, False, True)
    if case == "future":
        cutoff = 1
    if case == "state":
        object.__setattr__(state, "decision_index", 1)
    if case == "action":
        announced = Action(
            name="lateral_left", delta_forward=0.0, delta_lateral=-0.5, delta_yaw=0.0
        )
    if case == "availability":
        available = (1, True)  # type: ignore[assignment]
    if case == "rational":
        object.__setattr__(source, "announced", (Q(0), Q(1, 3), Q(0)))
    if case == "truncation":
        object.__setattr__(source, "frames", source.frames[1:])
    with pytest.raises((ValueError, PermissionError)):
        m.Projection(source, state, ACCESS, trust, available, "a" * 32, "b" * 40, cutoff, announced)
    assert provider.calls == []


def test_genuine_action_bytes_preserve_signed_zero() -> None:
    negative_zero = Action(
        name="lateral_right", delta_forward=-0.0, delta_lateral=0.5, delta_yaw=-0.0
    )
    p, _, state, _ = example(action=negative_zero)
    assert b"-0.0" in state.executed_commands[0]
    assert p.revalidate().executed == (CMD, CMD)
    assert json.loads(p.feature_bytes())["executed_action"] == ["0", "1/2", "0"]
    assert p._state == state.canonical_bytes()


def normalized(p: m.Projection, renamed: dict[str, str]) -> Any:
    f = json.loads(p.feature_bytes())
    nodes = {
        renamed.get(name, name): data
        for name, data in zip(p.alignment, f.pop("nodes"), strict=True)
    }
    endpoints = []
    for e in f.pop("endpoints"):
        pairs = (
            None
            if e["pairs"] is None
            else sorted(
                tuple(
                    sorted(
                        (
                            renamed.get(p.alignment[a], p.alignment[a]),
                            renamed.get(p.alignment[b], p.alignment[b]),
                        )
                    )
                )
                for a, b in e["pairs"]
            )
        )
        endpoints.append({**e, "pairs": pairs})
    return nodes, endpoints, f


def test_renaming_equivariance_and_immutable_snapshots() -> None:
    p, source, _, _ = example()
    renamed = tuple(reversed(NAMES))
    other, _, _, _ = example(names=renamed)
    assert normalized(p, {}) == normalized(other, dict(zip(renamed, NAMES, strict=True)))
    before = p.feature_bytes()
    detached = json.loads(before)
    detached["nodes"].clear()
    with pytest.raises(ValueError):
        source.frames[0].masks[0][1][0, 0] = False
    assert p.feature_bytes() == before
    assert p.storage()["full_prefix_context_bytes"] > 0


@pytest.mark.parametrize("case", ["binding", "action", "canonical", "inventory", "unknown_key"])
def test_candidate_rejected_before_target_fetch(case: str) -> None:
    p, source, state, provider = example()
    saved = json.loads(p.persistence())
    if case == "binding":
        saved["binding_sha256"] = "c" * 64
    if case == "inventory":
        del saved["candidate"]["known"][NAMES[0]]
    if case == "unknown_key":
        saved["extra"] = 1
    raw = canonical_json_bytes(saved)
    if case == "canonical":
        raw += b"\n"
    if case == "action":
        alternative = m.Projection(
            source, state, ACCESS, TRUST, (True, True), "d" * 32, "b" * 40, 2, ACT
        )
        raw = alternative.persistence()
    with pytest.raises(ValueError):
        p.evaluate(raw, provider)
    assert provider.calls == []


def test_existing_neutral_target_denominators_unknown_and_new() -> None:
    p, _, _, provider = example()
    unknown = p.save(tuple((name, None) for name in p.alignment), None)
    report = p.evaluate(unknown, provider).report
    assert provider.calls == [3] and (report["N"], report["U"], report["C"], report["E"]) == (
        20,
        20,
        0,
        0,
    )
    assert report["ignorance_interval"] == (Q(0), Q(1))
    empty, _, _, empty_provider = example(empty=True)
    empty_provider.frames[3] = frame(3, [1, 0, 0, 0])
    evaluated = empty.evaluate(empty.persistence(), empty_provider)
    assert (evaluated.report["N"], evaluated.report["E"]) == (4, 1)
    assert evaluated.report["first_observed_pixels"] == 1
    assert empty.alignment == ()


def test_public_smoke_validation_and_inspection() -> None:
    p, _, _, provider = example()
    saved = p.persistence()
    result = p.evaluate(saved, provider)
    assert provider.calls == [3]
    assert result.report["complete_assertion"] and result.report["N"] == 20
    print(
        "Shared-mask public fake smoke:",
        json.dumps(
            {
                "version": m.VERSION,
                "nodes": len(p.alignment),
                "storage": p.storage(),
                "N": result.report["N"],
                "E": result.report["E"],
                "model": "NONE",
                "qualification": "SOURCE_ONLY",
            },
            sort_keys=True,
        ),
    )


@pytest.mark.parametrize("case", ["older_inventory", "executed_numeric", "feature_pairs"])
def test_valid_but_inconsistent_context_rejects(case: str) -> None:
    p, source, state, provider = example()
    if case == "older_inventory":
        from epsbench.diagnostics.visible_forecast_contract import CausalInput, TokenFrame

        first = TokenFrame(0, source.shape, source.frames[0].masks[:2])
        source = CausalInput(
            (first, *source.frames[1:]), source.executed, source.announced, ACCESS, source.limits
        )
    elif case == "executed_numeric":
        from epsbench.diagnostics.visible_forecast_contract import CausalInput

        source = CausalInput(
            source.frames, ((Q(0), Q(-1, 2), Q(0)), CMD), source.announced, ACCESS, source.limits
        )
    else:
        f = json.loads(p.feature_bytes())
        f["endpoints"][1]["pairs"] = []
        object.__setattr__(p, "_features", canonical_json_bytes(f))
        with pytest.raises(ValueError):
            p.evaluate(p.persistence(), provider)
        assert provider.calls == []
        return
    with pytest.raises(ValueError):
        m.Projection(source, state, ACCESS, TRUST, (True, True), "a" * 32, "b" * 40, 2, ACT)
    assert provider.calls == []
