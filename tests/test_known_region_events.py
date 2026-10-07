"""Public handwritten rasters; no renderer, dataset, model or study execution."""

from __future__ import annotations

import itertools
import json
from fractions import Fraction as Q
from typing import Any

import numpy as np
import pytest

from epsbench.diagnostics import known_region_events as m
from epsbench.diagnostics.boundary_observation import VisibleRaster
from epsbench.diagnostics.causal_region_lifecycle import TrustedObservation, advance
from epsbench.diagnostics.restricted_mask_projection import Projection, Trust, numeric
from epsbench.diagnostics.visible_forecast_contract import REQUIRED, CausalView, Limits
from epsbench.schema import Action, ModalityPermissionSet
from epsbench.utils.canonical import canonical_json_bytes

NAMES = tuple(f"surface-{i:016x}" for i in range(1, 5))
ACCESS = ModalityPermissionSet(allowed=REQUIRED)
ACTION = Action(name="lateral_right", delta_forward=0.0, delta_lateral=0.5, delta_yaw=0.0)


class Fake:
    def __init__(self, names: tuple[str, ...] = NAMES) -> None:
        self.names = names
        self.calls: list[int] = []
        self.labels = {0: [1, 2, 3, 0], 1: [1, 2, 0, 0], 2: [0, 2, 3, 4]}
        self.complete = True

    def raster(self, index: int) -> VisibleRaster:
        self.calls.append(index)
        labels = self.labels[index]
        return VisibleRaster(
            index,
            np.array([labels], dtype=np.int32),
            tuple((i, self.names[i - 1]) for i in sorted(set(labels) - {0})),
        )

    def observation(self, index: int) -> TrustedObservation:
        return TrustedObservation(self.raster(index), self.complete, self.complete, self.complete)


def example(
    names: tuple[str, ...] = NAMES,
    availability: tuple[bool, bool] = (True, True),
    action: Action = ACTION,
    count: int = 2,
) -> tuple[Projection, Fake]:
    provider = Fake(names)
    command = numeric(action)
    source = CausalView(provider, ACCESS, count - 1, Limits(3, 4, 4)).materialize(
        (command,) * (count - 1), command
    )
    state = None
    for index in range(count):
        state = advance(
            state, provider, ACCESS, index, count - 1, None if index == 0 else action
        ).state
    assert state is not None
    projection = Projection(
        source,
        state,
        ACCESS,
        Trust(True, True, True, True),
        availability,
        "a" * 32,
        "b" * 40,
        count - 1,
        action,
    )
    provider.calls.clear()
    return projection, provider


def test_overlapping_predicates_exhaustive_two_pixel_supports() -> None:
    for before, after in itertools.product(itertools.product((False, True), repeat=2), repeat=2):
        truth = (
            not any(before) and any(after),
            any(before) and not any(after),
            any(a and not b for b, a in zip(before, after, strict=True)),
            any(b and not a for b, a in zip(before, after, strict=True)),
        )
        assert m.events(np.array([before]), np.array([after])) == truth
    assert m.events(np.array([[True, False]]), np.array([[False, True]])) == (
        False,
        False,
        True,
        True,
    )


def test_target_remembered_stable_disappearance_and_new_exclusion() -> None:
    p, provider = example()
    result = m.evaluate(p, m.control(p, "stable"), provider)
    assert provider.calls == [2]
    assert result.target == ((False, True, False, True), (False,) * 4, (True, False, True, False))
    assert len(result.target) == 3 and NAMES[3] not in p.alignment
    assert result.report()["brier"] == "1/3"
    assert result.report()["event_frequency"]["appearance"] == "1/3"
    assert result.report()["strata"]["REMEMBERED_ABSENT"] == {"nodes": 1, "brier": "1/2"}
    assert result.report()["N"] == result.report()["C"] == 12
    assert result.report()["U"] == 0


def test_exact_score_half_perfect_and_float_convention() -> None:
    p, provider = example()
    half = m.evaluate(p, m.control(p, "half"), provider)
    assert half.report()["brier"] == "1/4"
    assert set(half.report()["event_brier"].values()) == {"1/4"}
    perfect = tuple(
        (name, tuple(map(int, row))) for name, row in zip(p.alignment, half.target, strict=True)
    )
    assert m.evaluate(p, m.save(p, perfect), provider).report()["brier"] == "0"
    assert m.probability(0.1) == Q(0.1)  # Exact IEEE value, not decimal 1/10.
    assert m.save(p, tuple(reversed(perfect))) == m.save(p, perfect)


@pytest.mark.parametrize("method", ["stable", "half", "replay", "visibility_saturation"])
def test_four_untrained_controls_complete(method: str) -> None:
    p, provider = example()
    saved = m.control(p, method)
    assert len(m.validate(p, saved)) == 3 and provider.calls == []
    if method == "visibility_saturation":
        assert m.validate(p, saved) == ((Q(0), Q(0), Q(1), Q(1)),) * 2 + ((Q(0),) * 4,)
    if method == "replay":
        assert m.validate(p, saved)[2] == (Q(0), Q(1), Q(0), Q(1))
    with pytest.raises(ValueError):
        m.control(p, "train_frequency")


@pytest.mark.parametrize("value", [True, False, float("nan"), float("inf"), -0.1, 1.1, "0.5", None])
def test_invalid_numeric_rejected(value: Any) -> None:
    p, provider = example()
    with pytest.raises(ValueError):
        m.save(p, tuple((name, (value,) * 4) for name in p.alignment))
    assert provider.calls == []


@pytest.mark.parametrize(
    "case",
    [
        "binding",
        "missing",
        "extra",
        "duplicate",
        "few_events",
        "bool",
        "nan",
        "range",
        "noncanonical",
        "key",
        "duplicate_key",
    ],
)
def test_candidate_rejection_before_fetch(case: str) -> None:
    p, provider = example()
    data = json.loads(m.control(p, "half"))
    if case == "binding":
        data["binding_sha256"] = "c" * 64
    elif case == "missing":
        data["rows"].pop()
    elif case == "extra":
        data["rows"].append([NAMES[3], ["0"] * 4])
    elif case == "duplicate":
        data["rows"][1] = data["rows"][0]
    elif case == "few_events":
        data["rows"][0][1].pop()
    elif case in ("bool", "nan", "range"):
        data["rows"][0][1][0] = {"bool": True, "nan": "NaN", "range": "2"}[case]
    elif case == "key":
        data["extra"] = 1
    raw = canonical_json_bytes(data)
    if case == "noncanonical":
        raw += b"\n"
    if case == "duplicate_key":
        raw = raw[:-1] + b',"version":"known-region-events-v1:candidate"}'
    with pytest.raises(ValueError):
        m.evaluate(p, raw, provider)
    assert provider.calls == []


def test_changed_action_binding_before_future() -> None:
    p, provider = example()
    action = Action(name="lateral_left", delta_forward=0.0, delta_lateral=-0.5, delta_yaw=0.0)
    other, _ = example(action=action)
    with pytest.raises(ValueError):
        m.evaluate(p, m.control(other, "half"), provider)
    assert provider.calls == []


@pytest.mark.parametrize(
    "case", ["unavailable", "late_prefix", "permission", "empty", "wrong_type"]
)
def test_input_domain_fail_closed(case: str) -> None:
    p, provider = example(
        availability=(False, True) if case == "unavailable" else (True, True),
        count=3 if case == "late_prefix" else 2,
    )
    if case == "permission":
        object.__setattr__(p, "access", ModalityPermissionSet(allowed=frozenset()))
    if case == "empty":
        object.__setattr__(p, "alignment", ())
    if case == "wrong_type":
        p = object()  # type: ignore[assignment]
    with pytest.raises((ValueError, PermissionError)):
        m.evaluate(p, b"bad JSON", provider)
    assert provider.calls == []


@pytest.mark.parametrize("case", ["incomplete", "none", "unresolved", "shape", "chronology"])
def test_future_qualification_rejected_once(case: str) -> None:
    p, provider = example()
    if case == "incomplete":
        provider.complete = False
    elif case == "shape":
        provider.labels[2] = [1, 2]
    elif case == "chronology":

        class WrongIndex(Fake):
            def observation(self, index: int) -> TrustedObservation:
                value = super().observation(index)
                assert value.raster is not None
                return TrustedObservation(
                    VisibleRaster(1, value.raster.segmentation, value.raster.identities),
                    True,
                    True,
                    True,
                )

        provider = WrongIndex()
    else:

        class Unresolved(Fake):
            def observation(self, index: int) -> TrustedObservation:
                value = super().observation(index)
                return TrustedObservation(
                    None if case == "none" else value.raster, True, True, True, unresolved_labels=1
                )

        provider = Unresolved()
    with pytest.raises(ValueError):
        m.evaluate(p, m.control(p, "half"), provider)
    assert provider.calls == [2]


def test_lossless_encodings_and_joint_token_row_permutation() -> None:
    p, provider = example()
    assert m.reconstruct(m.unstructured(p)) == m.structured(p)
    f = json.loads(m.structured(p))["features"]
    assert f == json.loads(p.feature_bytes())
    assert f["endpoints"] == [
        {"age": 1, "available": True, "pairs": [[0, 1], [1, 2]]},
        {"age": 0, "available": True, "pairs": [[0, 1]]},
    ]
    renamed, _ = example(names=tuple(reversed(NAMES[:3])) + NAMES[3:])
    assert m.structured(renamed, (2, 1, 0)) == m.structured(p)
    assert m.unstructured(renamed, (2, 1, 0)) == m.unstructured(p)
    for order in itertools.permutations(range(3)):
        assert m.reconstruct(m.unstructured(p, order)) == m.structured(p, order)
    for raw in (m.structured(p), m.unstructured(p)):
        assert all(name.encode() not in raw for name in NAMES)
        assert all(
            key not in raw for key in (b"episode", b"source_head", b"sha256", b"depth", b"pose")
        )
    assert provider.calls == []


def test_explicit_relation_field_ablation_keeps_all_other_information() -> None:
    p, _ = example()
    full = json.loads(m.structured(p))
    ablated = json.loads(m.structured(p, omit_relations=True))
    for e in full["features"]["endpoints"]:
        e["pairs"] = None
    full["relations_omitted"] = True
    assert full == ablated
    assert m.reconstruct(m.unstructured(p, omit_relations=True)) == canonical_json_bytes(ablated)
    with pytest.raises(ValueError):
        m.structured(p, (0, 0, 1))
    with pytest.raises(ValueError):
        m.reconstruct(m.unstructured(p) + b"\n")


def test_public_smoke_validation_inspection() -> None:
    p, provider = example()
    saved = m.control(p, "half")
    m.validate(p, saved)
    assert m.reconstruct(m.unstructured(p)) == m.structured(p)
    result = m.evaluate(p, saved, provider)
    assert provider.calls == [2]
    print("Known-region handwritten smoke:", json.dumps(result.report(), sort_keys=True))


def test_future_provider_cannot_change_evaluator_prefix_snapshot() -> None:
    p, provider = example()
    saved = m.control(p, "half")
    binding = json.loads(saved)["binding_sha256"]

    class SideEffect(Fake):
        def observation(self, index: int) -> TrustedObservation:
            result = super().observation(index)
            object.__setattr__(p, "_binding", b"changed by provider")
            object.__setattr__(p, "_features", b"invalid JSON")
            return result

    provider = SideEffect()
    result = m.evaluate(p, saved, provider)
    assert json.loads(result.receipt_bytes)["binding_sha256"] == binding
    assert result.statuses == ("VISIBLE", "VISIBLE", "REMEMBERED_ABSENT")
    assert provider.calls == [2]
