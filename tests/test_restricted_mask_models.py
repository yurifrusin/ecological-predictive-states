"""Handwritten forward/gradient checks only; no optimizer trials or collection."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from epsbench.diagnostics import restricted_mask_readers as m
from epsbench.diagnostics import restricted_mask_training as t
from epsbench.diagnostics.boundary_observation import VisibleRaster
from epsbench.diagnostics.causal_region_lifecycle import TrustedObservation, advance
from epsbench.diagnostics.restricted_mask_fixtures import ACCESS, fixtures, public_key
from epsbench.diagnostics.restricted_mask_projection import Projection, Trust, numeric
from epsbench.diagnostics.visible_forecast_contract import CausalView, Limits
from epsbench.schema import Action, ModalityPermissionSet
from epsbench.utils.canonical import canonical_json_bytes


def projected(n: int = 3, reverse: bool = False, available: bool = True) -> Projection:
    names = tuple(f"surface-{j:016x}" for j in range(1, n + 1))
    if reverse:
        names = tuple(reversed(names))
    grid = np.zeros((32, 32), dtype=np.int32)
    for j in range(1, n + 1):
        grid[5 * j : 5 * j + 2, 4:6] = j

    class Prefix:
        def raster(self, index: int) -> VisibleRaster:
            return VisibleRaster(index, grid, tuple(enumerate(names, 1)))

        def observation(self, index: int) -> TrustedObservation:
            return TrustedObservation(self.raster(index), True, True, True)

    provider = Prefix()
    action = Action(name="lateral_right", delta_forward=-0.0, delta_lateral=0.5, delta_yaw=-0.0)
    source = CausalView(provider, ACCESS, 1, Limits(2, 1024, 3)).materialize(
        (numeric(action),), numeric(action)
    )
    state = advance(None, provider, ACCESS, 0, 1, None).state
    assert state is not None
    state = advance(state, provider, ACCESS, 1, 1, action).state
    assert state is not None
    return Projection(
        source,
        state,
        ACCESS,
        Trust(True, True, True, True),
        (available, available),
        "a" * 32,
        "b" * 40,
        1,
        action,
    )


def test_count_initialization_dtype_and_literal_draw() -> None:
    key = public_key()
    e, control = m.Reader("E", key), m.Reader("L", key)
    assert sum(p.numel() for p in e.parameters()) == 4531
    assert m.weights(e) == m.weights(control)
    assert all(p.dtype == torch.float64 and p.device.type == "cpu" for p in e.parameters())
    import hashlib

    b = int.from_bytes(hashlib.sha256(key + b"|node.conv1.weight|0").digest()[:8], "big") >> 11
    expected = (2 * (b / 2**53) - 1) * math.sqrt(6 / (11 * 25))
    assert float(e.named_scalars()[0][1].flatten()[0].detach()) == expected
    assert all(
        not torch.signbit(p).any() and not (p != 0).any()
        for name, p in e.named_scalars()
        if name.endswith(".bias")
    )


@pytest.mark.parametrize("ordered", [False, True])
def test_scalar_empty_signedzero_reduction_and_gradient(ordered: bool) -> None:
    a = m.tensor([-0.0])
    out = m.reduce(a, ordered=ordered)
    assert out.shape == () and out.item() == 0 and not torch.signbit(out)
    assert m.reduce(m.tensor([]), ordered=ordered).item() == 0
    x = m.tensor([[1e16, -0.0], [1.0, 0.0], [-1e16, 2.0]])
    x.requires_grad_(True)
    y = m.reduce(x, ordered=ordered)
    y.backward(m.tensor([3.0, 5.0]))  # type: ignore[no-untyped-call]
    assert x.grad is not None
    assert torch.equal(x.grad, m.tensor([[3.0, 5.0]] * 3))
    if ordered:
        assert torch.equal(y, m.tensor([0.0, 2.0]))


@pytest.mark.parametrize("arm", ["E", "L"])
def test_exact_fixed_reader_renaming_empty_and_max(arm: m.Arm) -> None:
    reader = m.Reader(arm, public_key())
    p, q = projected(), projected(reverse=True)
    with torch.no_grad():
        a, b = reader(p, ACCESS), reader(q, ACCESS)
        assert a is not None and b is not None
        assert torch.equal(a[:3], b[:3].flip(0)) and torch.equal(a[3:], b[3:])
        empty = reader(projected(0), ACCESS)
        assert empty is not None and empty.shape == (2, 32, 32)
    assert b"-0.0" in p._state
    assert json.loads(p.feature_bytes())["executed_action"] == ["0", "1/2", "0"]


def test_actual_nonadjacent_message_intervention() -> None:
    e, control = m.Reader("E", public_key()), m.Reader("L", public_key())
    # Fixed hand weights isolate a nonadjacent message into the known logit.
    for reader in (e, control):
        with torch.no_grad():
            for parameter in reader.parameters():
                parameter.zero_()
            cast(m.Layer, reader.layers["message_fc2"]).bias.fill_(1.0)
            cast(m.Layer, reader.layers["known_fc1"]).weight[0, 24, 0, 0] = 1.0
            cast(m.Layer, reader.layers["known_fc2"]).weight[0, 0, 0, 0] = 1.0
    p = projected(2)
    a, b = e(p, ACCESS), control(p, ACCESS)
    assert a is not None and b is not None
    assert not (a[:2] != 0).any() and bool((b[:2] == 1).all())


def test_ties_retainedprecision_serialization_and_unknown() -> None:
    p = projected(2)
    zero = torch.zeros((4, 32, 32), dtype=torch.float64)
    saved = m.hard_candidate(p, zero)
    assert not any(any(row) for row in json.loads(saved)["candidate"]["new"])
    zero[0, 0, 0] = np.nextafter(0.0, 1.0)
    assert json.loads(m.hard_candidate(p, zero))["candidate"]["known"][p.alignment[0]][0][0]
    assert torch.equal(m.unpack(m.pack(zero), tuple(zero.shape)), zero)
    unavailable = projected(2, available=False)
    reader = m.Reader("E", public_key())
    pred = m.prediction(reader, unavailable, ACCESS)
    assert json.loads(pred)["logits"] is None
    assert all(
        value is None
        for value in json.loads(m.candidate(pred, unavailable, ACCESS))["candidate"][
            "known"
        ].values()
    )
    with pytest.raises(ValueError):
        m.candidate(pred, p, ACCESS)


def test_permissions_binding_nonfinite_and_canonical_reject_before_fetch() -> None:
    p = projected(1)
    reader = m.Reader("E", public_key())
    with pytest.raises(PermissionError):
        reader(p, ModalityPermissionSet(allowed=frozenset()))
    with pytest.raises(ValueError):
        m.commands(["1/3"])
    with pytest.raises(ValueError):
        m.finite(m.tensor([float("nan")]))
    saved = m.prediction(reader, p, ACCESS)
    with pytest.raises(ValueError):
        m.candidate(saved + b"\n", p, ACCESS)
    changed = json.loads(saved)
    changed["candidate"]["binding_sha256"] = "c" * 64
    with pytest.raises(ValueError):
        m.candidate(canonical_json_bytes(changed), p, ACCESS)
    with torch.no_grad():
        reader.named_scalars()[0][1].flatten()[0] = float("inf")
    with pytest.raises(ValueError):
        reader(p, ACCESS)


def test_hand_loss_gradient_and_adam_atomic_checkpoint() -> None:
    x = torch.zeros((2, 32, 32), dtype=torch.float64, requires_grad=True)
    labels = torch.zeros((32, 32), dtype=torch.int64)
    result = t.loss(x, labels)
    expected = 0.0
    for _ in range(1024):
        expected += math.log(2.0)
    assert result.item() == expected / 1024
    result.backward()  # type: ignore[no-untyped-call]
    assert x.grad is not None
    assert torch.equal(x.grad[0], torch.full((32, 32), -0.5 / 1024, dtype=torch.float64))
    reader = m.Reader("E", public_key())
    opt = t.Adam(reader)
    with torch.no_grad():
        for _, p in reader.named_scalars():
            p.zero_()
            p.grad = torch.full_like(p, 2.0)
    opt.update(reader)  # One analytic constant-gradient operation, not a fitted trial.
    assert reader.named_scalars()[0][1].flatten()[0].item() == -0.001 * 2 / (2 + 1e-8)
    saved = t.checkpoint(reader, opt)
    other = m.Reader("E", public_key())
    loaded = t.load_checkpoint(saved, other)
    assert t.checkpoint(other, loaded) == saved
    before = m.weights(reader)
    gradient = reader.named_scalars()[-1][1].grad
    assert gradient is not None
    gradient.fill_(float("nan"))
    with pytest.raises(ValueError):
        opt.update(reader)
    assert m.weights(reader) == before and opt.step == 1


def test_literal_public_smoke_validation_inspection_and_onefetch() -> None:
    cases = fixtures("b" * 40)
    for case in cases:
        case.check()
        target = case.fetch(case.projection)
        with pytest.raises(PermissionError):
            case.fetch(case.projection)

        class Cached:
            def __init__(self, target: VisibleRaster) -> None:
                self.target = target

            def raster(self, index: int) -> VisibleRaster:
                assert index == 2
                return self.target

        report = case.projection.evaluate(case.projection.persistence(), Cached(target)).report
        assert report["E"] == 4 and report["N"] == (len(case.projection.alignment) + 1) * 1024
        assert report["C"] == report["N"] and report["U"] == 0
    print("Public literal fixture smoke/validation/inspection: 4 cases, complete truth; no fit")


def test_no_fit_and_no_realization_in_source_checks() -> None:
    with pytest.raises(RuntimeError, match="fit denied"):
        t.public_readiness("E", "b" * 40, None)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="fit denied"):
        t.fit_comparative(None, (), None, None, None)  # type: ignore[arg-type]


def test_optional_actual_imports_denied() -> None:
    import importlib
    import importlib.util

    assert importlib.util.find_spec("tensorflow") is None
    with pytest.raises(RuntimeError, match="forbidden"):
        importlib.import_module("tensorflow")


def test_forward_gradient_and_saved_score_prefetch() -> None:
    p = projected(1)
    reader = m.Reader("E", public_key())
    logits = reader(p, ACCESS)
    assert logits is not None
    m.reduce(logits.reshape(-1), ordered=False).backward()  # type: ignore[no-untyped-call]
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in reader.parameters()
    )

    class Denied:
        calls = 0

        def raster(self, index: int) -> VisibleRaster:
            self.calls += 1
            raise AssertionError("target fetch must remain denied")

    denied = Denied()
    saved = m.prediction(reader, p, ACCESS)
    with pytest.raises(ValueError):
        m.evaluate_prediction(saved, p, ACCESS, "c" * 64, denied)
    assert denied.calls == 0
    with pytest.raises(PermissionError):
        t.Example(p, p.persistence(), "training")  # type: ignore[arg-type]


def test_reduction_overflow_is_typed_arithmetic_failure() -> None:
    with pytest.raises(m.ArithmeticFailure):
        m.reduce(m.tensor([1.7e308, 1.7e308]), ordered=False)


def test_canonical_tensor_decode_and_terminal_reservation(tmp_path: Path) -> None:
    value = m.pack(m.tensor([1.0]))
    with pytest.raises(ValueError):
        m.unpack({"shape": [True], "bytes": value["bytes"]}, (1,))
    with pytest.raises(ValueError):
        m.unpack({"shape": [1], "bytes": value["bytes"] + "="}, (1,))
    sink = t._Sink(tmp_path / "exclusive-public-fake")
    sink.count = 192 * (1 << 20) - (1 << 16)
    with pytest.raises(OSError):
        sink.write("ordinary.json", b"x")
    sink.write("terminal.json", b"{}", terminal=True)
    assert (sink.root / "terminal.json").read_bytes() == b"{}"
    assert not (sink.root / "ordinary.json").exists()
