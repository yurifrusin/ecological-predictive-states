"""Lawful SYNTHETIC_SOURCE_ONLY masks/seeds; never actual membership or data."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from typing import Any, cast

import pytest

pytest.importorskip("torch", reason="optional restricted-models dependency group")
import torch

from epsbench.diagnostics.restricted_learning_contract import CONDITIONS, Observation, boundaries
from epsbench.diagnostics.restricted_model_export import Example
from epsbench.diagnostics.restricted_model_resources import WorkTrace, matched_reports
from epsbench.diagnostics.restricted_model_training import (
    Adam,
    ModelSource,
    cheap_control,
    prediction,
    train_frequencies,
    update,
)
from epsbench.diagnostics.restricted_models import RestrictedModel, initialized


@pytest.fixture(scope="session", autouse=True)
def cpu() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def example() -> Example:
    masks = cast(
        tuple[bytes, bytes, bytes],
        tuple(bytes(int((i % 32) // 11 == j) for i in range(1024)) for j in range(3)),
    )
    observations = []
    for t in range(3):
        current = masks if t != 2 else (masks[0], bytes(1024), masks[2])
        observations.append(
            Observation(
                t,
                tuple(f"surface-{i:016x}" for i in range(3)),
                (0, 0, 0),
                current,
                (1, 1, 1),
                (1, int(t != 2), 1),
                boundaries(current),
                (Fraction(0), Fraction(t, 2), Fraction(0)),
            )
        )
    return Example(
        (observations[0], observations[1], observations[2]), (0.0, 0.75, 0.0), (True, True, False)
    )


@pytest.mark.parametrize("condition", CONDITIONS)
def test_parameters_gradients_state_and_diagnostics(condition: str) -> None:
    model = RestrictedModel(condition, 17)  # public dummy init; never an allocated seed
    x = example()
    result = prediction(model, x)
    ((result[0] - 1) ** 2 + (result[1] - 1) ** 2 + result[2] ** 2).backward()  # type: ignore[no-untyped-call]
    expected = 99984 if condition == "dense" else 99913
    assert sum(p.numel() for p in model.parameters()) == expected
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    if condition in ("relational", "dense"):
        assert all(bool(torch.ne(cast(torch.Tensor, p.grad), 0).any()) for p in model.parameters())
    state = model.initial_state()
    for o in x.observations:
        state = model.advance(state, o)
    assert sum(s.numel() for s in state.streams) == 384
    assert len(state.handles) == 3
    with pytest.raises(PermissionError):
        model.advance(state, x.observations[2])
    if condition == "action-zero":
        changed = replace(
            x,
            announced=(0.0, -0.75, 0.0),
            observations=cast(
                tuple[Observation, Observation, Observation],
                tuple(
                    replace(o, executed=(Fraction(0), Fraction(-o.index), Fraction(0)))
                    for o in x.observations
                ),
            ),
        )
        assert torch.equal(result, prediction(model, changed))
    if condition == "memory-reset":
        changed = replace(
            x,
            observations=(
                replace(
                    x.observations[0],
                    masks=cast(
                        tuple[bytes, bytes, bytes], tuple(m[::-1] for m in x.observations[0].masks)
                    ),
                    boundary=boundaries(
                        cast(
                            tuple[bytes, bytes, bytes],
                            tuple(m[::-1] for m in x.observations[0].masks),
                        )
                    ),
                ),
                *x.observations[1:],
            ),
        )
        assert torch.allclose(result, prediction(model, changed), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("condition", ("relational", "dense"))
def test_all_slot_permutations_and_opaque_spelling(condition: str) -> None:
    from itertools import permutations

    model = RestrictedModel(condition, 17)
    x = example()
    expected = prediction(model, x)
    for order in permutations(range(3)):
        observations = []
        for o in x.observations:
            masks = cast(tuple[bytes, bytes, bytes], tuple(o.masks[i] for i in order))
            observations.append(
                replace(
                    o,
                    handles=tuple(f"surface-{99 - i:016x}" for i in order),
                    masks=masks,
                    visible=cast(tuple[int, int, int], tuple(o.visible[i] for i in order)),
                    boundary=boundaries(masks),
                )
            )
        y = replace(
            x,
            observations=(observations[0], observations[1], observations[2]),
            truth=tuple(x.truth[i] for i in order),
        )
        assert torch.allclose(prediction(model, y), expected[list(order)], atol=1e-6, rtol=1e-5)


def test_initializer_exact_and_cpu_and_chronology() -> None:
    import hashlib
    import math
    import struct

    from epsbench.diagnostics.restricted_models import DOMAIN

    name = b"encoder/layer1/W"
    raw = (
        struct.pack("<Q", len(DOMAIN))
        + DOMAIN
        + struct.pack("<Q", len(name))
        + name
        + struct.pack("<QQ", 17, 0)
    )
    n = int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")
    expected = torch.tensor((2 * ((n + 0.5) / 2**64) - 1) / math.sqrt(1024), dtype=torch.float32)
    assert initialized(name.decode(), 1, 1024, 17)[0, 0] == expected
    model = RestrictedModel("relational", 17)
    with pytest.raises(PermissionError):
        model.readout(model.initial_state(), (0.0, 0.75, 0.0))
    with pytest.raises(ValueError):
        initialized("x", 1, 1, -1)


def test_one_synthetic_batch_records_actual_reverse_and_adam() -> None:
    model = RestrictedModel("relational", 17)
    adam = Adam(model)
    trace = WorkTrace()
    before = model.weight("encoder/layer1/W").detach().clone()
    with trace:
        loss = update(model, adam, (example(),) * 16, trace)
    assert 0 <= loss <= 1 and adam.k == 1
    assert not torch.equal(before, model.weight("encoder/layer1/W"))
    assert all(trace.work[p] > 0 for p in ("forward", "backward", "adam", "loss"))
    assert matched_reports({}, {}) == "INCONCLUSIVE"
    print(
        "SYNTHETIC_SOURCE_ONLY dispatched work",
        dict(trace.work),
        "unclassified",
        dict(trace.unknown),
    )


def test_fixed_controls_and_laplace() -> None:
    x = example()
    assert cheap_control("persistence", x) == (1.0, 0.0, 1.0)
    assert cheap_control("absent", x) == (0.0,) * 3
    assert cheap_control("visible", x) == (1.0,) * 3
    assert cheap_control("half", x) == (0.5,) * 3
    table = train_frequencies((x,))
    assert cheap_control("train-frequency", x, table) == (0.5, 2 / 3, 0.5)
    with pytest.raises(PermissionError):
        cheap_control("train-frequency", x)
    with pytest.raises(ValueError):
        ModelSource("bad", "bad", "0" * 64)


@pytest.mark.parametrize("condition", ("relational", "dense"))
def test_representative_synthetic_timing(condition: str) -> None:
    import time

    model = RestrictedModel(condition, 17)
    adam = Adam(model)
    trace = WorkTrace()
    started_cpu, started_wall = time.process_time(), time.monotonic()
    with trace:
        loss = update(model, adam, (example(),) * 16, trace)
    cpu, wall = time.process_time() - started_cpu, time.monotonic() - started_wall
    assert loss >= 0
    fits = 6 if condition == "dense" else 18
    print(
        f"SYNTHETIC_SOURCE_ONLY {condition}: batch16, three observations, N=9/example; "
        f"CPU={cpu:.6f}s wall={wall:.6f}s; work={dict(trace.work)}; "
        f"rough{fits}fitsx1000updates CPU={cpu * fits * 1000:.3f}s; trace overhead; no fits"
    )


def test_malformed_state_and_invisible_first_arrival() -> None:
    from epsbench.diagnostics.restricted_models import State

    model = RestrictedModel("relational", 17)
    initial = model.initial_state()
    with pytest.raises(ValueError):
        model.advance(
            replace(initial, streams=cast(tuple[torch.Tensor, ...], list(initial.streams))),
            example().observations[0],
        )
    with pytest.raises(ValueError):
        model.advance(replace(initial, streams=(torch.zeros(63),) * 6), example().observations[0])
    invisible = replace(
        example().observations[0],
        masks=(bytes(1024),) * 3,
        visible=(0, 0, 0),
        boundary=boundaries((bytes(1024),) * 3),
    )
    with pytest.raises(ValueError):
        model.advance(initial, invisible)
    with pytest.raises(ValueError):
        replace(example(), observations=(invisible, *example().observations[1:]))
    state = initial
    for o in example().observations:
        state = model.advance(state, o)
    with pytest.raises(ValueError):
        model.readout(replace(state, flags=((1, 1),) * 2), (0.0, 0.75, 0.0))
    with pytest.raises(PermissionError):
        model.readout(state, cast(tuple[float, float, float], [0.0, 0.75, 0.0]))
    with pytest.raises(ValueError):
        model.advance(
            State(initial.streams, 0, example().observations[0].handles, ((1, 1),) * 3, (1, 1, 1)),
            example().observations[1],
        )


def synthetic_material() -> object:
    from epsbench.diagnostics.restricted_model_export import TrainingMaterial

    return TrainingMaterial(
        16,
        (example(),) * 128,
        (example(),) * 128,
        tuple(i % 128 for i in range(16000)),
        (17, 18, 19),
        "0" * 64,
        "1" * 64,
        "2" * 64,
        "3" * 64,
    )


def test_checkpoint_exact_bindings_and_no_partial_or_extra() -> None:
    import struct

    from epsbench.diagnostics.restricted_learning_contract import Fit
    from epsbench.diagnostics.restricted_model_export import TrainingMaterial
    from epsbench.diagnostics.restricted_model_training import checkpoint, restore
    from epsbench.diagnostics.visible_forecast_contract import _json

    material = cast(TrainingMaterial, synthetic_material())
    model = RestrictedModel("relational", 17)
    fit = Fit(16, "init-0", "relational")
    source = ModelSource("a" * 40, "b" * 40, "c" * 64)
    with pytest.raises(PermissionError):
        checkpoint(model, fit, source, material, 999)
    data = checkpoint(
        model, fit, source, material, 1000
    )  # synthetic final-counter format check only; no fit/readiness
    length = struct.unpack("<Q", data[9:17])[0]
    expected = _json(data[17 : 17 + length])
    restored = RestrictedModel("relational", 18)
    restore(data, restored, expected)
    assert all(
        torch.equal(a, b) for a, b in zip(model.parameters(), restored.parameters(), strict=True)
    )
    with pytest.raises(ValueError):
        restore(data + b"extra", restored, expected)
    with pytest.raises(ValueError):
        restore(data, restored, {**expected, "source": source.__dict__ | {"head": "f" * 40}})


def test_synthetic_all_roster_seal_denies_targets_and_partial(tmp_path: object) -> None:
    from pathlib import Path

    import numpy as np

    from epsbench.diagnostics.restricted_learning_contract import (
        LIMITS,
        REQUIRED,
        ROSTER,
        ZERO,
        Forecast,
        InputEvidence,
        digest,
    )
    from epsbench.diagnostics.restricted_learning_retention import Archive
    from epsbench.diagnostics.restricted_model_export import TrainingMaterial
    from epsbench.diagnostics.restricted_model_forecasts import ForecastPlan, ForecastSession
    from epsbench.diagnostics.restricted_model_training import checkpoint
    from epsbench.diagnostics.visible_forecast_contract import CausalInput, TokenFrame
    from epsbench.schema import ModalityPermissionSet

    root = cast(Path, tmp_path)
    x = example()
    frames = tuple(
        TokenFrame(
            o.index,
            (32, 32),
            tuple(
                (h, np.frombuffer(o.masks[i], dtype=np.uint8).astype(np.bool_).reshape(32, 32))
                for i, h in enumerate(o.handles)
                if o.visible[i]
            ),
        )
        for o in x.observations
    )
    source_input = InputEvidence(
        CausalInput(
            frames,
            (ZERO, ZERO),
            (Fraction(0), Fraction(3, 4), Fraction(0)),
            ModalityPermissionSet(allowed=REQUIRED),
            LIMITS,
        )
    )
    modelsource = ModelSource("a" * 40, "b" * 40, "c" * 64)
    plan = ForecastPlan(
        "d" * 40,
        "e" * 40,
        "0" * 64,
        "2" * 64,
        "4" * 64,
        modelsource,
        (("SYNTHETIC_SOURCE_ONLY", source_input),),
        tuple(f.key + "/SYNTHETIC_SOURCE_ONLY" for f in ROSTER),
        True,
    )
    session = ForecastSession(plan, Archive(root / "complete"))
    called: list[str] = []

    def target_probe(decision: str) -> bytes:
        called.append(decision)
        return b""

    with pytest.raises(PermissionError):
        session.score(target_probe, b"dummy")
    assert not called
    material = cast(TrainingMaterial, synthetic_material())
    models = {c: RestrictedModel(c, 17) for c in CONDITIONS}
    records = {}
    for f in ROSTER:
        if f.condition in CONDITIONS:
            data = checkpoint(
                models[f.condition],
                f,
                modelsource,
                replace(material, budget=f.budget, examples=(x,) * (f.budget * 8)),
                1000,
            )
            record = {
                "checkpoint": digest(data),
                "updates": 1000,
                "batch": 16,
                "schedule": material.schedule_root,
                "data": material.data_root,
            }
            run = b"{}"
            if f.key == "16/init-0/relational":
                for malformed in (data[:-4], data + b"extra", data[:-4] + b"\x00\x00\xc0\x7f"):
                    with pytest.raises(ValueError):
                        session.retain_fit(
                            f, malformed, {**record, "checkpoint": digest(malformed)}, run
                        )
            session.retain_fit(f, data, record, run)
            records[f.key] = (digest(data), digest(run))
    forecasts = []
    for f in ROSTER:
        checkpoint_hash, run_hash = records.get(f.key, ("0" * 64, "1" * 64))
        values = (
            (0.5,) * 3
            if f.condition in CONDITIONS
            else cheap_control(f.condition, x, {(x.announced, 0): 0.5, (x.announced, 1): 0.5})
        )
        forecast = Forecast(
            source_input.digest,
            checkpoint_hash,
            run_hash,
            f,
            tuple(zip(x.observations[-1].handles, values, strict=True)),
        )
        forecasts.append((f.key + "/SYNTHETIC_SOURCE_ONLY", forecast))
    seal = session.seal(tuple(forecasts))
    assert len(seal) == 64
    with pytest.raises(PermissionError):
        session.score(target_probe, b"dummy")
    assert not called
    with pytest.raises(PermissionError):
        session.seal(tuple(forecasts))
    partial = ForecastSession(plan, Archive(root / "partial"))
    with pytest.raises(ValueError):
        partial.seal(())
    assert partial.archive.failed and (root / "partial" / "operator-failure.json").is_file()


def test_bulk_export_verifies_committed_schedule_then_sanitizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import numpy as np

    from epsbench.diagnostics.restricted_learning_contract import (
        INITIALIZATIONS,
        LIMITS,
        REQUIRED,
        ZERO,
        InputEvidence,
        digest,
    )
    from epsbench.diagnostics.restricted_learning_membership import INIT_DOMAIN, decisions
    from epsbench.diagnostics.restricted_learning_sampling import commitment
    from epsbench.diagnostics.restricted_model_export import ReadOnlyExport, lawful_example
    from epsbench.diagnostics.visible_forecast_contract import CausalInput, TokenFrame
    from epsbench.schema import ModalityPermissionSet
    from epsbench.utils.canonical import canonical_json_bytes

    # Public symbolic identities only; no descriptors, membership generator or retained seeds.
    names = tuple(f"{i:064x}" for i in range(80))
    raw_schedules = tuple(
        tuple(
            f"{names[g]}/prefix-{(cycle % 8) // 4}/action-{cycle % 4}"
            for cycle in range(16000 // budget)
            for g in range(budget)
        )
        for budget in (16, 64)
    )
    schedules = (raw_schedules[0], raw_schedules[1])
    values = (17, 18, 19)
    lock = SimpleNamespace(
        train=tuple(SimpleNamespace(identity=n) for n in names[:64]),
        nested_train=names[:16],
        development=tuple(SimpleNamespace(identity=n) for n in names[64:]),
        initialization_commitments=tuple(
            (i, commitment(INIT_DOMAIN, v.to_bytes(8, "little")))
            for i, v in zip(INITIALIZATIONS, values, strict=True)
        ),
        schedule_commitments=tuple(
            (b, digest(canonical_json_bytes(list(s))))
            for b, s in zip((16, 64), schedules, strict=True)
        ),
    )
    x = example()
    frames = tuple(
        TokenFrame(
            o.index,
            (32, 32),
            tuple(
                (h, np.frombuffer(o.masks[i], dtype=np.uint8).astype(np.bool_).reshape(32, 32))
                for i, h in enumerate(o.handles)
                if o.visible[i]
            ),
        )
        for o in x.observations
    )
    source = InputEvidence(
        CausalInput(
            frames,
            (ZERO, ZERO),
            (Fraction(0), Fraction(3, 4), Fraction(0)),
            ModalityPermissionSet(allowed=REQUIRED),
            LIMITS,
        )
    )
    boundary = ReadOnlyExport.__new__(ReadOnlyExport)
    boundary._lock = lock  # type: ignore[assignment]
    boundary._seal_a = "0" * 64
    monkeypatch.setattr(
        boundary,
        "_material",
        lambda d: (source, tuple(zip(x.observations[-1].handles, x.truth, strict=True))),
    )
    material = boundary.export(schedules, values)
    assert tuple(len(m.examples) for m in material) == (128, 512)
    assert all(type(i) is int for m in material for i in m.order)
    assert all(not hasattr(m, "geometry") and not hasattr(m, "decision") for m in material)
    assert material[0].examples[0] == lawful_example(
        source, tuple(zip(x.observations[-1].handles, x.truth, strict=True))
    )
    bad = (schedules[0][::-1], schedules[1])
    with pytest.raises(ValueError):
        boundary.export(bad, values)
    with pytest.raises(ValueError):
        boundary.export(schedules, (17, 18, 20))
    assert len(decisions(names[0])) == 8


def test_untrusted_or_missing_resource_evidence_cannot_match() -> None:
    from epsbench.diagnostics.restricted_model_resources import valid_measurement

    assert not valid_measurement({"status": "VERIFIED", "used_parameters": 99913})
    assert matched_reports({"status": "VERIFIED"}, {"status": "VERIFIED"}) == "INCONCLUSIVE"


def test_hand_derived_reset_after_gru_and_adam_numerics() -> None:
    model = RestrictedModel("relational", 17)
    with torch.no_grad():
        for name in model._names:
            if name.startswith("candidate/token-GRU/"):
                model.weight(name).zero_()
    hidden = torch.arange(16, dtype=torch.float32)
    assert torch.equal(
        model.gru("candidate/token-GRU", torch.ones(53, dtype=torch.float32), hidden), hidden / 2
    )
    fresh = RestrictedModel("dense", 17)
    adam = Adam(fresh)
    before = fresh.weight("encoder/layer1/W").detach().clone()
    for p in fresh.parameters():
        p.grad = torch.full_like(p, 0.25)
    adam.step()
    expected = before - 0.001 * 0.25 / (0.25 + 1e-8)
    assert torch.allclose(fresh.weight("encoder/layer1/W"), expected, atol=1e-7, rtol=0)
    assert adam.k == 1
    assert all(p.grad is None for p in fresh.parameters())


def test_portable_memory_capability_missing_and_fake_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ctypes
    import platform

    from epsbench.diagnostics import restricted_model_resources as resources

    monkeypatch.setattr(platform, "system", lambda: "Linux")
    assert resources.observed_peak_bytes() is None
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(ctypes, "windll", None, raising=False)
    assert resources.observed_peak_bytes() is None
    current = ctypes.CFUNCTYPE(ctypes.c_void_p)(lambda: 123)
    api = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong)(
        lambda h, p, n: 0
    )

    class FakeLibrary(ctypes.CDLL):
        GetCurrentProcess = current
        GetProcessMemoryInfo = api

        def __init__(self, name: str) -> None:
            pass

    loader = ctypes.LibraryLoader(FakeLibrary)
    monkeypatch.setattr(ctypes, "windll", loader, raising=False)
    assert resources.observed_peak_bytes() is None
    assert current.restype is ctypes.wintypes.HANDLE
    assert current.argtypes == ()
    assert api.argtypes[0] is ctypes.wintypes.HANDLE
    assert len(api.argtypes) == 3

    class FakeCounters(ctypes.Structure):
        cb: int
        peak_working: int
        _fields_ = [("cb", ctypes.wintypes.DWORD), ("faults", ctypes.wintypes.DWORD)] + [
            (name, ctypes.c_size_t)
            for name in (
                "peak_working",
                "working",
                "peak_paged",
                "paged",
                "peak_nonpaged",
                "nonpaged",
                "pagefile",
                "peak_pagefile",
            )
        ]

    def fill(h: int, p: int, n: int) -> int:
        memory = ctypes.cast(p, ctypes.POINTER(FakeCounters)).contents
        assert memory.cb == n
        memory.peak_working = 123456
        return 1

    successful = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong)(
        fill
    )
    FakeLibrary.GetProcessMemoryInfo = successful
    assert resources.observed_peak_bytes() == 123456


def test_fit_setup_clocks_include_both_constructors(monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    from epsbench.diagnostics import restricted_model_training as training

    clocks = [100.0, 200.0]
    monkeypatch.setattr(time, "process_time", lambda: clocks[0])
    monkeypatch.setattr(time, "monotonic", lambda: clocks[1])

    def model(condition: str, seed: int) -> object:
        clocks[0] += 5.0
        clocks[1] += 7.0
        return object()

    def adam(value: object) -> object:
        clocks[0] += 3.0
        clocks[1] += 4.0
        return object()

    monkeypatch.setattr(training, "RestrictedModel", model)
    monkeypatch.setattr(training, "Adam", adam)
    monkeypatch.setattr(training, "provenance", lambda: {})
    _, _, _, _, cpu, wall = training._fit_setup("relational", 17)
    assert (cpu, wall) == (100.0, 200.0)
    assert (clocks[0] - cpu, clocks[1] - wall) == (8.0, 11.0)


def test_required_scalar_derivation_and_matching_totals() -> None:
    from epsbench.diagnostics.restricted_model_resources import matching_totals, scalar_work

    for condition in ("relational", "action-zero", "memory-reset"):
        measured = scalar_work(condition, 1, 1)
        # Hand count: three p/2 divides + two (zero-start three-add sums +divide),
        # repeated six streams/three observations; Adam2pow+2sub each38 tensors.
        assert measured == {
            "python_scalar_forward_units": 6 * 3 * (3 + 2 * (3 + 1)),
            "python_scalar_adam_bias_correction_units": 38 * 4,
            "python_scalar_adam_step_units": 1,
        }
        forward, total = matching_totals(
            {
                "trace": {"work": {"forward": 1000, "backward": 2000, "adam": 3000, "loss": 40}},
                **measured,
            }
        )
        assert (forward, total) == (1198, 6040 + 198 + 152 + 1)
    dense = scalar_work("dense", 16000, 1000)
    assert dense == {
        "python_scalar_forward_units": 0,
        "python_scalar_adam_bias_correction_units": 80000,
        "python_scalar_adam_step_units": 1000,
    }
    with pytest.raises(ValueError):
        scalar_work("unknown", 1, 1)
    with pytest.raises(ValueError):
        scalar_work("relational", True, 1)


def test_full_synthetic_measurement_requires_known_scalar_work() -> None:
    from copy import deepcopy

    from epsbench.diagnostics.restricted_model_resources import (
        SCALAR_CONVENTION,
        scalar_work,
        valid_measurement,
    )

    # A hand-built SYNTHETIC_SOURCE_ONLY schema witness, never an actual fit receipt.
    n = 99913
    report: dict[str, Any] = {
        "fit": "16/init-0/relational",
        "model_source": {"head": "a" * 40, "tree": "b" * 40, "launch": "c" * 64},
        **{
            k: "0" * 64
            for k in ("initialization", "labels", "order", "schedule", "data", "checkpoint")
        },
        "optimizer": {
            "name": "Adam",
            "lr": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
            "weight_decay": 0,
            "clipping": False,
            "amsgrad": False,
            "dtype": "float32",
        },
        "trace": {
            "convention": "executed-dispatch-v1; nonlinear unit; not CPU instructions",
            "unclassified": {},
            "work": {"forward": 1000000, "backward": 2000000, "adam": 3000000, "loss": 40},
            "arithmetic_by_operator": {
                "forward:synthetic": 1000000,
                "backward:synthetic": 2000000,
                "adam:synthetic": 3000000,
                "loss:synthetic": 40,
            },
            "operators": {"forward:aten.mv.default": 534 * 16000, "adam:aten.sub_.Tensor": 38000},
        },
        "used_parameters": n,
        "parameter_bytes": n * 4,
        "gradient_bytes": n * 4,
        "adam_moment_bytes": n * 8,
        "gradient_connected": True,
        "state_floats": 384,
        "recurrent_bytes": 1536,
        "streams": 6,
        "updates": 1000,
        "batch": 16,
        "trainer_batch_examples": 16,
        "prediction_values": 3,
        "shared_encoder": True,
        "saved_activation_bytes_cumulative_not_peak": 1000,
        "saved_activation_peak_conservative_bytes": 100,
        "buffers": {
            "current_mask_bytes": 12288,
            "shared_features_bytes": 192,
            "current_boundary_bytes": 48,
            "flags_bytes": 24,
            "commands_bytes": 24,
            "candidate_context_bytes": 4240,
            "prediction_bytes": 12,
            "batch_examples": 16,
            "schedule_index_bytes": 128000,
            "lawful_material_mask_bytes": 128,
        },
        "observed_mask_encodings": 0,
        "provenance": {
            "threads": 1,
            "interop_threads": 1,
            "deterministic": True,
            "device": "cpu",
            "cuda": None,
            **{
                k: "SYNTHETIC_SOURCE_ONLY"
                for k in ("cpu", "os", "python", "framework", "runtime_blas")
            },
        },
        "memory_peak": 100000,
        "memory_complete": True,
        "elapsed_cpu": 1.0,
        "elapsed_wall": 1.0,
        **scalar_work("relational", 16000, 1000),
        "python_scalar_scope": SCALAR_CONVENTION,
    }
    assert valid_measurement(report)  # schema/accounting only, never actual evidence acceptance
    for field in scalar_work("relational", 16000, 1000):
        missing = deepcopy(report)
        del missing[field]
        assert not valid_measurement(missing)
        wrong = deepcopy(report)
        wrong[field] += 1
        assert not valid_measurement(wrong)
        wrong[field] = True
        assert not valid_measurement(wrong)
    wrong = deepcopy(report)
    wrong["python_scalar_scope"] = "unknown power accounting"
    assert not valid_measurement(wrong)
