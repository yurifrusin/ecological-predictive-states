"""SYNTHETIC_SOURCE_ONLY health packet/operator fakes: no child or health training."""

from __future__ import annotations

import platform
import struct
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import pytest

pytest.importorskip("torch", reason="optional restricted-models dependency group")

from epsbench.diagnostics import restricted_model_health as health
from epsbench.diagnostics.restricted_learning_contract import CONDITIONS, boundaries, digest
from epsbench.diagnostics.restricted_learning_membership import MembershipLock
from epsbench.diagnostics.restricted_learning_sampling import shuffled
from epsbench.diagnostics.restricted_model_export import TrainingMaterial
from epsbench.diagnostics.restricted_model_resources import (
    SCALAR_CONVENTION,
    provenance,
    scalar_work,
)
from epsbench.diagnostics.restricted_model_training import ModelSource
from epsbench.diagnostics.visible_forecast_contract import _json
from epsbench.utils.canonical import canonical_json_bytes
from tests.test_restricted_models_source import example

SOURCE = ModelSource("1" * 40, "2" * 40, "3" * 64)


@pytest.fixture(autouse=True)
def fake_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(health, "_environment_bytes", lambda: 662146229)


def payload() -> bytes:
    x = example()
    material = TrainingMaterial(
        16,
        (x,) * 128,
        (x,) * 128,
        (127,) * 16000,
        (17, 18, 19),
        "a" * 64,
        "b" * 64,
        "c" * 64,
        "d" * 64,
    )
    return health.packet(material, SOURCE, ("4" * 40, "5" * 40))


def admission() -> health.HealthAdmission:
    return health.HealthAdmission(
        SOURCE, "e" * 64, "c" * 64, "f" * 64, 1.0, 662146229, 256 * 1024**2
    )


def result(raw: bytes, condition: str, final: float = 0.0) -> bytes:
    x, _, _ = health.parse_packet(raw, digest(raw))
    count = 99984 if condition == "dense" else 99913
    state = (b"EPSHEALTH1" + struct.pack("<Q", 200) + bytes(3 * count * 4)).hex()
    runtime = provenance() | {
        "cpu": "synthetic CPU",
        "threads": 1,
        "interop_threads": 1,
        "deterministic": True,
    }
    trace = {
        "convention": "executed-dispatch-v1; nonlinear unit; not CPU instructions",
        "work": dict.fromkeys(("forward", "backward", "loss", "adam"), 1),
        "arithmetic_by_operator": {
            phase + ":aten.add.Tensor": 1 for phase in ("forward", "backward", "loss", "adam")
        },
        "operators": {
            phase + ":aten.add.Tensor": 1 for phase in ("forward", "backward", "loss", "adam")
        },
        "unclassified": {},
        "temporary_output_bytes_cumulative_not_peak": {},
    }
    return canonical_json_bytes(
        {
            "version": health.VERSION,
            "packet": digest(raw),
            "condition": condition,
            "updates": 200,
            "source": SOURCE.__dict__,
            "initial": 0.25,
            "final": final,
            "floor": health.floor(x, condition),
            "connected_finite": True,
            "state": state,
            "trace": trace,
            "scalar": scalar_work(condition, 3202, 200),
            "scalar_convention": SCALAR_CONVENTION,
            "provenance": runtime,
            "elapsed_cpu": 0.5,
            "elapsed_wall": 0.7,
            "peak_ram": 1000000,
        }
    )


def fake(raw: bytes, condition: str, directory: Path, remaining: float) -> health.ProcessOutcome:
    return health.ProcessOutcome(result(raw, condition), 1.0, 1.1)


def test_packet_selected_ordinal_and_strict_causal_modal_boundary() -> None:
    raw = payload()
    selected, seed, p = health.parse_packet(raw, digest(raw))
    assert seed == 17 and selected == example() and p["ordinal"] == 0
    assert "order" not in p and "development" not in p and "membership" not in p
    for key, value in (
        ("depth", []),
        ("ordinal", True),
        ("seed", False),
        ("initialization", "init-1"),
    ):
        bad = p | {key: value}
        data = canonical_json_bytes(bad)
        with pytest.raises(ValueError):
            health.parse_packet(data, digest(data))
    bad = _json(raw)
    bad["example"]["observations"][0]["masks"][1] = bytes(1024).hex()
    bad["example"]["observations"][0]["visible"][1] = 0
    data = canonical_json_bytes(bad)
    with pytest.raises(ValueError, match="causal"):
        health.parse_packet(data, digest(data))
    with pytest.raises(ValueError):
        health.parse_packet(raw + b" ", digest(raw + b" "))
    with pytest.raises(ValueError):
        health.parse_packet(raw, "0" * 64)


def test_floor_and_fixed_initial_final_threshold() -> None:
    x = example()
    assert health.floor(x, "memory-reset") == 0.0
    # Two absent tokens with opposite labels: p=1/2, both strata => floor1/8.
    last = x.observations[-1]
    masks = (last.masks[0], bytes(1024), bytes(1024))
    altered = replace(
        x,
        observations=(
            *x.observations[:2],
            replace(last, masks=masks, visible=(1, 0, 0), boundary=boundaries(masks)),
        ),
    )
    assert health.floor(altered, "memory-reset") == 0.125
    assert health.floor(altered, "action-zero") == 0.0
    absent = replace(
        altered,
        observations=(
            *x.observations[:2],
            replace(
                last,
                masks=(bytes(1024),) * 3,
                visible=(0, 0, 0),
                boundary=boundaries((bytes(1024),) * 3),
            ),
        ),
    )
    assert health.floor(absent, "memory-reset") == pytest.approx(2 / 9)
    assert health.criterion(0.25, 0.009, 0.0)
    assert not health.criterion(0.25, 0.011, 0.0)
    assert not health.criterion(0.05, 0.009, 0.0)
    assert health.criterion(0.005, 0.009, 0.0)
    with pytest.raises(ValueError):
        health.criterion(float("nan"), 0.0, 0.0)


def test_recipe_reproduces_complete_synthetic_commitments(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLock:
        nested_train = tuple(digest(f"SYNTHETIC-{i}".encode()) for i in range(16))
        train = tuple(
            type("Unit", (), {"identity": digest(f"SYNTHETIC-{i}".encode())})() for i in range(64)
        )
        schedule_commitments: tuple[tuple[int, str], ...] = ()

    lock = FakeLock()
    seed = b"SYNTHETIC_SOURCE_ONLY_SEED".ljust(32, b"!")
    expected = []
    for budget in (16, 64):
        ids = tuple(g.identity for g in lock.train[:budget])
        orders = {
            g: shuffled(tuple(range(8)), health.SCHEDULE_DOMAIN + "/decisions/" + g, seed)
            for g in ids
        }
        expanded = tuple(
            f"{g}/prefix-{orders[g][cycle % 8] // 4}/action-{orders[g][cycle % 8] % 4}"
            for cycle in range(16000 // budget)
            for g in shuffled(ids, health.SCHEDULE_DOMAIN + f"/budget-{budget}/cycle-{cycle}", seed)
        )
        expected.append(expanded)
    lock.schedule_commitments = tuple(
        (b, digest(canonical_json_bytes(list(s)))) for b, s in zip((16, 64), expected, strict=True)
    )
    monkeypatch.setattr(health, "MembershipLock", FakeLock)
    assert health.schedules(cast(MembershipLock, lock), seed) == tuple(expected)
    lock.schedule_commitments = ((16, "0" * 64), (64, "1" * 64))
    with pytest.raises(ValueError, match="commitment"):
        health.schedules(cast(MembershipLock, lock), seed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("updates", 201),
        ("packet", "0" * 64),
        ("connected_finite", False),
        ("scalar_convention", "proxy"),
        ("elapsed_cpu", float("inf")),
        ("peak_ram", False),
    ],
)
def test_child_result_rejection(field: str, value: Any) -> None:
    raw = payload()
    p = _json(result(raw, "relational"))
    p[field] = value
    # Canonical encoder rejects nonfinite itself; both boundaries fail closed.
    with pytest.raises(ValueError):
        health.validate_result(canonical_json_bytes(p), raw, "relational")


def test_health_state_never_comparative_and_complete() -> None:
    raw = payload()
    p = _json(result(raw, "relational"))
    assert bytes.fromhex(p["state"]).startswith(b"EPSHEALTH1")
    for changed in (p["state"][:-8], p["state"] + "00000000", (b"EPSMODEL1" + bytes(12)).hex()):
        with pytest.raises(ValueError):
            health.validate_state(changed, "relational")
    p["state"] = p["state"][:36] + "0000807f" + p["state"][44:]
    with pytest.raises(ValueError):
        health.validate_result(canonical_json_bytes(p), raw, "relational")


def test_fake_operator_fixed_sequence_and_missing_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(health, "observed_peak_bytes", lambda: 1000000)
    calls = []

    def execute(
        raw: bytes, condition: str, directory: Path, remaining: float
    ) -> health.ProcessOutcome:
        calls.append(condition)
        return fake(raw, condition, directory, remaining)

    report = health._operate(tmp_path / "new", tmp_path / "closed", admission(), payload, execute)
    assert calls == list(CONDITIONS) and report["completed"] == list(CONDITIONS)
    assert report["status"] == "HEALTH_READY" and report["health_only"]
    assert report["child_cpu"] == 4.0
    with pytest.raises(FileExistsError):
        health._operate(tmp_path / "new", tmp_path / "closed", admission(), payload, execute)

    def missing(
        raw: bytes, condition: str, directory: Path, remaining: float
    ) -> health.ProcessOutcome:
        return health.ProcessOutcome(result(raw, condition), None, 1.0)

    assert (
        health._operate(tmp_path / "missing", tmp_path / "closed", admission(), payload, missing)[
            "status"
        ]
        == "INCONCLUSIVE"
    )


@pytest.mark.parametrize("phase", ("setup", "child", "retention", "criterion"))
def test_durable_failure_and_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    original = health._write

    def write(directory: Path, name: str, data: bytes) -> None:
        if phase == "retention" and name == "completed.json":
            raise OSError("synthetic retention failure")
        original(directory, name, data)

    monkeypatch.setattr(health, "_write", write)

    def prepare() -> bytes:
        if phase == "setup":
            raise RuntimeError("synthetic setup failure")
        return payload()

    def execute(
        raw: bytes, condition: str, directory: Path, remaining: float
    ) -> health.ProcessOutcome:
        if phase == "child":
            raise RuntimeError("synthetic child failure")
        return health.ProcessOutcome(
            result(raw, condition, 0.2 if phase == "criterion" else 0.0), 1.0, 1.1
        )

    output = tmp_path / "new"
    with pytest.raises((RuntimeError, OSError)):
        health._operate(output, tmp_path / "closed", admission(), prepare, execute)
    assert (output / "pending.json").exists() and (output / "failure.json").exists()
    assert not (output / "ready.json").exists()
    if phase in ("criterion", "retention"):
        assert (output / "relational.json").exists()


def test_disjoint_before_any_write_and_planning_denial(tmp_path: Path) -> None:
    closed = tmp_path / "closed"
    closed.mkdir()
    for output in (closed, closed / "child", closed / ".." / "closed", tmp_path):
        with pytest.raises(PermissionError):
            health._operate(output, closed, admission(), payload, fake)
    assert list(closed.iterdir()) == []
    with pytest.raises(RuntimeError):
        replace(admission(), admission_cpu_debit=3000.0)
    with pytest.raises(ValueError):
        replace(admission(), packaging_reserve_bytes=0)


def test_missing_process_clock_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    read, close = health._process_clock(123)
    assert read() is None
    close()


def test_remaining_deadline_shrinks_and_missing_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(health, "observed_peak_bytes", lambda: 1000000)
    clocks = []

    def execute(
        raw: bytes, condition: str, directory: Path, remaining: float
    ) -> health.ProcessOutcome:
        clocks.append(remaining)
        return fake(raw, condition, directory, remaining)

    health._operate(tmp_path / "shrinks", tmp_path / "closed", admission(), payload, execute)
    assert all(a > b for a, b in pairwise(clocks))
    calls = []

    def missing(
        raw: bytes, condition: str, directory: Path, remaining: float
    ) -> health.ProcessOutcome:
        calls.append(condition)
        return health.ProcessOutcome(result(raw, condition), None, 1.0)

    report = health._operate(
        tmp_path / "missing", tmp_path / "closed", admission(), payload, missing
    )
    assert calls == ["relational"] and report["remaining"] == list(CONDITIONS[1:])
    assert report["historical_wp2_cpu"] == "UNKNOWN"
    assert report["whole_wp2_resource_compliance"] == "INCONCLUSIVE"


def test_preserve_initiating_failure_if_marker_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = health._write

    def write(directory: Path, name: str, data: bytes) -> None:
        if name == "failure.json":
            raise OSError("synthetic marker failure")
        original(directory, name, data)

    monkeypatch.setattr(health, "_write", write)

    def prepare() -> bytes:
        raise RuntimeError("original initiating exception")

    with pytest.raises(RuntimeError, match="original initiating"):
        health._operate(tmp_path / "new", tmp_path / "closed", admission(), prepare, fake)
    assert (tmp_path / "new" / "pending.json").exists()
    assert not (tmp_path / "new" / "ready.json").exists()


def test_environment_change_retained_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(health, "_environment_bytes", lambda: 1)
    with pytest.raises(ValueError, match="environment"):
        health._operate(tmp_path / "new", tmp_path / "closed", admission(), payload, fake)
    assert (tmp_path / "new" / "failure.json").exists()


def test_windows_process_clock_fake_handle_and_signatures(monkeypatch: pytest.MonkeyPatch) -> None:
    import ctypes
    from ctypes import wintypes

    pointer = ctypes.POINTER(wintypes.FILETIME)
    open_api = ctypes.CFUNCTYPE(wintypes.HANDLE, wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)(
        lambda access, inherit, pid: 123
    )
    calls = []

    def times(handle: int, created: Any, exited: Any, kernel: Any, user: Any) -> int:
        kernel.contents.dwLowDateTime = 10000000
        user.contents.dwLowDateTime = 20000000
        return 1

    times_api = ctypes.CFUNCTYPE(
        wintypes.BOOL, wintypes.HANDLE, pointer, pointer, pointer, pointer
    )(times)

    def closed(handle: int) -> int:
        calls.append(handle)
        return 1

    close_api = ctypes.CFUNCTYPE(wintypes.BOOL, wintypes.HANDLE)(closed)
    functions = {"OpenProcess": open_api, "GetProcessTimes": times_api, "CloseHandle": close_api}

    class FakeLibrary(ctypes.CDLL):
        def __init__(self, name: str) -> None:
            pass  # no DLL or native process is loaded

        def __getattr__(self, name: str) -> Any:
            return functions[name]

    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(ctypes, "windll", ctypes.LibraryLoader(FakeLibrary), raising=False)
    read, close = health._process_clock(123)
    assert read() == 3.0
    assert open_api.restype is wintypes.HANDLE
    assert times_api.argtypes == (wintypes.HANDLE, pointer, pointer, pointer, pointer)
    close()
    assert calls == [123]
    monkeypatch.setattr(ctypes, "windll", None)
    read, close = health._process_clock(123)
    assert read() is None
    close()


def test_fake_child_timeout_preserves_exception_if_process_retention_fails() -> None:
    import subprocess

    class FakeChild:
        returncode = None

        def communicate(self, data: bytes, timeout: float) -> None:
            assert data == b"SYNTHETIC" and timeout == 2.0
            raise subprocess.TimeoutExpired("SYNTHETIC fake child", timeout)

        def poll(self) -> None:
            return None

        def kill(self) -> None:
            self.killed = True

        def wait(self) -> int:
            return -9

    child = FakeChild()

    def retain(cpu: float | None) -> None:
        raise OSError("synthetic process retention failure")

    with pytest.raises(subprocess.TimeoutExpired):
        health._communicate(
            cast(subprocess.Popen[bytes], child), b"SYNTHETIC", 2.0, lambda: 1.0, retain
        )
    assert child.killed
