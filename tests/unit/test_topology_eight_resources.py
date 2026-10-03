"""Mocked process/clock tests: no executor, graphics context, or GPU queries."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from epsbench.diagnostics import topology_eight_resources as resources


def phases(root: Path, count: int = 16, duration: float = 1.0) -> None:
    operator = root / "operator"
    operator.mkdir(parents=True, exist_ok=True)
    for sequence in range(count):
        (operator / f"resource-phase-{sequence:04d}.json").write_text(
            json.dumps(
                {
                    "ordinal": sequence // 2,
                    "event": "start" if sequence % 2 == 0 else "complete",
                    "monotonic_seconds": 1.0 + sequence * duration,
                }
            )
        )


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    state: dict[str, Any] = {
        "root": tmp_path / "study",
        "launches": [],
        "kills": [],
        "now": 20.0,
        "rss": 1,
        "output": 1,
        "count": 16,
        "code": 0,
    }

    class FakeProcess:
        pid = 42

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            state["launches"].append((args, kwargs))
            phases(state["root"], state["count"])
            (state["root"] / "retained.lock").write_text("owned")
            kwargs["stdout"].write(b"partial worker output\n")
            kwargs["stdout"].flush()
            kwargs["stderr"].write(b"partial native diagnostics\n")
            kwargs["stderr"].flush()

        def poll(self) -> int | None:
            value = state["code"]
            assert value is None or isinstance(value, int)
            return value

        def wait(self, timeout: int) -> int:
            state["waited"] = True
            return -9

    times = iter([0.0])
    monkeypatch.setattr(resources, "require_linux_capabilities", lambda: None)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {3, 4, 7}, raising=False)
    monkeypatch.setattr(subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(time, "monotonic", lambda: next(times, state["now"]))
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    monkeypatch.setattr(resources, "_tree_usage", lambda *args: (state["rss"], {42: 100}))
    monkeypatch.setattr(resources, "_output_bytes", lambda root: state["output"])
    monkeypatch.setattr(resources, "_kill", lambda pid, known: state["kills"].append(pid))
    monkeypatch.setattr(resources, "_group_alive", lambda pid: False)
    return state


def run(state: dict[str, Any]) -> int:
    root = state["root"]
    return resources.supervise(
        ["python", "--internal-worker"], root, root.parent / (root.name + ".supervisor.json")
    )


def test_one_session_and_controlled_cpu_environment(harness: dict[str, Any]) -> None:
    assert run(harness) == 0
    assert len(harness["launches"]) == 1
    _, kwargs = harness["launches"][0]
    assert kwargs["start_new_session"] is True
    assert kwargs["preexec_fn"] is resources._preexec_limits
    assert kwargs["stdout"].name.endswith(".worker-stdout.log")
    assert kwargs["stderr"].name.endswith(".worker-stderr.log")
    assert kwargs["env"]["EPS_TOPOLOGY_SUPERVISOR_PID"] == str(os.getpid())
    assert all(kwargs["env"][key] == value for key, value in resources.THREAD_ENV.items())
    assert harness["kills"] == []
    records = [
        json.loads(line)
        for line in (harness["root"].parent / "study.supervisor.json").read_text().splitlines()
    ]
    assert records[0]["cpus"] == [3, 4]
    assert records[-1]["event"] == "complete"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("rss", resources.MEMORY_BYTES + 1, "RSS"),
        ("output", resources.OUTPUT_BYTES + 1, "output"),
        ("now", resources.TOTAL_SECONDS + 1, "total wall"),
        ("count", 1, "cell wall"),
        ("count", 0, "incomplete"),
        ("code", 2, "worker failed"),
    ],
)
def test_breach_kills_once_preserves_namespace(
    harness: dict[str, Any], field: str, value: int, message: str
) -> None:
    harness[field] = value
    if message == "cell wall":
        harness["now"] = resources.CELL_SECONDS + 2
    with pytest.raises(resources.ResourceFailure, match=message):
        run(harness)
    assert harness["kills"] == [42]
    assert harness["waited"]
    assert harness["root"].exists()
    assert (harness["root"] / "retained.lock").read_text() == "owned"
    assert (harness["root"].parent / "study.worker-stderr.log").read_bytes() == (
        b"partial native diagnostics\n"
    )
    log = harness["root"].parent / "study.supervisor.json"
    assert json.loads(log.read_text().splitlines()[-1])["event"] == "permanent_stop"
    with pytest.raises(resources.ResourceFailure, match="fresh"):
        run(harness)
    assert len(harness["launches"]) == 1


def test_outer_log_never_overwritten(harness: dict[str, Any]) -> None:
    log = harness["root"].parent / "study.supervisor.json"
    log.write_text("retained")
    with pytest.raises(FileExistsError):
        run(harness)
    assert log.read_text() == "retained"
    assert harness["launches"] == []


def test_worker_stream_never_overwritten(harness: dict[str, Any]) -> None:
    stream = harness["root"].parent / "study.worker-stderr.log"
    stream.write_text("preserved diagnostics")
    with pytest.raises(FileExistsError):
        run(harness)
    assert stream.read_text() == "preserved diagnostics"
    assert harness["launches"] == []


def test_non_linux_never_launches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(resources.ResourceFailure, match="Linux"):
        resources.require_linux_capabilities()


@pytest.mark.parametrize("mutation", ["gap", "order", "nan", "future", "backwards", "fields"])
def test_phase_corruption_is_closed(tmp_path: Path, mutation: str) -> None:
    phases(tmp_path, 2)
    path = tmp_path / "operator/resource-phase-0001.json"
    value = json.loads(path.read_text())
    if mutation == "gap":
        path.rename(path.with_name("resource-phase-0002.json"))
    else:
        if mutation == "order":
            value["ordinal"] = 1
        elif mutation == "nan":
            value["monotonic_seconds"] = float("nan")
        elif mutation == "future":
            value["monotonic_seconds"] = 21
        elif mutation == "backwards":
            value["monotonic_seconds"] = 0
        else:
            value["other"] = "undeclared"
        path.write_text(json.dumps(value))
    with pytest.raises(resources.ResourceFailure):
        resources._phases(tmp_path, 0, 20)


def test_completed_cell_elapsed_is_checked_even_if_polls_missed_start(tmp_path: Path) -> None:
    phases(tmp_path, 2, resources.CELL_SECONDS + 1)
    with pytest.raises(resources.ResourceFailure, match="cell wall"):
        resources._phases(tmp_path, 0, 400)


def test_final_assessment_remains_under_total_clock(harness: dict[str, Any]) -> None:
    harness["now"] = resources.TOTAL_SECONDS + 1
    with pytest.raises(resources.ResourceFailure, match="total wall"):
        run(harness)


def test_output_counts_all_files(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "a").write_bytes(b"a" * 7)
    (tmp_path / "nested/b").write_bytes(b"b" * 8)
    assert resources._output_bytes(tmp_path) == 15


def test_output_hardlink_rejected(tmp_path: Path) -> None:
    original = tmp_path / "a"
    original.write_bytes(b"a")
    os.link(original, tmp_path / "b")
    with pytest.raises(resources.ResourceFailure, match="hardlink"):
        resources._output_bytes(tmp_path)


def test_own_publication_pair_is_counted_twice(tmp_path: Path) -> None:
    original = tmp_path / "artifact.json"
    original.write_bytes(b"abc")
    os.link(original, tmp_path / ".artifact.json.0123456789abcdef.tmp")
    assert resources._output_bytes(tmp_path) == 6


def test_child_limits_before_native_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, Any] = {"cpus": {3, 4, 7}}
    fake = SimpleNamespace(
        RLIMIT_AS=9,
        setrlimit=lambda key, value: observed.update(limit=value),
        getrlimit=lambda key: observed["limit"],
    )
    monkeypatch.setitem(sys.modules, "resource", fake)
    monkeypatch.setattr(resources, "require_linux_capabilities", lambda: None)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: observed["cpus"], raising=False)
    monkeypatch.setattr(
        os,
        "sched_setaffinity",
        lambda pid, cpus: observed.update(cpus=cpus),
        raising=False,
    )
    for key in resources.THREAD_ENV:
        monkeypatch.setenv(key, "999")
    facts = resources.child_apply_limits()
    assert observed["cpus"] == {3, 4}
    assert observed["limit"] == (resources.MEMORY_BYTES, resources.MEMORY_BYTES)
    assert facts["cpus"] == [3, 4]
    assert all(os.environ[key] == value for key, value in resources.THREAD_ENV.items())


def test_surviving_unobserved_process_group_fails(
    harness: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(resources, "_group_alive", lambda pid: True)
    with pytest.raises(resources.ResourceFailure, match="surviving descendants"):
        run(harness)
    assert harness["kills"] == [42]


@pytest.mark.parametrize("escape", ["group", "thread", "address"])
def test_process_and_each_thread_containment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, escape: str
) -> None:
    proc = tmp_path / "42"
    (proc / "task/42").mkdir(parents=True)
    (proc / "task/43").mkdir()
    (proc / "status").write_text("VmRSS:\t12 kB\n")
    cap = resources.MEMORY_BYTES if escape != "address" else resources.MEMORY_BYTES + 1
    (proc / "limits").write_text(f"Max address space         {cap} {cap} bytes\n")
    monkeypatch.setattr(resources, "_PROC_ROOT", tmp_path)
    monkeypatch.setattr(resources, "_owned_processes", lambda *args: {42: 100})
    monkeypatch.setattr(os, "getpgid", lambda pid: 42 if escape != "group" else 99, raising=False)
    monkeypatch.setattr(
        os,
        "sched_getaffinity",
        lambda tid: {3, 4, 7} if escape == "thread" and tid == 43 else {3, 4},
        raising=False,
    )
    known: dict[int, int] = {}
    with pytest.raises(resources.ResourceFailure):
        resources._tree_usage(42, known, {3, 4})
    assert known == {42: 100}


def test_kill_preserves_reused_unrelated_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(os, "killpg", lambda pid, sig: calls.append((pid, sig)), raising=False)
    monkeypatch.setattr(os, "kill", lambda pid, sig: calls.append((pid, sig)))
    monkeypatch.setattr(resources, "_identity", lambda pid: 100 if pid == 43 else 999)
    resources._kill(42, {43: 100, 44: 101})
    assert calls == [(42, 9), (43, 9)]


def test_scoped_descendants_and_pid_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for pid, children, identity in [
        (42, "43", 100),
        (43, "44", 101),
        (44, "", 102),
        (99, "100", 999),
    ]:
        proc = tmp_path / str(pid)
        (proc / "task" / str(pid)).mkdir(parents=True)
        (proc / "task" / str(pid) / "children").write_text(children)
        (proc / "stat").write_text(f"{pid} (process name) S " + "0 " * 18 + str(identity))
    monkeypatch.setattr(resources, "_PROC_ROOT", tmp_path)
    assert resources._owned_processes(42, {99: 200}) == {42: 100, 43: 101, 44: 102}
