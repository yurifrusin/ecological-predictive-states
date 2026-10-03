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
        "events": [],
        "survivors": [],
    }

    class FakeProcess:
        pid = 42

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            state["events"].append("popen")
            state["launches"].append((args, kwargs))
            phases(state["root"], state["count"])
            (state["root"] / "retained.lock").write_text("owned")
            kwargs["stdout"].write(b"partial worker output\n")
            kwargs["stdout"].flush()
            kwargs["stderr"].write(b"partial native diagnostics\n")
            kwargs["stderr"].flush()

        def poll(self) -> int | None:
            raise AssertionError("poll would reap the process-group anchor")

        def wait(self, timeout: int) -> int:
            state["waited"] = True
            state["reaped"] = True
            state["events"].append("reap")
            value = state["code"]
            assert isinstance(value, int)
            return value

    times = iter([0.0])
    monkeypatch.setattr(resources, "require_linux_capabilities", lambda: None)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {3, 4, 7}, raising=False)
    monkeypatch.setattr(subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(time, "monotonic", lambda: next(times, state["now"]))
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    monkeypatch.setattr(resources, "_tree_usage", lambda *args: (state["rss"], {42: 100}))
    monkeypatch.setattr(resources, "_output_bytes", lambda root: state["output"])

    def kill(pid: int, known: dict[int, int]) -> None:
        assert not state.get("reaped", False)
        state["events"].append("group_cleanup")
        state["kills"].append(pid)

    monkeypatch.setattr(resources, "_kill", kill)
    monkeypatch.setattr(resources, "_identity", lambda pid: 100)
    monkeypatch.setattr(resources, "_wait_status", lambda pid: state["code"])
    monkeypatch.setattr(resources, "_observed_survivors", lambda *args: state["survivors"])
    monkeypatch.setattr(
        resources, "_flush_parent_directory", lambda parent: state["events"].append("parent_flush")
    )
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
    assert harness["kills"] == [42]
    assert harness["reaped"]
    assert harness["events"].index("group_cleanup") < harness["events"].index("reap")
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


def test_external_evidence_fsynced_before_launch(
    harness: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    real_fsync = os.fsync

    def fsync(descriptor: int) -> None:
        harness["events"].append("fsync")
        real_fsync(descriptor)

    def flush_parent(parent: Path) -> None:
        assert parent == harness["root"].parent
        assert (parent / "study.worker-stdout.log").exists()
        assert (parent / "study.worker-stderr.log").exists()
        records = (parent / "study.supervisor.json").read_text().splitlines()
        assert json.loads(records[-1])["event"] == "launch"
        harness["events"].append("parent_flush")

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(resources, "_flush_parent_directory", flush_parent)
    assert run(harness) == 0
    assert harness["events"][:5] == ["fsync", "fsync", "fsync", "parent_flush", "popen"]


def test_failed_parent_flush_prevents_launch_preserves_failure(
    harness: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(parent: Path) -> None:
        raise OSError("parent fsync injected failure")

    monkeypatch.setattr(resources, "_flush_parent_directory", fail)
    with pytest.raises(OSError, match="parent fsync injected"):
        run(harness)
    assert harness["launches"] == []
    assert not harness["root"].exists()
    parent = harness["root"].parent
    assert (parent / "study.worker-stdout.log").exists()
    assert (parent / "study.worker-stderr.log").exists()
    terminal = json.loads((parent / "study.supervisor.json").read_text().splitlines()[-1])
    assert terminal["event"] == "permanent_stop"
    assert terminal["error"] == "parent fsync injected failure"


@pytest.mark.parametrize("failure", [False, True])
def test_linux_parent_flush_readonly_descriptor_closed_even_on_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: bool
) -> None:
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os, "O_DIRECTORY", 65536, raising=False)
    monkeypatch.setattr(os, "O_NOFOLLOW", 131072, raising=False)

    def open_directory(path: Path, flags: int) -> int:
        calls.append(("open", (path, flags)))
        return 17

    def fsync(descriptor: int) -> None:
        calls.append(("fsync", descriptor))
        if failure:
            raise OSError("injected directory failure")

    monkeypatch.setattr(os, "open", open_directory)
    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "close", lambda descriptor: calls.append(("close", descriptor)))
    if failure:
        with pytest.raises(OSError, match="injected directory failure"):
            resources._flush_parent_directory(tmp_path)
    else:
        resources._flush_parent_directory(tmp_path)
    assert calls == [
        ("open", (tmp_path, os.O_RDONLY | 65536 | 131072)),
        ("fsync", 17),
        ("close", 17),
    ]


@pytest.mark.parametrize("bad_flag", [None, False, 0])
def test_directory_flags_fail_closed_without_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bad_flag: object
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os, "O_DIRECTORY", bad_flag, raising=False)
    monkeypatch.setattr(os, "O_NOFOLLOW", 131072, raising=False)
    with pytest.raises(resources.ResourceFailure, match="flags"):
        resources._flush_parent_directory(tmp_path)


def test_missing_directory_flag_has_no_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delattr(os, "O_DIRECTORY", raising=False)
    with pytest.raises(AttributeError):
        resources._flush_parent_directory(tmp_path)


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


def test_surviving_group_member_fails_before_successful_exit_cleanup(
    harness: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    harness["survivors"] = [43]
    monkeypatch.setattr(resources, "_confirm_cleanup", lambda *args: None)
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
    closed: list[int] = []
    monkeypatch.setattr(signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(os, "killpg", lambda pid, sig: calls.append((pid, sig)), raising=False)
    monkeypatch.setattr(os, "getpgid", lambda pid: 42, raising=False)
    monkeypatch.setattr(os, "getsid", lambda pid: 42, raising=False)
    monkeypatch.setattr(os, "pidfd_open", lambda pid, flags: pid + 1000, raising=False)
    monkeypatch.setattr(
        signal,
        "pidfd_send_signal",
        lambda fd, sig, info, flags: calls.append((fd, sig)),
        raising=False,
    )
    monkeypatch.setattr(os, "close", lambda fd: closed.append(fd))
    monkeypatch.setattr(resources, "_wait_status", lambda pid: None)
    monkeypatch.setattr(resources, "_identity", lambda pid: {42: 100, 43: 101}.get(pid, 999))
    resources._kill(42, {42: 100, 43: 101, 44: 102})
    assert calls == [(42, 9), (1043, 9)]
    assert closed == [1043, 1044]


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
    assert resources._owned_processes(42, {42: 100, 99: 200}) == {42: 100, 43: 101, 44: 102}


def fake_process(
    root: Path,
    pid: int,
    identity: int,
    group: int,
    session: int,
    children: str = "",
    state: str = "S",
) -> None:
    proc = root / str(pid)
    (proc / "task" / str(pid)).mkdir(parents=True)
    fields = [state, "1", str(group), str(session), *(["0"] * 15), str(identity)]
    (proc / "stat").write_text(f"{pid} (process name) " + " ".join(fields))
    (proc / "status").write_text(f"State:\t{state}\n")
    (proc / "task" / str(pid) / "children").write_text(children)


@pytest.mark.parametrize("reused", [42, 43])
def test_discovery_never_relabels_reused_root_or_member(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, reused: int
) -> None:
    fake_process(tmp_path, 42, 999 if reused == 42 else 100, 42, 42, "43")
    fake_process(tmp_path, 43, 999 if reused == 43 else 101, 42, 42, "44")
    fake_process(tmp_path, 44, 102, 42, 42)
    monkeypatch.setattr(resources, "_PROC_ROOT", tmp_path)
    known = {42: 100, 43: 101}
    if reused == 42:
        with pytest.raises(resources.ResourceFailure, match="root identity changed"):
            resources._owned_processes(42, known)
    else:
        assert resources._owned_processes(42, known) == {42: 100}
    assert known == {42: 100, 43: 101}


def test_anchored_group_detects_unobserved_reparented_survivor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_process(tmp_path, 42, 100, 42, 42, state="Z")
    fake_process(tmp_path, 43, 101, 42, 42)
    fake_process(tmp_path, 44, 102, 42, 42, state="Z")
    fake_process(tmp_path, 99, 999, 99, 99)
    monkeypatch.setattr(resources, "_PROC_ROOT", tmp_path)
    monkeypatch.setattr(resources, "_assert_leader_anchor", lambda *args: None)
    known = {42: 100}
    assert resources._observed_survivors(42, known) == [43]
    assert known == {42: 100, 43: 101, 44: 102}


@pytest.mark.parametrize("code,status,expected", [(1, 7, 7), (2, 9, -9), (3, 11, -11)])
def test_nonreaping_waitid_observation(
    monkeypatch: pytest.MonkeyPatch, code: int, status: int, expected: int
) -> None:
    constants = {
        "P_PID": 1,
        "WEXITED": 4,
        "WNOHANG": 1,
        "WNOWAIT": 16,
        "CLD_EXITED": 1,
        "CLD_KILLED": 2,
        "CLD_DUMPED": 3,
    }
    for name, value in constants.items():
        monkeypatch.setattr(os, name, value, raising=False)
    calls: list[tuple[int, int, int]] = []

    def waitid(kind: int, pid: int, flags: int) -> SimpleNamespace:
        calls.append((kind, pid, flags))
        return SimpleNamespace(si_pid=pid, si_status=status, si_code=code)

    monkeypatch.setattr(os, "waitid", waitid, raising=False)
    assert resources._wait_status(42) == expected
    assert calls == [(1, 42, 4 | 1 | 16)]


def test_lost_wait_ownership_never_signals_group(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, int]] = []

    def lost(pid: int) -> int:
        raise resources.ResourceFailure("worker leader ownership lost")

    monkeypatch.setattr(resources, "_wait_status", lost)
    monkeypatch.setattr(os, "killpg", lambda pid, sig: calls.append((pid, sig)), raising=False)
    with pytest.raises(resources.ResourceFailure, match="ownership lost"):
        resources._kill(42, {42: 100})
    assert calls == []


def test_reused_leader_never_signals_group(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(resources, "_wait_status", lambda pid: 0)
    monkeypatch.setattr(resources, "_identity", lambda pid: 999)
    monkeypatch.setattr(os, "killpg", lambda pid, sig: calls.append((pid, sig)), raising=False)
    with pytest.raises(resources.ResourceFailure, match="identity unconfirmed"):
        resources._kill(42, {42: 100})
    assert calls == []


@pytest.mark.parametrize("race", ["reuse_before_verify", "exit_after_verify"])
def test_pidfd_signal_closes_handle_and_never_uses_numeric_pid(
    monkeypatch: pytest.MonkeyPatch, race: str
) -> None:
    events: list[str] = []

    def pidfd_open(pid: int, flags: int) -> int:
        events.append("open_bound_handle")
        return 17

    def identity(pid: int) -> int:
        assert events == ["open_bound_handle"]
        events.append("identity_verify")
        return 999 if race == "reuse_before_verify" else 100

    def send(fd: int, sig: int, info: object, flags: int) -> None:
        assert fd == 17
        events.append("pidfd_signal")
        raise ProcessLookupError("original handled task exited")

    def numeric_signal(pid: int, sig: int) -> None:
        raise AssertionError("reusable numeric PID must never be signalled")

    monkeypatch.setattr(os, "pidfd_open", pidfd_open, raising=False)
    monkeypatch.setattr(signal, "pidfd_send_signal", send, raising=False)
    monkeypatch.setattr(signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(os, "kill", numeric_signal)
    monkeypatch.setattr(os, "close", lambda fd: events.append("close_handle"))
    monkeypatch.setattr(resources, "_identity", identity)
    resources._signal_bound_process(43, 100)
    expected = ["open_bound_handle", "identity_verify"]
    if race == "exit_after_verify":
        expected.append("pidfd_signal")
    assert events == [*expected, "close_handle"]


def test_cleanup_uncertainty_is_logged_without_reaping(
    harness: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    harness["code"] = 2

    def lost(pid: int, known: dict[int, int]) -> None:
        raise resources.ResourceFailure("worker leader ownership lost")

    monkeypatch.setattr(resources, "_kill", lost)
    with pytest.raises(resources.ResourceFailure, match="termination is unconfirmed"):
        run(harness)
    assert not harness.get("reaped", False)
    terminal = json.loads(
        (harness["root"].parent / "study.supervisor.json").read_text().splitlines()[-1]
    )
    assert terminal["event"] == "permanent_stop"
    assert terminal["termination"] == "unconfirmed"
    assert terminal["cleanup_error"] == "worker leader ownership lost"


def test_postreap_evidence_failure_never_signals_group_again(
    harness: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    real_fsync = os.fsync
    injected = False

    def fsync(descriptor: int) -> None:
        nonlocal injected
        if harness.get("reaped", False) and not injected:
            injected = True
            raise OSError("postreap evidence failure")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fsync)
    with pytest.raises(OSError, match="postreap"):
        run(harness)
    assert harness["reaped"]
    assert harness["kills"] == [42]


def test_observed_termination_confirmation_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    times = iter([0.0, 1.0, 6.0])
    monkeypatch.setattr(time, "monotonic", lambda: next(times))
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    monkeypatch.setattr(resources, "_wait_status", lambda pid: 0)
    monkeypatch.setattr(resources, "_observed_survivors", lambda *args: [43])
    with pytest.raises(resources.ResourceFailure, match="termination remains unconfirmed"):
        resources._confirm_cleanup(42, {42: 100, 43: 101})
