"""Synthetic runtime identity/accounting only; no subprocess or learner execution."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest

from epsbench.diagnostics import restricted_health_runtime as runtime


def setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, same: bool = False
) -> tuple[Any, list[Any]]:
    pid = 111 if same else 222
    ready = tmp_path / "runtime.json"
    runtime.write(
        ready,
        runtime.identity()
        | {
            "pid": pid,
            "ppid": 111,
            "self_cpu_ready": 0.1,
            "source": runtime.sha(Path(runtime.__file__).read_bytes()),
        },
    )
    calls: list[Any] = []

    def opened(mask: int, inherit: bool, value: int) -> int:
        calls.append((mask, value))
        return value

    api = {
        "open": opened,
        "close": lambda handle: calls.append(("close", handle)),
        "wait": lambda handle, timeout: 0,
        "last_error": lambda: 0,
    }
    monkeypatch.setattr(runtime, "native", lambda: api)
    monkeypatch.setattr(
        runtime,
        "observe",
        lambda api, handle: {
            "creation_100ns": handle,
            "exit_100ns": 1,
            "image": runtime.identity()["base_executable"],
            "cpu": 1.0,
            "peak_working_set": 1000,
            "memory_api_success": True,
        },
    )

    class Child:
        pid = 111

        def poll(self) -> None:
            return None

    child = cast(subprocess.Popen[bytes], Child())
    return runtime.OwnedProcess(child, ready, 1.0), calls


@pytest.mark.parametrize("same,expected", [(False, 2.0), (True, 1.0)])
def test_unique_owned_interpreter_and_launcher_counted_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, same: bool, expected: float
) -> None:
    owned, calls = setup(tmp_path, monkeypatch, same)
    final = owned.finish(1.0)
    assert final["cpu"] == expected and final["peak"] == 1000
    assert all(mask == 0x100411 for mask, value in calls)
    owned.close()
    assert len([c for c in calls if c[0] == "close"]) == (1 if same else 2)


@pytest.mark.parametrize("change", ["creation", "cpu", "exit"])
def test_terminal_identity_and_measurement_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    owned, _ = setup(tmp_path, monkeypatch)
    original = runtime.observe

    def observed(api: Any, handle: Any) -> dict[str, Any]:
        result = original(api, handle)
        result[{"creation": "creation_100ns", "cpu": "cpu", "exit": "exit_100ns"}[change]] = None
        return result

    monkeypatch.setattr(runtime, "observe", observed)
    with pytest.raises(RuntimeError, match="terminal process"):
        owned.finish(1.0)
    owned.close()


def test_filtered_environment_excludes_private_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        os,
        "environ",
        {
            "USERNAME": "public-user",
            "SECRET": "DENIED",
            "PYTHONPATH": "DENIED",
            "OMP_NUM_THREADS": "64",
            "PATH": "public-path",
        },
    )
    environment = runtime.filtered_environment()
    assert environment["USERNAME"] == "public-user" and environment["OMP_NUM_THREADS"] == "1"
    assert "SECRET" not in environment and "PYTHONPATH" not in environment


def test_handshake_hash_and_existing_ack_denied(tmp_path: Path) -> None:
    ready, go = tmp_path / "ready", tmp_path / "go"
    with pytest.raises(ValueError, match="source/control"):
        runtime.handshake(ready, go, "0" * 64)
    assert not ready.exists()
    go.write_bytes(b"WRONG")
    with pytest.raises(ValueError, match="source/control"):
        runtime.handshake(ready, go, runtime.sha(Path(runtime.__file__).read_bytes()))
    assert not ready.exists()


def test_missing_actual_peak_never_replaced_by_launcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owned, _ = setup(tmp_path, monkeypatch)
    original = runtime.observe

    def observed(api: Any, handle: Any) -> dict[str, Any]:
        result = original(api, handle)
        if handle == 222:
            result["peak_working_set"] = None
        return result

    monkeypatch.setattr(runtime, "observe", observed)
    assert owned.finish(1.0)["peak"] is None
    owned.close()


def test_atomic_ack_not_visible_while_staged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "ack"
    original = os.link

    def link(staged: Path, final: Path) -> None:
        assert not final.exists() and staged.read_bytes() == b"COMPLETE_PUBLIC_ACK"
        original(staged, final)

    monkeypatch.setattr(os, "link", link)
    runtime.publish(destination, b"COMPLETE_PUBLIC_ACK")
    assert destination.read_bytes() == b"COMPLETE_PUBLIC_ACK"
    monkeypatch.setattr(os, "link", original)
    with pytest.raises(FileExistsError):
        runtime.publish(destination, b"NO_OVERWRITE")
    assert destination.read_bytes() == b"COMPLETE_PUBLIC_ACK"


def test_owned_actual_cleanup_independent_of_exited_launcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owned, _ = setup(tmp_path, monkeypatch)
    terminated: list[int] = []
    owned.api["wait"] = lambda handle, timeout: 0 if handle == 111 or timeout else 258

    def terminate(handle: int, code: int) -> bool:
        terminated.append(handle)
        return True

    owned.api["terminate"] = terminate
    report = owned.cleanup()
    assert report["complete"] and terminated == [222]
    owned.close()


def test_actual_cpu_must_dominate_its_own_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owned, _ = setup(tmp_path, monkeypatch)
    owned.ready["self_cpu_ready"] = 2.0
    with pytest.raises(RuntimeError, match="terminal process"):
        owned.finish(1.0)
    owned.close()
