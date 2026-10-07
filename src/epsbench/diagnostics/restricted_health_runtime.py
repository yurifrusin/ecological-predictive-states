"""Narrow owned-process startup and observation; no scientific input in handshake."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path
from typing import Any


def raw(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def publish(path: Path, data: bytes) -> None:
    """Atomic visibility and exclusive destination; control bytes only."""
    staged = path.with_name(path.name + ".writing")
    with staged.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(staged, path)
    staged.unlink()


def write(path: Path, value: Any) -> None:
    publish(path, raw(value))


def identity() -> dict[str, Any]:
    return {
        "pid": os.getpid(),
        "self_cpu_ready": time.process_time(),
        "ppid": os.getppid(),
        "executable": sys.executable,
        "base_executable": getattr(sys, "_base_executable", None),
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "virtual_env": os.environ.get("VIRTUAL_ENV"),
        "login_variable_presence": {
            k: k in os.environ for k in ("LOGNAME", "USER", "LNAME", "USERNAME")
        },
    }


def native() -> dict[str, Any]:
    loader = getattr(ctypes, "windll", None)
    if not isinstance(loader, ctypes.LibraryLoader):
        raise RuntimeError("Windows native owned-process observations required")
    kernel, psapi = loader.LoadLibrary("kernel32"), loader.LoadLibrary("psapi")

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("faults", wintypes.DWORD),
            *[
                (n, ctypes.c_size_t)
                for n in (
                    "peak",
                    "working",
                    "peak_pool",
                    "pool",
                    "peak_nonpool",
                    "nonpool",
                    "page",
                    "peak_page",
                )
            ],
        ]

    opened, clock, image = (
        kernel.OpenProcess,
        kernel.GetProcessTimes,
        kernel.QueryFullProcessImageNameW,
    )
    memory, close = psapi.GetProcessMemoryInfo, kernel.CloseHandle
    opened.argtypes, opened.restype = (
        (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD),
        wintypes.HANDLE,
    )
    clock.argtypes, clock.restype = (
        (wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)),
        wintypes.BOOL,
    )
    image.argtypes, image.restype = (
        (wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)),
        wintypes.BOOL,
    )
    memory.argtypes, memory.restype = (
        (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD),
        wintypes.BOOL,
    )
    close.argtypes, close.restype = (wintypes.HANDLE,), wintypes.BOOL
    wait, terminate = kernel.WaitForSingleObject, kernel.TerminateProcess
    wait.argtypes, wait.restype = (wintypes.HANDLE, wintypes.DWORD), wintypes.DWORD
    terminate.argtypes, terminate.restype = (wintypes.HANDLE, wintypes.UINT), wintypes.BOOL
    last_error = kernel.GetLastError
    last_error.argtypes, last_error.restype = (), wintypes.DWORD
    return {
        "last_error": last_error,
        "wait": wait,
        "terminate": terminate,
        "open": opened,
        "clock": clock,
        "image": image,
        "memory": memory,
        "close": close,
        "Counters": Counters,
    }


def observe(api: dict[str, Any], handle: Any) -> dict[str, Any]:
    values = tuple(wintypes.FILETIME() for _ in range(4))
    success = bool(api["clock"](handle, *(ctypes.byref(v) for v in values)))
    counts = [(v.dwHighDateTime << 32) + v.dwLowDateTime for v in values]
    image = ctypes.create_unicode_buffer(32768)
    size = wintypes.DWORD(len(image))
    image_ok = bool(api["image"](handle, 0, image, ctypes.byref(size)))
    counters = api["Counters"]()
    counters.cb = ctypes.sizeof(counters)
    memory_ok = bool(api["memory"](handle, ctypes.byref(counters), counters.cb))
    return {
        "cpu": (counts[2] + counts[3]) / 10**7 if success else None,
        "creation_100ns": counts[0] if success else None,
        "exit_100ns": counts[1] if success else None,
        "image": image.value if image_ok else None,
        "memory_api_success": memory_ok,
        "peak_working_set": int(counters.peak) if memory_ok else None,
    }


class OwnedProcess:
    """Retained owned launcher/interpreter; no scientific input in this handshake."""

    def __init__(self, child: subprocess.Popen[bytes], ready: Path, timeout: float) -> None:
        self.api = native()
        self.handles: dict[int, Any] = {}
        self.before: dict[int, dict[str, Any]] = {}
        self.ready: dict[str, Any] = {}
        deadline = time.monotonic() + min(timeout, 20.0)
        try:
            while not ready.exists():
                if child.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("owned runtime readiness unavailable")
                time.sleep(0.005)
            if ready.stat().st_size > 8192:
                raise ValueError("bounded public runtime identity required")
            self.ready = json.loads(ready.read_bytes())
            if set(self.ready) != {
                "pid",
                "ppid",
                "executable",
                "base_executable",
                "prefix",
                "base_prefix",
                "virtual_env",
                "login_variable_presence",
                "source",
                "self_cpu_ready",
            }:
                raise ValueError("runtime identity fields differ")
            if (
                self.ready["source"] != sha(Path(__file__).read_bytes())
                or type(self.ready["pid"]) is not int
                or self.ready["ppid"] not in (child.pid, os.getpid())
                or Path(self.ready["executable"]).resolve() != Path(sys.executable).resolve()
            ):
                raise ValueError("owned parent/executable/source identity differs")
            for pid in dict.fromkeys((child.pid, self.ready["pid"])):
                handle = self.api["open"](0x100411, False, pid)
                if not handle:
                    raise RuntimeError("owned runtime handle unavailable")
                self.handles[pid] = handle
                self.before[pid] = observe(self.api, handle)
                if (
                    not self.before[pid]["creation_100ns"]
                    or not self.before[pid]["image"]
                    or self.before[pid]["cpu"] is None
                ):
                    raise RuntimeError("owned live process identity unavailable")
        except BaseException:
            try:
                write(
                    ready.with_name(ready.name + "-failure.json"),
                    {"cleanup": self.cleanup(), "before": self.before},
                )
            except BaseException:
                pass
            self.close()
            raise

    def finish(self, timeout: float) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        deadline = time.monotonic() + timeout
        for pid, handle in self.handles.items():
            result = self.api["wait"](handle, max(1, int(1000 * (deadline - time.monotonic()))))
            if result != 0:
                raise RuntimeError(
                    f"owned runtime wait return={result} error={self.api['last_error']()}"
                )
            final = observe(self.api, handle)
            if (
                final["creation_100ns"] != self.before[pid]["creation_100ns"]
                or not final["exit_100ns"]
                or final["cpu"] is None
                or final["cpu"] < self.before[pid]["cpu"]
                or (pid == self.ready["pid"] and final["cpu"] + 1e-7 < self.ready["self_cpu_ready"])
            ):
                raise RuntimeError("terminal process identity/CPU unavailable")
            records.append({"pid": pid, "before": self.before[pid], "final": final})
        return {
            "ready": self.ready,
            "processes": records,
            "cpu": sum(r["final"]["cpu"] for r in records),
            "peak": (
                max(r["final"]["peak_working_set"] for r in records)
                if all(
                    r["final"]["peak_working_set"] is not None
                    and r["final"]["peak_working_set"] > 0
                    for r in records
                )
                else None
            ),
        }

    def cleanup(self) -> dict[str, Any]:
        records = []
        for pid, handle in self.handles.items():
            initial = self.api["wait"](handle, 0)
            record = {
                "pid": pid,
                "initial_wait": initial,
                "terminated": False,
                "terminal_wait": initial,
                "error": None,
            }
            if initial == 258:  # WAIT_TIMEOUT proves retained owned process still live.
                record["terminated"] = bool(self.api["terminate"](handle, 1))
                if not record["terminated"]:
                    record["error"] = self.api["last_error"]()
                record["terminal_wait"] = self.api["wait"](handle, 20000)
            elif initial != 0:
                record["error"] = self.api["last_error"]()
            records.append(record)
        return {
            "processes": records,
            "complete": bool(records)
            and all(r["terminal_wait"] == 0 and r["error"] is None for r in records),
        }

    def close(self) -> None:
        for handle in self.handles.values():
            self.api["close"](handle)
        self.handles.clear()


def filtered_environment() -> dict[str, str]:
    """Ordinary login setup only; no private data or inherited Python configuration."""
    allowed = {
        "SYSTEMROOT",
        "WINDIR",
        "TEMP",
        "TMP",
        "PATH",
        "LOGNAME",
        "USER",
        "LNAME",
        "USERNAME",
    }
    result = {k: v for k, v in os.environ.items() if k.upper() in allowed}
    result.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    if os.name == "nt" and not any(result.get(k) for k in ("LOGNAME", "USER", "LNAME", "USERNAME")):
        raise RuntimeError("ordinary account-name startup setup unavailable")
    return result


def handshake(ready: Path, go: Path, expected: str) -> None:
    """Only public runtime identity; selected learner packet remains stdin only."""
    if sha(Path(__file__).read_bytes()) != expected or ready.exists() or go.exists():
        raise ValueError("exact unused runtime source/control paths required")
    write(ready, identity() | {"source": expected})
    deadline = time.monotonic() + 20
    while not go.exists():
        if time.monotonic() >= deadline:
            raise RuntimeError("owned runtime ACK deadline")
        time.sleep(0.005)
    if go.read_bytes() != expected.encode():
        raise ValueError("exact runtime source ACK required")


if __name__ == "__main__":
    if len(sys.argv) != 7 or sys.argv[1] != "--learner":
        raise PermissionError("no default runtime operation")
    handshake(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from epsbench.diagnostics.restricted_model_health import main

    sys.argv = [sys.argv[0], "--learner", sys.argv[5], sys.argv[6]]
    raise SystemExit(main())
