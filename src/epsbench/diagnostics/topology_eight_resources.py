"""Linux process-tree supervisor for the finite development invocation.

No renderer is imported or constructed here. RSS and output are polled; address
space and inherited CPU affinity are kernel-enforced. A breach permanently kills
the invocation and leaves its study namespace and ownership files untouched.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

MEMORY_BYTES = 8 * 1024**3
OUTPUT_BYTES = 1024**3
CELL_SECONDS = 300
TOTAL_SECONDS = 2700
POLL_SECONDS = 0.1
_PROC_ROOT = Path("/proc")
THREAD_ENV = {"LP_NUM_THREADS": "2", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}


class ResourceFailure(RuntimeError):
    """Resource enforcement failed; restarting this namespace is forbidden."""


def _linux_attribute(module: object, name: str) -> Any:
    return getattr(module, name)


def _affinity(pid: int) -> set[int]:
    # Dynamic lookup keeps the inert module importable/typecheckable on Windows.
    getter = cast(Callable[[int], set[int]], _linux_attribute(os, "sched_getaffinity"))
    return getter(pid)


def require_linux_capabilities() -> None:
    """Fail before launching unless the required Linux interfaces are accessible."""
    if sys.platform != "linux":
        raise ResourceFailure("resource enforcement requires Linux/WSL")
    resource = importlib.import_module("resource")

    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise ResourceFailure("CPU affinity is unavailable")
    if not _affinity(0):
        raise ResourceFailure("no permitted logical CPUs")
    resource.getrlimit(resource.RLIMIT_AS)
    own = _PROC_ROOT / str(os.getpid())
    for path in (own / "status", own / "limits", own / "task" / str(os.getpid()) / "children"):
        path.read_text(encoding="ascii")


def child_apply_limits() -> dict[str, object]:
    """Apply inherited limits before any graphics/native imports in the worker."""
    require_linux_capabilities()
    resource = importlib.import_module("resource")

    cpus = set(sorted(_affinity(0))[:2])
    _linux_attribute(os, "sched_setaffinity")(0, cpus)
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
    os.environ.update(THREAD_ENV)
    if _affinity(0) != cpus:
        raise ResourceFailure("CPU affinity could not be enforced")
    if resource.getrlimit(resource.RLIMIT_AS) != (MEMORY_BYTES, MEMORY_BYTES):
        raise ResourceFailure("address-space limit could not be enforced")
    return {"cpus": sorted(cpus), "address_space_bytes": MEMORY_BYTES, "thread_env": THREAD_ENV}


def _preexec_limits() -> None:
    child_apply_limits()


def _mark(root: Path, ordinal: int, event: str) -> None:
    if type(ordinal) is not int or not 0 <= ordinal < 8:
        raise ResourceFailure("invalid resource phase ordinal")
    sequence = ordinal * 2 + int(event == "complete")
    from epsbench.diagnostics.revision_capture import canonical_json_bytes, publish_bytes

    publish_bytes(
        root / "operator" / f"resource-phase-{sequence:04d}.json",
        canonical_json_bytes(
            {"ordinal": ordinal, "event": event, "monotonic_seconds": time.monotonic()}
        ),
    )


def mark_cell_start(root: Path, ordinal: int) -> None:
    """Mark immediately before reservation, including constructor failures."""
    _mark(root, ordinal, "start")


def mark_cell_complete(root: Path, ordinal: int) -> None:
    """Mark only after release confirmation was observed without exception."""
    _mark(root, ordinal, "complete")


def _identity(pid: int) -> int:
    return int((_PROC_ROOT / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[19])


def _owned_processes(pid: int, known: dict[int, int]) -> dict[int, int]:
    """Discover only children of the worker or previously observed descendants."""
    pending = [pid]
    for descendant, identity in list(known.items()):
        try:
            if _identity(descendant) == identity:
                pending.append(descendant)
        except FileNotFoundError:
            continue
    live: dict[int, int] = {}
    while pending:
        current = pending.pop()
        if current in live:
            continue
        proc = _PROC_ROOT / str(current)
        if not proc.exists():
            continue
        try:
            live[current] = _identity(current)
        except FileNotFoundError:
            continue
        try:
            tasks = list((proc / "task").iterdir())
        except FileNotFoundError:
            continue
        for task in tasks:
            try:
                pending.extend(int(value) for value in (task / "children").read_text().split())
            except FileNotFoundError:
                continue
    return live


def _tree_usage(pid: int, known: dict[int, int], cpus: set[int]) -> tuple[int, dict[int, int]]:
    live = _owned_processes(pid, known)
    known.update(live)
    rss = 0
    for current in live:
        proc = _PROC_ROOT / str(current)
        try:
            if _linux_attribute(os, "getpgid")(current) != pid:
                raise ResourceFailure("worker descendant escaped its process group")
            status = (proc / "status").read_text()
            values = {line.split(":", 1)[0]: line.split(":", 1)[1] for line in status.splitlines()}
            if "VmRSS" not in values and not values.get("State", "").lstrip().startswith("Z"):
                raise ResourceFailure("worker RSS cannot be observed")
            rss += int(values.get("VmRSS", "0 kB").split()[0]) * 1024
            limits = (proc / "limits").read_text().splitlines()
            address = next(line for line in limits if line.startswith("Max address space"))
            if address.split()[3:5] != [str(MEMORY_BYTES), str(MEMORY_BYTES)]:
                raise ResourceFailure("worker descendant changed address-space limit")
            for task in (proc / "task").iterdir():
                if not _affinity(int(task.name)) <= cpus:
                    raise ResourceFailure("worker thread escaped CPU affinity")
        except (FileNotFoundError, ProcessLookupError):
            continue
    return rss, live


def _output_bytes(root: Path) -> int:
    total = 0
    entries: dict[tuple[int, int], list[Path]] = {}
    linked: set[tuple[int, int]] = set()
    if not root.exists():
        return total

    def walk_error(error: OSError) -> None:
        raise error

    for directory, dirs, files in os.walk(root, followlinks=False, onerror=walk_error):
        for name in [*dirs, *files]:
            path = Path(directory) / name
            try:
                metadata = path.lstat()
            except FileNotFoundError:
                if re.fullmatch(r"\..+\.[0-9a-f]{16}\.tmp", name):
                    continue  # publisher has just removed its own transient file
                raise
            if stat.S_ISLNK(metadata.st_mode):
                raise ResourceFailure("output contains a symlink")
            if stat.S_ISREG(metadata.st_mode):
                identity = metadata.st_dev, metadata.st_ino
                entries.setdefault(identity, []).append(path)
                if metadata.st_nlink != 1:
                    if metadata.st_nlink != 2:
                        raise ResourceFailure("output contains a hardlink")
                    linked.add(identity)
                total += metadata.st_size
            elif not stat.S_ISDIR(metadata.st_mode):
                raise ResourceFailure("output contains a non-regular object")
    for identity in linked:
        paths = entries[identity]
        if len(paths) == 2:
            first, second = paths
            pairs = ((first, second), (second, first))
            if any(
                temporary.parent == final.parent
                and re.fullmatch(
                    r"\." + re.escape(final.name) + r"\.[0-9a-f]{16}\.tmp", temporary.name
                )
                for temporary, final in pairs
            ):
                continue
        # A publication pair can disappear between lstat and visiting the other
        # entry. Recheck the final link count rather than accepting external links.
        surviving = []
        for path in paths:
            if path.exists():
                surviving.append(path.lstat())
            else:
                temporary = re.fullmatch(r"\.(.+)\.[0-9a-f]{16}\.tmp", path.name)
                if temporary and (final := path.with_name(temporary.group(1))).exists():
                    surviving.append(final.lstat())
        if surviving and all(
            (metadata.st_dev, metadata.st_ino) == identity and metadata.st_nlink == 1
            for metadata in surviving
        ):
            continue
        raise ResourceFailure("output contains an unrecognised hardlink")
    return total


def _phases(root: Path, launch: float, now: float) -> tuple[int, float | None]:
    paths = sorted((root / "operator").glob("resource-phase-*.json"))
    if len(paths) > 16:
        raise ResourceFailure("excess resource phase markers")
    previous = launch
    active: float | None = None
    for sequence, path in enumerate(paths):
        if path.name != f"resource-phase-{sequence:04d}.json":
            raise ResourceFailure("noncontiguous resource phase markers")
        marker: Any = json.loads(path.read_text())
        expected = "start" if sequence % 2 == 0 else "complete"
        if (
            not isinstance(marker, dict)
            or set(marker) != {"ordinal", "event", "monotonic_seconds"}
            or type(marker["ordinal"]) is not int
            or marker["ordinal"] != sequence // 2
            or marker["event"] != expected
            or type(marker["monotonic_seconds"]) not in (int, float)
        ):
            raise ResourceFailure("invalid resource phase marker")
        timestamp = float(marker["monotonic_seconds"])
        if not previous <= timestamp <= now:
            raise ResourceFailure("resource phase clock is not monotonic")
        if expected == "start":
            active = timestamp
        else:
            if active is None or timestamp - active > CELL_SECONDS:
                raise ResourceFailure("cell wall-time limit exceeded")
            active = None
        previous = timestamp
    if active is not None and now - active > CELL_SECONDS:
        raise ResourceFailure("cell wall-time limit exceeded")
    return len(paths), active


def _kill(pid: int, known: dict[int, int]) -> None:
    try:
        _linux_attribute(os, "killpg")(pid, _linux_attribute(signal, "SIGKILL"))
    except ProcessLookupError:
        pass
    for descendant, identity in known.items():
        try:
            if _identity(descendant) == identity:
                os.kill(descendant, _linux_attribute(signal, "SIGKILL"))
        except (FileNotFoundError, ProcessLookupError):
            pass


def _group_alive(pid: int) -> bool:
    try:
        _linux_attribute(os, "killpg")(pid, 0)
    except ProcessLookupError:
        return False
    return True


def supervise(argv: list[str], root: Path, outer_log: Path) -> int:
    """Run exactly one new session, fail closed, and preserve all failure evidence.

    The caller validates source, owner preflight and canonical output paths.
    Polling has finite detection latency; it is not a cgroup allocation guarantee.
    """
    require_linux_capabilities()
    if root.exists() or not argv or outer_log != root.parent / (root.name + ".supervisor.json"):
        raise ResourceFailure("invalid fresh resource-supervisor invocation")
    cpus = set(sorted(_affinity(0))[:2])
    environment = os.environ.copy()
    environment.update(THREAD_ENV)
    environment["EPS_TOPOLOGY_SUPERVISOR_PID"] = str(os.getpid())
    launch = time.monotonic()
    prior_phases: tuple[bytes, ...] = ()
    known: dict[int, int] = {}
    child: subprocess.Popen[bytes] | None = None
    stdout_path = root.parent / (root.name + ".worker-stdout.log")
    stderr_path = root.parent / (root.name + ".worker-stderr.log")
    with (
        outer_log.open("x", encoding="utf-8") as log,
        stdout_path.open("xb") as stdout,
        stderr_path.open("xb") as stderr,
    ):

        def record(value: dict[str, object]) -> None:
            log.write(json.dumps(value, allow_nan=False, sort_keys=True) + "\n")
            log.flush()
            os.fsync(log.fileno())

        record({"event": "launch", "cpus": sorted(cpus), "monotonic_seconds": launch})
        try:
            child = subprocess.Popen(
                argv,
                env=environment,
                start_new_session=True,
                preexec_fn=_preexec_limits,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
            )
            while True:
                now = time.monotonic()
                rss, live = _tree_usage(child.pid, known, cpus)
                known.update(live)
                output = _output_bytes(root) + sum(
                    path.stat().st_size for path in (outer_log, stdout_path, stderr_path)
                )
                phases, _ = _phases(root, launch, now)
                snapshots = tuple(
                    path.read_bytes()
                    for path in sorted((root / "operator").glob("resource-phase-*.json"))
                )
                if snapshots[: len(prior_phases)] != prior_phases:
                    raise ResourceFailure("resource phase history changed")
                prior_phases = snapshots
                if now - launch > TOTAL_SECONDS:
                    raise ResourceFailure("total wall-time limit exceeded")
                if rss > MEMORY_BYTES:
                    raise ResourceFailure("aggregate RSS limit exceeded")
                if output > OUTPUT_BYTES:
                    raise ResourceFailure("aggregate output limit exceeded")
                record(
                    {
                        "event": "sample",
                        "monotonic_seconds": now,
                        "rss_bytes": rss,
                        "output_bytes": output,
                        "phase_count": phases,
                        "pids": sorted(live),
                    }
                )
                result = child.poll()
                if result is not None:
                    if result != 0 or phases != 16 or _group_alive(child.pid):
                        raise ResourceFailure(
                            "worker failed, incomplete cells or surviving descendants"
                        )
                    os.fsync(stdout.fileno())
                    os.fsync(stderr.fileno())
                    record({"event": "complete", "returncode": result})
                    return result
                time.sleep(POLL_SECONDS)
        except BaseException as error:
            if child is not None:
                _kill(child.pid, known)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    record({"event": "termination_unconfirmed", "error": str(error)})
                    raise ResourceFailure("worker termination is unconfirmed") from error
            os.fsync(stdout.fileno())
            os.fsync(stderr.fileno())
            record(
                {"event": "permanent_stop", "error": str(error), "error_type": type(error).__name__}
            )
            raise
