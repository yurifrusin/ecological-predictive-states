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
import threading
import time
from collections.abc import Callable
from contextlib import ExitStack
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
    if (
        threading.active_count() != 1
        or signal.getsignal(_linux_attribute(signal, "SIGCHLD")) != signal.SIG_DFL
    ):
        raise ResourceFailure("supervisor requires sole child-reaping authority")
    for module, name in ((os, "waitid"), (os, "pidfd_open"), (signal, "pidfd_send_signal")):
        if not callable(_linux_attribute(module, name)):
            raise ResourceFailure("identity-bound process lifecycle is unavailable")
    own_handle = _linux_attribute(os, "pidfd_open")(os.getpid(), 0)
    os.close(own_handle)
    for name in (
        "P_PID",
        "WEXITED",
        "WNOHANG",
        "WNOWAIT",
        "CLD_EXITED",
        "CLD_KILLED",
        "CLD_DUMPED",
    ):
        value = _linux_attribute(os, name)
        if type(value) is not int or value <= 0:
            raise ResourceFailure("required Linux waitid constants are unavailable")

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
    """Discover scoped children without relabelling any previously bound PID."""
    if pid not in known:
        raise ResourceFailure("worker root identity was not bound")
    if _identity(pid) != known[pid]:
        raise ResourceFailure("worker root identity changed")
    pending: list[int] = []
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
            identity = _identity(current)
            if current in known and known[current] != identity:
                continue
            live[current] = identity
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
    for current, identity in live.items():
        if known.setdefault(current, identity) != identity:
            raise ResourceFailure("previously bound process identity changed")
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


def _wait_status(pid: int) -> int | None:
    """Observe exit without reaping: the original leader still reserves its PID."""
    flags = (
        _linux_attribute(os, "WEXITED")
        | _linux_attribute(os, "WNOHANG")
        | _linux_attribute(os, "WNOWAIT")
    )
    try:
        result = _linux_attribute(os, "waitid")(_linux_attribute(os, "P_PID"), pid, flags)
    except ChildProcessError as error:
        raise ResourceFailure("worker leader ownership lost; termination uncertain") from error
    if result is None:
        return None
    if result.si_pid != pid or type(result.si_status) is not int:
        raise ResourceFailure("unexpected child wait observation")
    if result.si_code == _linux_attribute(os, "CLD_EXITED"):
        return result.si_status
    if result.si_code in (_linux_attribute(os, "CLD_KILLED"), _linux_attribute(os, "CLD_DUMPED")):
        return -result.si_status
    raise ResourceFailure("unexpected child wait event")


def _signal_bound_process(pid: int, identity: int) -> None:
    """Signal a bound task through a pidfd, never through a reusable PID number."""
    try:
        descriptor = _linux_attribute(os, "pidfd_open")(pid, 0)
    except ProcessLookupError:
        return
    try:
        if _identity(pid) == identity:
            _linux_attribute(signal, "pidfd_send_signal")(
                descriptor, _linux_attribute(signal, "SIGKILL"), None, 0
            )
    except (FileNotFoundError, ProcessLookupError):
        pass
    finally:
        os.close(descriptor)


def _assert_leader_anchor(pid: int, known: dict[int, int]) -> None:
    # No poll/wait occurs before this non-reaping observation. The synchronous,
    # single-threaded supervisor with default SIGCHLD alone controls reaping.
    _wait_status(pid)
    if pid not in known or _identity(pid) != known[pid]:
        raise ResourceFailure("worker leader identity unconfirmed; termination uncertain")
    if _linux_attribute(os, "getpgid")(pid) != pid or _linux_attribute(os, "getsid")(pid) != pid:
        raise ResourceFailure("worker session/group anchor changed; termination uncertain")


def _kill(pid: int, known: dict[int, int]) -> None:
    _assert_leader_anchor(pid, known)
    try:
        _linux_attribute(os, "killpg")(pid, _linux_attribute(signal, "SIGKILL"))
    except ProcessLookupError:
        pass
    for descendant, identity in known.items():
        if descendant != pid:
            _signal_bound_process(descendant, identity)


def _anchored_group_members(pid: int, known: dict[int, int]) -> dict[int, int]:
    """Discover the anchored private session/group, including reparented members.

    Only stat metadata is read while filtering proc entries; identities/status
    retained for assessment must belong to the still-owned session and group.
    """
    _assert_leader_anchor(pid, known)
    members: dict[int, int] = {}
    for path in _PROC_ROOT.iterdir():
        if not path.name.isdecimal():
            continue
        try:
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        except FileNotFoundError:
            continue
        if int(fields[2]) != pid or int(fields[3]) != pid:
            continue
        current = int(path.name)
        identity = int(fields[19])
        if known.setdefault(current, identity) != identity:
            raise ResourceFailure("anchored group member PID was reused")
        members[current] = identity
    return members


def _observed_survivors(pid: int, known: dict[int, int]) -> list[int]:
    survivors = []
    members = _owned_processes(pid, known)
    members.update(_anchored_group_members(pid, known))
    for descendant in members:
        if descendant == pid:
            continue
        try:
            status = (_PROC_ROOT / str(descendant) / "status").read_text()
            state = next(
                line.split(":", 1)[1] for line in status.splitlines() if line.startswith("State:")
            )
            if not state.lstrip().startswith("Z"):
                survivors.append(descendant)
        except FileNotFoundError:
            continue
    return survivors


def _confirm_cleanup(pid: int, known: dict[int, int]) -> None:
    """Bound confirmation to observed tasks while the original leader is unreaped."""
    deadline = time.monotonic() + 5
    while _wait_status(pid) is None or _observed_survivors(pid, known):
        if time.monotonic() >= deadline:
            raise ResourceFailure("observed worker termination remains unconfirmed")
        time.sleep(POLL_SECONDS)


def _flush_parent_directory(parent: Path) -> None:
    """Durably retain external evidence names before a worker may start."""
    if sys.platform != "linux":
        raise ResourceFailure("parent-directory durability requires Linux/WSL")
    directory = _linux_attribute(os, "O_DIRECTORY")
    nofollow = _linux_attribute(os, "O_NOFOLLOW")
    if type(directory) is not int or type(nofollow) is not int or directory <= 0 or nofollow <= 0:
        raise ResourceFailure("required Linux directory-open flags are unavailable")
    descriptor = os.open(parent, os.O_RDONLY | directory | nofollow)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def supervise(argv: list[str], root: Path, outer_log: Path) -> int:
    """Run exactly one new session, fail closed, and preserve all failure evidence.

    The caller validates source, owner preflight and canonical output paths.
    Polling has finite detection latency; it is not a cgroup allocation guarantee.
    Discovery covers the anchored private session/group and observed escaped
    descendants. No subreaper or exhaustive escaped-orphan claim is made.
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
    reaped = False
    stdout_path = root.parent / (root.name + ".worker-stdout.log")
    stderr_path = root.parent / (root.name + ".worker-stderr.log")
    with outer_log.open("x", encoding="utf-8") as log, ExitStack() as streams:
        stdout = None
        stderr = None

        def record(value: dict[str, object]) -> None:
            log.write(json.dumps(value, allow_nan=False, sort_keys=True) + "\n")
            log.flush()
            os.fsync(log.fileno())

        try:
            stdout = streams.enter_context(stdout_path.open("xb"))
            stderr = streams.enter_context(stderr_path.open("xb"))
            record({"event": "launch", "cpus": sorted(cpus), "monotonic_seconds": launch})
            os.fsync(stdout.fileno())
            os.fsync(stderr.fileno())
            _flush_parent_directory(root.parent)
            child = subprocess.Popen(
                argv,
                env=environment,
                start_new_session=True,
                preexec_fn=_preexec_limits,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
            )
            known[child.pid] = _identity(child.pid)
            while True:
                now = time.monotonic()
                rss, live = _tree_usage(child.pid, known, cpus)
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
                result = _wait_status(child.pid)
                if result is not None:
                    survivors = _observed_survivors(child.pid, known)
                    if result != 0 or phases != 16 or survivors:
                        raise ResourceFailure(
                            "worker failed, incomplete cells or surviving descendants"
                        )
                    _kill(child.pid, known)
                    _confirm_cleanup(child.pid, known)
                    actual_result = child.wait(timeout=5)
                    reaped = True
                    if actual_result != result:
                        raise ResourceFailure("worker reap status differs from exit observation")
                    os.fsync(stdout.fileno())
                    os.fsync(stderr.fileno())
                    record(
                        {
                            "event": "complete",
                            "returncode": result,
                            "descendant_coverage": "anchored_group_and_observed_descendants",
                            "group_cleanup": "unreaped_leader_anchored",
                        }
                    )
                    return result
                time.sleep(POLL_SECONDS)
        except BaseException as error:
            if child is not None and not reaped:
                try:
                    _kill(child.pid, known)
                    _confirm_cleanup(child.pid, known)
                    child.wait(timeout=5)
                    reaped = True
                except BaseException as cleanup_error:
                    record(
                        {
                            "event": "permanent_stop",
                            "error": str(error),
                            "error_type": type(error).__name__,
                            "termination": "unconfirmed",
                            "cleanup_error": str(cleanup_error),
                        }
                    )
                    raise ResourceFailure("worker termination is unconfirmed") from error
            record(
                {"event": "permanent_stop", "error": str(error), "error_type": type(error).__name__}
            )
            if stdout is not None:
                os.fsync(stdout.fileno())
            if stderr is not None:
                os.fsync(stderr.fileno())
            raise
