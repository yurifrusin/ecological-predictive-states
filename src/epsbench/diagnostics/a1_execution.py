"""Fixed A1 launches: one archive-file bind, bounded external history witness."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, BinaryIO, cast

from epsbench.diagnostics.a1_action_contrast import cases
from epsbench.diagnostics.a1_lifecycle import digest
from epsbench.diagnostics.a1_retention import (
    CHUNK,
    MIB,
    Budget,
    RetainedArchive,
    _digest,
    _safe_file,
    recover,
)

LABEL = "eps.a1.execution-owner"
BUDGET = Budget(archive=767 * MIB)
RECEIPT_LIMIT = MIB
CONTROL_BOUND = 16 * 1024
ZERO_CHAIN = "0" * 64
DUMMY_TASKS = {
    "dummy-complete",
    "dummy-interrupted",
    "dummy-sink-failure",
    "dummy-timeout",
    "dummy-limits-overflow",
}


@dataclass(frozen=True)
class Binding:
    source_head: str
    source_tree: str
    configuration_root: str
    image: str
    purpose: str = "a1_native_study_v1"

    def __post_init__(self) -> None:
        if self.purpose not in {"a1_native_study_v1", "a1_dummy_qualification_v1"}:
            raise ValueError("typed native/dummy purpose required")
        if any(
            re.fullmatch("[0-9a-f]{40}", v) is None for v in (self.source_head, self.source_tree)
        ):
            raise ValueError("exact source head/tree required")
        if re.fullmatch("[0-9a-f]{64}", self.configuration_root) is None:
            raise ValueError("configuration root required")
        if re.fullmatch("sha256:[0-9a-f]{64}", self.image) is None:
            raise ValueError("immutable image required")

    @property
    def root(self) -> str:
        return digest(vars(self))


def source_binding(source: Path, image: str, purpose: str = "a1_native_study_v1") -> Binding:
    def git(*args: str) -> str:
        return (
            subprocess.check_output(["git", "-C", str(source), *args], timeout=10).decode().strip()
        )

    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("committed clean source required")
    return Binding(
        git("rev-parse", "HEAD"),
        git("rev-parse", "HEAD^{tree}"),
        digest([c.config.model_dump(mode="json") for c in cases(source)]),
        image,
        purpose,
    )


def history(path: Path) -> str:
    _safe_file(path)
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(CHUNK):
            result.update(data)
    return result.hexdigest()


@dataclass
class ReceiptState:
    checkpoint: str | None = None
    phase: str | None = None
    completed: tuple[str, ...] = ()
    tasks: tuple[str, ...] = ()
    task_index: int = 0
    attempt: str | None = None
    elapsed: float = 0.0
    failed: bool = False


def _elapsed(value: Any) -> float:
    try:
        valid = type(value) in {int, float} and math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("finite nonnegative elapsed required")
    return float(value)


def replay_receipts(records: list[dict[str, Any]], binding: Binding) -> ReceiptState:
    """Replay only this driver's finite operational vocabulary, including dummy attempts."""
    state = ReceiptState()
    previous = ZERO_CHAIN
    commands: list[dict[str, Any]] = []
    dummy_seen: set[str] = set()
    for index, record in enumerate(records):
        if (
            type(record.get("sequence")) is not int
            or record["sequence"] != index
            or record.get("previous_sha256") != previous
        ):
            raise ValueError("receipt chain sequence/parent differs")
        body = {k: v for k, v in record.items() if k != "sha256"}
        if record.get("sha256") != digest(body):
            raise ValueError("receipt checksum differs")
        previous = record["sha256"]
        value = {k: v for k, v in body.items() if k not in {"sequence", "previous_sha256"}}
        kind = value.get("kind")
        if state.failed and kind != "FAILURE":
            raise ValueError("terminal receipt history cannot continue")
        if index == 0:
            if value != {
                "kind": "BINDING",
                "version": "a1_host_receipts_v2",
                "binding": vars(binding),
                "budget": vars(BUDGET),
                "host_receipt_bytes": RECEIPT_LIMIT,
            }:
                raise ValueError("receipt binding/version differs")
        elif kind == "CHECKPOINT":
            if set(value) != {"kind", "task", "history"}:
                raise ValueError("invalid checkpoint fields")
            _digest(value["history"])
            if index == 1 and value["task"] == "initialize":
                state.checkpoint = value["history"]
            else:
                if (
                    state.attempt != value["task"]
                    or not commands
                    or any(c["exit"] != 0 or c["timeout"] or c["overflow"] for c in commands)
                ):
                    raise ValueError("task checkpoint lacks successful attempt")
                if binding.purpose == "a1_native_study_v1" and tuple(
                    c["args"][0] for c in commands
                ) != ("create", "inspect", "start", "inspect", "inspect", "rm"):
                    raise ValueError("native task command chronology differs")
                state.checkpoint = value["history"]
                state.attempt = None
                state.task_index += 1
                commands = []
        elif kind == "PHASE":
            fields = {
                "kind",
                "phase",
                "decision",
                "ordinals",
                "contexts",
                "render_read_pairs",
                "cumulative_seconds",
                "binding",
                "prior_history",
                "history",
                "receipt_history",
            }
            phase = value.get("phase")
            if (
                set(value) != fields
                or binding.purpose != "a1_native_study_v1"
                or phase not in {"development", "continuation"}
                or state.phase is not None
                or state.attempt is not None
                or state.checkpoint is None
                or value["prior_history"] != state.checkpoint
                or value["receipt_history"] != record["previous_sha256"]
                or value["binding"] != vars(binding)
                or not isinstance(value["decision"], str)
                or not value["decision"].strip()
                or state.completed != (() if phase == "development" else ("development",))
            ):
                raise ValueError("illegal phase chronology")
            ordinals = [0, 1] if phase == "development" else list(range(2, 8))
            if (
                value["ordinals"] != ordinals
                or type(value["contexts"]) is not int
                or value["contexts"] != len(ordinals)
                or type(value["render_read_pairs"]) is not int
                or value["render_read_pairs"] != 6 * len(ordinals)
                or type(value["cumulative_seconds"]) is not int
                or value["cumulative_seconds"] != 2700
            ):
                raise ValueError("fixed phase reservation differs")
            _digest(value["history"])
            state.phase = phase
            state.checkpoint = value["history"]
            state.tasks = (
                *(f"cell-{i:02d}" for i in ordinals),
                "inspect-development" if phase == "development" else "evaluate",
            )
            state.task_index = 0
        elif kind == "ATTEMPT":
            if (
                set(value) != {"kind", "task", "name", "token", "expected_history", "seconds"}
                or state.attempt is not None
                or state.checkpoint is None
                or value["expected_history"] != state.checkpoint
                or type(value["seconds"]) is not int
                or value["seconds"] != 300
                or not isinstance(value["token"], str)
                or re.fullmatch("[0-9a-f]{32}", value["token"]) is None
                or value["name"] != "eps-a1-" + value["token"]
            ):
                raise ValueError("illegal attempt chronology")
            task = value["task"]
            if binding.purpose == "a1_native_study_v1":
                if (
                    state.phase is None
                    or state.task_index >= len(state.tasks)
                    or task != state.tasks[state.task_index]
                ):
                    raise ValueError("fixed native attempt order differs")
            elif task not in DUMMY_TASKS or task in dummy_seen:
                raise ValueError("unknown/repeated dummy attempt")
            dummy_seen.add(task)
            state.attempt = task
            commands = []
        elif kind == "COMMAND":
            if (
                set(value) != {"kind", "args", "exit", "timeout", "overflow", "stdout", "stderr"}
                or state.attempt is None
                or not isinstance(value["args"], list)
                or not value["args"]
                or value["args"][0] not in {"create", "inspect", "start", "rm"}
                or any(not isinstance(a, str) or len(a) > 8192 for a in value["args"])
                or type(value["exit"]) is not int
                or type(value["timeout"]) is not bool
                or type(value["overflow"]) is not bool
                or any(
                    not isinstance(value[k], str) or len(value[k]) > CONTROL_BOUND
                    for k in ("stdout", "stderr")
                )
            ):
                raise ValueError("illegal command receipt")
            commands.append(value)
        elif kind == "COMPLETE":
            if (
                set(value) != {"kind", "phase", "elapsed"}
                or state.phase is None
                or state.phase != value["phase"]
                or state.attempt is not None
                or state.task_index != len(state.tasks)
            ):
                raise ValueError("premature or unknown phase completion")
            state.elapsed += _elapsed(value["elapsed"])
            if state.elapsed > 2700:
                raise ValueError("cumulative active deadline exceeded")
            state.completed += (state.phase,)
            state.phase = None
        elif kind == "FAILURE":
            if set(value) == {"kind", "task", "error", "observed_history"}:
                if value["task"] != state.attempt:
                    raise ValueError("failure task differs")
            elif set(value) == {"kind", "phase", "error", "elapsed", "observed_history"}:
                next_phase = "development" if not state.completed else "continuation"
                if value["phase"] != (state.phase or next_phase):
                    raise ValueError("failure phase differs")
                _elapsed(value["elapsed"])
            else:
                raise ValueError("invalid terminal receipt")
            if not isinstance(value["error"], str) or not value["error"]:
                raise ValueError("failure reason required")
            _digest(value["observed_history"])
            state.failed = True
        else:
            raise ValueError("unknown or misplaced receipt kind")
    return state


class HostReceipts:
    """Fixed 1 MiB append-only witness; missing/torn history denies resume."""

    def __init__(self, path: Path, binding: Binding, *, initial: bool) -> None:
        self.path = path
        self.binding = binding
        self.poisoned = False
        self.records: list[dict[str, Any]] = []
        if initial:
            if path.exists() or path.is_symlink():
                raise ValueError("existing receipt anchor denies initialization")
            for parent in path.parents:
                if parent.is_symlink():
                    raise ValueError("linked receipt anchor denied")
            self.stream: BinaryIO = path.open("xb+")
            self._lock()
            self.append(
                {
                    "kind": "BINDING",
                    "version": "a1_host_receipts_v2",
                    "binding": vars(binding),
                    "budget": vars(BUDGET),
                    "host_receipt_bytes": RECEIPT_LIMIT,
                }
            )
        else:
            _safe_file(path)
            if path.stat().st_size > RECEIPT_LIMIT:
                raise ValueError("host receipt capacity exceeded")
            self.stream = path.open("r+b")
            self._lock()
            self.stream.seek(0)
            try:
                for line in self.stream:
                    if not line.endswith(b"\n"):
                        raise ValueError("torn witness denies continuation")
                    record = json.loads(line)
                    if (
                        not isinstance(record, dict)
                        or json.dumps(
                            record, sort_keys=True, separators=(",", ":"), allow_nan=False
                        ).encode()
                        + b"\n"
                        != line
                    ):
                        raise ValueError("noncanonical witness record")
                    self.records.append(record)
                if not self.records:
                    raise ValueError("missing receipt history")
                replay_receipts(self.records, binding)
            except BaseException:
                self.stream.close()
                raise
            self.stream.seek(0, os.SEEK_END)

    def _lock(self) -> None:
        """One cooperative host writer; no parallel scientific archive launches."""
        try:
            self.stream.seek(0)
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.stream.close()
            raise

    def append(self, value: dict[str, Any]) -> None:
        if self.poisoned:
            raise ValueError("failed witness cannot acknowledge")
        if set(value) & {"sequence", "previous_sha256", "sha256"}:
            raise ValueError("reserved receipt chain fields")
        body = {"sequence": len(self.records), "previous_sha256": self.root, **value}
        record = {**body, "sha256": digest(body)}
        payload = (
            json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            + b"\n"
        )
        record = json.loads(payload)
        replay_receipts([*self.records, record], self.binding)
        reserve = 0 if value.get("kind") == "FAILURE" else 64 * 1024
        if self.stream.tell() + len(payload) > RECEIPT_LIMIT - reserve:
            raise ValueError("host receipt admission denied")
        try:
            if self.stream.write(payload) != len(payload):
                raise OSError("short witness write")
            self.stream.flush()
            os.fsync(self.stream.fileno())
        except BaseException:
            self.poisoned = True
            raise
        self.records.append(record)

    @property
    def root(self) -> str:
        return str(self.records[-1]["sha256"]) if self.records else ZERO_CHAIN

    def close(self) -> None:
        self.stream.close()


def create_arguments(
    name: str, token: str, binding: Binding, archive: Path, task: str, expected: str
) -> list[str]:
    _safe_file(archive)
    return [
        "create",
        "--pull=never",
        "--name",
        name,
        "--label",
        LABEL + "=" + token,
        "--read-only",
        "--network=none",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--log-driver=none",
        "--cpuset-cpus=0,1",
        "--cpus=2",
        "--pids-limit=64",
        "--memory=8g",
        "--memory-swap=8g",
        "--shm-size=1m",
        "--tmpfs=/output:rw,noexec,nosuid,nodev,size=255m",
        "--mount",
        f"type=bind,src={archive.resolve()},dst=/retained/history",
        "--env=EPS_A1_RUNTIME=docker_candidate_v1",
        "--env=MUJOCO_GL=osmesa",
        "--env=PYOPENGL_PLATFORM=osmesa",
        "--env=LP_NUM_THREADS=2",
        "--env=OMP_NUM_THREADS=2",
        "--env=PYTHONDONTWRITEBYTECODE=1",
        "--env=TMPDIR=/output",
        binding.image,
        "python",
        "scripts/a1_docker_entry.py",
        "--task",
        task,
        "--expected-history",
        expected,
        "--binding",
        json.dumps(vars(binding), separators=(",", ":")),
    ]


def validate_archive_mount(info: dict[str, Any], archive: Path) -> None:
    """Submitted source is exact; only its deterministic Desktop translation is allowed."""
    expected = str(archive.resolve())
    definitions = info["HostConfig"].get("Mounts", [])
    mounts = info["Mounts"]
    if (
        len(definitions) != 1
        or definitions[0].get("Type") != "bind"
        or definitions[0].get("Source") != expected
        or definitions[0].get("Target") != "/retained/history"
        or definitions[0].get("ReadOnly", False) is not False
        or len(mounts) != 1
    ):
        raise ValueError("archive mount definition differs")
    accepted = {expected}
    windows = PureWindowsPath(expected)
    if re.fullmatch("[A-Za-z]:", windows.drive):
        suffix = windows.drive[0].lower() + "/" + "/".join(windows.parts[1:])
        accepted.update({"/run/desktop/mnt/host/" + suffix, "/host_mnt/" + suffix})
    actual = mounts[0]
    if (
        actual.get("Type") != "bind"
        or actual.get("Source") not in accepted
        or actual.get("Destination") != "/retained/history"
        or actual.get("RW") is not True
    ):
        raise ValueError("archive mount resolved source differs")


class DockerController:
    def __init__(
        self, docker: str, archive: Path, binding: Binding, receipts: HostReceipts
    ) -> None:
        self.docker, self.archive, self.binding, self.receipts = docker, archive, binding, receipts

    def command(
        self, args: list[str], deadline: float, *, timeout: float = 15, cleanup: bool = False
    ) -> str:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("active deadline exhausted")
        proc = subprocess.Popen(
            [self.docker, *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        buffers = [bytearray(), bytearray()]
        overflow = threading.Event()

        def read(pipe: BinaryIO, result: bytearray) -> None:
            while data := pipe.read(4096):
                room = CONTROL_BOUND - len(result)
                result.extend(data[: max(0, room)])
                if len(data) > room:
                    overflow.set()
                    proc.kill()
                    return

        assert proc.stdout is not None and proc.stderr is not None
        readers = [
            threading.Thread(target=read, args=(pipe, result), daemon=True)
            for pipe, result in zip((proc.stdout, proc.stderr), buffers, strict=True)
        ]
        for reader in readers:
            reader.start()
        timed_out = False
        try:
            proc.wait(timeout=min(timeout, remaining))
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            proc.wait(timeout=2)
        finally:
            for reader in readers:
                reader.join(timeout=1)
            try:
                self.receipts.append(
                    {
                        "kind": "COMMAND",
                        "args": args,
                        "exit": proc.returncode,
                        "timeout": timed_out,
                        "overflow": overflow.is_set(),
                        "stdout": buffers[0].decode(errors="replace"),
                        "stderr": buffers[1].decode(errors="replace"),
                    }
                )
            except (OSError, ValueError):
                if not cleanup:
                    raise
        if timed_out or overflow.is_set() or any(r.is_alive() for r in readers) or proc.returncode:
            raise RuntimeError("Docker command unsuccessful or uncertain")
        return buffers[0].decode().strip()

    def owned(
        self, identity: str, name: str, token: str, deadline: float, *, cleanup: bool = False
    ) -> dict[str, Any]:
        record = json.loads(
            self.command(["inspect", "--format", "{{json .}}", identity], deadline, cleanup=cleanup)
        )
        if (
            re.fullmatch("[0-9a-f]{64}", record["Id"]) is None
            or record["Name"] != "/" + name
            or record["Config"]["Labels"].get(LABEL) != token
        ):
            raise ValueError("ownership mismatch; cleanup denied")
        return cast(dict[str, Any], record)

    def launch(self, task: str, expected: str, deadline: float) -> str:
        if task.startswith("dummy-") != (self.binding.purpose == "a1_dummy_qualification_v1"):
            raise ValueError("dummy/native identity domains differ")
        token = uuid.uuid4().hex
        name = "eps-a1-" + token
        identity = name
        self.receipts.append(
            {
                "kind": "ATTEMPT",
                "task": task,
                "name": name,
                "token": token,
                "expected_history": expected,
                "seconds": 300,
            }
        )
        failure: BaseException | None = None
        try:
            identity = self.command(
                create_arguments(name, token, self.binding, self.archive, task, expected), deadline
            )
            info = self.owned(identity, name, token, deadline)
            validate_archive_mount(info, self.archive)
            hc, mounts = info["HostConfig"], info["Mounts"]
            if (
                info["Image"] != self.binding.image
                or len(mounts) != 1
                or mounts[0]["Type"] != "bind"
                or mounts[0]["Destination"] != "/retained/history"
                or not mounts[0]["RW"]
                or hc["Binds"]
                or hc["Memory"] != 8 * 1024**3
                or hc["MemorySwap"] != hc["Memory"]
                or hc["NanoCpus"] != 2_000_000_000
                or hc["CpusetCpus"] != "0,1"
                or hc["PidsLimit"] != 64
                or not hc["ReadonlyRootfs"]
                or hc["NetworkMode"] != "none"
                or hc["LogConfig"]["Type"] != "none"
                or hc["CapDrop"] != ["ALL"]
                or hc["Privileged"]
                or "no-new-privileges" not in hc["SecurityOpt"]
                or hc["ShmSize"] != MIB
                or hc["Tmpfs"] != {"/output": "rw,noexec,nosuid,nodev,size=255m"}
                or hc["RestartPolicy"]["Name"] != "no"
            ):
                raise ValueError("pre-start confinement differs")
            self.command(
                ["start", "--attach", identity], min(deadline, time.monotonic() + 300), timeout=300
            )
            state = self.owned(identity, name, token, deadline)["State"]
            if state["Running"] or state["ExitCode"] != 0 or state["OOMKilled"]:
                raise ValueError("completion unsuccessful")
        except BaseException as error:
            failure = error
        finally:
            cleanup = time.monotonic() + 20
            try:
                owned = self.owned(identity, name, token, cleanup, cleanup=True)
                self.command(["rm", "--force", owned["Id"]], cleanup, cleanup=True)
            except BaseException as error:
                failure = error
        observed = history(self.archive)
        if failure is not None:
            self.receipts.append(
                {
                    "kind": "FAILURE",
                    "task": task,
                    "error": type(failure).__name__,
                    "observed_history": observed,
                }
            )
            raise RuntimeError("stopped attempt; no retry") from failure
        state = recover(self.archive, self.binding.root, BUDGET, observed)
        if state.incomplete or state.suffix_bytes:
            raise ValueError("incomplete archive denies continuation")
        names = {p for p, _, _ in state.committed}
        required = f"operations/{task}-complete.json"
        if required not in names:
            raise ValueError("task completion receipt missing")
        self.receipts.append({"kind": "CHECKPOINT", "task": task, "history": observed})
        return observed

    def phase(self, phase: str, decision: str, expected: str, expected_receipt_history: str) -> str:
        if self.binding.purpose != "a1_native_study_v1":
            raise ValueError("dummy history cannot become a native study")
        records = self.receipts.records
        replay_receipts(records, self.binding)
        _digest(expected_receipt_history)
        if expected_receipt_history != self.receipts.root:
            raise ValueError("stale receipt-bound authorization")
        if phase not in {"development", "continuation"} or not decision.strip():
            raise ValueError("explicit phase decision required")
        if any(r["kind"] == "FAILURE" for r in records):
            raise ValueError("failed attempt denies continuation")
        if any(r["kind"] == "PHASE" and r["phase"] == phase for r in records):
            raise ValueError("phase already attempted; no retries")
        checkpoints = [r["history"] for r in records if r["kind"] == "CHECKPOINT"]
        if not checkpoints or expected != checkpoints[-1] or history(self.archive) != expected:
            raise ValueError("external witnessed history mismatch")
        if phase == "continuation" and not any(
            r["kind"] == "COMPLETE" and r["phase"] == "development" for r in records
        ):
            raise ValueError("first two accepted captures required")
        used = sum(float(r["elapsed"]) for r in records if r["kind"] == "COMPLETE")
        started = time.monotonic()
        deadline = started + max(0, 2700 - used)
        ordinals = (0, 1) if phase == "development" else tuple(range(2, 8))
        try:
            # Durable consumption precedes both the host PHASE receipt and any container.
            archive = RetainedArchive(
                self.archive, self.binding.root, BUDGET, expected_history=expected
            )
            try:
                archive.put(
                    f"operations/phase-{phase}-consumed.json",
                    json.dumps(
                        {
                            "phase": phase,
                            "binding": vars(self.binding),
                            "prior_history": expected,
                            "receipt_history": expected_receipt_history,
                            "decision": decision,
                        },
                        sort_keys=True,
                    ).encode(),
                )
                prior_history, expected = expected, archive.history()
            finally:
                archive.close()
            self.receipts.append(
                {
                    "kind": "PHASE",
                    "phase": phase,
                    "decision": decision,
                    "ordinals": ordinals,
                    "contexts": len(ordinals),
                    "render_read_pairs": 6 * len(ordinals),
                    "cumulative_seconds": 2700,
                    "binding": vars(self.binding),
                    "prior_history": prior_history,
                    "history": expected,
                    "receipt_history": expected_receipt_history,
                }
            )
            for ordinal in ordinals:
                expected = self.launch(f"cell-{ordinal:02d}", expected, deadline)
            expected = self.launch(
                "inspect-development" if phase == "development" else "evaluate", expected, deadline
            )
            self.receipts.append(
                {"kind": "COMPLETE", "phase": phase, "elapsed": time.monotonic() - started}
            )
        except BaseException as error:
            self.receipts.append(
                {
                    "kind": "FAILURE",
                    "phase": phase,
                    "error": type(error).__name__,
                    "elapsed": time.monotonic() - started,
                    "observed_history": history(self.archive),
                }
            )
            raise
        return expected


def initialize(path: Path, binding: Binding, receipts: HostReceipts) -> str:
    archive = RetainedArchive(path, binding.root, BUDGET)
    try:
        archive.put("operations/binding.json", json.dumps(vars(binding), sort_keys=True).encode())
        expected = archive.history()
        receipts.append({"kind": "CHECKPOINT", "task": "initialize", "history": expected})
        return expected
    finally:
        archive.close()
