"""Specific nine-slot causal controller, independent of the exposed A1 lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, BinaryIO

from epsbench.diagnostics.a1_retention import (
    CHUNK,
    MAX_RECORD,
    MIB,
    Budget,
    RetainedArchive,
    _identity,
    _safe_file,
    recover,
)
from epsbench.diagnostics.causal_history_lifecycle import put_once
from epsbench.diagnostics.causal_history_sequence import (
    CONFIG_SHA256,
    MAX_ARTIFACT_BYTES,
    MAX_SEQUENCE_BYTES,
    MEMBERS,
    Progress,
    RetainedFlow,
    RetainedFrame,
    SequenceEvidence,
    candidate,
    canonical,
    digest,
    encode_sequence,
    parse,
)

BUDGET = Budget(archive=767 * MIB)
TASKS = ("compile-only", *("capture-" + m for m in MEMBERS), "seal", "evaluate-inspect")
DUMMY_TASKS = (
    "dummy-complete",
    "dummy-interrupted",
    "dummy-sink-failure",
    "dummy-timeout",
    "dummy-overflow",
)
SLOT_SECONDS = 300
WORK_SECONDS = 260
HOST_WORK_SECONDS = 275
CLEANUP_SECONDS = 20
DUMMY_SECONDS = 8
MAX_FILES = 2048
MAX_MATERIALIZED = 240 * MIB
MAX_CONTROL_BYTES = 32 * MIB
MAX_PARTIAL_BYTES = 64 * MIB
HOST_LIMIT = MIB
CONTROL_BOUND = 8192
LABEL = "eps.causal-history.execution-owner"
ZERO = "0" * 64


def maximum_bytes() -> dict[str, int]:
    """Bound every accepted write, not an estimate of typical native artifacts.

    The full prewrite archive cap includes arbitrary accepted CHUNK/STAGE framing,
    base64 and terminal records; no transfer-record count is assumed. Control and
    journal charges share cumulative materialization, including empty copy charges.
    """
    staging = 144 * MIB + MAX_CONTROL_BYTES + MAX_PARTIAL_BYTES
    archive = BUDGET.archive
    total = staging + archive + BUDGET.shared + HOST_LIMIT
    if staging > BUDGET.staging or archive > BUDGET.archive or total > 1024 * MIB:
        raise ValueError("complete output proof fails")
    return {
        "staging": staging,
        "archive_including_terminal": archive,
        "shared": BUDGET.shared,
        "host": HOST_LIMIT,
        "total": total,
        "tasks": 9,
        "reserved_seconds": 9 * SLOT_SECONDS,
    }


@dataclass(frozen=True)
class Binding:
    source_head: str
    source_tree: str
    configuration_root: str
    image: str
    purpose: str = "causal_history_native_v1"
    dummy_task: str | None = None

    def __post_init__(self) -> None:
        if (
            self.purpose not in {"causal_history_native_v1", "causal_history_dummy_v1"}
            or any(
                re.fullmatch(r"[0-9a-f]{40}", v) is None
                for v in (self.source_head, self.source_tree)
            )
            or self.configuration_root != CONFIG_SHA256
            or re.fullmatch(r"sha256:[0-9a-f]{64}", self.image) is None
        ):
            raise ValueError("strict source/tree/config/image/purpose binding")
        if (self.purpose == "causal_history_native_v1" and self.dummy_task is not None) or (
            self.purpose == "causal_history_dummy_v1" and self.dummy_task not in DUMMY_TASKS
        ):
            raise ValueError("dummy qualification task must be separately bound")

    @property
    def tasks(self) -> tuple[str, ...]:
        return TASKS if self.dummy_task is None else (self.dummy_task,)

    @property
    def root(self) -> str:
        return digest(canonical(vars(self)))


def source_binding(
    source: Path,
    image: str,
    purpose: str = "causal_history_native_v1",
    dummy_task: str | None = None,
) -> Binding:
    def git(*args: str) -> str:
        value = subprocess.check_output(["git", "-C", str(source), *args], timeout=10)
        if len(value) > CONTROL_BOUND:
            raise ValueError("source observation bound")
        return value.decode().strip()

    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("exact committed clean source required")
    data = (source / "configs/development/causal_history_fixture_design_v1.json").read_bytes()
    config, _, _ = candidate(data, next(iter(MEMBERS)))
    return Binding(
        git("rev-parse", "HEAD"),
        git("rev-parse", "HEAD^{tree}"),
        digest(canonical(config)),
        image,
        purpose,
        dummy_task,
    )


def history(path: Path) -> str:
    _safe_file(path)
    if path.stat().st_size > BUDGET.archive:
        raise ValueError("bounded archive history")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(CHUNK):
            h.update(data)
    return h.hexdigest()


def verify_prefix(path: Path, length: int, expected: str) -> None:
    """Completion must extend the exact consumed prefix, never substitute a history."""
    h = hashlib.sha256()
    remaining = length
    with path.open("rb") as stream:
        while remaining:
            data = stream.read(min(CHUNK, remaining))
            if not data:
                raise ValueError("archive rolled back")
            h.update(data)
            remaining -= len(data)
    if h.hexdigest() != expected:
        raise ValueError("archive consumed prefix changed")


class FiniteArchive(RetainedArchive):
    """Apply fixed causal count/category bounds without changing archive contracts."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.categories: dict[str, int] = {}
        # Bounded streaming replay; no materialized full archive/dataset/journal.
        with self.path.open("rb") as stream:
            active = ""
            while header := stream.read(4):
                size = struct.unpack("!I", header)[0]
                if not 0 < size <= MAX_RECORD:
                    raise ValueError("bounded accounting record")
                record = json.loads(stream.read(size))
                if record["kind"] == "START":
                    active = record["path"]
                    self.category(active)
                elif record["kind"] == "STAGE":
                    key = self.category(active)
                    self.categories[key] = self.categories.get(key, 0) + record["size"]
                elif record["kind"] == "FAILURE":
                    self.close()
                    raise ValueError("terminal failed archive cannot resume")
        if self.staging_reserved > MAX_MATERIALIZED or len(self.used) > MAX_FILES:
            raise ValueError("causal count/materialization exceeded")
        for key, amount in self.categories.items():
            limit = (
                MAX_PARTIAL_BYTES
                if key == "partial"
                else MAX_CONTROL_BYTES
                if key == "control"
                else MAX_SEQUENCE_BYTES
            )
            if amount > limit:
                self.close()
                raise ValueError("resumed category capacity exceeded")

    @staticmethod
    def category(path: str) -> str:
        if path.startswith("sequences/"):
            member = path.split("/")[1]
            if member not in MEMBERS:
                raise ValueError("fixed sequence path required")
            return member
        if path.startswith("partial/"):
            return "partial"
        if path.startswith(
            (
                "operations/",
                "control/",
                "forecasts/",
                "results/",
                "charges/",
                "inspection/",
                "dummy/",
            )
        ):
            return "control"
        raise ValueError("unknown causal output namespace")

    def start(self, path: str) -> None:
        self.category(path)
        if len(self.used) >= MAX_FILES:
            raise ValueError("fixed artifact count bound")
        super().start(path)

    def reserve_staging(self, size: int) -> None:
        if self.pending is None:
            raise ValueError("pending reservation required")
        key = self.category(self.pending)
        limit = (
            MAX_PARTIAL_BYTES
            if key == "partial"
            else MAX_CONTROL_BYTES
            if key == "control"
            else MAX_SEQUENCE_BYTES
        )
        if (
            size > MAX_ARTIFACT_BYTES
            or self.artifact_reserved + size > MAX_ARTIFACT_BYTES
            or self.categories.get(key, 0) + size > limit
            or self.staging_reserved + size > MAX_MATERIALIZED
        ):
            raise ValueError("finite category/artifact/materialization bound")
        super().reserve_staging(size)
        self.categories[key] = self.categories.get(key, 0) + size

    def failure(self, reason: str) -> None:
        super().failure(reason)
        self.poisoned = True  # No later read/seal/continuation, even after committed output.


class CaptureRetention:
    """Shared prefix encoder emits final names once and preserves early native reads."""

    def __init__(
        self,
        sink: FiniteArchive,
        payload: bytes,
        member: str,
        binding: Binding,
        renderer_identity: dict[str, Any],
    ):
        self.sink, self.payload, self.member, self.binding = sink, payload, member, binding
        self.renderer_identity = renderer_identity
        self.sequence: SequenceEvidence | None = None
        self.frames: list[RetainedFrame] = []
        self.flows: list[RetainedFlow] = []
        self.events = 0
        self.failed = False

    def __call__(self, event: Progress) -> None:
        if self.failed:
            raise ValueError("failed callback is terminal")
        try:
            if event.stage not in {
                "sequence",
                "frame",
                "flow",
                "failure",
                "capture_attempt",
                "rgb_attempt",
                "rgb_read",
                "paired_draw_attempt",
                "paired_draw_complete",
                "paired_read_attempt",
                "paired_read_complete",
                "paired_draw_input",
                "paired_draw_output",
                "paired_read_input",
                "paired_read_output",
                "frame_operations",
            }:
                raise ValueError("unknown finite progress stage")
            if self.events >= 128 or type(event.index) is not int or not 0 <= event.index <= 4:
                raise ValueError("bounded progress events")
            if event.index >= MEMBERS[self.member][3]:
                raise ValueError("progress index outside fixed captured poses")
            if event.stage == "sequence":
                self.sequence = event.value
            elif event.stage == "frame":
                if event.index != len(self.frames):
                    raise ValueError("ordered validated frames")
                self.frames.append(event.value)
            elif event.stage == "flow":
                if event.index != len(self.flows):
                    raise ValueError("ordered validated flows")
                self.flows.append(event.value)
            elif event.stage == "rgb_read":
                put_once(
                    self.sink,
                    f"sequences/{self.member}/frame-{event.index}/rgb.bin",
                    event.value.tobytes(),
                )
            elif event.stage == "paired_read_complete":
                if (
                    not isinstance(event.value, tuple)
                    or len(event.value) != 2
                    or tuple(map(len, event.value)) != (120 * 160 * 3, 120 * 160 * 4)
                ):
                    raise ValueError("bounded returned paired bytes")
                for name, data in zip(("native-id", "native-depth"), event.value, strict=True):
                    put_once(
                        self.sink, f"partial/{self.member}/frame-{event.index}-{name}.bin", data
                    )
            elif event.stage in {
                "paired_draw_input",
                "paired_draw_output",
                "paired_read_input",
                "paired_read_output",
                "frame_operations",
            }:
                if type(event.value) is not bytes or len(event.value) > 256 * 1024:
                    raise ValueError("bounded immutable operational snapshot")
                parse(event.value, 256 * 1024)
                put_once(
                    self.sink,
                    f"partial/{self.member}/frame-{event.index}-{event.stage}.json",
                    event.value,
                )
            if event.stage in {"sequence", "frame", "flow"}:
                self.encode(partial=True)
            record = {
                "stage": event.stage,
                "index": event.index,
                "failure": event.value if event.stage == "failure" else None,
            }
            data = canonical(record)
            if len(data) > 16 * 1024:
                raise ValueError("progress receipt bound")
            self.sink.put(f"operations/{self.member}-progress-{self.events:03d}.json", data)
            self.events += 1
            if event.stage == "failure":
                self.failed = True
        except BaseException:
            self.failed = True
            raise

    def encode(self, *, partial: bool) -> None:
        if self.failed:
            raise ValueError("failed capture cannot encode a new envelope")
        if self.sequence is None:
            raise ValueError("committed mapping metadata required")
        config, _, _ = candidate(self.payload, self.member)
        manifest, artifacts = encode_sequence(
            config=config,
            member=self.member,
            source_head=self.binding.source_head,
            source_tree=self.binding.source_tree,
            image_digest=self.binding.image,
            renderer_identity=self.renderer_identity,
            sequence=self.sequence,
            frames=tuple(self.frames),
            flows=tuple(self.flows),
            partial=partial,
        )
        for path, data in artifacts.items():
            put_once(self.sink, f"sequences/{self.member}/{path}", data)
        if manifest:
            self.sink.put(f"sequences/{self.member}/manifest.json", manifest)


@dataclass
class AttemptState:
    history: str | None = None
    index: int = 0
    task: str | None = None
    name: str | None = None
    token: str | None = None
    failed: bool = False
    reserved: int = 0


def replay(records: list[dict[str, Any]], binding: Binding, archive: Path) -> AttemptState:
    state = AttemptState()
    previous = ZERO
    schedule = binding.tasks
    commands: list[str] = []
    for i, r in enumerate(records):
        body = {k: v for k, v in r.items() if k != "sha256"}
        if (
            r.get("sequence") != i
            or type(r.get("sequence")) is not int
            or r.get("previous") != previous
            or r.get("sha256") != digest(canonical(body))
        ):
            raise ValueError("receipt chain changed")
        previous = r["sha256"]
        v = {k: x for k, x in body.items() if k not in {"sequence", "previous"}}
        if state.failed:
            raise ValueError("terminal failed attempt cannot continue")
        if i == 0:
            if v != {
                "kind": "BINDING",
                "binding": vars(binding),
                "archive": str(archive.resolve()),
                "schedule": list(schedule),
                "budget": vars(BUDGET),
                "max_bytes": maximum_bytes(),
            }:
                raise ValueError("exact roots/task membership/source binding changed")
        elif v["kind"] == "INITIAL":
            if i != 1 or set(v) != {"kind", "history"}:
                raise ValueError("initial exact history required")
            state.history = v["history"]
        elif v["kind"] == "ATTEMPT":
            seconds = (
                SLOT_SECONDS if binding.purpose == "causal_history_native_v1" else DUMMY_SECONDS
            )
            if (
                set(v) != {"kind", "task", "name", "token", "history", "seconds"}
                or state.task is not None
                or state.index >= len(schedule)
                or v["task"] != schedule[state.index]
                or v["history"] != state.history
                or type(v["seconds"]) is not int
                or v["seconds"] != seconds
                or re.fullmatch(r"[0-9a-f]{32}", v["token"]) is None
                or v["name"] != "eps-causal-" + v["token"]
            ):
                raise ValueError("stale/reused/unordered attempt denied")
            state.task, state.name, state.token = v["task"], v["name"], v["token"]
            state.reserved += seconds
            commands = []
        elif v["kind"] == "COMMAND":
            if (
                set(v) != {"kind", "operation", "exit", "timeout", "overflow", "stdout", "stderr"}
                or state.task is None
                or v["operation"] not in {"create", "inspect", "start", "rm"}
                or type(v["exit"]) is not int
                or type(v["timeout"]) is not bool
                or type(v["overflow"]) is not bool
                or any(
                    type(v[k]) is not str or len(v[k].encode()) > CONTROL_BOUND
                    for k in ("stdout", "stderr")
                )
            ):
                raise ValueError("bounded owned command record required")
            commands.append(
                v["operation"]
                if v["exit"] == 0 and not v["timeout"] and not v["overflow"]
                else "failed"
            )
        elif v["kind"] == "COMPLETE":
            if (
                set(v) != {"kind", "task", "history"}
                or state.task != v["task"]
                or commands != ["create", "inspect", "start", "inspect", "inspect", "rm"]
                or v["history"] == state.history
            ):
                raise ValueError("owned cleanup/append-only completion required")
            state.history, state.task = v["history"], None
            state.index += 1
        elif v["kind"] == "FAILURE":
            if set(v) != {"kind", "task", "error", "history"} or v["task"] != state.task:
                raise ValueError("bounded task terminal failure")
            state.failed = True
        else:
            raise ValueError("unknown finite attempt event")
        if "history" in v and re.fullmatch(r"[0-9a-f]{64}", v["history"]) is None:
            raise ValueError("exact history digest required")
    return state


class HostReceipts:
    def __init__(self, path: Path, binding: Binding, archive: Path, *, initial: bool):
        self.path, self.binding, self.archive = path, binding, archive
        self.records: list[dict[str, Any]] = []
        self.poisoned = False
        if initial:
            for p in (path, *path.parents):
                if p.is_symlink():
                    raise ValueError("linked receipt path denied")
            self.stream: BinaryIO = path.open("xb+")
        else:
            _safe_file(path)
            if path.stat().st_size > HOST_LIMIT:
                raise ValueError("receipt byte limit")
            self.stream = path.open("r+b")
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            if initial:
                self.append(
                    {
                        "kind": "BINDING",
                        "binding": vars(binding),
                        "archive": str(archive.resolve()),
                        "schedule": list(binding.tasks),
                        "budget": vars(BUDGET),
                        "max_bytes": maximum_bytes(),
                    }
                )
            else:
                for line in self.stream:
                    obj = parse(line, 64 * 1024)
                    if canonical(obj) + b"\n" != line:
                        raise ValueError("torn/noncanonical receipt history")
                    self.records.append(obj)
                replay(self.records, binding, archive)
                self.stream.seek(0, os.SEEK_END)
                self.file_identity = _identity(path.stat())
        except BaseException:
            self.stream.close()
            raise

    @property
    def root(self) -> str:
        return self.records[-1]["sha256"] if self.records else ZERO

    def append(self, value: dict[str, Any]) -> None:
        if self.poisoned or set(value) & {"sequence", "previous", "sha256"}:
            raise ValueError("failed/reserved witness fields")
        if self.records:
            _safe_file(self.path)
            if (
                _identity(self.path.stat()) != self.file_identity
                or _identity(os.fstat(self.stream.fileno())) != self.file_identity
            ):
                self.poisoned = True
                raise ValueError("host receipt changed or replaced")
        body = {"sequence": len(self.records), "previous": self.root, **value}
        record = parse(canonical({**body, "sha256": digest(canonical(body))}))
        replay([*self.records, record], self.binding, self.archive)
        data = canonical(record) + b"\n"
        reserve = 0 if value["kind"] == "FAILURE" else 64 * 1024
        if len(data) > 64 * 1024 or self.stream.tell() + len(data) > HOST_LIMIT - reserve:
            raise ValueError("bounded host receipt admission")
        try:
            if self.stream.write(data) != len(data):
                raise OSError("short witness write")
            self.stream.flush()
            os.fsync(self.stream.fileno())
        except BaseException:
            self.poisoned = True
            raise
        self.records.append(record)
        self.file_identity = _identity(self.path.stat())

    def close(self) -> None:
        self.stream.close()


def initialize(path: Path, binding: Binding, receipts: HostReceipts) -> str:
    sink = FiniteArchive(path, binding.root, BUDGET)
    try:
        sink.put("operations/binding.json", canonical(vars(binding)))
        expected = sink.history()
        receipts.append({"kind": "INITIAL", "history": expected})
        return expected
    finally:
        sink.close()


def create_arguments(
    name: str, token: str, binding: Binding, path: Path, task: str, expected: str
) -> list[str]:
    _safe_file(path)
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
        f"type=bind,src={path.resolve()},dst=/retained/history",
        "--env=EPS_CAUSAL_RUNTIME=docker_candidate_v1",
        "--env=MUJOCO_GL=osmesa",
        "--env=PYOPENGL_PLATFORM=osmesa",
        "--env=LP_NUM_THREADS=2",
        "--env=OMP_NUM_THREADS=2",
        "--env=PYTHONDONTWRITEBYTECODE=1",
        "--env=TMPDIR=/output",
        binding.image,
        "python",
        "scripts/causal_history_entry.py",
        "--task",
        task,
        "--expected-history",
        expected,
        "--binding",
        canonical(vars(binding)).decode(),
    ]


def validate_mount(info: dict[str, Any], path: Path) -> None:
    expected = str(path.resolve())
    definitions, mounts = info["HostConfig"].get("Mounts", []), info["Mounts"]
    accepted = {expected}
    windows = PureWindowsPath(expected)
    if re.fullmatch(r"[A-Za-z]:", windows.drive):
        suffix = windows.drive[0].lower() + "/" + "/".join(windows.parts[1:])
        accepted.update({"/run/desktop/mnt/host/" + suffix, "/host_mnt/" + suffix})
    if (
        len(definitions) != 1
        or len(mounts) != 1
        or definitions[0].get("Source") != expected
        or definitions[0].get("Type") != "bind"
        or definitions[0].get("Target") != "/retained/history"
        or definitions[0].get("ReadOnly", False)
        or mounts[0].get("Source") not in accepted
        or mounts[0].get("Type") != "bind"
        or mounts[0].get("Destination") != "/retained/history"
        or mounts[0].get("RW") is not True
    ):
        raise ValueError("exact retained-file mount required")


class DockerController:
    def __init__(self, docker: str, path: Path, binding: Binding, receipts: HostReceipts):
        self.docker, self.path, self.binding, self.receipts = docker, path, binding, receipts

    def command(self, args: list[str], deadline: float, timeout: float = 15) -> str:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("owned task deadline")
        proc = subprocess.Popen(
            [self.docker, *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        buffers = [bytearray(), bytearray()]
        overflow = threading.Event()

        def read(pipe: BinaryIO, result: bytearray) -> None:
            while chunk := pipe.read(4096):
                room = CONTROL_BOUND - len(result)
                result.extend(chunk[: max(0, room)])
                if len(chunk) > room:
                    overflow.set()
                    proc.kill()
                    return

        assert proc.stdout is not None and proc.stderr is not None
        readers = [
            threading.Thread(target=read, args=(p, b), daemon=True)
            for p, b in zip((proc.stdout, proc.stderr), buffers, strict=True)
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
            self.receipts.append(
                {
                    "kind": "COMMAND",
                    "operation": args[0],
                    "exit": proc.returncode,
                    "timeout": timed_out,
                    "overflow": overflow.is_set(),
                    "stdout": buffers[0].decode("ascii", errors="replace"),
                    "stderr": buffers[1].decode("ascii", errors="replace"),
                }
            )
        if timed_out or overflow.is_set() or proc.returncode or any(r.is_alive() for r in readers):
            raise RuntimeError("failed/uncertain command")
        return buffers[0].decode().strip()

    def owned(self, identity: str, name: str, token: str, deadline: float) -> dict[str, Any]:
        info = parse(
            self.command(["inspect", "--format", "{{json .}}", identity], deadline).encode(),
            CONTROL_BOUND,
        )
        if (
            re.fullmatch(r"[0-9a-f]{64}", info["Id"]) is None
            or info["Name"] != "/" + name
            or info["Config"]["Labels"].get(LABEL) != token
        ):
            raise ValueError("unowned container cleanup denied")
        return dict(info)

    def launch(self, task: str, expected: str, receipt_root: str) -> str:
        state = replay(self.receipts.records, self.binding, self.path)
        if (
            state.failed
            or state.task is not None
            or state.history != expected
            or receipt_root != self.receipts.root
            or history(self.path) != expected
        ):
            raise ValueError("stale/incomplete external witness")
        seconds = (
            SLOT_SECONDS if self.binding.purpose == "causal_history_native_v1" else DUMMY_SECONDS
        )
        started = time.monotonic()
        deadline = started + seconds
        token = uuid.uuid4().hex
        name = "eps-causal-" + token
        # Full nonrefundable reservation fsyncs BEFORE create/start can be attempted.
        self.receipts.append(
            {
                "kind": "ATTEMPT",
                "task": task,
                "name": name,
                "token": token,
                "history": expected,
                "seconds": seconds,
            }
        )
        try:
            sink = FiniteArchive(self.path, self.binding.root, BUDGET, expected_history=expected)
            try:
                sink.put(
                    f"operations/{task}-reserved.json",
                    canonical(
                        {
                            "task": task,
                            "seconds": seconds,
                            "receipt": self.receipts.root,
                            "prior_history": expected,
                            "binding": vars(self.binding),
                        }
                    ),
                )
                consumed = sink.history()
                consumed_length = self.path.stat().st_size
            finally:
                sink.close()
        except BaseException as failure:
            self.receipts.append(
                {
                    "kind": "FAILURE",
                    "task": task,
                    "error": type(failure).__name__,
                    "history": history(self.path),
                }
            )
            raise
        identity = name
        error: BaseException | None = None
        try:
            identity = self.command(
                create_arguments(name, token, self.binding, self.path, task, consumed), deadline
            )
            info = self.owned(identity, name, token, deadline)
            validate_mount(info, self.path)
            hc = info["HostConfig"]
            if (
                info["Image"] != self.binding.image
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
                raise ValueError("exact prestart resource profile required")
            work = HOST_WORK_SECONDS if seconds == SLOT_SECONDS else 3
            self.command(
                ["start", "--attach", identity], min(deadline, started + work), timeout=work
            )
            completed = self.owned(identity, name, token, deadline)["State"]
            if completed["Running"] or completed["ExitCode"] != 0 or completed["OOMKilled"]:
                raise ValueError("unsuccessful task")
        except BaseException as failure:
            error = failure
        finally:
            try:
                cleanup = min(
                    deadline, time.monotonic() + (CLEANUP_SECONDS if seconds == SLOT_SECONDS else 3)
                )
                info = self.owned(identity, name, token, cleanup)
                self.command(["rm", "--force", info["Id"]], cleanup)
            except BaseException as failure:
                error = failure
        observed = history(self.path)
        if error is None:
            try:
                retained = recover(self.path, self.binding.root, BUDGET, observed)
                verify_prefix(self.path, consumed_length, consumed)
                if (
                    retained.incomplete
                    or retained.suffix_bytes
                    or f"operations/{task}-complete.json"
                    not in {p for p, _, _ in retained.committed}
                ):
                    raise ValueError("missing durable complete receipt")
                self.receipts.append({"kind": "COMPLETE", "task": task, "history": observed})
            except BaseException as failure:
                error = failure
        if error is not None:
            self.receipts.append(
                {
                    "kind": "FAILURE",
                    "task": task,
                    "error": type(error).__name__,
                    "history": observed,
                }
            )
            raise RuntimeError("terminal retained attempt; no retry") from error
        return observed
