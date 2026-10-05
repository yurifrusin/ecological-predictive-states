"""Bounded append-only retained archive, independent of native capture and evaluation."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import stat
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

CHUNK = 64 * 1024
MAX_RECORD = 100 * 1024
MIB = 1024 * 1024


def _digest(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("SHA-256 required")


def _path(value: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode()) > 1024
        or "\\" in value
        or ":" in value
        or "\0" in value
        or PurePosixPath(value).is_absolute()
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError("noncanonical artifact path")


@dataclass(frozen=True)
class Budget:
    """Logical application bytes, including duplicates and framing; not an OS quota."""

    staging: int = 255 * MIB
    shared: int = MIB
    archive: int = 768 * MIB
    terminal_reserve: int = 8 * MIB

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 0 for value in vars(self).values()):
            raise ValueError("nonnegative integer budget required")
        if not 512 <= self.terminal_reserve < self.archive:
            raise ValueError("terminal reserve must fit archive")
        if self.staging + self.shared + self.archive > 1024 * MIB:
            raise ValueError("logical total exceeds 1 GiB")


@dataclass(frozen=True)
class Recovery:
    committed: tuple[tuple[str, int, str], ...]
    incomplete: tuple[str, ...]
    sequence: int
    committed_prefix_bytes: int
    suffix_bytes: int
    history_sha256: str
    staging_reserved: int


def _safe_file(path: Path) -> None:
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError("linked archive denied")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("non-owned regular file denied")


def _identity(info: os.stat_result) -> tuple[int, ...]:
    # Reads may update atime; modification, replacement and link changes may not.
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def recover(path: Path, study: str, budget: Budget, expected_history: str) -> Recovery:
    """Validate bounded history, retaining (never truncating) an interrupted suffix."""
    _digest(study)
    _digest(expected_history)
    _safe_file(path)
    before = path.stat()
    if before.st_size > budget.archive:
        raise ValueError("archive exceeds logical budget")
    history = hashlib.sha256()
    committed: list[tuple[str, int, str]] = []
    used: set[str] = set()
    incomplete: list[str] = []
    pending: str | None = None
    length = 0
    content = hashlib.sha256()
    sequence = 0
    prefix = 0
    staging_reserved = 0
    artifact_reserved = 0
    with path.open("rb") as stream:
        if _identity(os.fstat(stream.fileno())) != _identity(before):
            raise ValueError("history changed before read")
        while True:
            header = stream.read(4)
            history.update(header)
            if not header:
                break
            if len(header) != 4:
                break
            size = struct.unpack("!I", header)[0]
            if size > MAX_RECORD or size == 0:
                raise ValueError("invalid record length")
            payload = stream.read(size)
            history.update(payload)
            if len(payload) != size:
                break
            record = json.loads(payload)
            if (
                not isinstance(record, dict)
                or type(record.get("sequence")) is not int
                or record.get("sequence") != sequence
                or json.dumps(
                    record, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode()
                != payload
            ):
                raise ValueError("record sequence differs")
            kind = record.get("kind")
            if sequence == 0:
                if record != {
                    "sequence": 0,
                    "kind": "STUDY",
                    "study": study,
                    "budget": vars(budget),
                }:
                    raise ValueError("study or capacity contract differs")
            elif kind == "START":
                if set(record) != {"sequence", "kind", "path"} or pending is not None:
                    raise ValueError("invalid reservation")
                pending = record["path"]
                _path(pending)
                if pending in used:
                    raise ValueError("duplicate artifact")
                used.add(pending)
                incomplete.append(pending)
                length = 0
                artifact_reserved = 0
                content = hashlib.sha256()
            elif kind == "STAGE":
                if (
                    set(record) != {"sequence", "kind", "size"}
                    or pending is None
                    or type(record["size"]) is not int
                    or record["size"] <= 0
                    or staging_reserved + record["size"] > budget.staging
                ):
                    raise ValueError("invalid staging reservation")
                staging_reserved += record["size"]
                artifact_reserved += record["size"]
            elif kind == "CHUNK":
                if set(record) != {"sequence", "kind", "data"} or pending is None:
                    raise ValueError("unreserved data")
                data = base64.b64decode(record["data"], validate=True)
                if not 0 < len(data) <= CHUNK:
                    raise ValueError("chunk bound differs")
                length += len(data)
                if length > artifact_reserved:
                    raise ValueError("unreserved staging bytes")
                content.update(data)
            elif kind == "COMMIT":
                if set(record) != {"sequence", "kind", "length", "sha256"} or pending is None:
                    raise ValueError("unreserved commit")
                _digest(record["sha256"])
                if type(record["length"]) is not int or (record["length"], record["sha256"]) != (
                    length,
                    content.hexdigest(),
                ):
                    raise ValueError("artifact length/hash differs")
                committed.append((pending, length, content.hexdigest()))
                incomplete.remove(pending)
                pending = None
            elif kind == "FAILURE":
                if (
                    set(record) != {"sequence", "kind", "reason"}
                    or not isinstance(record["reason"], str)
                    or len(record["reason"]) > 128
                ):
                    raise ValueError("invalid failure receipt")
                pending = None
            else:
                raise ValueError("unknown record")
            sequence += 1
            prefix = stream.tell()
    after = path.stat()
    if (
        _identity(before) != _identity(after)
        or history.hexdigest() != expected_history
        or sequence == 0
    ):
        raise ValueError("missing, changed or unstable study history")
    return Recovery(
        tuple(committed),
        tuple(incomplete),
        sequence,
        prefix,
        before.st_size - prefix,
        history.hexdigest(),
        staging_reserved,
    )


class RetainedArchive:
    """Single cooperative sink. A failed flush poisons it; no ACK or further writes."""

    def __init__(
        self, path: Path, study: str, budget: Budget, *, expected_history: str | None = None
    ) -> None:
        _digest(study)
        self.path = path
        self.study = study
        self.budget = budget
        self.staging_limit = budget.staging
        self.sequence = 0
        self.used: set[str] = set()
        self.pending: str | None = None
        self.poisoned = False
        self.staging_reserved = 0
        self.artifact_reserved = 0
        self.artifact_length = 0
        self.artifact_digest = hashlib.sha256()
        self.committed: dict[str, tuple[int, str, int, int]] = {}
        self.artifact_start = 0
        if expected_history is None:
            if path.exists() or path.is_symlink():
                raise ValueError("existing history requires exact resume receipt")
            for part in path.parents:
                if part.is_symlink():
                    raise ValueError("linked archive denied")
            self.stream: BinaryIO = path.open("xb")
            self._append({"kind": "STUDY", "study": study, "budget": vars(budget)})
        else:
            state = recover(path, study, budget, expected_history)
            if state.suffix_bytes or state.incomplete:
                raise ValueError("INCOMPLETE_UNCOMMITTED history requires read-only recovery")
            self.sequence = state.sequence
            self.staging_reserved = state.staging_reserved
            self.used = {item[0] for item in state.committed}
            self.stream = path.open("ab")
            start = 0
            name = ""
            with path.open("rb") as existing:
                while header := existing.read(4):
                    position = existing.tell() - 4
                    record = json.loads(existing.read(struct.unpack("!I", header)[0]))
                    if record["kind"] == "START":
                        name, start = record["path"], position
                    elif record["kind"] == "COMMIT":
                        self.committed[name] = (
                            record["length"],
                            record["sha256"],
                            start,
                            existing.tell(),
                        )

    def _flush(self) -> None:
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def _append(self, values: dict[str, Any], *, terminal: bool = False) -> None:
        if self.poisoned:
            raise ValueError("failed sink cannot acknowledge")
        payload = json.dumps(
            {"sequence": self.sequence, **values},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        if len(payload) > MAX_RECORD:
            raise ValueError("record bound exceeded")
        limit = self.budget.archive - (0 if terminal else self.budget.terminal_reserve)
        if self.stream.tell() + 4 + len(payload) > limit:
            raise ValueError("archive admission denied before write")
        try:
            framed = struct.pack("!I", len(payload)) + payload
            if self.stream.write(framed) != len(framed):
                raise OSError("short archive write")
            self._flush()
        except Exception:
            self.poisoned = True
            raise
        self.sequence += 1

    def start(self, path: str) -> None:
        _path(path)
        if self.pending is not None or path in self.used:
            raise ValueError("unfinished or duplicate artifact")
        self.artifact_start = self.stream.tell()
        self._append({"kind": "START", "path": path})
        self.used.add(path)
        self.pending = path
        self.artifact_reserved = 0
        self.artifact_length = 0
        self.artifact_digest = hashlib.sha256()

    def chunk(self, data: bytes) -> None:
        """Admit a bounded transfer chunk through the same archive rules."""
        if (
            self.pending is None
            or not isinstance(data, bytes)
            or not 0 < len(data) <= CHUNK
            or self.artifact_length + len(data) > self.artifact_reserved
        ):
            raise ValueError("unreserved or oversized transfer")
        self._append({"kind": "CHUNK", "data": base64.b64encode(data).decode("ascii")})
        self.artifact_length += len(data)
        self.artifact_digest.update(data)

    def commit_transfer(self, length: int, sha256: str) -> None:
        _digest(sha256)
        if (
            self.pending is None
            or type(length) is not int
            or (length, sha256) != (self.artifact_length, self.artifact_digest.hexdigest())
        ):
            raise ValueError("transfer length/hash differs")
        self._append({"kind": "COMMIT", "length": length, "sha256": sha256})
        self.committed[self.pending] = (length, sha256, self.artifact_start, self.stream.tell())
        self.pending = None

    def reserve_staging(self, size: int) -> None:
        if size == 0:
            return
        if (
            self.pending is None
            or type(size) is not int
            or size < 0
            or self.staging_reserved + size > self.staging_limit
        ):
            raise ValueError("staging admission denied before write")
        self._append({"kind": "STAGE", "size": size})
        self.staging_reserved += size
        self.artifact_reserved += size

    def publish(self, path: Path, relative: str) -> None:
        if self.pending is None or relative != self.pending:
            raise ValueError("artifact reservation required")
        _safe_file(path)
        before = path.stat()
        if before.st_size > self.artifact_reserved:
            raise ValueError("staging bound differs")
        digest = hashlib.sha256()
        length = 0
        with path.open("rb") as stream:
            if _identity(os.fstat(stream.fileno())) != _identity(before):
                raise ValueError("artifact changed before transfer")
            while data := stream.read(CHUNK):
                if length + len(data) > before.st_size:
                    raise ValueError("artifact changed during transfer")
                self.chunk(data)
                digest.update(data)
                length += len(data)
        if _identity(path.stat()) != _identity(before) or length != before.st_size:
            raise ValueError("artifact changed during transfer")
        self.commit_transfer(length, digest.hexdigest())

    def failure(self, reason: str) -> None:
        if not isinstance(reason, str) or len(reason) > 128:
            raise ValueError("bounded failure reason required")
        self._append({"kind": "FAILURE", "reason": reason}, terminal=True)
        self.pending = None

    def put(self, relative: str, data: bytes) -> None:
        """Retain small control-plane content; charge its in-memory materialization."""
        self.start(relative)
        self.reserve_staging(len(data))
        for offset in range(0, len(data), CHUNK):
            self.chunk(data[offset : offset + CHUNK])
        self.commit_transfer(len(data), hashlib.sha256(data).hexdigest())

    def entries(self) -> tuple[tuple[str, int, str], ...]:
        return tuple((name, value[0], value[1]) for name, value in self.committed.items())

    def history(self) -> str:
        self._flush()
        result = hashlib.sha256()
        with self.path.open("rb") as stream:
            while data := stream.read(CHUNK):
                result.update(data)
        return result.hexdigest()

    def read_chunks(self, relative: str) -> Iterator[bytes]:
        """Yield validated committed content using bounded reads, never bulk arrays."""
        if relative not in self.committed:
            raise KeyError(relative)
        length, sha, start, end = self.committed[relative]
        observed = hashlib.sha256()
        count = 0
        with self.path.open("rb") as stream:
            stream.seek(start)
            while stream.tell() < end:
                header = stream.read(4)
                size = struct.unpack("!I", header)[0]
                if not 0 < size <= MAX_RECORD:
                    raise ValueError("changed indexed frame")
                record = json.loads(stream.read(size))
                if record["kind"] == "CHUNK":
                    data = base64.b64decode(record["data"], validate=True)
                    if not 0 < len(data) <= CHUNK:
                        raise ValueError("changed indexed chunk bound")
                    count += len(data)
                    observed.update(data)
                    yield data
        if count != length or observed.hexdigest() != sha:
            raise ValueError("changed indexed artifact")

    def reserve_copy(self, relative: str, length: int) -> None:
        """Durable charge before re-materializing retained bytes into staging."""
        self.start(relative)
        self.reserve_staging(length)
        self.commit_transfer(0, hashlib.sha256(b"").hexdigest())

    def close(self) -> None:
        self.stream.close()
