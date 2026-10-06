"""Bounded private retention and globally sealed chronology; synthetic-testable."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from epsbench.diagnostics.boundary_observation import VisibleRaster
from epsbench.diagnostics.visible_forecast_contract import (
    REQUIRED,
    CausalInput,
    CausalView,
    Command,
    Forecast,
    Limits,
    TokenFrame,
    _json,
    evaluate,
    forecast,
    storage,
)
from epsbench.schema import ModalityPermissionSet
from epsbench.utils.canonical import canonical_json_bytes

VERSION = "occupancy-qualification-development-20261007-v2"
MEMBERS = ("R", "H", "F", "P")
UNITS = ("RH", "F", "P")
LIMITS = Limits(2, 1024, 3)
TOTAL_BYTES = 16 * 1024 * 1024
FAILURE_BYTES = 256 * 1024


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode(value: Any) -> bytes:
    # Closed writers construct fields; fractions in evaluator output remain exact strings.
    from fractions import Fraction

    def exact(v: Any) -> str:
        if type(v) is not Fraction:
            raise TypeError("unsupported retained value")
        return str(v)

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=exact
    ).encode()


def flatten(source: CausalInput) -> bytes:
    """Lossless unstructured row-major bytes plus exactly the permitted metadata."""
    masks = [
        (frame.index, token, a.tobytes(order="C"))
        for frame in source.frames
        for token, a in frame.masks
    ]
    header = {
        "source": source.digest,
        "shape": source.shape,
        "frames": [[t for t, _ in f.masks] for f in source.frames],
        "executed": [[str(v) for v in c] for c in source.executed],
        "announced": [str(v) for v in source.announced],
    }
    return encode(header) + b"\n" + b"".join(a for _, _, a in masks)


def unflatten(data: bytes, source: CausalInput) -> CausalInput:
    header_bytes, buffer = data.split(b"\n", 1)
    header = _json(header_bytes)
    if header != _json(flatten(source).split(b"\n", 1)[0]):
        raise ValueError("flat metadata differs")
    pixels = source.shape[0] * source.shape[1]
    count = sum(len(f) for f in header["frames"])
    if len(buffer) != count * pixels or any(v not in (0, 1) for v in buffer):
        raise ValueError("flat Boolean buffer differs")
    offset = 0
    frames = []
    for index, tokens in enumerate(header["frames"]):
        masks = []
        for token in tokens:
            a = np.frombuffer(buffer[offset : offset + pixels], dtype=np.bool_).reshape(
                source.shape
            )
            masks.append((token, a))
            offset += pixels
        frames.append(TokenFrame(index, source.shape, tuple(masks)))
    return CausalInput(
        tuple(frames), source.executed, source.announced, source.permissions, source.limits
    )


def reject_reparse(path: Path) -> None:
    for candidate in (path, *path.parents):
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            continue
        if candidate.is_symlink() or getattr(info, "st_file_attributes", 0) & getattr(
            stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024
        ):
            raise ValueError("reparse/junction retained path denied")


class Retention:
    """One fresh private directory, no archive/copy allowance outside TOTAL_BYTES."""

    def __init__(self, root: Path, clock: Callable[[], float] = time.monotonic) -> None:
        if root.exists():
            raise ValueError("fresh retention directory required; no resume")
        reject_reparse(root)
        root.mkdir(parents=True)
        self.root = root
        self.clock = clock
        self.start = clock()
        self.used = 0
        self.calls = 0
        self.entries: list[dict[str, Any]] = []
        self.failed = False
        self.stage = "INITIALIZATION"

    def check(self) -> None:
        if self.failed or self.clock() - self.start >= 60:
            raise RuntimeError("closed or cooperative deadline exceeded")

    def read(self, name: str) -> bytes:
        self.check()
        path = self.root / name
        reject_reparse(path)
        if (
            self.root.is_symlink()
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size > TOTAL_BYTES
        ):
            raise ValueError("bounded retained regular file required")
        return path.read_bytes()

    def producer_call(self) -> None:
        self.check()
        if self.calls >= 10:
            raise RuntimeError("ten producer calls exhausted")
        self.calls += 1

    def write(self, name: str, data: bytes, failure: bool = False) -> str:
        if not failure:
            self.check()
        if re.fullmatch(r"[a-zA-Z0-9_-]+[.](json|bin)", name) is None:
            raise ValueError("flat retained filename required")
        cap = TOTAL_BYTES if failure else TOTAL_BYTES - FAILURE_BYTES
        if type(data) is not bytes or self.used + len(data) > cap:
            raise RuntimeError("total retained byte budget exceeded")
        path = self.root / name
        reject_reparse(path)
        if self.root.is_symlink() or path.exists():
            raise ValueError("retained path replacement denied")
        self.used += len(data)  # Reserve before opening, including partial/failed writes.
        with path.open("xb") as handle:
            count = handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if count != len(data):
            raise OSError("partial retained write")
        if not failure:
            self.check()
        return digest(data)

    def event(self, kind: str, member: str, binding: str) -> None:
        if kind not in {"LOCK", "PREFIX", "INPUT", "FORECAST", "SEAL", "TARGET", "COMPLETE"}:
            raise ValueError("closed chronology kind")
        entry = {
            "index": len(self.entries),
            "kind": kind,
            "member": member,
            "binding": binding,
            "previous": digest(encode(self.entries[-1])) if self.entries else None,
        }
        self.write(f"event-{len(self.entries)}.json", encode(entry))
        self.entries.append(entry)
        self.stage = kind

    def fail(self, pending: list[str], error: Exception | None = None) -> None:
        # Fixed non-reconstructive error code; exception text/host times not scientific fields.
        self.failed = True
        self.write(
            "failure-context.json",
            encode({"private_error": str(error)[:2048] if error else "unspecified"}),
            True,
        )
        self.write(
            "failure.json",
            encode(
                {
                    "version": VERSION,
                    "status": "FAILED_CLOSED",
                    "pending": pending,
                    "producer_calls": self.calls,
                    "stage": self.stage,
                    "failure_kind": "VALUE"
                    if isinstance(error, ValueError)
                    else "IO"
                    if isinstance(error, OSError)
                    else "LIMIT_OR_CLOSED",
                    "last_event": digest(encode(self.entries[-1])) if self.entries else None,
                }
            ),
            True,
        )


class Provider(Protocol):
    def prefix(self, unit: str, index: int) -> VisibleRaster: ...
    def target(self, member: str) -> VisibleRaster: ...


class RetainedProvider:
    def __init__(self, frames: tuple[VisibleRaster, ...]) -> None:
        self.frames = frames

    def raster(self, sequence_index: int) -> VisibleRaster:
        if type(sequence_index) is not int or not 0 <= sequence_index < len(self.frames):
            raise PermissionError("retained chronology unavailable before fetch")
        return self.frames[sequence_index]


class Qualification:
    """Single-use controller. No retries, predictor callbacks, or target prefetch."""

    def __init__(
        self,
        provider: Provider,
        retained: Retention,
        plan: dict[str, Any],
        commands: dict[str, tuple[tuple[Command, ...], Command]],
    ) -> None:
        if set(commands) != set(MEMBERS):
            raise ValueError("complete four-member command plan required")
        self.provider = provider
        self.retained = retained
        self.plan = {
            **plan,
            "actions": {
                m: {"executed": [[str(v) for v in c] for c in e], "announced": [str(v) for v in a]}
                for m, (e, a) in commands.items()
            },
        }
        self.commands = commands
        self.sealed = False
        self.started = False
        self.targets: dict[str, VisibleRaster] = {}

    def target(self, member: str) -> VisibleRaster:
        if not self.sealed or member not in MEMBERS:
            raise PermissionError("complete eight-forecast seal required before target fetch")
        self.retained.check()
        self.verify_seal()
        if member not in self.targets:
            self.targets[member] = self.provider.target(member)
        return self.targets[member]

    def verify_seal(self) -> None:
        validate_seal(self.retained.read, self.retained.entries, self.plan, self.commands)

    def run(self) -> dict[str, Any]:
        if self.started:
            raise RuntimeError("single use; no retry/resume")
        self.started = True
        pending = list(MEMBERS)
        try:
            self.retained.stage = "LOCK"
            self.retained.event("LOCK", "ALL", self.retained.write("plan.json", encode(self.plan)))
            prefixes: dict[str, tuple[VisibleRaster, ...]] = {}
            self.retained.stage = "PREFIX"
            for unit in UNITS:
                frames = []
                for index in range(2):
                    r = self.provider.prefix(unit, index)
                    if type(r) is not VisibleRaster or r.sequence_index != index:
                        raise ValueError("prefix chronology differs")
                    LIMITS.check(index + 1, r.segmentation.shape, len(r.identities))
                    frames.append(r)
                    payload = encode(
                        {
                            "index": index,
                            "shape": r.segmentation.shape,
                            "identities": r.identities,
                            "labels": r.segmentation.tolist(),
                        }
                    )
                    sha = self.retained.write(f"prefix-{unit}-{index}.json", payload)
                    self.retained.event("PREFIX", unit, sha)
                prefixes[unit] = tuple(frames)
            sources: dict[str, CausalInput] = {}
            saved: dict[tuple[str, str], bytes] = {}
            bindings = []
            self.retained.stage = "FORECAST"
            for member in MEMBERS:
                executed, announced = self.commands[member]
                source = CausalView(
                    RetainedProvider(prefixes["RH" if member in ("R", "H") else member]),
                    ModalityPermissionSet(allowed=REQUIRED),
                    1,
                    LIMITS,
                ).materialize(executed, announced)
                sources[member] = source
                data = source.canonical_bytes()
                flat_data = flatten(source)
                flat = unflatten(flat_data, source)
                flat_sha = self.retained.write(f"flat-{member}.bin", flat_data)
                roundtrip = CausalInput.from_bytes(data, source.permissions, LIMITS)
                if roundtrip.canonical_bytes() != data:
                    raise ValueError("lossless input fidelity failed")
                self.retained.event(
                    "INPUT", member, self.retained.write(f"input-{member}.json", data)
                )
                for rule, window in (("current-mask-persistence", 1), ("k-frame-agreement", 2)):
                    data = forecast(source, rule, window).canonical_bytes()
                    # Lossless row-major unstructured control has exactly the same rule/information.
                    if forecast(flat, rule, window).canonical_bytes() != data:
                        raise ValueError("equal-information fidelity failed")
                    Forecast.from_bytes(data, source)
                    saved[member, rule] = data
                    sha = self.retained.write(f"forecast-{member}-{window}.json", data)
                    self.retained.event("FORECAST", member, sha)
                    bindings.append(
                        {
                            "member": member,
                            "rule": rule,
                            "window": window,
                            "input": source.digest,
                            "forecast": sha,
                            "inventory": source.inventory,
                            "storage": storage(source, data),
                            "flat_input": flat_sha,
                        }
                    )
            if len(bindings) != 8:
                raise ValueError("all eight bindings required")
            seal = self.retained.write(
                "seal.json",
                encode(
                    {
                        "version": VERSION,
                        "members": list(MEMBERS),
                        "bindings": bindings,
                        "plan": digest(encode(self.plan)),
                        "chronology": digest(encode(self.retained.entries)),
                    }
                ),
            )
            self.retained.event("SEAL", "ALL", seal)
            self.sealed = True
            self.verify_seal()
            self.retained.stage = "TARGET"
            reports = {}
            for member in MEMBERS:
                target = self.target(member)
                if type(target) is not VisibleRaster or target.sequence_index != 2:
                    raise ValueError("target chronology differs")
                LIMITS.check(1, target.segmentation.shape, len(target.identities))
                sha = self.retained.write(
                    f"target-{member}.json",
                    encode(
                        {
                            "index": 2,
                            "shape": target.segmentation.shape,
                            "identities": target.identities,
                            "labels": target.segmentation.tolist(),
                        }
                    ),
                )
                self.retained.event("TARGET", member, sha)
                reports[member] = {
                    rule: evaluate(
                        sources[member],
                        data,
                        RetainedProvider(
                            (
                                prefixes["RH" if member in ("R", "H") else member][0],
                                prefixes["RH" if member in ("R", "H") else member][1],
                                target,
                            )
                        ),
                    )
                    for (m, rule), data in saved.items()
                    if m == member
                }
                current = dict(sources[member].frames[-1].masks)
                for report in reports[member].values():
                    report["current_strata"] = {
                        name: [
                            row for row in report["tokens"] if (row["token"] in current) == visible
                        ]
                        for name, visible in (("current_visible", True), ("current_absent", False))
                    }
                    report["reappearance"] = [
                        {
                            "token": row["token"],
                            "reappeared": row["token"] not in current and row["target_visible"],
                        }
                        for row in report["tokens"]
                    ]
                pending.remove(member)
            self.retained.event(
                "COMPLETE", "ALL", self.retained.write("reports.json", encode(reports))
            )
            return reports
        except Exception as error:
            self.sealed = False
            self.retained.fail(pending, error)
            raise


def inspect(root: Path) -> dict[str, Any]:
    """Synthetic or private local inspection; never a public item-level receipt."""
    reject_reparse(root)
    files = sorted(root.iterdir())
    for p in files:
        reject_reparse(p)
    if any(not p.is_file() or p.is_symlink() for p in files):
        raise ValueError("flat regular retained files required")
    size = sum(p.stat().st_size for p in files)
    roots = {
        p.name: digest(p.read_bytes())
        for p in files
        if p.name not in {"runtime-metadata.json", "failure-context.json"}
    }
    if size > TOTAL_BYTES:
        raise ValueError("retention exceeds total")
    events: list[dict[str, Any]] = []
    for i in range(len([p for p in files if p.name.startswith("event-")])):
        e = _json((root / f"event-{i}.json").read_bytes())
        if (
            set(e) != {"index", "kind", "member", "binding", "previous"}
            or type(e["index"]) is not int
            or e["index"] != i
        ):
            raise ValueError("chronology fields differ")
        if e["previous"] != (digest(encode(events[-1])) if events else None):
            raise ValueError("chronology chain differs")
        if e["kind"] == "TARGET" and not any(v["kind"] == "SEAL" for v in events):
            raise ValueError("unsealed target")
        if e["kind"] not in {
            "LOCK",
            "PREFIX",
            "INPUT",
            "FORECAST",
            "SEAL",
            "TARGET",
            "COMPLETE",
        } or e["member"] not in {*MEMBERS, "RH", "ALL"}:
            raise ValueError("closed event kind/member differs")
        member, kind = e["member"], e["kind"]
        seen = sum(v["kind"] == kind and v["member"] == member for v in events)
        expected = {"LOCK": "plan.json", "SEAL": "seal.json", "COMPLETE": "reports.json"}.get(kind)
        if kind == "PREFIX":
            if member not in UNITS or seen >= 2:
                raise ValueError("prefix membership differs")
            expected = f"prefix-{member}-{seen}.json"
        elif kind == "INPUT":
            if member not in MEMBERS or seen:
                raise ValueError("input membership differs")
            expected = f"input-{member}.json"
        elif kind == "FORECAST":
            if member not in MEMBERS or seen >= 2:
                raise ValueError("forecast membership differs")
            expected = f"forecast-{member}-{seen + 1}.json"
        elif kind == "TARGET":
            if member not in MEMBERS or seen:
                raise ValueError("target membership differs")
            expected = f"target-{member}.json"
        elif seen or member != "ALL":
            raise ValueError("global event membership differs")
        if expected is None or roots.get(expected) != e["binding"]:
            raise ValueError("bound retained artifact missing or changed")
        events.append(e)
    if any(e["kind"] == "SEAL" for e in events):
        plan = _json((root / "plan.json").read_bytes())
        from fractions import Fraction

        commands: dict[str, tuple[tuple[Command, ...], Command]] = {}
        for m in MEMBERS:
            a = plan["actions"][m]

            def command(values: list[str]) -> Command:
                if len(values) != 3 or any(type(v) is not str for v in values):
                    raise ValueError("strict planned action required")
                return Fraction(values[0]), Fraction(values[1]), Fraction(values[2])

            commands[m] = (tuple(command(c) for c in a["executed"]), command(a["announced"]))

        def read(name: str) -> bytes:
            path = root / name
            reject_reparse(path)
            if not path.is_file() or path.stat().st_size > TOTAL_BYTES:
                raise ValueError("sealed artifact missing/oversized")
            return path.read_bytes()

        validate_seal(read, events, plan, commands)
    return {
        "status": "FAILED_CLOSED"
        if (root / "failure.json").exists()
        else "COMPLETE"
        if events and events[-1]["kind"] == "COMPLETE"
        else "PENDING",
        "retained_bytes": size,
        "files": len(files),
        "events": len(events),
        "root": digest(canonical_json_bytes(roots)),
    }


def validate_seal(
    read: Callable[[str], bytes],
    events: list[dict[str, Any]],
    plan: dict[str, Any],
    commands: dict[str, tuple[tuple[Command, ...], Command]],
) -> None:
    data = read("seal.json")
    seal_events = [e for e in events if e["kind"] == "SEAL"]
    if len(seal_events) != 1 or digest(data) != seal_events[0]["binding"]:
        raise PermissionError("seal absent or changed before target fetch")
    p = _json(data)
    if (
        type(p) is not dict
        or set(p) != {"version", "members", "bindings", "plan", "chronology"}
        or p["version"] != VERSION
        or p["members"] != list(MEMBERS)
    ):
        raise ValueError("closed seal fields/membership differ")
    if p["plan"] != digest(read("plan.json")) or p["plan"] != digest(encode(plan)):
        raise ValueError("source plan changed")
    before = events[: seal_events[0]["index"]]
    if {
        kind: sum(e["kind"] == kind for e in before)
        for kind in ("LOCK", "PREFIX", "INPUT", "FORECAST")
    } != {"LOCK": 1, "PREFIX": 6, "INPUT": 4, "FORECAST": 8}:
        raise ValueError("complete lock/prefix/input/forecast chronology required")
    if p["chronology"] != digest(encode(before)):
        raise ValueError("seal chronology differs")
    if type(p["bindings"]) is not list or len(p["bindings"]) != 8:
        raise ValueError("eight retained forecasts required")
    seen = set()
    for b in p["bindings"]:
        if set(b) != {
            "member",
            "rule",
            "window",
            "input",
            "forecast",
            "inventory",
            "storage",
            "flat_input",
        }:
            raise ValueError("closed forecast binding fields required")
        m, rule, w = b["member"], b["rule"], b["window"]
        if type(m) is not str or type(rule) is not str or type(w) is not int:
            raise ValueError("strict binding scalar types required")
        if (
            m not in MEMBERS
            or (rule, w) not in (("current-mask-persistence", 1), ("k-frame-agreement", 2))
            or (m, rule) in seen
        ):
            raise ValueError("forecast membership differs")
        seen.add((m, rule))
        source = CausalInput.from_bytes(
            read(f"input-{m}.json"),
            ModalityPermissionSet(allowed=REQUIRED),
            LIMITS,
        )
        if (
            (source.executed, source.announced) != commands[m]
            or source.digest != b["input"]
            or list(source.inventory) != b["inventory"]
        ):
            raise ValueError("retained input/action/inventory differs")
        flat_data = read(f"flat-{m}.bin")
        flat = unflatten(flat_data, source)
        if (
            digest(flat_data) != b["flat_input"]
            or flat.canonical_bytes() != source.canonical_bytes()
        ):
            raise ValueError("lossless flat binding differs")
        saved = read(f"forecast-{m}-{w}.json")
        f = Forecast.from_bytes(saved, source)
        if (
            f.rule != rule
            or f.window != w
            or digest(saved) != b["forecast"]
            or storage(source, saved) != b["storage"]
        ):
            raise ValueError("retained forecast differs")
