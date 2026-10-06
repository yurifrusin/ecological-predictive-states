"""Private write-once evidence, qualification SealA and later forecast SealB.

Capabilities assume isolated author/evaluator processes and private archive ACLs;
Python objects/filenames cannot defend against a hostile same-process caller.
"""

from __future__ import annotations

import os
import re
import stat
import time
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

from epsbench.diagnostics.restricted_learning_contract import (
    ARCHITECTURE,
    INITIALIZATIONS,
    REQUIRED,
    ROSTER,
    VERSION,
    Forecast,
    InputEvidence,
    digest,
    sha,
)
from epsbench.diagnostics.restricted_learning_membership import (
    ANNOUNCED,
    BOOTSTRAP_DOMAIN,
    PREFIXES,
    MembershipLock,
    decisions,
)
from epsbench.diagnostics.restricted_learning_sampling import commitment
from epsbench.diagnostics.restricted_learning_scoring import GeometryScore, Score, score
from epsbench.diagnostics.visible_forecast_contract import _json, _keys
from epsbench.schema import ModalityPermissionSet
from epsbench.utils.canonical import canonical_json_bytes


def reject_reparse(path: Path) -> None:
    for candidate in (path, *path.parents):
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            continue
        if candidate.is_symlink() or getattr(info, "st_file_attributes", 0) & getattr(
            stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024
        ):
            raise ValueError("symlink/junction/reparse archive denied")


def _control_rule(
    condition: str,
    budget: int,
    lock: MembershipLock,
    inputs: dict[str, InputEvidence],
    targets: dict[str, QualifiedTarget],
) -> tuple[str, dict[tuple[tuple[str, ...], bool], float]]:
    if (
        type(lock) is not MembershipLock
        or condition not in ("persistence", "absent", "visible", "half", "train-frequency")
        or budget not in (16, 64)
    ):
        raise ValueError("actual declared cheap-control scope required")
    geometries = lock.train[:16] if budget == 16 else lock.train
    train_root = None
    if condition == "train-frequency":
        train_root = digest(
            canonical_json_bytes(
                [inputs[d].digest for g in geometries for d in decisions(g.identity)]
            )
        )
    rule = {
        "version": VERSION,
        "condition": condition,
        "train_data": train_root if condition == "train-frequency" else None,
        "smoothing": "Laplace(1,1)" if condition == "train-frequency" else None,
    }
    frequencies: dict[tuple[tuple[str, ...], bool], list[int]] = {}
    if condition == "train-frequency":
        for g in geometries:
            for d in decisions(g.identity):
                source = inputs[d]
                visible = {h for h, _ in source.prefix.frames[-1].masks}
                command_class = tuple(str(v) for v in source.prefix.announced)
                for h, truth in targets[d].truth:
                    pair = (command_class, h in visible)
                    count = frequencies.setdefault(pair, [0, 0])
                    count[0] += int(truth)
                    count[1] += 1
    return digest(canonical_json_bytes(rule)), {
        k: (v[0] + 1) / (v[1] + 2) for k, v in frequencies.items()
    }


def _check_control(
    forecast: Forecast,
    source: InputEvidence,
    rules: dict[tuple[str, int], tuple[str, dict[tuple[tuple[str, ...], bool], float]]],
) -> None:
    if forecast.fit.condition not in (
        "persistence",
        "absent",
        "visible",
        "half",
        "train-frequency",
    ):
        return
    root, frequencies = rules[(forecast.fit.condition, forecast.fit.budget)]
    if forecast.checkpoint_sha256 != root:
        raise ValueError("cheap-control canonical rule/training-data binding required")
    visible = {h for h, _ in source.prefix.frames[-1].masks}
    for h, value in forecast.channels:
        expected = {
            "persistence": float(h in visible),
            "absent": 0.0,
            "visible": 1.0,
            "half": 0.5,
            "train-frequency": frequencies.get(
                (tuple(str(v) for v in source.prefix.announced), h in visible), 0.5
            ),
        }[forecast.fit.condition]
        if value is not None and value != expected:
            raise ValueError("cheap control differs from fixed train-only rule")


class Archive:
    """Bounded payloads/deadline, not a hostwide resource guarantee or supervisor."""

    def __init__(
        self,
        root: Path,
        max_bytes: int = 1024**3,
        seconds: float = 3600.0,
        clock: Callable[[], float] = time.monotonic,
        failure_reserve: int = 32768,
    ) -> None:
        if (
            root.exists()
            or type(max_bytes) is not int
            or max_bytes < 65536
            or seconds <= 0
            or type(failure_reserve) is not int
            or not 32768 <= failure_reserve < max_bytes
        ):
            raise ValueError("fresh bounded private archive required")
        reject_reparse(root)
        root.mkdir(parents=True)
        self.root, self.max_bytes, self.seconds, self.clock = root, max_bytes, seconds, clock
        self.failure_reserve = failure_reserve
        self.start, self.used, self.failed = clock(), 0, False
        self.events: list[dict[str, Any]] = []

    def check(self) -> None:
        if self.failed or self.clock() - self.start >= self.seconds:
            raise RuntimeError("terminal failure/cooperative deadline")

    def write(self, name: str, data: bytes, failure: bool = False) -> str:
        if not failure:
            self.check()
        if re.fullmatch(r"[a-zA-Z0-9_-]+[.](json|bin)", name) is None:
            raise ValueError("flat closed retained path required")
        cap = self.max_bytes if failure else self.max_bytes - self.failure_reserve
        if type(data) is not bytes or len(data) > 4 * 1024 * 1024 or self.used + len(data) > cap:
            raise RuntimeError("payload/retained byte bound")
        path = self.root / name
        reject_reparse(path)
        self.used += len(data)  # Includes partial/failed writes.
        with path.open("xb") as f:
            if f.write(data) != len(data):
                raise OSError("partial write")
            f.flush()
            os.fsync(f.fileno())
        return digest(data)

    def event(self, kind: str, name: str, data: bytes) -> str:
        binding = self.write(name, data)
        record = {
            "index": len(self.events),
            "kind": kind,
            "path": name,
            "sha256": binding,
            "previous": digest(canonical_json_bytes(self.events[-1])) if self.events else None,
        }
        self.write(f"event-{len(self.events)}.json", canonical_json_bytes(record))
        self.events.append(record)
        return binding

    def read(self, name: str) -> bytes:
        if re.fullmatch(r"[a-zA-Z0-9_-]+[.](json|bin)", name) is None:
            raise ValueError("closed path required")
        path = self.root / name
        reject_reparse(path)
        if not path.is_file() or path.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("bounded regular artifact required")
        return path.read_bytes()

    def fail(self, pending: tuple[str, ...], error: Exception) -> None:
        self.failed = True
        kinds = [e["kind"] for e in self.events]
        phase = (
            "SCORING" if "SEAL_B" in kinds else "FORECAST" if "SEAL_A" in kinds else "QUALIFICATION"
        )
        membership = self.read("membership-lock.json")
        version = _json(membership)["version"]
        data = canonical_json_bytes(
            {
                "version": version,
                "disposition": "INCONCLUSIVE",
                "status": "FAILED_CLOSED",
                "pending": list(pending),
                "failure_kind": "PERMISSION"
                if isinstance(error, PermissionError)
                else "VALUE"
                if isinstance(error, ValueError)
                else "IO"
                if isinstance(error, OSError)
                else "LIMIT_OR_CLOSED"
                if isinstance(error, RuntimeError)
                else "OTHER",
                "last_event": digest(canonical_json_bytes(self.events[-1]))
                if self.events
                else None,
                "used_bytes": self.used,
                "phase": phase,
                "membership": digest(membership),
            }
        )
        binding = self.write("failure.json", data, True)
        record = {
            "index": len(self.events),
            "kind": "FAILURE",
            "path": "failure.json",
            "sha256": binding,
            "previous": digest(canonical_json_bytes(self.events[-1])) if self.events else None,
        }
        self.write(f"event-{len(self.events)}.json", canonical_json_bytes(record), True)
        self.events.append(record)

    def inspect(self) -> dict[str, Any]:
        used = 0
        records: list[dict[str, Any]] = []
        for i in range(len(self.events)):
            data = self.read(f"event-{i}.json")
            p = _json(data)
            _keys(p, {"index", "kind", "path", "sha256", "previous"})
            if (
                p != self.events[i]
                or p["index"] != i
                or p["previous"] != (digest(canonical_json_bytes(records[-1])) if records else None)
                or digest(self.read(p["path"])) != p["sha256"]
            ):
                raise ValueError("retained evidence/history changed")
            used += len(data) + len(self.read(p["path"]))
            records.append(p)
        if records and records[0]["kind"] == "MEMBERSHIP_LOCK":
            result = inspect_archive(self.root, allow_synthetic=True)
            if result["failed"] != self.failed:
                raise ValueError("live/durable failure state differs")
            return result
        return {
            "version": VERSION,
            "events": len(records),
            "validated_bytes": used,
            "failed": self.failed,
            "history_root": digest(canonical_json_bytes(records)),
        }


def _prefix_audit(data: bytes) -> None:
    annotation = _json(data)
    _keys(annotation, {"version", "audits", "decision_occlusion"})
    if (
        annotation["version"] != VERSION
        or type(annotation["audits"]) is not list
        or len(annotation["audits"]) != 3
    ):
        raise ValueError("all three independent prefix audits required")
    for audit in annotation["audits"]:
        _keys(audit, {"raw", "reference", "ties", "boundaries", "statuses"})
        if audit["raw"] != audit["reference"] or audit["ties"] != []:
            raise ValueError("prefix independent unique raster qualification failed")
        sha(audit["raw"])
    if annotation["decision_occlusion"] != "COMPLETE_IN_FRUSTUM_OCCLUSION":
        raise ValueError("previously observed background pure in-frame hiding required")


def _readiness(
    lock: MembershipLock, inputs: dict[str, InputEvidence], targets: dict[str, QualifiedTarget]
) -> None:
    if type(lock) is not MembershipLock:
        raise ValueError("actual lock required")
    classes = set()
    for g in lock.evaluation:
        absent = visible = 0
        for d in decisions(g.identity):
            source = inputs[d]
            currently_visible = {h for h, _ in source.prefix.frames[-1].masks}
            for h, y in targets[d].truth:
                if h in currently_visible:
                    visible += 1
                else:
                    absent += 1
                    classes.add(y)
        if not absent or not visible:
            raise ValueError("empty required geometry stratum; all32 retained INCONCLUSIVE")
    if classes != {False, True}:
        raise ValueError("both future classes required in current-absent evaluation inventory")


def inspect_archive(root: Path, allow_synthetic: bool = False) -> dict[str, Any]:
    """Reconstruct durable semantic bindings read-only; no resume or recollection.

    The returned archive_root includes terminal failure content. An independently
    retained immutable bundle receipt authenticates original bytes against coherent
    wholesale rewrites; this validator establishes semantics at those bytes.
    """
    reject_reparse(root)
    if not root.is_dir():
        raise ValueError("retained regular directory required")

    def read(name: str) -> bytes:
        if type(name) is not str or re.fullmatch(r"[a-zA-Z0-9_-]+[.](json|bin)", name) is None:
            raise ValueError("flat artifact name required")
        path = root / name
        reject_reparse(path)
        if not path.is_file() or path.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("bounded retained regular artifact required")
        return path.read_bytes()

    retained_bytes = 0
    for path in root.iterdir():
        reject_reparse(path)
        if not path.is_file() or re.fullmatch(r"[a-zA-Z0-9_-]+[.](json|bin)", path.name) is None:
            raise ValueError("flat bounded regular retained files required")
        if path.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("retained file bound including partial/orphan files")
        retained_bytes += path.stat().st_size
        if retained_bytes > 1024**3:
            raise ValueError("retained inclusive byte budget exceeded")
    names = {p.name for p in root.glob("event-*.json")}
    if not names or len(names) > 100000 or names != {f"event-{i}.json" for i in range(len(names))}:
        raise ValueError("complete consecutive bounded durable journal required")
    records: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    used = 0
    preceding_bytes: list[int] = []
    for i in range(len(names)):
        event = read(f"event-{i}.json")
        record = _json(event)
        _keys(record, {"index", "kind", "path", "sha256", "previous"})
        data = read(record["path"])
        sha(record["sha256"])
        if (
            type(record["index"]) is not int
            or record["index"] != i
            or record["previous"]
            != (digest(canonical_json_bytes(records[-1])) if records else None)
            or digest(data) != record["sha256"]
            or canonical_json_bytes(record) != event
        ):
            raise ValueError("durable artifact or history changed")
        if record["path"] in payloads:
            raise ValueError("duplicate durable artifact event")
        payloads[record["path"]] = data
        preceding_bytes.append(used)
        used += len(event) + len(data)
        records.append(record)
    if records[0]["kind"] != "MEMBERSHIP_LOCK" or records[0]["path"] != "membership-lock.json":
        raise ValueError("precollection membership event required")
    membership = payloads["membership-lock.json"]
    synthetic = _json(membership).get("version") == "SYNTHETIC_SOURCE_ONLY"
    lock: MembershipLock | SyntheticLock
    if synthetic:
        if not allow_synthetic or membership != SyntheticLock().canonical_bytes():
            raise PermissionError("explicit fixed source-smoke inspection only")
        lock = SyntheticLock()
    else:
        lock = MembershipLock.from_bytes(membership)
    decision_roster = lock.decision_roster
    forecast_roster = lock.forecast_roster
    decision_set = set(decision_roster)
    forecast_set = set(forecast_roster)
    version = "SYNTHETIC_SOURCE_ONLY" if synthetic else VERSION
    inputs: dict[str, InputEvidence] = {}
    input_by_digest: dict[str, str] = {}
    targets: dict[str, QualifiedTarget] = {}
    prefix_audits: set[str] = set()
    forecasts: dict[str, Forecast] = {}
    bindings: dict[str, list[str]] = {}
    seal_a = seal_b = failure_sha = None
    report_seen = scoring_seen = False
    control_rules: dict[tuple[str, int], tuple[str, dict[tuple[tuple[str, ...], bool], float]]] = {}
    phase = "QUALIFICATION"
    fixed = {
        "MEMBERSHIP_LOCK": "membership-lock.json",
        "COLLECTION_AUTHORITY": "collection-authority.json",
        "PRIVATE_IDENTITIES": "identity-private.json",
        "SEAL_A": "seal-a.json",
        "SEAL_B": "seal-b.json",
        "SCORING_ACCESS": "scoring-access.json",
        "REPORT": "report.json",
        "FAILURE": "failure.json",
    }
    private = {
        "RAW_PRIVATE": r"raw-[0-9]+[.]bin",
        "REFERENCE_PRIVATE": r"reference-[0-9]+[.]json",
        "AUDIT_PRIVATE": r"audit-[0-9]+[.]json",
    }
    for index, record in enumerate(records):
        kind, path = record["kind"], record["path"]
        data = payloads[path]
        if kind in fixed and path != fixed[kind]:
            raise ValueError("closed phase artifact path differs")
        if failure_sha is not None:
            raise ValueError("terminal failure forbids subsequent events")
        if kind == "MEMBERSHIP_LOCK":
            if index != 0:
                raise ValueError("duplicate membership lock")
        elif kind in ("COLLECTION_AUTHORITY", "PRIVATE_IDENTITIES"):
            if phase != "QUALIFICATION" or inputs or synthetic:
                raise ValueError("collection instrumentation outside initial actual phase")
        elif kind in private:
            if phase != "QUALIFICATION" or synthetic or re.fullmatch(private[kind], path) is None:
                raise ValueError("private qualification path/phase differs")
        elif kind in ("INPUT", "PREFIX_AUDIT", "QUALIFIED_TARGET"):
            prefix = {
                "INPUT": "input",
                "PREFIX_AUDIT": "prefix-audit",
                "QUALIFIED_TARGET": "target",
            }[kind]
            match = re.fullmatch(prefix + r"-([0-9]+)[.]json", path)
            if phase != "QUALIFICATION" or match is None:
                raise ValueError("qualification path/phase differs")
            ordinal = int(match[1])
            if ordinal >= len(decision_roster):
                raise ValueError("extra qualification member")
            decision = decision_roster[ordinal]
            if kind == "INPUT":
                if ordinal != len(inputs):
                    raise ValueError("missing/duplicate input ordinal")
                source = InputEvidence.from_bytes(data, ModalityPermissionSet(allowed=REQUIRED))
                if data != source.canonical_bytes():
                    raise ValueError("noncanonical retained input")
                if not synthetic:
                    _, prefix_name, action_name = decision.split("/")
                    poses = PREFIXES[int(prefix_name.removeprefix("prefix-"))]
                    executed = tuple(
                        (Fraction(0), poses[j + 1] - poses[j], Fraction(0)) for j in range(2)
                    )
                    announced = (
                        Fraction(0),
                        ANNOUNCED[int(action_name.removeprefix("action-"))],
                        Fraction(0),
                    )
                    if source.prefix.executed != executed or source.prefix.announced != announced:
                        raise ValueError("committed prefix/action chronology differs")
                input_root = source.digest
                if input_root in input_by_digest:
                    raise ValueError("distinct episode input identity required")
                input_by_digest[input_root] = decision
                inputs[decision] = source
            elif kind == "PREFIX_AUDIT":
                if decision not in inputs or decision in prefix_audits:
                    raise ValueError("audit before input or duplicate audit")
                _prefix_audit(data)
                prefix_audits.add(decision)
            else:
                if decision not in prefix_audits or decision in targets:
                    raise ValueError("target before frozen audited input or duplicate target")
                target = QualifiedTarget.from_bytes(data)
                if {h for h, _ in target.truth} != set(inputs[decision].prefix.inventory):
                    raise ValueError("retained target observed inventory differs")
                if (
                    not synthetic
                    and _json(target.annotations)["descriptor"] != decision.split("/")[0]
                ):
                    raise ValueError("retained target geometry binding differs")
                targets[decision] = target
        elif kind == "SEAL_A":
            if (
                phase != "QUALIFICATION"
                or set(inputs) != decision_set
                or set(targets) != decision_set
            ):
                raise ValueError("complete qualification required before single SealA")
            if type(lock) is MembershipLock:
                _readiness(lock, inputs, targets)
            a = _json(data)
            _keys(
                a,
                {
                    "version",
                    "architecture",
                    "membership",
                    "qualification_history",
                    "inputs",
                    "targets",
                    "readiness",
                },
            )
            expected: dict[str, Any] = {
                "version": version,
                "architecture": ARCHITECTURE,
                "membership": lock.digest,
                "qualification_history": digest(canonical_json_bytes(records[:index])),
                "inputs": {d: v.digest for d, v in inputs.items()},
                "targets": {d: digest(v.canonical_bytes()) for d, v in targets.items()},
                "readiness": "SOURCE_SMOKE_ONLY" if synthetic else "QUALIFIED",
            }
            if a != expected or data != canonical_json_bytes(expected):
                raise ValueError("complete immutable SealA semantic bindings differ")
            seal_a, phase = digest(data), "FORECAST"
            if type(lock) is MembershipLock:
                control_rules = {
                    (c, b): _control_rule(c, b, lock, inputs, targets)
                    for b in (16, 64)
                    for c in ("persistence", "absent", "visible", "half", "train-frequency")
                }
        elif kind == "FORECAST":
            if phase != "FORECAST" or path != f"forecast-{len(forecasts)}.json":
                raise ValueError("forecast ordinal/phase differs")
            p = _json(data)
            forecast_input = p.get("input")
            if type(forecast_input) is not str:
                raise ValueError("strict forecast input identity required")
            matched_decision = input_by_digest.get(forecast_input)
            if matched_decision is None:
                raise ValueError("unique retained input binding required")
            decision = matched_decision
            forecast = Forecast.from_bytes(data, inputs[decision])
            if not synthetic:
                _check_control(forecast, inputs[decision], control_rules)
            key = forecast.fit.key + "/" + decision
            if key not in forecast_set or key in forecasts or data != forecast.canonical_bytes():
                raise ValueError("missing/extra/duplicate/noncanonical forecast binding")
            pair = [forecast.checkpoint_sha256, forecast.run_sha256]
            if forecast.fit.key in bindings and bindings[forecast.fit.key] != pair:
                raise ValueError("retained fit checkpoint/run differs")
            bindings[forecast.fit.key] = pair
            forecasts[key] = forecast
        elif kind == "SEAL_B":
            if phase != "FORECAST" or set(forecasts) != forecast_set:
                raise ValueError("all declared forecasts required before single SealB")
            b = _json(data)
            _keys(b, {"version", "seal_a", "membership", "forecasts", "bindings", "roster"})
            expected = {
                "version": version,
                "seal_a": seal_a,
                "membership": lock.digest,
                "forecasts": {k: digest(v.canonical_bytes()) for k, v in forecasts.items()},
                "bindings": bindings,
                "roster": list(forecast_roster),
            }
            if b != expected or data != canonical_json_bytes(expected):
                raise ValueError("complete immutable SealB semantic bindings differ")
            seal_b, phase = digest(data), "SCORING"
        elif kind == "SCORING_ACCESS":
            if phase != "SCORING" or scoring_seen or synthetic:
                raise ValueError("single actual scoring access after SealB required")
            a = _json(data)
            _keys(a, {"seal_a", "seal_b", "resource_receipt"})
            sha(a["resource_receipt"])
            if a["seal_a"] != seal_a or a["seal_b"] != seal_b:
                raise ValueError("scoring seal binding differs")
            scoring_seen = True
        elif kind == "REPORT":
            if not scoring_seen or report_seen:
                raise ValueError("single report after scoring access required")
            report_seen = True
        elif kind == "FAILURE":
            f = _json(data)
            _keys(
                f,
                {
                    "version",
                    "disposition",
                    "status",
                    "pending",
                    "failure_kind",
                    "last_event",
                    "used_bytes",
                    "phase",
                    "membership",
                },
            )
            roster = decision_roster if phase == "QUALIFICATION" else forecast_roster
            done = (
                set(targets)
                if phase == "QUALIFICATION"
                else set(forecasts)
                if phase == "FORECAST"
                else set()
            )
            pending = [k for k in roster if k not in done]
            if (
                f["version"] != version
                or f["disposition"] != "INCONCLUSIVE"
                or f["status"] != "FAILED_CLOSED"
                or (
                    type(f["pending"]) is not list
                    or f["pending"] != pending
                    or f["phase"] != phase
                    or f["membership"] != lock.digest
                    or f["last_event"] != record["previous"]
                    or type(f["used_bytes"]) is not int
                    or not preceding_bytes[index] <= f["used_bytes"] <= 1024**3
                    or f["failure_kind"]
                    not in ("VALUE", "PERMISSION", "IO", "LIMIT_OR_CLOSED", "OTHER")
                )
                or data != canonical_json_bytes(f)
            ):
                raise ValueError("closed terminal failure schema/chronology/coverage differs")
            failure_sha = digest(data)
        else:
            raise ValueError("unknown durable phase event kind")
    if (root / "failure.json").exists() != (failure_sha is not None):
        raise ValueError("orphan/unbound failure record cannot establish failure evidence")
    history = digest(canonical_json_bytes(records))
    return {
        "version": version,
        "events": len(records),
        "validated_bytes": used,
        "history_root": history,
        "failure_sha256": failure_sha,
        "archive_root": digest(canonical_json_bytes({"history": history, "failure": failure_sha})),
        "failed": failure_sha is not None,
        "seal_a": seal_a,
        "seal_b": seal_b,
    }


@dataclass(frozen=True)
class QualifiedTarget:
    truth: tuple[tuple[str, bool], ...]
    annotations: bytes

    def __post_init__(self) -> None:
        if (
            type(self.truth) is not tuple
            or any(type(pair) is not tuple or len(pair) != 2 for pair in self.truth)
            or any(type(y) is not bool for _, y in self.truth)
            or (len({h for h, _ in self.truth}) != len(self.truth))
        ):
            raise ValueError("typed qualified truth required")
        p = _json(self.annotations)
        _keys(p, {"version", "descriptor", "raw", "reference", "ties", "boundaries", "statuses"})
        if p["version"] != VERSION or p["raw"] != p["reference"] or p["ties"]:
            raise ValueError("unique independent exact reference agreement required")
        VisibilityLabels(self.truth)
        sha(p["descriptor"])
        sha(p["raw"])
        # Clipping/boundaries and support UNKNOWN_DOMAIN are retained, not discarded.

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(
            {"truth": dict(self.truth), "annotations": _json(self.annotations)}
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> QualifiedTarget:
        p = _json(data)
        _keys(p, {"truth", "annotations"})
        if type(p["truth"]) is not dict:
            raise ValueError("closed retained truth map required")
        result = cls(tuple(p["truth"].items()), canonical_json_bytes(p["annotations"]))
        if data != result.canonical_bytes():
            raise ValueError("canonical retained target required")
        return result


@dataclass(frozen=True)
class VisibilityLabels:
    channels: tuple[tuple[str, bool], ...]

    def __post_init__(self) -> None:
        from epsbench.diagnostics.visible_forecast_contract import token

        if (
            type(self.channels) is not tuple
            or any(type(pair) is not tuple or len(pair) != 2 for pair in self.channels)
            or len({h for h, _ in self.channels}) != len(self.channels)
        ):
            raise ValueError("exact immutable unique label channels required")
        for h, y in self.channels:
            token(h)
            if type(y) is not bool:
                raise ValueError("visibility labels only")


class QualificationProvider(Protocol):
    def prefix(self, decision: str) -> tuple[InputEvidence, bytes]: ...
    def target(self, decision: str) -> QualifiedTarget: ...


@dataclass(frozen=True)
class SyntheticLock:
    """Tiny symbolic fixture; never an actual MembershipLock or readiness claim."""

    decision_roster: tuple[str, ...] = ("synthetic/source-only",)
    forecast_roster: tuple[str, ...] = ("16/init-0/relational/synthetic/source-only",)

    def __post_init__(self) -> None:
        if (
            self.decision_roster != ("synthetic/source-only",)
            or self.forecast_roster != ("16/init-0/relational/synthetic/source-only",)
            or type(self.decision_roster) is not tuple
            or type(self.forecast_roster) is not tuple
        ):
            raise ValueError("fixed symbolic source-smoke scope only")

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(
            {
                "version": "SYNTHETIC_SOURCE_ONLY",
                "decisions": list(self.decision_roster),
                "forecasts": list(self.forecast_roster),
            }
        )

    @property
    def digest(self) -> str:
        return digest(self.canonical_bytes())


@dataclass(frozen=True)
class ExpectedLaunch:
    model_source_head: str
    model_source_tree: str
    launch_sha256: str

    def __post_init__(self) -> None:
        sha(self.launch_sha256)
        for value in (self.model_source_head, self.model_source_tree):
            if (
                type(value) is not str
                or len(value) != 40
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError("separate exact model source identity required")


class Lifecycle:
    def __init__(self, lock: MembershipLock | SyntheticLock, archive: Archive) -> None:
        if type(lock) not in (MembershipLock, SyntheticLock):
            raise ValueError("typed precollection membership required")
        # Largest closed pending roster plus fixed failure fields and one event.
        required_reserve = max(
            32768,
            len(canonical_json_bytes(list(lock.forecast_roster))) + 4096,
            len(canonical_json_bytes(list(lock.decision_roster))) + 4096,
        )
        if archive.used + required_reserve >= archive.max_bytes:
            raise ValueError("complete bounded terminal failure reserve required before work")
        archive.failure_reserve = max(archive.failure_reserve, required_reserve)
        self.lock, self.archive = lock, archive
        self.started = False
        self.inputs: dict[str, InputEvidence] = {}
        self._targets: dict[str, QualifiedTarget] = {}
        self._forecasts: dict[str, Forecast] = {}
        self.seal_a: str | None = None
        self.seal_b: str | None = None
        self.archive.event("MEMBERSHIP_LOCK", "membership-lock.json", lock.canonical_bytes())

    def qualify(self, provider: QualificationProvider) -> str:
        if self.started:
            raise PermissionError("single qualification attempt; no rerun/replacement")
        self.started = True
        pending = list(self.lock.decision_roster)
        try:
            for i, decision in enumerate(self.lock.decision_roster):
                self.archive.check()
                source, prefix_audit = provider.prefix(decision)
                if type(source) is not InputEvidence:
                    raise ValueError("typed qualified prefix required")
                annotation = _json(prefix_audit)
                _keys(annotation, {"version", "audits", "decision_occlusion"})
                if annotation["version"] != VERSION or len(annotation["audits"]) != 3:
                    raise ValueError("all three independent prefix audits required")
                for audit in annotation["audits"]:
                    _keys(audit, {"raw", "reference", "ties", "boundaries", "statuses"})
                    if audit["raw"] != audit["reference"] or audit["ties"]:
                        raise ValueError("prefix independent unique raster qualification failed")
                    sha(audit["raw"])
                if annotation["decision_occlusion"] != "COMPLETE_IN_FRUSTUM_OCCLUSION":
                    raise ValueError("previously observed background pure in-frame hiding required")
                self.archive.event("INPUT", f"input-{i}.json", source.canonical_bytes())
                self.archive.event("PREFIX_AUDIT", f"prefix-audit-{i}.json", prefix_audit)
                self.inputs[decision] = source  # Freeze before its target callback.
                target = provider.target(decision)
                if type(target) is not QualifiedTarget or {h for h, _ in target.truth} != set(
                    source.prefix.inventory
                ):
                    raise ValueError("exact observed qualified target inventory required")
                self.archive.event("QUALIFIED_TARGET", f"target-{i}.json", target.canonical_bytes())
                self._targets[decision] = target
                pending.remove(decision)
            if type(self.lock) is MembershipLock:
                self._readiness()
            self.seal_a = self.archive.event(
                "SEAL_A",
                "seal-a.json",
                canonical_json_bytes(
                    {
                        "version": VERSION
                        if type(self.lock) is MembershipLock
                        else "SYNTHETIC_SOURCE_ONLY",
                        "architecture": ARCHITECTURE,
                        "membership": self.lock.digest,
                        "qualification_history": self.archive.inspect()["history_root"],
                        "inputs": {d: v.digest for d, v in self.inputs.items()},
                        "targets": {
                            d: digest(v.canonical_bytes()) for d, v in self._targets.items()
                        },
                        "readiness": "QUALIFIED"
                        if type(self.lock) is MembershipLock
                        else "SOURCE_SMOKE_ONLY",
                    }
                ),
            )
            return self.seal_a
        except Exception as e:
            try:
                self.archive.fail(tuple(pending), e)
            except Exception as preservation_error:
                e.add_note("failure preservation error: " + type(preservation_error).__name__)
            raise

    def _readiness(self) -> None:
        if type(self.lock) is not MembershipLock:
            raise ValueError("actual lock required")
        _readiness(self.lock, self.inputs, self._targets)

    def training_material(
        self, geometry: str
    ) -> tuple[tuple[InputEvidence, VisibilityLabels], ...]:
        if (
            type(self.lock) is not MembershipLock
            or not self.seal_a
            or self.lock.split(geometry) == "evaluation"
        ):
            raise PermissionError("qualified train/development only; evaluation denied")
        self.verify_a()
        return tuple(
            (self.inputs[d], VisibilityLabels(self._targets[d].truth)) for d in decisions(geometry)
        )

    def verify_a(self) -> None:
        self.archive.inspect()
        if not self.seal_a or digest(self.archive.read("seal-a.json")) != self.seal_a:
            raise PermissionError("actual immutable qualification SealA required")
        p = _json(self.archive.read("seal-a.json"))
        if (
            p["membership"] != self.lock.digest
            or p["inputs"] != {d: v.digest for d, v in self.inputs.items()}
            or (p["targets"] != {d: digest(v.canonical_bytes()) for d, v in self._targets.items()})
        ):
            raise ValueError("qualified retained content changed")

    def control_rule(
        self, condition: str, budget: int
    ) -> tuple[str, dict[tuple[tuple[str, ...], bool], float]]:
        if type(self.lock) is not MembershipLock:
            raise ValueError("actual declared cheap-control scope required")
        return _control_rule(condition, budget, self.lock, self.inputs, self._targets)

    def seal_forecasts(self, forecasts: tuple[tuple[str, Forecast], ...]) -> str:
        self.verify_a()
        if self.seal_b is not None or self.archive.failed:
            raise PermissionError("single SealB; no rerun/reseal")
        expected_roster = self.lock.forecast_roster
        pending = list(expected_roster)
        try:
            if (
                type(forecasts) is not tuple
                or len(forecasts) != len(expected_roster)
                or ({k for k, _ in forecasts} != set(expected_roster))
            ):
                raise ValueError(
                    "complete all24 learned fits and all scoped cheap controls required"
                )
            control_rules = (
                {
                    (condition, budget): self.control_rule(condition, budget)
                    for budget in (16, 64)
                    for condition in ("persistence", "absent", "visible", "half", "train-frequency")
                }
                if type(self.lock) is MembershipLock
                else {}
            )
            bindings: dict[str, tuple[str, str]] = {}
            roots = {}
            for i, (key, f) in enumerate(forecasts):
                d = key.removeprefix(f.fit.key + "/")
                if key != f.fit.key + "/" + d or d not in self.inputs:
                    raise ValueError("exact fit/decision binding required")
                f.validate_input(self.inputs[d])
                if type(self.lock) is MembershipLock:
                    _check_control(f, self.inputs[d], control_rules)
                pair = (f.checkpoint_sha256, f.run_sha256)
                if f.fit.key in bindings and bindings[f.fit.key] != pair:
                    raise ValueError("fit checkpoint/run changed within roster")
                bindings[f.fit.key] = pair
                roots[key] = self.archive.event(
                    "FORECAST", f"forecast-{i}.json", f.canonical_bytes()
                )
                self._forecasts[key] = f
                pending.remove(key)
            self.seal_b = self.archive.event(
                "SEAL_B",
                "seal-b.json",
                canonical_json_bytes(
                    {
                        "version": VERSION
                        if type(self.lock) is MembershipLock
                        else "SYNTHETIC_SOURCE_ONLY",
                        "seal_a": self.seal_a,
                        "membership": self.lock.digest,
                        "forecasts": roots,
                        "bindings": bindings,
                        "roster": list(expected_roster),
                    }
                ),
            )
            return self.seal_b
        except Exception as e:
            try:
                self.archive.fail(tuple(pending), e)
            except Exception as preservation_error:
                e.add_note("failure preservation error: " + type(preservation_error).__name__)
            raise

    def evaluate(
        self,
        bootstrap_seed: bytes,
        resource_receipt: bytes,
        expected_launch: ExpectedLaunch | None = None,
    ) -> dict[str, Any]:
        if type(self.lock) is not MembershipLock or not self.seal_b:
            raise PermissionError("actual all-roster SealB required; synthetic cannot evaluate")
        if self.archive.failed:
            raise PermissionError("failed terminal evidence")
        try:
            return self._evaluate(bootstrap_seed, resource_receipt, expected_launch)
        except Exception as error:
            try:
                self.archive.fail(self.lock.forecast_roster, error)
            except Exception as preservation_error:
                error.add_note("failure preservation error: " + type(preservation_error).__name__)
            raise

    def _evaluate(
        self, bootstrap_seed: bytes, resource_receipt: bytes, expected_launch: ExpectedLaunch | None
    ) -> dict[str, Any]:
        self.verify_a()
        if type(self.lock) is not MembershipLock or not self.seal_b:
            raise PermissionError("actual all-roster SealB required; synthetic cannot evaluate")
        self.archive.inspect()
        if digest(self.archive.read("seal-b.json")) != self.seal_b:
            raise ValueError("immutable SealB changed")
        b = _json(self.archive.read("seal-b.json"))
        if (
            b["roster"] != list(self.lock.forecast_roster)
            or b["forecasts"]
            != {k: digest(v.canonical_bytes()) for k, v in self._forecasts.items()}
            or b["seal_a"] != self.seal_a
            or b["membership"] != self.lock.digest
        ):
            raise ValueError("complete bound forecast seal required")
        if commitment(BOOTSTRAP_DOMAIN, bootstrap_seed) != self.lock.bootstrap_commitment:
            raise ValueError("bootstrap precommitment differs")
        if type(expected_launch) is not ExpectedLaunch:
            raise PermissionError("later separately accepted model/launch bindings required")
        receipt = _json(resource_receipt)
        _keys(
            receipt,
            {
                "architecture",
                "collection_source_head",
                "model_source_head",
                "model_source_tree",
                "launch",
                "fits",
                "provenance",
            },
        )
        if (
            receipt["architecture"] != ARCHITECTURE
            or receipt["collection_source_head"] != self.lock.source_head
            or (
                receipt["model_source_head"] != expected_launch.model_source_head
                or receipt["model_source_tree"] != expected_launch.model_source_tree
                or receipt["launch"] != expected_launch.launch_sha256
                or type(receipt["fits"]) is not dict
                or set(receipt["fits"]) != {f.key for f in ROSTER}
                or type(receipt["provenance"]) is not dict
            )
        ):
            raise ValueError("separate collection/model source and complete bound roster required")
        for fit in ROSTER:
            record = receipt["fits"][fit.key]
            _keys(record, {"checkpoint", "run", "measurement_root", "schedule_root"})
            for field in record.values():
                sha(field)
            if record["schedule_root"] != dict(self.lock.schedule_commitments)[fit.budget]:
                raise ValueError("committed training schedule differs")
            for g in self.lock.evaluation:
                for decision in decisions(g.identity):
                    forecast = self._forecasts[fit.key + "/" + decision]
                    if (forecast.checkpoint_sha256, forecast.run_sha256) != (
                        record["checkpoint"],
                        record["run"],
                    ):
                        raise ValueError(
                            "model/checkpoint/run receipt differs from immutable forecasts"
                        )
        # Measurement roots bind future evidence; they are deliberately NOT a fairness pass.
        self.archive.event(
            "SCORING_ACCESS",
            "scoring-access.json",
            canonical_json_bytes(
                {
                    "seal_a": self.seal_a,
                    "seal_b": self.seal_b,
                    "resource_receipt": digest(resource_receipt),
                }
            ),
        )
        by_fit: dict[str, dict[str, dict[str, Score]]] = {}
        for f in ROSTER:
            rows = {}
            for g in self.lock.evaluation:
                ds = decisions(g.identity)
                result = GeometryScore(
                    g.identity,
                    f.key,
                    tuple(
                        (
                            d,
                            score(
                                self._forecasts[f.key + "/" + d],
                                self.inputs[d],
                                self._targets[d].truth,
                            ),
                        )
                        for d in ds
                    ),
                ).aggregate(ds)
                rows[g.identity] = result
            by_fit[f.key] = rows
        retained_rows = {
            f: {
                g: {
                    s: {
                        "N": v.denominator,
                        "U": v.unknown,
                        "sum": v.known_sum,
                        "coverage": v.coverage,
                        "conditional": v.conditional,
                        "interval": v.interval,
                    }
                    for s, v in scores.items()
                }
                for g, scores in geometries.items()
            }
            for f, geometries in by_fit.items()
        }
        incomplete = any(
            v[s].risk is None
            for rows in by_fit.values()
            for v in rows.values()
            for s in ("all", "current-absent", "current-visible")
        )
        if incomplete:
            report = {
                "disposition": "INCONCLUSIVE",
                "reason": "COVERAGE_OR_EMPTY_STRATUM",
                "phase_gate_effect": "NONE",
                "rows": retained_rows,
            }
            self.archive.event("REPORT", "report.json", canonical_json_bytes(report))
            return report
        evaluation_geometries = self.lock.evaluation

        def risks(budget: int, condition: str, stratum: str) -> tuple[float, ...]:
            values = []
            for g in evaluation_geometries:
                r = [
                    by_fit[f"{budget}/{i}/{condition}"][g.identity][stratum].risk
                    for i in INITIALIZATIONS
                ]
                if any(v is None for v in r):
                    raise ValueError("missing paired initialization risk")
                values.append(sum(v for v in r if v is not None) / 3)
            return tuple(values)

        def difference(a: tuple[float, ...], c: tuple[float, ...]) -> tuple[float, ...]:
            return tuple(x - y for x, y in zip(a, c, strict=True))

        low = risks(16, "relational", "current-absent")
        report = {
            "disposition": "INCONCLUSIVE",
            "reason": "RESOURCE_CHECKER_PENDING_WP2",
            "phase_gate_effect": "NONE",
            "rows": retained_rows,
            "paired_geometry_endpoints": {
                "low_improvement": difference(risks(16, "dense", "current-absent"), low),
                "high_absent_penalty": difference(
                    risks(64, "relational", "current-absent"), risks(64, "dense", "current-absent")
                ),
                "high_visible_penalty": difference(
                    risks(64, "relational", "current-visible"),
                    risks(64, "dense", "current-visible"),
                ),
                "low_candidate": low,
                "control_improvements": {
                    c: difference(risks(16, c, "current-absent"), low)
                    for c in ("persistence", "absent", "visible", "half", "train-frequency")
                },
                "ablation_penalties": {
                    c: difference(risks(16, c, "current-absent"), low)
                    for c in ("action-zero", "memory-reset")
                },
            },
        }
        self.archive.event("REPORT", "report.json", canonical_json_bytes(report))
        return report
