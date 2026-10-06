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
from pathlib import Path
from typing import Any, Protocol

from epsbench.diagnostics.restricted_learning_contract import (
    ARCHITECTURE,
    INITIALIZATIONS,
    ROSTER,
    VERSION,
    Forecast,
    InputEvidence,
    digest,
    sha,
)
from epsbench.diagnostics.restricted_learning_membership import (
    BOOTSTRAP_DOMAIN,
    MembershipLock,
    decisions,
)
from epsbench.diagnostics.restricted_learning_sampling import commitment
from epsbench.diagnostics.restricted_learning_scoring import GeometryScore, Score, score
from epsbench.diagnostics.visible_forecast_contract import _json, _keys
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


class Archive:
    """Bounded payloads/deadline, not a hostwide resource guarantee or supervisor."""

    def __init__(
        self,
        root: Path,
        max_bytes: int = 1024**3,
        seconds: float = 3600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if root.exists() or type(max_bytes) is not int or max_bytes < 65536 or seconds <= 0:
            raise ValueError("fresh bounded private archive required")
        reject_reparse(root)
        root.mkdir(parents=True)
        self.root, self.max_bytes, self.seconds, self.clock = root, max_bytes, seconds, clock
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
        cap = self.max_bytes if failure else self.max_bytes - 32768
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
        self.write(
            "failure.json",
            canonical_json_bytes(
                {
                    "version": VERSION,
                    "disposition": "INCONCLUSIVE",
                    "status": "FAILED_CLOSED",
                    "pending": list(pending),
                    "failure_kind": type(error).__name__,
                    "last_event": digest(canonical_json_bytes(self.events[-1]))
                    if self.events
                    else None,
                    "used_bytes": self.used,
                }
            ),
            True,
        )

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
        return {
            "version": VERSION,
            "events": len(records),
            "validated_bytes": used,
            "failed": self.failed,
            "history_root": digest(canonical_json_bytes(records)),
        }


def inspect_archive(root: Path, allow_synthetic: bool = False) -> dict[str, Any]:
    """Read-only durable history verification; never resumes a producer or forecast."""
    reject_reparse(root)
    if not root.is_dir():
        raise ValueError("retained regular directory required")

    def read(name: str) -> bytes:
        if re.fullmatch(r"[a-zA-Z0-9_-]+[.](json|bin)", name) is None:
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
    event_names = {p.name for p in root.glob("event-*.json")}
    if (
        not event_names
        or len(event_names) > 100000
        or event_names != {f"event-{i}.json" for i in range(len(event_names))}
    ):
        raise ValueError("complete consecutive bounded durable journal required")
    records: list[dict[str, Any]] = []
    used = 0
    for i in range(len(event_names)):
        event = read(f"event-{i}.json")
        record = _json(event)
        _keys(record, {"index", "kind", "path", "sha256", "previous"})
        payload = read(record["path"])
        if (
            type(record["index"]) is not int
            or record["index"] != i
            or record["previous"]
            != (digest(canonical_json_bytes(records[-1])) if records else None)
            or digest(payload) != record["sha256"]
            or canonical_json_bytes(record) != event
        ):
            raise ValueError("durable artifact or history changed")
        used += len(event) + len(payload)
        if used > 1024**3:
            raise ValueError("durable payload budget exceeded")
        records.append(record)
    if records[0]["kind"] != "MEMBERSHIP_LOCK" or records[0]["path"] != "membership-lock.json":
        raise ValueError("precollection membership event required")
    membership = read("membership-lock.json")
    p = _json(membership)
    synthetic = p.get("version") == "SYNTHETIC_SOURCE_ONLY"
    if synthetic:
        if not allow_synthetic or membership != SyntheticLock().canonical_bytes():
            raise PermissionError("explicit source-smoke inspection only")
    else:
        MembershipLock.from_bytes(membership)
    kinds = [r["kind"] for r in records]
    if kinds.count("SEAL_A") > 1 or kinds.count("SEAL_B") > 1 or kinds.count("REPORT") > 1:
        raise ValueError("immutable single-use seals/report required")
    if "SEAL_A" in kinds:
        a_index = kinds.index("SEAL_A")
        a = _json(read("seal-a.json"))
        if (
            a["membership"] != digest(membership)
            or a["qualification_history"] != digest(canonical_json_bytes(records[:a_index]))
            or a["version"] != ("SYNTHETIC_SOURCE_ONLY" if synthetic else VERSION)
        ):
            raise ValueError("durable SealA differs from precommitted qualified history")
        if any(
            k
            in (
                "INPUT",
                "PREFIX_AUDIT",
                "QUALIFIED_TARGET",
                "RAW_PRIVATE",
                "REFERENCE_PRIVATE",
                "AUDIT_PRIVATE",
            )
            for k in kinds[a_index + 1 :]
        ):
            raise ValueError("qualification artifacts after SealA denied")
    if "SEAL_B" in kinds:
        if "SEAL_A" not in kinds or kinds.index("SEAL_B") <= kinds.index("SEAL_A"):
            raise ValueError("qualification must precede complete forecast SealB")
        b = _json(read("seal-b.json"))
        if (
            b["membership"] != digest(membership)
            or b["seal_a"] != digest(read("seal-a.json"))
            or (b["version"] != ("SYNTHETIC_SOURCE_ONLY" if synthetic else VERSION))
        ):
            raise ValueError("durable SealB binding differs")
        if any(k == "FORECAST" for k in kinds[kinds.index("SEAL_B") + 1 :]):
            raise ValueError("forecast after SealB denied")
    if "SCORING_ACCESS" in kinds and (
        "SEAL_B" not in kinds or kinds.index("SCORING_ACCESS") < kinds.index("SEAL_B")
    ):
        raise ValueError("scoring before global forecast seal denied")
    return {
        "version": "SYNTHETIC_SOURCE_ONLY" if synthetic else VERSION,
        "events": len(records),
        "validated_bytes": used,
        "history_root": digest(canonical_json_bytes(records)),
        "failed": (root / "failure.json").is_file(),
        "seal_a": digest(read("seal-a.json")) if "SEAL_A" in kinds else None,
        "seal_b": digest(read("seal-b.json")) if "SEAL_B" in kinds else None,
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
        sha(p["descriptor"])
        sha(p["raw"])
        # Clipping/boundaries and support UNKNOWN_DOMAIN are retained, not discarded.

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(
            {"truth": dict(self.truth), "annotations": _json(self.annotations)}
        )


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
        classes = set()
        for g in self.lock.evaluation:
            absent = visible = 0
            for d in decisions(g.identity):
                source = self.inputs[d]
                currently_visible = {h for h, _ in source.prefix.frames[-1].masks}
                for h, y in self._targets[d].truth:
                    if h in currently_visible:
                        visible += 1
                    else:
                        absent += 1
                        classes.add(y)
            if not absent or not visible:
                raise ValueError("empty required geometry stratum; all32 retained INCONCLUSIVE")
        if classes != {False, True}:
            raise ValueError("both future classes required in current-absent evaluation inventory")

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
        if (
            type(self.lock) is not MembershipLock
            or condition not in ("persistence", "absent", "visible", "half", "train-frequency")
            or budget not in (16, 64)
        ):
            raise ValueError("actual declared cheap-control scope required")
        geometries = self.lock.train[:16] if budget == 16 else self.lock.train
        train_root = None
        if condition == "train-frequency":
            train_root = digest(
                canonical_json_bytes(
                    [self.inputs[d].digest for g in geometries for d in decisions(g.identity)]
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
                    source = self.inputs[d]
                    visible = {h for h, _ in source.prefix.frames[-1].masks}
                    command_class = tuple(str(v) for v in source.prefix.announced)
                    for h, truth in self._targets[d].truth:
                        pair = (command_class, h in visible)
                        count = frequencies.setdefault(pair, [0, 0])
                        count[0] += int(truth)
                        count[1] += 1
        return digest(canonical_json_bytes(rule)), {
            k: (v[0] + 1) / (v[1] + 2) for k, v in frequencies.items()
        }

    def seal_forecasts(self, forecasts: tuple[tuple[str, Forecast], ...]) -> str:
        self.verify_a()
        if self.seal_b is not None or self.archive.failed:
            raise PermissionError("single SealB; no rerun/reseal")
        pending = self.lock.forecast_roster
        try:
            if (
                type(forecasts) is not tuple
                or len(forecasts) != len(pending)
                or ({k for k, _ in forecasts} != set(pending))
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
                if type(self.lock) is MembershipLock and f.fit.condition in (
                    "persistence",
                    "absent",
                    "visible",
                    "half",
                    "train-frequency",
                ):
                    root, frequency = control_rules[(f.fit.condition, f.fit.budget)]
                    if f.checkpoint_sha256 != root:
                        raise ValueError(
                            "cheap-control canonical rule/training-data binding required"
                        )
                    source = self.inputs[d]
                    visible = {h for h, _ in source.prefix.frames[-1].masks}
                    for h, value in f.channels:
                        expected = {
                            "persistence": float(h in visible),
                            "absent": 0.0,
                            "visible": 1.0,
                            "half": 0.5,
                            "train-frequency": frequency.get(
                                (tuple(str(v) for v in source.prefix.announced), h in visible), 0.5
                            ),
                        }[f.fit.condition]
                        if value is not None and value != expected:
                            raise ValueError("cheap control differs from fixed train-only rule")
                pair = (f.checkpoint_sha256, f.run_sha256)
                if f.fit.key in bindings and bindings[f.fit.key] != pair:
                    raise ValueError("fit checkpoint/run changed within roster")
                bindings[f.fit.key] = pair
                roots[key] = self.archive.event(
                    "FORECAST", f"forecast-{i}.json", f.canonical_bytes()
                )
                self._forecasts[key] = f
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
                        "roster": list(pending),
                    }
                ),
            )
            return self.seal_b
        except Exception as e:
            try:
                self.archive.fail(pending, e)
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
