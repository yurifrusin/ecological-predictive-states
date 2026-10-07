"""One fresh private finite scalar comparison; offline verification never produces frames."""

from __future__ import annotations

import json
import re
import secrets
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, is_dataclass
from fractions import Fraction as Q
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from epsbench.diagnostics.boundary_observation import VisibleRaster
from epsbench.diagnostics.causal_region_lifecycle import (
    REQUIRED,
    State,
    TrustedObservation,
    action_bytes,
    keys,
)
from epsbench.diagnostics.causal_region_lifecycle import advance as observe
from epsbench.diagnostics.fraction_extrapolation import (
    Input,
    TargetEvidence,
    binding,
    decode,
    evaluate,
    forecast,
)
from epsbench.diagnostics.restricted_exact_raster import Box, Calibration
from epsbench.schema import Action, ModalityPermissionSet
from epsbench.utils.canonical import canonical_json_bytes, sha256_bytes

VERSION = "fraction-finite-study-v1"
LIMIT = 16 * 1024 * 1024
REVIEW_RESERVE = 4 * 1024 * 1024
TERMINAL_RESERVE = 64 * 1024
PERMISSIONS = ModalityPermissionSet(allowed=REQUIRED)
INFORMATIVE = (
    "nonempty_known",
    "prefix_known_change",
    "targets_novel_to_both_prefixes",
    "known_action_contrast",
)
SOURCE_FILES = (
    "src/epsbench/diagnostics/fraction_finite_collector.py",
    "src/epsbench/diagnostics/fraction_extrapolation.py",
    "src/epsbench/diagnostics/causal_region_lifecycle.py",
    "src/epsbench/diagnostics/boundary_observation.py",
    "src/epsbench/diagnostics/restricted_exact_raster.py",
    "src/epsbench/diagnostics/occupancy_reference.py",
    "src/epsbench/schema.py",
    "src/epsbench/utils/canonical.py",
    "src/epsbench/annotations/derive.py",
    "src/epsbench/__init__.py",
    "src/epsbench/diagnostics/__init__.py",
    "src/epsbench/annotations/__init__.py",
    "src/epsbench/utils/__init__.py",
)


def validate_context(context: dict[str, Any]) -> None:
    keys(context, {"head", "tree", "manifest_sha256", "sources", "versions", "purpose"})
    if context["purpose"] != VERSION or type(context["versions"]) is not dict:
        raise ValueError("exact runtime purpose/versions required")
    for name, length in (("head", 40), ("tree", 40), ("manifest_sha256", 64)):
        if (
            type(context[name]) is not str
            or re.fullmatch(r"[0-9a-f]{" + str(length) + "}", context[name]) is None
        ):
            raise ValueError("exact runtime source/manifest identity required")
    sources = keys(context["sources"], set(SOURCE_FILES))
    if any(
        type(v) is not str or re.fullmatch(r"[0-9a-f]{64}", v) is None for v in sources.values()
    ):
        raise ValueError("complete source byte hashes required")


def verify_source(root: Path, context: dict[str, Any]) -> None:
    validate_context(context)
    for name, digest in context["sources"].items():
        if sha256_bytes((root / name).read_bytes()) != digest:
            raise ValueError("reviewed source bytes differ")
    observed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD", "HEAD^{tree}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout.splitlines()
    if observed != [context["head"], context["tree"]]:
        raise ValueError("reviewed source head/tree differs")


def rational(value: Any) -> Q:
    if type(value) is not str or len(value) > 64:
        raise ValueError("bounded canonical rational string required")
    q = Q(value)
    if str(q) != value:
        raise ValueError("reduced rational spelling required")
    return q


def plain(value: Any) -> Any:
    if type(value) is Q:
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def action(q: Q) -> Action:
    f = float(q)
    if Q(f) != q or not f:
        raise ValueError("nonzero exactly binary64-representable lateral command required")
    return Action(
        name="lateral_right" if f > 0 else "lateral_left",
        delta_forward=0.0,
        delta_lateral=f,
        delta_yaw=0.0,
    )


@dataclass(frozen=True)
class Unit:
    boxes: tuple[Box, Box]
    prefix: tuple[Q, Q]
    actions: tuple[Q, Q]


@dataclass(frozen=True)
class Study:
    payload: bytes
    proposal_sha256: str
    adoption_sha256: str
    units: tuple[Unit, Unit]
    extent: tuple[Q, Q]


def parse(payload: bytes) -> Study:
    if type(payload) is not bytes or len(payload) > 64 * 1024:
        raise ValueError("bounded exact manifest bytes required")
    p = keys(
        json.loads(payload),
        {
            "version",
            "proposal_sha256",
            "adoption_sha256",
            "calibration",
            "support",
            "units",
            "criterion",
            "limits",
        },
    )
    calibration = {
        "size": 32,
        "forward": "-3",
        "elevation": "1",
        "up_y": "0",
        "fovy": "90",
        "clipping": "NONE",
    }
    criterion = {
        "margin": "1/1024",
        "informative": list(INFORMATIVE),
        "precedence": "FAIL_PASS_INCONCLUSIVE",
    }
    limits = {
        "producer_calls": 8,
        "audits": 8,
        "seconds": 60,
        "inclusive_bytes": LIMIT,
        "review_reserve_bytes": REVIEW_RESERVE,
    }
    if (
        p["version"] != VERSION
        or p["calibration"] != calibration
        or p["criterion"] != criterion
        or p["limits"] != limits
    ):
        raise ValueError("fixed calibration, criterion and resource contract required")
    for name in ("proposal_sha256", "adoption_sha256"):
        value = p[name]
        if (
            type(value) is not str
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("definition/adoption SHA256 required")
    if (
        type(p["support"]) is not list
        or len(p["support"]) != 2
        or type(p["units"]) is not list
        or len(p["units"]) != 2
    ):
        raise ValueError("two units and support extents required")
    extent = cast(tuple[Q, Q], tuple(rational(x) for x in p["support"]))
    if not all(0 < x <= 32 for x in extent):
        raise ValueError("bounded positive support required")
    units = []
    for value in p["units"]:
        u = keys(value, {"boxes", "prefix", "actions"})
        if any(type(u[k]) is not list or len(u[k]) != 2 for k in ("boxes", "prefix", "actions")):
            raise ValueError("two boxes/prefix positions/announced commands required")
        boxes = []
        for value in u["boxes"]:
            coords = keys(value, {"lower", "upper"})
            if any(type(coords[k]) is not list or len(coords[k]) != 3 for k in coords):
                raise ValueError("three box coordinates required")
            boxes.append(
                Box(
                    cast(tuple[Q, Q, Q], tuple(rational(x) for x in coords["lower"])),
                    cast(tuple[Q, Q, Q], tuple(rational(x) for x in coords["upper"])),
                )
            )
        prefix, commands = (tuple(rational(x) for x in u[k]) for k in ("prefix", "actions"))
        action(prefix[1] - prefix[0])
        for q in commands:
            action(q)
        positions = (*prefix, *(prefix[1] + q for q in commands))
        if len(set(positions)) != 4 or commands[0] <= 0 or commands[1] >= 0:
            raise ValueError("distinct non-return ordered signed continuations required")
        if not boxes[0].upper[1] < boxes[1].lower[1]:
            raise ValueError("separated ordered positive-depth boxes required")
        for b in boxes:
            if b.lower[1] <= -3 or b.lower[2] <= 0:
                raise ValueError("boxes must be in front and strictly above support")
            # Conservative corner domain bound, without generating observations.
            for lateral in positions:
                for x in (b.lower[0], b.upper[0]):
                    for y in (b.lower[1], b.upper[1]):
                        for z in (b.lower[2], b.upper[2]):
                            if max(abs((x - lateral) / (y + 3)), abs((z - 1) / (y + 3))) > 8:
                                raise ValueError("reference footprint work domain exceeded")
        units.append(
            Unit(
                cast(tuple[Box, Box], tuple(boxes)),
                cast(tuple[Q, Q], prefix),
                cast(tuple[Q, Q], commands),
            )
        )
    if p["units"][0] == p["units"][1] or canonical_json_bytes(p) != payload:
        raise ValueError("distinct units and canonical complete manifest required")
    return Study(
        payload,
        p["proposal_sha256"],
        p["adoption_sha256"],
        cast(tuple[Unit, Unit], tuple(units)),
        extent,
    )


class Adapter(Protocol):
    def produce(
        self, unit: Unit, lateral: Q, extent: tuple[Q, Q], check: Callable[[], None]
    ) -> Any: ...
    def audit(
        self, unit: Unit, lateral: Q, extent: tuple[Q, Q], check: Callable[[], None]
    ) -> Any: ...


class ExactAdapter:
    def produce(
        self, unit: Unit, lateral: Q, extent: tuple[Q, Q], check: Callable[[], None]
    ) -> Any:
        from epsbench.diagnostics.restricted_exact_raster import raster

        return raster(lateral, unit.boxes, extent, check)

    def audit(self, unit: Unit, lateral: Q, extent: tuple[Q, Q], check: Callable[[], None]) -> Any:
        from epsbench.diagnostics.occupancy_reference import Solid, audit

        return audit(
            Calibration().origin(lateral),
            cast(tuple[Solid, Solid], tuple(Solid(b.lower, b.upper) for b in unit.boxes)),
            extent,
            32,
            check,
        )


class Sink:
    def __init__(self, path: Path, clock: Callable[[], float]) -> None:
        self.path, self.clock = path, clock
        self.started = clock()
        self.hashes: dict[str, str] = {}
        self.bytes = 0

    def check(self) -> None:
        if self.clock() - self.started > 60:
            raise TimeoutError("cooperative elapsed budget exceeded")

    def write(self, name: str, data: bytes, terminal: bool = False) -> None:
        if not terminal:
            self.check()
        reserve = 0 if terminal else TERMINAL_RESERVE
        if 2 * (self.bytes + len(data) + reserve) + REVIEW_RESERVE > LIMIT:
            raise OSError("inclusive originals/archive/review budget exceeded")
        with (self.path / name).open("xb") as f:
            f.write(data)
            f.flush()
        self.bytes += len(data)
        self.hashes[name] = sha256_bytes(data)

    def json(self, name: str, value: Any, terminal: bool = False) -> None:
        self.write(name, canonical_json_bytes(plain(value)), terminal)

    def verify(self) -> None:
        self.check()
        for name, digest in self.hashes.items():
            self.check()
            if sha256_bytes((self.path / name).read_bytes()) != digest:
                raise ValueError("retained bytes changed before dependent use")


def rows(value: Any) -> tuple[tuple[int, ...], ...]:
    if (
        type(value) is not list
        or len(value) != 32
        or any(type(row) is not list or len(row) != 32 for row in value)
    ):
        raise ValueError("complete 32x32 raster required")
    if any(type(v) is not int or not 0 <= v <= 3 for row in value for v in row):
        raise ValueError("closed controlled role labels required")
    return tuple(tuple(row) for row in value)


def qualified(raw: dict[str, Any], audit: dict[str, Any]) -> tuple[tuple[int, ...], ...]:
    keys(raw, {"labels", "ties"})
    keys(audit, {"labels", "ties", "boundaries", "supports", "footprints"})
    labels = rows(raw["labels"])
    if labels != rows(audit["labels"]) or raw["ties"] or audit["ties"] or audit["boundaries"]:
        raise ValueError("tie/boundary/disagreement denies truth")
    if (
        type(audit["supports"]) is not list
        or len(audit["supports"]) != 2
        or type(audit["footprints"]) is not list
        or len(audit["footprints"]) != 2
    ):
        raise ValueError("complete reference domain evidence required")
    for support, footprint in zip(audit["supports"], audit["footprints"], strict=True):
        if (
            type(support) is not list
            or len(support) != 3
            or support[2] is not False
            or any(type(x) is not int or x < 0 for x in support[:2])
            or support[1] > support[0]
        ):
            raise ValueError("sampled footprint boundary or support contradiction")
        keys(footprint, {"polygon", "depths", "domain"})
        if (
            footprint["domain"] is not True
            or len(footprint["polygon"]) < 3
            or len(footprint["depths"]) != 2
            or not 0 < rational(footprint["depths"][0]) <= rational(footprint["depths"][1])
        ):
            raise ValueError("unsupported reference footprint domain")
    return labels


def visible(
    labels: tuple[tuple[int, ...], ...], index: int, tokens: tuple[str, ...]
) -> VisibleRaster:
    observed = sorted({tokens[v - 1] for row in labels for v in row if v})
    local = {token: i + 1 for i, token in enumerate(observed)}
    converted = [[0 if v == 0 else local[tokens[v - 1]] for v in row] for row in labels]
    return VisibleRaster(
        index, np.array(converted, dtype=np.int32), tuple((i, t) for t, i in local.items())
    )


class Observation:
    def __init__(self, raster: VisibleRaster) -> None:
        self.raster = raster

    def observation(self, index: int) -> TrustedObservation:
        return TrustedObservation(self.raster, True, True, True)


class Target:
    def __init__(self, source: Input, raster: VisibleRaster) -> None:
        self.evidence = TargetEvidence(binding(source), raster, True, True, True, True)

    def target(self, index: int) -> TargetEvidence:
        return self.evidence


def before(
    study: Study,
    unit: int,
    frames: tuple[tuple[tuple[int, ...], ...], ...],
    identity: dict[str, Any],
    head: str,
) -> tuple[Input, Input]:
    tokens = tuple(identity["tokens"])
    if len(tokens) != 3 or len(set(tokens)) != 3:
        raise ValueError("fresh bijective private controlled identity required")
    state: State | None = None
    for i, labels in enumerate(frames):
        result = observe(
            state,
            Observation(visible(labels, i, tokens)),
            PERMISSIONS,
            i,
            i,
            None if i == 0 else action(study.units[unit].prefix[1] - study.units[unit].prefix[0]),
        )
        if result.state is None or result.unresolved:
            raise ValueError("prefix lifecycle unresolved")
        state = result.state
    assert state is not None and state.decision_index == 1
    return cast(
        tuple[Input, Input],
        tuple(
            Input(
                state, action_bytes(action(q)), identity["episode_key"], head, True, PERMISSIONS, 1
            )
            for q in study.units[unit].actions
        ),
    )


def identity() -> dict[str, Any]:
    return {
        "episode_key": secrets.token_hex(16),
        "tokens": ["surface-" + secrets.token_hex(8) for _ in range(3)],
    }


def load(path: Path, name: str) -> Any:
    data = (path / name).read_bytes()
    value = json.loads(data)
    if canonical_json_bytes(value) != data:
        raise ValueError("retained JSON must be canonical")
    return value


def result(units: list[dict[str, Any]]) -> str:
    if any(u["status"] == "FAIL" for u in units):
        return "FAIL"
    return (
        "PASS" if len(units) == 2 and all(u["status"] == "PASS" for u in units) else "INCONCLUSIVE"
    )


def inspect(path: Path, context: dict[str, Any], terminal: bool = True) -> dict[str, Any]:
    """Only retained bytes and scalar mathematics; no producer/reference calls."""
    validate_context(context)
    study = parse((path / "manifest.json").read_bytes())
    if (
        sha256_bytes(study.payload) != context["manifest_sha256"]
        or sha256_bytes((path / "proposal.md").read_bytes()) != study.proposal_sha256
        or sha256_bytes((path / "adoption.md").read_bytes()) != study.adoption_sha256
        or load(path, "context.json") != context
    ):
        raise ValueError("retained definition/source context differs")
    finish = load(path, "terminal.json") if terminal else None
    if finish is not None:
        keys(
            finish,
            {
                "version",
                "completion",
                "first_failure",
                "analysis",
                "calls",
                "unattempted_producer_calls",
                "elapsed_seconds",
                "files",
                "bytes_before_terminal",
            },
        )
        if finish["version"] != VERSION:
            raise ValueError("terminal contract differs")
        allowed = {
            "manifest.json",
            "proposal.md",
            "adoption.md",
            "context.json",
            "identities.json",
            "attempt.json",
            "seal.json",
            "u0-analysis.json",
            "u1-analysis.json",
        }
        allowed.update(
            f"u{i}-{phase}{j}-{kind}.json"
            for i in range(2)
            for phase in ("p", "t")
            for j in range(2)
            for kind in ("raw", "audit")
        )
        allowed.update(f"u{i}-q{j}-forecast.json" for i in range(2) for j in range(2))
        allowed.update(
            f"call-{i:02d}-{phase}.json" for i in range(1, 17) for phase in ("attempt", "complete")
        )
        if type(finish["files"]) is not dict or not set(finish["files"]) <= allowed:
            raise ValueError("closed retained file vocabulary required")
        for name, digest in finish["files"].items():
            if sha256_bytes((path / name).read_bytes()) != digest:
                raise ValueError("terminal retained file hash differs")
        if {p.name for p in path.iterdir()} != set(finish["files"]) | {"terminal.json"}:
            raise ValueError("unbound retained file or missing evidence")
        calls = finish["calls"]
        if type(calls) is not list or len(calls) > 16:
            raise ValueError("bounded call accounting required")
        for kind in ("raw", "audit"):
            if sum(c["kind"] == kind for c in calls) > 8:
                raise ValueError("per-algorithm call cap exceeded")
        expected_frames = [
            f"u{i}-{phase}{j}-{kind}.json"
            for phase in ("p", "t")
            for i in range(2)
            for j in range(2)
            for kind in ("raw", "audit")
        ]
        for index, call in enumerate(calls):
            keys(call, {"kind", "frame", "completed", "retained"})
            if (
                call["frame"] != expected_frames[index]
                or call["kind"] != ("raw" if index % 2 == 0 else "audit")
                or type(call["completed"]) is not bool
                or type(call["retained"]) is not bool
            ):
                raise ValueError("fixed ordered calls required")
            if call["retained"] and (not call["completed"] or call["frame"] not in finish["files"]):
                raise ValueError("call return/retention contradiction")
        if finish["unattempted_producer_calls"] != 8 - sum(c["kind"] == "raw" for c in calls):
            raise ValueError("unattempted call count differs")
        complete = (
            len(calls) == 16
            and all(c["completed"] and c["retained"] for c in calls)
            and finish["first_failure"] is None
        )
        if finish["completion"] != ("COMPLETE" if complete else "STOPPED_INCOMPLETE"):
            raise ValueError("completion differs from operational facts")
    units: list[dict[str, Any]] = [
        {"status": "INCONCLUSIVE", "reason": "incomplete evidence"} for _ in range(2)
    ]
    seal_path = path / "seal.json"
    if seal_path.exists():
        seal = load(path, "seal.json")
        keys(seal, {"members", "manifest_sha256"})
        expected = {
            "manifest.json",
            "proposal.md",
            "adoption.md",
            "context.json",
            "identities.json",
            "attempt.json",
        }
        expected.update(
            f"u{i}-p{j}-{kind}.json"
            for i in range(2)
            for j in range(2)
            for kind in ("raw", "audit")
        )
        expected.update(f"u{i}-q{j}-forecast.json" for i in range(2) for j in range(2))
        if (
            set(seal["members"]) != expected
            or seal["manifest_sha256"] != context["manifest_sha256"]
        ):
            raise ValueError("complete global forecast membership seal required")
        for name, digest in seal["members"].items():
            if sha256_bytes((path / name).read_bytes()) != digest:
                raise ValueError("global forecast seal changed")
        identities = load(path, "identities.json")
        if len(identities) != 2 or identities[0]["episode_key"] == identities[1]["episode_key"]:
            raise ValueError("distinct per-unit episodes required")
        for i in range(2):
            prefix = tuple(
                qualified(load(path, f"u{i}-p{j}-raw.json"), load(path, f"u{i}-p{j}-audit.json"))
                for j in range(2)
            )
            sources = before(study, i, prefix, identities[i], context["head"])
            for j, source in enumerate(sources):
                decode((path / f"u{i}-q{j}-forecast.json").read_bytes(), source)
            targets, reports = [], []
            for j, source in enumerate(sources):
                names = [f"u{i}-t{j}-{kind}.json" for kind in ("raw", "audit")]
                if not all((path / name).exists() for name in names):
                    break
                try:
                    labels = qualified(load(path, names[0]), load(path, names[1]))
                    evaluation = evaluate(
                        source,
                        (path / f"u{i}-q{j}-forecast.json").read_bytes(),
                        Target(source, visible(labels, 2, tuple(identities[i]["tokens"]))),
                    )
                    if evaluation.report is None or evaluation.unresolved:
                        raise ValueError("target evaluation unresolved")
                    targets.append(labels)
                    reports.append(evaluation.report)
                except (ValueError, TypeError, KeyError) as error:
                    units[i] = {"status": "INCONCLUSIVE", "reason": str(error)}
                    break
            if len(reports) != 2:
                continue
            s = sources[0].state
            changed_prefix = any(
                n.previous_mask != n.current_mask
                and int.from_bytes(n.previous_mask or b"").bit_count()
                != int.from_bytes(n.current_mask).bit_count()
                for n in s.nodes
            )
            informative = (
                bool(s.inventory)
                and changed_prefix
                and all(t != p for t in targets for p in prefix)
                and reports[0].targets != reports[1].targets
            )
            deltas = [r.strata[0].difference for r in reports]
            complete = all(
                r.strata[0].predicted == len(s.inventory)
                and not sources[j].state.inventory == ()
                and deltas[j] is not None
                for j, r in enumerate(reports)
            )
            delta = sum((d for d in deltas if d is not None), Q()) / 2 if complete else None
            status = (
                "INCONCLUSIVE"
                if not informative or delta is None
                else "PASS"
                if delta <= -Q(1, 1024)
                else "FAIL"
            )
            units[i] = {
                "status": status,
                "qualified": True,
                "informative": informative,
                "complete_coverage": complete,
                "delta": plain(delta),
                "reports": plain(reports),
            }
    output: dict[str, Any] = {"scientific_outcome": result(units), "units": units}
    if finish is not None:
        if finish["analysis"] != output:
            raise ValueError("terminal analysis differs from retained-only mathematics")
        actual_bytes = sum(p.stat().st_size for p in path.iterdir())
        if 2 * actual_bytes + REVIEW_RESERVE > LIMIT:
            raise ValueError("inclusive retained budget exceeded")
        output.update(
            {
                "completion": finish["completion"],
                "first_failure": finish["first_failure"],
                "calls": finish["calls"],
                "original_bytes": actual_bytes,
                "inclusive_accounted_bytes": 2 * actual_bytes + REVIEW_RESERVE,
            }
        )
    return output


def collect(
    manifest: bytes,
    proposal: bytes,
    adoption: bytes,
    context: dict[str, Any],
    output: Path,
    protect: Callable[[Path], None],
    source_check: Callable[[], None],
    adapter: Adapter,
    clock: Callable[[], float] = time.monotonic,
    allocate: Callable[[], dict[str, Any]] = identity,
) -> dict[str, Any]:
    started = clock()
    context = json.loads(canonical_json_bytes(context))
    validate_context(context)
    study = parse(manifest)
    if (
        context["manifest_sha256"] != sha256_bytes(manifest)
        or sha256_bytes(proposal) != study.proposal_sha256
        or sha256_bytes(adoption) != study.adoption_sha256
    ):
        raise ValueError("exact accepted manifest/definition required")
    source_check()
    output.mkdir()  # An existing attempted namespace is never resumed or overwritten.
    protect(output)  # Actual invocation supplies reviewed owner-restricted protection.
    sink = Sink(output, clock)
    sink.started = started
    calls: list[dict[str, Any]] = []
    first_failure: dict[str, Any] | None = None
    pending_return: dict[str, Any] | None = None
    analysis: dict[str, Any] = {
        "scientific_outcome": "INCONCLUSIVE",
        "units": [{"status": "INCONCLUSIVE", "reason": "incomplete evidence"} for _ in range(2)],
    }
    try:
        for name, payload in (
            ("manifest.json", manifest),
            ("proposal.md", proposal),
            ("adoption.md", adoption),
        ):
            sink.write(name, payload)
        sink.json("context.json", context)
        sink.json(
            "attempt.json",
            {
                "version": VERSION,
                "status": "ONE_ATTEMPT",
                "manifest_sha256": context["manifest_sha256"],
            },
        )
        identities = [allocate(), allocate()]
        sink.json("identities.json", identities)

        def frame(i: int, phase: str, j: int, lateral: Q) -> tuple[tuple[int, ...], ...]:
            nonlocal pending_return
            sink.verify()
            source_check()
            products = []
            for kind, operation in (("raw", adapter.produce), ("audit", adapter.audit)):
                sink.check()
                if sum(c["kind"] == kind for c in calls) >= 8:
                    raise ValueError("call budget exceeded")
                name = f"u{i}-{phase}{j}-{kind}.json"
                call = {"kind": kind, "frame": name, "completed": False, "retained": False}
                calls.append(call)
                sink.json(f"call-{len(calls):02d}-attempt.json", call)
                produced = plain(operation(study.units[i], lateral, study.extent, sink.check))
                call["completed"] = True
                pending_return = {"frame": name, "record": produced}
                sink.json(name, produced)
                call["retained"] = True
                pending_return = None
                sink.json(f"call-{len(calls):02d}-complete.json", call)
                products.append(produced)
            return qualified(products[0], products[1])

        prefixes = [
            tuple(frame(i, "p", j, pos) for j, pos in enumerate(unit.prefix))
            for i, unit in enumerate(study.units)
        ]
        for i, prefix in enumerate(prefixes):
            for j, source in enumerate(before(study, i, prefix, identities[i], context["head"])):
                sink.write(f"u{i}-q{j}-forecast.json", forecast(source).canonical_bytes())
        members = {n: h for n, h in sink.hashes.items() if not n.startswith("call-")}
        sink.json("seal.json", {"manifest_sha256": context["manifest_sha256"], "members": members})
        # Complete disk-bound all-unit/all-action validation precedes ANY target call.
        inspect(output, context, terminal=False)
        sink.verify()
        for i, unit in enumerate(study.units):
            for j, q in enumerate(unit.actions):
                inspect(output, context, terminal=False)
                frame(i, "t", j, unit.prefix[1] + q)
            analysis = inspect(output, context, terminal=False)
            sink.json(f"u{i}-analysis.json", analysis["units"][i])
    except Exception as error:
        first_failure = {"type": type(error).__name__, "message": str(error)}
        if pending_return is not None:
            first_failure["unretained_return"] = pending_return
        try:
            source_check()
            analysis = inspect(output, context, terminal=False)
        except Exception as inspection_error:
            first_failure["inspection_error"] = str(inspection_error)
            first_failure["prior_analysis_unverified"] = analysis
            analysis = {
                "scientific_outcome": "INCONCLUSIVE",
                "units": [
                    {"status": "INCONCLUSIVE", "reason": "global evidence unverifiable"}
                    for _ in range(2)
                ],
            }
    completion = (
        "COMPLETE"
        if first_failure is None
        and len(calls) == 16
        and all(c["completed"] and c["retained"] for c in calls)
        else "STOPPED_INCOMPLETE"
    )
    # Include any bytes left by a failed write, never erase partial evidence.
    actual_files = {p.name: sha256_bytes(p.read_bytes()) for p in output.iterdir() if p.is_file()}
    sink.bytes = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
    terminal = {
        "version": VERSION,
        "completion": completion,
        "first_failure": first_failure,
        "analysis": analysis,
        "calls": calls,
        "unattempted_producer_calls": 8 - sum(c["kind"] == "raw" for c in calls),
        "elapsed_seconds": clock() - sink.started,
        "files": actual_files,
        "bytes_before_terminal": sink.bytes,
    }
    sink.json("terminal.json", terminal, terminal=True)
    return terminal
