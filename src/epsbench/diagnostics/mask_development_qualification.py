"""Fixed mask development collector; no model or closed-controller import."""

from __future__ import annotations

import json
import re
import secrets
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from fractions import Fraction as Q
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from epsbench.diagnostics.boundary_observation import BoundaryObservationView, VisibleRaster
from epsbench.diagnostics.causal_region_lifecycle import (
    REQUIRED,
    State,
    TrustedObservation,
    advance,
    keys,
)
from epsbench.diagnostics.neutral_observation_target import Candidate
from epsbench.diagnostics.restricted_exact_raster import Box, Calibration
from epsbench.diagnostics.restricted_mask_projection import Projection, Trust, numeric
from epsbench.diagnostics.visible_forecast_contract import CausalInput, Limits, TokenFrame, _json
from epsbench.schema import Action, BoundaryAxis, ModalityPermissionSet
from epsbench.utils.canonical import canonical_json_bytes, sha256_bytes

VERSION = "mask-route-development-v1"
LIMIT = 16 * 1024 * 1024
REVIEW_RESERVE = 4 * 1024 * 1024
TERMINAL_RESERVE = 64 * 1024
LOG_RESERVE = 128 * 1024
ACCESS = ModalityPermissionSet(allowed=REQUIRED)
SIGNALS = (
    "known_inventory",
    "before_mask_change",
    "both_targets_novel",
    "action_contrast",
    "both_persistence_nonzero",
    "remembered_channel",
    "new_positive",
)
SOURCE_FILES = (
    "src/epsbench/diagnostics/mask_development_qualification.py",
    "src/epsbench/diagnostics/restricted_mask_projection.py",
    "src/epsbench/diagnostics/visible_forecast_contract.py",
    "src/epsbench/diagnostics/neutral_observation_target.py",
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


def context_check(context: dict[str, Any]) -> None:
    keys(context, {"head", "tree", "manifest_sha256", "sources", "versions", "purpose"})
    if context["purpose"] != VERSION or type(context["versions"]) is not dict:
        raise ValueError("reviewed purpose/runtime versions required")
    for name, n in (("head", 40), ("tree", 40), ("manifest_sha256", 64)):
        if (
            type(context[name]) is not str
            or re.fullmatch(r"[0-9a-f]{" + str(n) + "}", context[name]) is None
        ):
            raise ValueError("exact source/manifest binding required")
    if any(
        type(v) is not str or re.fullmatch(r"[0-9a-f]{64}", v) is None
        for v in keys(context["sources"], set(SOURCE_FILES)).values()
    ):
        raise ValueError("complete runtime source byte binding required")


def verify_source(root: Path, context: dict[str, Any]) -> None:
    context_check(context)
    if any(sha256_bytes((root / n).read_bytes()) != v for n, v in context["sources"].items()):
        raise ValueError("reviewed source bytes differ")
    observed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD", "HEAD^{tree}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout.splitlines()
    if observed != [context["head"], context["tree"]]:
        raise ValueError("source head/tree differs")


def rational(value: Any) -> Q:
    if type(value) is not str or len(value) > 64:
        raise ValueError("bounded rational string required")
    q = Q(value)
    if str(q) != value:
        raise ValueError("reduced rational spelling required")
    return q


def plain(value: Any) -> Any:
    if type(value) is Q:
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return plain(asdict(value))
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


@dataclass(frozen=True)
class Study:
    payload: bytes
    definition_sha256: str
    adoption_sha256: str
    boxes: tuple[Box, Box]
    extent: tuple[Q, Q]
    prefix: tuple[Q, Q]
    executed: Action
    announced: tuple[Action, Action]
    steps: tuple[Q, Q]


def parse(payload: bytes) -> Study:
    if type(payload) is not bytes or len(payload) > 65536:
        raise ValueError("bounded manifest required")
    p = keys(
        _json(payload),
        {
            "version",
            "definition_sha256",
            "adoption_sha256",
            "calibration",
            "support",
            "boxes",
            "prefix",
            "executed_action",
            "announced_actions",
            "steps",
            "signals",
            "limits",
        },
    )
    if (
        p["version"] != VERSION
        or p["signals"] != list(SIGNALS)
        or p["calibration"]
        != {
            "size": 32,
            "forward": "-3",
            "elevation": "1",
            "up_y": "0",
            "fovy": "90",
            "clipping": "NONE",
        }
        or p["limits"]
        != {
            "producer_calls": 4,
            "audits": 4,
            "seconds": 60,
            "inclusive_bytes": LIMIT,
            "review_reserve_bytes": REVIEW_RESERVE,
        }
    ):
        raise ValueError("fixed task/calibration/resource contract required")
    for name in ("definition_sha256", "adoption_sha256"):
        if type(p[name]) is not str or re.fullmatch(r"[0-9a-f]{64}", p[name]) is None:
            raise ValueError("definition binding required")
    for name in ("support", "prefix", "steps", "boxes", "announced_actions"):
        if type(p[name]) is not list or len(p[name]) != 2:
            raise ValueError("fixed two-element membership required")
    extent = cast(tuple[Q, Q], tuple(rational(v) for v in p["support"]))
    prefix = cast(tuple[Q, Q], tuple(rational(v) for v in p["prefix"]))
    steps = cast(tuple[Q, Q], tuple(rational(v) for v in p["steps"]))
    executed = Action.model_validate(p["executed_action"])
    announced = cast(
        tuple[Action, Action], tuple(Action.model_validate(v) for v in p["announced_actions"])
    )
    if numeric(executed) != (Q(0), prefix[1] - prefix[0], Q(0)) or any(
        numeric(a) != (Q(0), q, Q(0)) for a, q in zip(announced, steps, strict=True)
    ):
        raise ValueError("genuine Actions and exact numeric positions/steps must agree")
    if not steps[0] < 0 < steps[1] or len(set((*prefix, *(prefix[1] + q for q in steps)))) != 4:
        raise ValueError("opposed distinct non-return targets required")
    if not all(0 < x <= 32 for x in extent):
        raise ValueError("bounded support required")
    boxes = []
    for b in p["boxes"]:
        b = keys(b, {"lower", "upper"})
        if any(type(v) is not list or len(v) != 3 for v in b.values()):
            raise ValueError("three coordinates required")
        boxes.append(
            Box(
                cast(tuple[Q, Q, Q], tuple(rational(x) for x in b["lower"])),
                cast(tuple[Q, Q, Q], tuple(rational(x) for x in b["upper"])),
            )
        )
    if not boxes[0].upper[1] < boxes[1].lower[1] or any(
        b.lower[2] <= 0 or b.lower[1] <= -3 for b in boxes
    ):
        raise ValueError("separated above-support forward boxes required")
    for lateral in (*prefix, *(prefix[1] + q for q in steps)):
        for b in boxes:
            for x in (b.lower[0], b.upper[0]):
                for y in (b.lower[1], b.upper[1]):
                    for z in (b.lower[2], b.upper[2]):
                        if max(abs((x - lateral) / (y + 3)), abs((z - 1) / (y + 3))) > 8:
                            raise ValueError("reference footprint work bound exceeded")
    study = Study(
        payload,
        p["definition_sha256"],
        p["adoption_sha256"],
        cast(tuple[Box, Box], tuple(boxes)),
        extent,
        prefix,
        executed,
        announced,
        steps,
    )
    if canonical_json_bytes(p) != payload:
        raise ValueError("canonical manifest required")
    return study


class Adapter(Protocol):
    def produce(self, study: Study, witness: dict[str, Any], check: Callable[[], None]) -> Any: ...
    def audit(self, study: Study, witness: dict[str, Any], check: Callable[[], None]) -> Any: ...


class ExactAdapter:
    def produce(self, study: Study, witness: dict[str, Any], check: Callable[[], None]) -> Any:
        from epsbench.diagnostics.restricted_exact_raster import raster

        return raster(rational(witness["producer_lateral"]), study.boxes, study.extent, check)

    def audit(self, study: Study, witness: dict[str, Any], check: Callable[[], None]) -> Any:
        from epsbench.diagnostics.occupancy_reference import Solid, audit

        return audit(
            Calibration().origin(rational(witness["audit_lateral"])),
            cast(tuple[Solid, Solid], tuple(Solid(b.lower, b.upper) for b in study.boxes)),
            study.extent,
            32,
            check,
        )


class Sink:
    def __init__(self, path: Path, clock: Callable[[], float], external: Callable[[], int]) -> None:
        self.path, self.clock, self.external = path, clock, external
        self.started = clock()
        self.bytes = 0
        self.hashes: dict[str, str] = {}

    def check(self) -> None:
        if self.clock() - self.started > 60:
            raise TimeoutError("cooperative budget exceeded")

    def write(self, name: str, raw: bytes, terminal: bool = False) -> None:
        if not terminal:
            self.check()
        overhead = self.external()
        if type(overhead) is not int or overhead < 0:
            raise ValueError("exact external byte accounting required")
        reserve = 0 if terminal else TERMINAL_RESERVE
        if 2 * (self.bytes + len(raw) + overhead + LOG_RESERVE + reserve) + REVIEW_RESERVE > LIMIT:
            raise OSError("inclusive originals/archive/wrapper/log/review budget exceeded")
        with (self.path / name).open("xb") as f:
            f.write(raw)
            f.flush()
        self.bytes += len(raw)
        self.hashes[name] = sha256_bytes(raw)

    def json(self, name: str, value: Any, terminal: bool = False) -> None:
        self.write(name, canonical_json_bytes(plain(value)), terminal)

    def verify(self) -> None:
        self.check()
        for name, digest in self.hashes.items():
            self.check()
            if sha256_bytes((self.path / name).read_bytes()) != digest:
                raise ValueError("retained bytes changed")


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
    if (
        labels != rows(audit["labels"])
        or raw["ties"] != []
        or audit["ties"] != []
        or audit["boundaries"] != []
    ):
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
            or type(footprint["polygon"]) is not list
            or not 3 <= len(footprint["polygon"]) <= 8
            or any(
                type(point) is not list
                or len(point) != 2
                or any(abs(rational(q)) > 8 for q in point)
                for point in footprint["polygon"]
            )
            or type(footprint["depths"]) is not list
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


def identity() -> dict[str, Any]:
    return {
        "episode": secrets.token_hex(16),
        "tokens": ["surface-" + secrets.token_hex(8) for _ in range(3)],
    }


def check_identity(value: dict[str, Any]) -> None:
    keys(value, {"episode", "tokens"})
    if type(value["episode"]) is not str or re.fullmatch(r"[0-9a-f]{32}", value["episode"]) is None:
        raise ValueError("opaque episode required")
    if (
        type(value["tokens"]) is not list
        or len(value["tokens"]) != 3
        or len(set(value["tokens"])) != 3
        or any(
            type(v) is not str or re.fullmatch(r"surface-[0-9a-f]{16}", v) is None
            for v in value["tokens"]
        )
    ):
        raise ValueError("unique static opaque REGION mapping required")


@dataclass(frozen=True)
class Prefix:
    frames: tuple[VisibleRaster, VisibleRaster]

    def raster(self, index: int) -> VisibleRaster:
        if type(index) is not int or not 0 <= index <= 1:
            raise PermissionError("future prefix access denied")
        return self.frames[index]

    def observation(self, index: int) -> TrustedObservation:
        return TrustedObservation(self.raster(index), True, True, True)


def lattice(raster: VisibleRaster) -> list[dict[str, Any]]:
    """Independent full-grid boundary expectation from original retained raster."""
    lookup = dict(raster.identities)
    a = raster.segmentation
    h, w = a.shape
    result = []
    for axis in (BoundaryAxis.HORIZONTAL, BoundaryAxis.VERTICAL):
        dr, dc = (0, 1) if axis == BoundaryAxis.HORIZONTAL else (1, 0)
        for r in range(h - dr):
            for c in range(w - dc):
                left, right = int(a[r, c]), int(a[r + dr, c + dc])
                if left != right:
                    result.append(
                        {
                            "axis": axis.value,
                            "row": r,
                            "column": c,
                            "negative_surface_id": lookup.get(left),
                            "positive_surface_id": lookup.get(right),
                            "ownership": "unknown",
                        }
                    )
    return sorted(result, key=lambda e: (e["axis"], e["row"], e["column"]))


def projections(
    study: Study, frames: tuple[VisibleRaster, VisibleRaster], mapping: dict[str, Any], head: str
) -> tuple[Projection, Projection]:
    provider = Prefix(frames)
    state: State | None = None
    for i in range(2):
        update = advance(state, provider, ACCESS, i, 1, None if i == 0 else study.executed)
        if update.unresolved is not None or update.state is None:
            raise ValueError("qualified prefix lifecycle rejected")
        state = update.state
        observed = BoundaryObservationView(provider, ACCESS, 1).observe(i)
        if json.loads(observed.canonical_bytes())["edges"] != lattice(frames[i]):
            raise ValueError("independent endpoint lattice disagreement")
    assert state is not None
    parts = tuple(
        TokenFrame(
            i,
            (32, 32),
            tuple((t, frames[i].segmentation == label) for label, t in frames[i].identities),
        )
        for i in range(2)
    )
    outputs = []
    for announced in study.announced:
        source = CausalInput(
            parts, (numeric(study.executed),), numeric(announced), ACCESS, Limits(2, 1024, 3)
        )
        projected = Projection(
            source,
            state,
            ACCESS,
            Trust(True, True, True, True),
            (True, True),
            mapping["episode"],
            head,
            1,
            announced,
        )
        f = json.loads(projected.feature_bytes())
        slots = {name: i for i, name in enumerate(projected.alignment)}
        for i, raster in enumerate(frames):
            expected = sorted(
                {
                    tuple(
                        sorted((slots[e["negative_surface_id"]], slots[e["positive_surface_id"]]))
                    )
                    for e in lattice(raster)
                    if e["negative_surface_id"] is not None and e["positive_surface_id"] is not None
                }
            )
            if f["endpoints"][i] != {
                "age": 1 - i,
                "available": True,
                "pairs": [list(x) for x in expected],
            }:
                raise ValueError("independent endpoint pair/availability/time disagreement")
        outputs.append(projected)
    return cast(tuple[Projection, Projection], tuple(outputs))


def before_records(p: Projection) -> dict[str, bytes]:
    return {
        "input": p._source,
        "state": p._state,
        "projection": canonical_json_bytes(
            {
                "features": json.loads(p.feature_bytes()),
                "alignment": list(p.alignment),
                "binding": json.loads(p.binding_bytes()),
                "previous_mask_ablation": json.loads(p.feature_bytes(True)),
            }
        ),
        "candidate": p.persistence(),
    }


def witness(
    study: Study,
    mapping: dict[str, Any],
    context: dict[str, Any],
    phase: str,
    branch: int,
    p: Projection | None = None,
) -> dict[str, Any]:
    if phase == "prefix":
        producer = audit = study.prefix[branch]
        announced = None
        projection = None
        target_index = branch
    else:
        if p is None:
            raise ValueError("action-specific sealed projection required")
        actual = Action.model_validate_json(p.announced)
        producer = study.prefix[1] + numeric(actual)[1]
        audit = study.prefix[1] + rational(json.loads(study.payload)["steps"][branch])
        if producer != audit or numeric(actual) != (Q(0), study.steps[branch], Q(0)):
            raise ValueError("independent exact branch endpoint/Action mismatch")
        announced = actual.model_dump(mode="json")
        projection = sha256_bytes(p.binding_bytes())
        target_index = 2
    return {
        "episode": mapping["episode"],
        "manifest_sha256": context["manifest_sha256"],
        "source_head": context["head"],
        "source_tree": context["tree"],
        "source_files_sha256": sha256_bytes(canonical_json_bytes(context["sources"])),
        "start_lateral": str(study.prefix[branch] if phase == "prefix" else study.prefix[1]),
        "calibration": json.loads(study.payload)["calibration"],
        "phase": phase,
        "branch": branch,
        "target_index": target_index,
        "producer_lateral": str(producer),
        "audit_lateral": str(audit),
        "announced_action": announced,
        "projection_sha256": projection,
        "prefix_sha256": None if p is None else sha256_bytes(p._source),
        "shape": [32, 32],
    }


class Target:
    def __init__(
        self, raster: VisibleRaster, observed: dict[str, Any], expected: dict[str, Any]
    ) -> None:
        if observed != expected or observed["phase"] != "target":
            raise ValueError("target episode/branch/action/source witness differs")
        self.value = raster
        self.used = False

    def raster(self, index: int) -> VisibleRaster:
        if self.used or type(index) is not int or index != 2:
            raise PermissionError("exact one branch-local target fetch required")
        self.used = True
        if self.value.sequence_index != 2:
            raise ValueError("target raster chronology differs")
        return self.value


def read(path: Path, name: str) -> Any:
    raw = (path / name).read_bytes()
    p = _json(raw)
    if raw != canonical_json_bytes(p):
        raise ValueError("canonical retained object required")
    return p


def empty_analysis(reason: str = "incomplete evidence") -> dict[str, Any]:
    return {
        "development_outcome": "INCONCLUSIVE",
        "qualification": "UNRESOLVED",
        "reason": reason,
        "branches": [],
    }


def inspect(
    path: Path, context: dict[str, Any], source_check: Callable[[], None], terminal: bool = True
) -> dict[str, Any]:
    context_check(context)
    source_check()
    finish = read(path, "terminal.json") if terminal else None
    if finish is not None:
        keys(
            finish,
            {
                "version",
                "completion",
                "first_failure",
                "analysis",
                "calls",
                "files",
                "elapsed_seconds",
                "external_bytes",
                "bytes_before_terminal",
            },
        )
        allowed = {
            "manifest.json",
            "definition.md",
            "adoption.md",
            "context.json",
            "identity.json",
            "attempt.json",
            "seal.json",
            "branch0-analysis.json",
            "branch1-analysis.json",
        }
        allowed |= {
            f"{phase}{i}-{suffix}.json"
            for phase in ("prefix", "target")
            for i in range(2)
            for suffix in ("raw", "audit", "witness")
        }
        allowed |= {
            f"action{i}-{suffix}.json"
            for i in range(2)
            for suffix in ("input", "state", "projection", "candidate")
        }
        allowed |= {
            f"call-{i:02d}-{suffix}.json" for i in range(1, 9) for suffix in ("attempt", "complete")
        }
        files = list(path.iterdir())
        if any(not f.is_file() or f.is_symlink() for f in files):
            raise ValueError("flat retained regular-file set required")
        actual = {f.name: sha256_bytes(f.read_bytes()) for f in files if f.name != "terminal.json"}
        size = sum(f.stat().st_size for f in files)
        if finish["version"] != VERSION or actual != finish["files"] or not set(actual) <= allowed:
            raise ValueError("terminal version/file/hash set differs")
        if (
            type(finish["external_bytes"]) is not int
            or finish["external_bytes"] < 0
            or type(finish["bytes_before_terminal"]) is not int
            or finish["bytes_before_terminal"] != size - (path / "terminal.json").stat().st_size
            or type(finish["elapsed_seconds"]) not in (int, float)
            or finish["elapsed_seconds"] < 0
            or 2 * (size + finish["external_bytes"] + LOG_RESERVE) + REVIEW_RESERVE > LIMIT
        ):
            raise ValueError("inclusive retained budget/accounting differs")
        calls = finish["calls"]
        if type(calls) is not list or len(calls) > 8:
            raise ValueError("fixed attempted call budget required")
        for i, c in enumerate(calls):
            keys(c, {"name", "kind", "witness", "completed", "retained"})
            phase = "prefix" if i < 4 else "target"
            kind = "raw" if i % 2 == 0 else "audit"
            if (
                c["name"] != f"{phase}{(i // 2) % 2}-{kind}.json"
                or c["kind"] != kind
                or type(c["completed"]) is not bool
                or type(c["retained"]) is not bool
                or read(path, f"call-{i + 1:02d}-attempt.json")
                != {**c, "completed": False, "retained": False}
            ):
                raise ValueError("ordered attempted call accounting differs")
            if c["retained"] and (
                not c["completed"]
                or c["name"] not in actual
                or read(path, f"call-{i + 1:02d}-complete.json") != c
            ):
                raise ValueError("returned object/completion accounting differs")
        if finish["completion"] not in ("COMPLETE", "STOPPED_INCOMPLETE"):
            raise ValueError("terminal completion required")
        if finish["completion"] == "COMPLETE" and (
            len(calls) != 8
            or finish["first_failure"] is not None
            or not all(c["completed"] and c["retained"] for c in calls)
        ):
            raise ValueError("complete attempt accounting differs")

    def conclusion(output: dict[str, Any]) -> dict[str, Any]:
        if finish is not None and output != finish["analysis"]:
            raise ValueError("terminal development analysis differs")
        return output

    required = ("manifest.json", "definition.md", "adoption.md", "context.json", "identity.json")
    if not all((path / n).is_file() for n in required):
        return conclusion(empty_analysis())
    study = parse((path / "manifest.json").read_bytes())
    if (
        sha256_bytes(study.payload) != context["manifest_sha256"]
        or sha256_bytes((path / "definition.md").read_bytes()) != study.definition_sha256
        or sha256_bytes((path / "adoption.md").read_bytes()) != study.adoption_sha256
        or read(path, "context.json") != context
    ):
        raise ValueError("exact retained manifest/definition/context required")
    mapping = read(path, "identity.json")
    check_identity(mapping)
    if finish is not None:
        for i, c in enumerate(finish["calls"][:4]):
            if c["witness"] != witness(study, mapping, context, "prefix", i // 2):
                raise ValueError("prefix attempted witness differs")
    for phase, branches in (("prefix", range(2)), ("target", range(2))):
        for j in branches:
            name = f"{phase}{j}"
            if (path / (name + "-raw.json")).exists() and (path / (name + "-audit.json")).exists():
                try:
                    qualified(read(path, name + "-raw.json"), read(path, name + "-audit.json"))
                except (ValueError, TypeError, KeyError):
                    if (
                        finish is not None
                        and finish["analysis"]["development_outcome"] != "INCONCLUSIVE"
                    ):
                        raise ValueError("invalid evidence cannot support outcome") from None
                    return conclusion(empty_analysis("invalid full-grid truth"))
    if not (path / "seal.json").is_file():
        return conclusion(empty_analysis())
    seal = read(path, "seal.json")
    keys(seal, {"members", "manifest_sha256"})
    if seal["manifest_sha256"] != context["manifest_sha256"] or type(seal["members"]) is not dict:
        raise ValueError("complete before seal required")
    base = set(required) | {"attempt.json"}
    prefix_names = {
        f"prefix{i}-{suffix}.json" for i in range(2) for suffix in ("raw", "audit", "witness")
    }
    event_names = {
        f"call-{i:02d}-{suffix}.json" for i in range(1, 5) for suffix in ("attempt", "complete")
    }
    saved_names = {
        f"action{i}-{suffix}.json"
        for i in range(2)
        for suffix in ("input", "state", "projection", "candidate")
    }
    if set(seal["members"]) != base | prefix_names | event_names | saved_names:
        raise ValueError("all-prefix/all-action global seal incomplete")
    if any(sha256_bytes((path / n).read_bytes()) != h for n, h in seal["members"].items()):
        raise ValueError("sealed before bytes differ")
    frames = cast(
        tuple[VisibleRaster, VisibleRaster],
        tuple(
            visible(
                qualified(read(path, f"prefix{i}-raw.json"), read(path, f"prefix{i}-audit.json")),
                i,
                tuple(mapping["tokens"]),
            )
            for i in range(2)
        ),
    )
    projected = projections(study, frames, mapping, context["head"])
    for i, p in enumerate(projected):
        for suffix, data in before_records(p).items():
            if (path / f"action{i}-{suffix}.json").read_bytes() != data:
                raise ValueError("immutable action before-state/candidate differs")
        if read(path, f"prefix{i}-witness.json") != witness(study, mapping, context, "prefix", i):
            raise ValueError("prefix actual arguments differ")
    calls = [] if finish is None else finish["calls"]
    if finish is not None:
        if type(calls) is not list or len(calls) > 8:
            raise ValueError("fixed attempted call budget required")
        for i, c in enumerate(calls):
            keys(c, {"name", "kind", "witness", "completed", "retained"})
            index = i // 2
            phase = "prefix" if index < 2 else "target"
            branch = index % 2
            kind = "raw" if i % 2 == 0 else "audit"
            expected = witness(
                study,
                mapping,
                context,
                phase,
                branch,
                None if phase == "prefix" else projected[branch],
            )
            if (
                c["name"] != f"{phase}{branch}-{kind}.json"
                or c["kind"] != kind
                or c["witness"] != expected
                or type(c["completed"]) is not bool
                or type(c["retained"]) is not bool
            ):
                raise ValueError("ordered branch-bound attempted arguments differ")
            attempted = {**c, "completed": False, "retained": False}
            if read(path, f"call-{i + 1:02d}-attempt.json") != attempted:
                raise ValueError("call attempt record differs")
            if c["retained"] and (
                not c["completed"] or read(path, f"call-{i + 1:02d}-complete.json") != c
            ):
                raise ValueError("call return/retention record differs")
        if finish["completion"] == "COMPLETE" and (
            len(calls) != 8
            or finish["first_failure"] is not None
            or not all(c["completed"] and c["retained"] for c in calls)
        ):
            raise ValueError("complete attempt accounting differs")
    reports: list[dict[str, Any]] = []
    payloads = []
    for i, p in enumerate(projected):
        name = f"target{i}"
        if not all((path / f"{name}-{s}.json").is_file() for s in ("raw", "audit", "witness")):
            return conclusion({**empty_analysis(), "branches": reports})
        labels = qualified(read(path, name + "-raw.json"), read(path, name + "-audit.json"))
        target = Target(
            visible(labels, 2, tuple(mapping["tokens"])),
            read(path, name + "-witness.json"),
            witness(study, mapping, context, "target", i, p),
        )
        evaluated = p.evaluate((path / f"action{i}-candidate.json").read_bytes(), target)
        if evaluated.report["C"] != evaluated.report["N"] or evaluated.report["U"] != 0:
            raise ValueError("full known/NEW assertion required")
        reports.append(plain(evaluated.report))
        payloads.append(evaluated.target.channels.payload())
    source = projected[0].revalidate()
    empty = np.zeros((32, 32), dtype=np.bool_)
    before = []
    for f in source.frames:
        lookup = dict(f.masks)
        before.append(
            Candidate(
                source, tuple((name, lookup.get(name, empty)) for name in source.inventory), empty
            ).channels.payload()
        )
    current = dict(source.frames[-1].masks)
    previous = dict(source.frames[-2].masks)
    checks = {
        "known_inventory": bool(source.inventory),
        "before_mask_change": before[0] != before[1],
        "both_targets_novel": all(p != b for p in payloads for b in before),
        "action_contrast": payloads[0] != payloads[1],
        "both_persistence_nonzero": all(r["E"] > 0 for r in reports),
        "remembered_channel": any(
            name in previous and name not in current for name in source.inventory
        ),
        "new_positive": any(r["first_observed_pixels"] > 0 for r in reports),
    }
    output = {
        "development_outcome": "PASS" if all(checks.values()) else "FAIL",
        "qualification": "QUALIFIED",
        "signals": checks,
        "branches": reports,
    }
    if finish is not None:
        if output != finish["analysis"]:
            raise ValueError("terminal development analysis differs")
        size = sum(f.stat().st_size for f in path.iterdir() if f.is_file())
        if 2 * (size + finish["external_bytes"] + LOG_RESERVE) + REVIEW_RESERVE > LIMIT:
            raise ValueError("inclusive retained budget exceeded")
    return output


def collect(
    manifest: bytes,
    definition: bytes,
    adoption: bytes,
    context: dict[str, Any],
    output: Path,
    protect: Callable[[Path], None],
    source_check: Callable[[], None],
    adapter: Adapter,
    external: Callable[[], int],
    clock: Callable[[], float] = time.monotonic,
    allocate: Callable[[], dict[str, Any]] = identity,
) -> dict[str, Any]:
    started = clock()
    context = json.loads(canonical_json_bytes(context))
    context_check(context)
    study = parse(manifest)
    if (
        sha256_bytes(manifest) != context["manifest_sha256"]
        or sha256_bytes(definition) != study.definition_sha256
        or sha256_bytes(adoption) != study.adoption_sha256
    ):
        raise ValueError("exact adopted definition/manifest required")
    source_check()
    output.mkdir()
    protect(output)
    sink = Sink(output, clock, external)
    sink.started = started
    calls: list[dict[str, Any]] = []
    failure = None
    pending = None
    analysis = empty_analysis()
    try:
        for name, data in (
            ("manifest.json", manifest),
            ("definition.md", definition),
            ("adoption.md", adoption),
        ):
            sink.write(name, data)
        sink.json("context.json", context)
        sink.json("attempt.json", {"version": VERSION, "status": "ONE_ATTEMPT"})
        mapping = allocate()
        sink.json("identity.json", mapping)
        check_identity(mapping)

        def frame(phase: str, branch: int, p: Projection | None = None) -> VisibleRaster:
            nonlocal pending
            sink.verify()
            source_check()
            w = witness(study, mapping, context, phase, branch, p)
            sink.json(f"{phase}{branch}-witness.json", w)
            returns = []
            for kind, operation in (("raw", adapter.produce), ("audit", adapter.audit)):
                sink.check()
                sink.verify()
                source_check()
                if len(calls) >= 8:
                    raise ValueError("fixed call budget exceeded")
                name = f"{phase}{branch}-{kind}.json"
                call = {
                    "name": name,
                    "kind": kind,
                    "witness": w,
                    "completed": False,
                    "retained": False,
                }
                calls.append(call)
                sink.json(f"call-{len(calls):02d}-attempt.json", call)
                returned = plain(operation(study, json.loads(canonical_json_bytes(w)), sink.check))
                call["completed"] = True
                pending = {"name": name, "returned": returned}
                sink.json(name, returned)
                call["retained"] = True
                pending = None
                sink.json(f"call-{len(calls):02d}-complete.json", call)
                returns.append(returned)
            return visible(
                qualified(returns[0], returns[1]),
                branch if phase == "prefix" else 2,
                tuple(mapping["tokens"]),
            )

        frames = cast(
            tuple[VisibleRaster, VisibleRaster], tuple(frame("prefix", i) for i in range(2))
        )
        projected = projections(study, frames, mapping, context["head"])
        for i, p in enumerate(projected):
            for suffix, data in before_records(p).items():
                sink.write(f"action{i}-{suffix}.json", data)
        sink.json(
            "seal.json",
            {"members": dict(sink.hashes), "manifest_sha256": context["manifest_sha256"]},
        )
        inspect(output, context, source_check, False)
        sink.verify()
        for i, p in enumerate(projected):
            inspect(output, context, source_check, False)
            frame("target", i, p)
            analysis = inspect(output, context, source_check, False)
            sink.json(f"branch{i}-analysis.json", analysis)
    except Exception as error:
        failure = cast(dict[str, Any], {"type": type(error).__name__, "message": str(error)})
        if pending is not None:
            failure["unretained_return"] = pending
        try:
            analysis = inspect(output, context, source_check, False)
        except Exception as inspection_error:
            failure["inspection_error"] = str(inspection_error)
            failure["prior_analysis_unverified"] = analysis
            analysis = empty_analysis("global source/evidence unverifiable")
    completion = (
        "COMPLETE"
        if failure is None
        and len(calls) == 8
        and all(c["completed"] and c["retained"] for c in calls)
        else "STOPPED_INCOMPLETE"
    )
    actual_files = {p.name: sha256_bytes(p.read_bytes()) for p in output.iterdir() if p.is_file()}
    sink.bytes = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
    terminal = {
        "version": VERSION,
        "completion": completion,
        "first_failure": failure,
        "analysis": analysis,
        "calls": calls,
        "files": actual_files,
        "elapsed_seconds": clock() - sink.started,
        "external_bytes": external(),
        "bytes_before_terminal": sink.bytes,
    }
    sink.json("terminal.json", terminal, True)
    return terminal
