"""One fixed public cell bridge; no launcher, model, or historical collector invocation."""

from __future__ import annotations

import json
import math
import re
import subprocess
import time
from collections.abc import Callable
from fractions import Fraction as Q
from pathlib import Path
from typing import Any, cast

from epsbench.diagnostics import known_region_events as events
from epsbench.diagnostics.causal_region_lifecycle import TrustedObservation
from epsbench.diagnostics.mask_development_qualification import (
    SOURCE_FILES,
    Adapter,
    Study,
    check_identity,
    identity,
    plain,
    projections,
    qualified,
    visible,
)
from epsbench.diagnostics.restricted_exact_raster import Box
from epsbench.diagnostics.restricted_mask_projection import Projection
from epsbench.schema import Action, Modality, ModalityPermissionSet
from epsbench.utils.canonical import canonical_json_bytes, sha256_bytes

VERSION = "event-cell-mapping-v1"
METHODS = ("stable", "replay", "half", "visibility_saturation")
POSITIONS = (Q(-1, 4), Q(0), Q(-1, 2), Q(1, 2))
LIMIT = 16 * 1024 * 1024
REVIEW = 4 * 1024 * 1024
LOG = 128 * 1024
RESERVE = 1024 * 1024  # Returned object and closure, reserved before every call.
MAX_RETURN = 512 * 1024
SECONDS = 60
PRIVILEGED = frozenset({Modality.PRIVILEGED_GENERATION_RECORDS})
SOURCES = (
    *SOURCE_FILES,
    "src/epsbench/diagnostics/known_region_events.py",
    "src/epsbench/diagnostics/event_cell_mapping.py",
    "src/epsbench/appearance.py",
    "src/epsbench/utils/seeding.py",
)
INITIAL = {"cell.json", "context.json", "identity.json"}
ALLOWED = INITIAL | {"seal.json", "before-0.json", "before-1.json", "score-0.json", "score-1.json"}
ALLOWED |= {
    f"{i}-{kind}-{suffix}.json"
    for i in range(4)
    for kind in ("raw", "audit")
    for suffix in ("attempt", "return")
}
CELL: dict[str, Any] = {
    "version": VERSION,
    "calibration": {
        "size": 32,
        "forward": "-3",
        "elevation": "1",
        "up_y": "0",
        "fovy": "90",
        "clipping": "NONE",
    },
    "boxes": [
        {"lower": ["27/10", "0", "1/5"], "upper": ["29/10", "1/4", "7/5"]},
        {"lower": ["22/5", "3", "1/5"], "upper": ["24/5", "13/4", "7/5"]},
    ],
    "support": ["1/10", "1/10"],
    "positions": [str(q) for q in POSITIONS],
    "limits": {
        "raw_calls": 4,
        "audit_calls": 4,
        "seconds": SECONDS,
        "inclusive_bytes": LIMIT,
        "review_bytes": REVIEW,
    },
}
CELL_BYTES = canonical_json_bytes(CELL)


def privileged(access: ModalityPermissionSet) -> None:
    if type(access) is not ModalityPermissionSet or access.allowed != PRIVILEGED:
        raise PermissionError("exact typed privileged generation-record permission required")
    checked = ModalityPermissionSet.model_validate_json(access.model_dump_json())
    if checked.allowed != PRIVILEGED:
        raise PermissionError("privileged permission revalidation failed")


def action(lateral: Q) -> Action:
    return Action(
        name="lateral_left" if lateral < 0 else "lateral_right",
        delta_forward=0,
        delta_lateral=float(lateral),
        delta_yaw=0,
    )


def study() -> Study:
    """Reuse physical adapter arguments, not the historical manifest/signals/collector."""
    boxes = tuple(
        Box(
            cast(tuple[Q, Q, Q], tuple(map(Q, b["lower"]))),
            cast(tuple[Q, Q, Q], tuple(map(Q, b["upper"]))),
        )
        for b in CELL["boxes"]
    )
    return Study(
        CELL_BYTES,
        sha256_bytes(CELL_BYTES),
        sha256_bytes(CELL_BYTES),
        cast(tuple[Box, Box], boxes),
        (Q(1, 10), Q(1, 10)),
        (POSITIONS[0], POSITIONS[1]),
        action(Q(1, 4)),
        (action(Q(-1, 2)), action(Q(1, 2))),
        (Q(-1, 2), Q(1, 2)),
    )


def expected(index: int) -> tuple[tuple[int, ...], ...]:
    """Handwritten paper support table; evaluator instrumentation, never a predictor."""
    if type(index) is not int or not 0 <= index < 4:
        raise ValueError("fixed four positions required")
    a = ((31,), (29, 30), (), (27, 28))[index]
    b = ((28,), (27, 28), (29,), (26,))[index]
    floor = ((17,), (15, 16), (18,), (13,))[index]
    rows = [[0] * 32 for _ in range(32)]
    for r in range(14, 20):
        for c in a:
            rows[r][c] = 2
    for r in range(15, 18):
        for c in b:
            rows[r][c] = 3
    for c in floor:
        rows[21][c] = 1
    return tuple(tuple(row) for row in rows)


def source_context(root: Path, versions: dict[str, str]) -> dict[str, Any]:
    """Read-only exact source identity; no admission/operation authority is inferred."""
    head, tree = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD", "HEAD^{tree}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout.splitlines()
    return {
        "version": VERSION,
        "head": head,
        "tree": tree,
        "versions": versions,
        "sources": {n: sha256_bytes((root / n).read_bytes()) for n in SOURCES},
        "cell_sha256": sha256_bytes(CELL_BYTES),
    }


def context_check(context: dict[str, Any]) -> None:
    if (
        set(context) != {"version", "head", "tree", "versions", "sources", "cell_sha256"}
        or context["version"] != VERSION
        or context["cell_sha256"] != sha256_bytes(CELL_BYTES)
        or type(context["versions"]) is not dict
        or not context["versions"]
        or any(type(k) is not str or type(v) is not str for k, v in context["versions"].items())
        or type(context["sources"]) is not dict
        or set(context["sources"]) != set(SOURCES)
    ):
        raise ValueError("exact source/cell/version context required")
    for value, width in [
        (context["head"], 40),
        (context["tree"], 40),
        *((h, 64) for h in context["sources"].values()),
    ]:
        if type(value) is not str or re.fullmatch(f"[0-9a-f]{{{width}}}", value) is None:
            raise ValueError("canonical source hashes required")


def verify_source(root: Path, context: dict[str, Any]) -> None:
    context_check(context)
    if source_context(root, context["versions"]) != context:
        raise ValueError("source head/tree/bytes differ")


def witness(
    context: dict[str, Any],
    mapping: dict[str, Any],
    index: int,
    projection: Projection | None = None,
) -> dict[str, Any]:
    return {
        "version": VERSION,
        "index": index,
        "phase": "prefix" if index < 2 else "target",
        "producer_lateral": str(POSITIONS[index]),
        "audit_lateral": str(POSITIONS[index]),
        "cell_sha256": sha256_bytes(CELL_BYTES),
        "source": context,
        "episode": mapping["episode"],
        "roles": mapping["tokens"],
        "projection_sha256": None
        if projection is None
        else sha256_bytes(projection.binding_bytes()),
    }


class Once:
    """One evaluator-local fetch of an already qualified immutable target."""

    def __init__(self, observation: TrustedObservation) -> None:
        self.observed, self.used = observation, False

    def observation(self, index: int) -> TrustedObservation:
        if type(index) is not int or index != 2 or self.used:
            raise PermissionError("one retained evaluator fetch required")
        self.used = True
        return self.observed


def before(p: Projection) -> dict[str, Any]:
    structured, unstructured = events.structured(p), events.unstructured(p)
    if events.reconstruct(unstructured) != structured:
        raise ValueError("lossless encoding failed")
    return {
        "input": json.loads(p._source),
        "state": json.loads(p._state),
        "features": json.loads(p.feature_bytes()),
        "binding": json.loads(p.binding_bytes()),
        "structured": json.loads(structured),
        "unstructured": json.loads(unstructured),
        "candidates": {m: json.loads(events.control(p, m)) for m in METHODS},
    }


class Contradiction(ValueError):
    """Qualified physical labels/events disagree with the frozen paper cell."""


def score(
    p: Projection,
    candidates: dict[str, Any],
    observation: TrustedObservation,
    tokens: list[str],
    branch: int,
) -> dict[str, Any]:
    target = {
        tokens[0]: (False, False, True, True),
        tokens[1]: (False, branch == 0, branch != 0, True),
        tokens[2]: (False, False, True, True),
    }
    outputs = {}
    for method in METHODS:
        result = events.evaluate(p, canonical_json_bytes(candidates[method]), Once(observation))
        outputs[method] = {
            "report": result.report(),
            "receipt": json.loads(result.receipt_bytes),
            "targets": [list(row) for row in result.target],
        }
        if result.target != tuple(target[t] for t in p.alignment):
            raise Contradiction("qualified event contradiction")
    return outputs


def conclusion(
    completion: str, qualification: str, outcome: str, branches: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    return {
        "version": VERSION,
        "completion": completion,
        "qualification": qualification,
        "cell_outcome": outcome,
        "branches": [] if branches is None else branches,
        "scope": "EXACT_PUBLIC_CELL_ONLY",
        "phase_gate_effect": "NONE",
    }


class Store:
    def __init__(self, path: Path, clock: Callable[[], float], external: Callable[[], int]) -> None:
        self.path, self.clock, self.external = path, clock, external
        self.started = clock()
        self.hashes: dict[str, str] = {}
        self.bytes = 0

    def check(self, reserve: int = RESERVE) -> None:
        if self.clock() - self.started > SECONDS:
            raise TimeoutError("inclusive cooperative 60-second limit exceeded")
        if self.account(reserve) > LIMIT:
            raise OSError("inclusive retained/archive/external/review limit exceeded")

    def account(self, additional: int = 0) -> int:
        external = self.external()
        if type(external) is not int or external < 0:
            raise ValueError("nonnegative measured external bytes required")
        return 2 * (self.bytes + external + LOG + additional) + REVIEW

    def write(self, name: str, value: Any, *, closure: bool = False) -> None:
        raw = canonical_json_bytes(value)
        if len(raw) > MAX_RETURN:
            raise ValueError("bounded retained object contract exceeded")
        if not closure:
            self.check()
        # Closure/returned-object writes consume the reserve even after timeout.
        if self.account(len(raw)) > LIMIT:
            raise OSError("retention reserve exhausted")
        with (self.path / name).open("xb") as stream:
            stream.write(raw)
            stream.flush()
        self.hashes[name] = sha256_bytes(raw)
        self.bytes += len(raw)

    def verify(self) -> None:
        for name, digest in self.hashes.items():
            path = self.path / name
            if path.is_symlink() or sha256_bytes(path.read_bytes()) != digest:
                raise ValueError("retained source bytes changed")


def read(path: Path, name: str) -> Any:
    raw = (path / name).read_bytes()
    result = json.loads(raw)
    if raw != canonical_json_bytes(result):
        raise ValueError("canonical retained bytes required")
    return result


def terminal_check(finish: Any) -> None:
    """Validate operational metadata before using any recorded scientific disposition."""
    if (
        type(finish) is not dict
        or set(finish)
        != {
            "version",
            "completion",
            "analysis",
            "failure",
            "attempted_calls",
            "elapsed_seconds",
            "external_bytes",
            "files",
        }
        or finish["version"] != VERSION
        or finish["completion"] not in ("COMPLETE", "STOPPED_INCOMPLETE")
        or type(finish["files"]) is not dict
        or not set(finish["files"]) <= ALLOWED
        or type(finish["attempted_calls"]) is not int
        or not 0 <= finish["attempted_calls"] <= 8
        or type(finish["external_bytes"]) is not int
        or finish["external_bytes"] < 0
        or type(finish["elapsed_seconds"]) not in (int, float)
        or not math.isfinite(finish["elapsed_seconds"])
        or finish["elapsed_seconds"] < 0
    ):
        raise ValueError("strict finite terminal contract required")
    analysis, failure = finish["analysis"], finish["failure"]
    if (
        type(analysis) is not dict
        or set(analysis) != set(conclusion("", "", ""))
        or analysis["version"] != VERSION
        or analysis["completion"] != finish["completion"]
        or analysis["qualification"]
        not in ("UNRESOLVED", "PARTIAL_EXACT_GRID_AGREEMENT", "EXACT_CELL_MAPPING_AGREEMENT")
        or analysis["cell_outcome"] not in ("PASS", "FAIL", "INCONCLUSIVE")
        or analysis["scope"] != "EXACT_PUBLIC_CELL_ONLY"
        or analysis["phase_gate_effect"] != "NONE"
        or type(analysis["branches"]) is not list
    ):
        raise ValueError("terminal completion/analysis vocabulary differs")
    if failure is not None and (
        type(failure) is not dict
        or set(failure) != {"type", "message"}
        or type(failure["type"]) is not str
        or not failure["type"]
        or type(failure["message"]) is not str
    ):
        raise ValueError("typed terminal failure record required")
    if failure is not None:
        if failure["type"] == "Contradiction":
            if analysis["cell_outcome"] != "FAIL":
                raise ValueError("qualified Contradiction must record FAIL")
        elif (
            analysis["cell_outcome"] != "INCONCLUSIVE" or analysis["qualification"] != "UNRESOLVED"
        ):
            raise ValueError("operational failure must remain INCONCLUSIVE")
    if analysis["cell_outcome"] == "PASS" and (
        failure is not None
        or finish["completion"] != "COMPLETE"
        or analysis["qualification"] != "EXACT_CELL_MAPPING_AGREEMENT"
        or len(analysis["branches"]) != 2
        or finish["elapsed_seconds"] > SECONDS
    ):
        raise ValueError("PASS requires complete qualified failure-free operation")
    if (
        analysis["cell_outcome"] == "FAIL"
        and analysis["qualification"] != "PARTIAL_EXACT_GRID_AGREEMENT"
    ):
        raise ValueError("FAIL requires qualified contradictory evidence")
    if (
        analysis["cell_outcome"] == "INCONCLUSIVE"
        and analysis["qualification"] == "EXACT_CELL_MAPPING_AGREEMENT"
    ):
        raise ValueError("INCONCLUSIVE cannot certify exact mapping")
    if finish["completion"] == "COMPLETE" and (
        finish["attempted_calls"] != 8
        or not {
            f"{i}-{kind}-{suffix}.json"
            for i in range(4)
            for kind in ("raw", "audit")
            for suffix in ("attempt", "return")
        }
        <= set(finish["files"])
    ):
        raise ValueError("COMPLETE requires all eight attempted and retained returns")


def inspect(
    path: Path,
    access: ModalityPermissionSet,
    source_check: Callable[[], None],
    *,
    terminal: bool = True,
) -> dict[str, Any]:
    """Reconstruct from retained returns; never invoke a producer/reference."""
    privileged(access)  # Before source callback or filesystem access.
    source_check()
    finish = read(path, "terminal.json") if terminal else None
    if finish is not None:
        terminal_check(finish)
        actual = {
            p.name: sha256_bytes(p.read_bytes())
            for p in path.iterdir()
            if p.name != "terminal.json" and p.is_file() and not p.is_symlink()
        }
        if (
            any(not p.is_file() or p.is_symlink() for p in path.iterdir())
            or actual != finish["files"]
        ):
            raise ValueError("retained terminal file/hash set differs")
        if finish["attempted_calls"] != sum(n.endswith("-attempt.json") for n in actual):
            raise ValueError("attempted call accounting differs")
        size = sum(p.stat().st_size for p in path.iterdir())
        if 2 * (size + finish["external_bytes"] + LOG) + REVIEW > LIMIT:
            raise ValueError("retained-inclusive final accounting exceeds limit")
        attempts = [f"{i}-{k}-attempt.json" for i in range(4) for k in ("raw", "audit")]
        if {n for n in actual if n.endswith("-attempt.json")} != set(
            attempts[: finish["attempted_calls"]]
        ):
            raise ValueError("calls must form a single ordered prefix")
        if any(
            n.endswith("-return.json") and n.replace("-return", "-attempt") not in actual
            for n in actual
        ):
            raise ValueError("return without attempted argument record")
        if any(n.startswith(("2-", "3-")) for n in actual) and "seal.json" not in actual:
            raise ValueError("future operations require the global before seal")
        if not INITIAL <= set(actual):
            output = conclusion("STOPPED_INCOMPLETE", "UNRESOLVED", "INCONCLUSIVE")
            if finish["analysis"] != output or finish["failure"] is None:
                raise ValueError("incomplete claim must retain an inconclusive failure")
            return output
    context, mapping = read(path, "context.json"), read(path, "identity.json")
    context_check(context)
    check_identity(mapping)
    if (path / "cell.json").read_bytes() != CELL_BYTES:
        raise ValueError("fixed prospective cell changed")
    labels = []
    projected: tuple[Projection, Projection] | None = None
    branches: list[dict[str, Any]] = []
    seal_members = {"cell.json", "context.json", "identity.json"}
    outcome = "INCONCLUSIVE"
    qualification = "UNRESOLVED"
    count = 0
    for index in range(4):
        if index == 2:
            if projected is None or not (path / "seal.json").exists():
                break
            seal = read(path, "seal.json")
            if set(seal) != seal_members or any(
                sha256_bytes((path / n).read_bytes()) != digest for n, digest in seal.items()
            ):
                raise ValueError("complete before-only global seal differs")
        p = None if index < 2 else cast(tuple[Projection, Projection], projected)[index - 2]
        expected_witness = witness(context, mapping, index, p)
        returns = []
        for kind in ("raw", "audit"):
            stem = f"{index}-{kind}"
            attempt_path = path / f"{stem}-attempt.json"
            if not attempt_path.exists():
                break
            if read(path, attempt_path.name) != {"kind": kind, "arguments": expected_witness}:
                raise ValueError("retained attempted arguments/order differ")
            count += 1
            name = f"{stem}-return.json"
            if not (path / name).exists():
                break
            returns.append(read(path, name))
            if index < 2:
                seal_members.update((attempt_path.name, name))
        if len(returns) != 2:
            break
        try:
            grid = qualified(*returns)
        except (ValueError, TypeError, KeyError):
            # Disagreement, ties or boundary evidence do not qualify truth.
            break
        qualification = "PARTIAL_EXACT_GRID_AGREEMENT"
        if grid != expected(index):
            outcome = "FAIL"
            break
        labels.append(grid)
        if index == 1:
            frames = tuple(
                visible(grid, i, tuple(mapping["tokens"])) for i, grid in enumerate(labels)
            )
            projected = projections(study(), cast(Any, frames), mapping, context["head"])
            for branch, projection in enumerate(projected):
                name = f"before-{branch}.json"
                if not (path / name).exists():
                    return conclusion("STOPPED_INCOMPLETE", qualification, "INCONCLUSIVE")
                if read(path, name) != before(projection):
                    raise ValueError("before-only projection/control bytes differ")
                seal_members.add(name)
        elif index >= 2:
            assert p is not None
            observation = TrustedObservation(
                visible(grid, 2, tuple(mapping["tokens"])), True, True, True
            )
            try:
                result = score(
                    p,
                    read(path, f"before-{index - 2}.json")["candidates"],
                    observation,
                    mapping["tokens"],
                    index - 2,
                )
            except Contradiction:
                outcome = "FAIL"
                break
            name = f"score-{index - 2}.json"
            if not (path / name).exists():
                break
            if read(path, name) != result:
                raise ValueError("retained exact event/control report differs")
            branches.append(result)
    if len(labels) == 4 and len(branches) == 2 and count == 8:
        qualification, outcome = "EXACT_CELL_MAPPING_AGREEMENT", "PASS"
    completion = "COMPLETE" if len(branches) == 2 and count == 8 else "STOPPED_INCOMPLETE"
    output = conclusion(completion, qualification, outcome, branches)
    if finish is not None:
        if finish["failure"] is not None and finish["failure"]["type"] != "Contradiction":
            output = conclusion(finish["completion"], "UNRESOLVED", "INCONCLUSIVE", branches)
        if output != finish["analysis"]:
            raise ValueError("retained terminal analysis differs")
        if finish["elapsed_seconds"] > SECONDS and output["cell_outcome"] != "INCONCLUSIVE":
            raise ValueError("time failure cannot qualify cell")
    return output


def deadline_failure(message: str, initiating: dict[str, str] | None) -> dict[str, str]:
    if initiating is not None:
        message += f"; initiating {initiating['type']}: {initiating['message']}"
    return {"type": "TimeoutError", "message": message}


def run_mapping(
    path: Path,
    access: ModalityPermissionSet,
    context: dict[str, Any],
    adapter: Adapter,
    source_check: Callable[[], None],
    external: Callable[[], int],
    protect: Callable[[Path], None],
    *,
    clock: Callable[[], float] = time.monotonic,
    allocate: Callable[[], dict[str, Any]] = identity,
) -> dict[str, Any]:
    """Callable source bridge. Invocation requires a later separately reviewed admission."""
    privileged(access)  # Before clock, identity, claim, callback or physical access.
    store = Store(path, clock, external)  # Clock starts before claiming output.
    store.check()  # Refuse a claim when closure/returned-object reserve cannot be funded.
    path.mkdir()  # Single claim: no reuse/refill/retry.
    failure = None
    analysis = conclusion("STOPPED_INCOMPLETE", "UNRESOLVED", "INCONCLUSIVE")
    calls = 0
    branches: list[dict[str, Any]] = []
    try:
        protect(path)
        source_check()
        context = json.loads(canonical_json_bytes(context))
        context_check(context)
        mapping = allocate()
        check_identity(mapping)
        for name, value in (
            ("cell.json", CELL),
            ("context.json", context),
            ("identity.json", mapping),
        ):
            store.write(name, value)
        s = study()
        grids = []
        projected: tuple[Projection, Projection] | None = None
        for index in range(4):
            p = None if index < 2 else cast(tuple[Projection, Projection], projected)[index - 2]
            w = witness(context, mapping, index, p)
            returned = []
            for kind, operation in (("raw", adapter.produce), ("audit", adapter.audit)):
                store.check()
                store.verify()
                source_check()
                if calls >= 8:
                    raise ValueError("fixed call cap exceeded")
                stem = f"{index}-{kind}"
                store.write(stem + "-attempt.json", {"kind": kind, "arguments": w})
                calls += 1
                value = plain(operation(s, json.loads(canonical_json_bytes(w)), store.check))
                store.write(stem + "-return.json", value, closure=True)
                returned.append(value)  # Persist before qualification or clock recheck.
                store.check()
            grid = qualified(*returned)
            if grid != expected(index):
                analysis = conclusion(
                    "STOPPED_INCOMPLETE", "PARTIAL_EXACT_GRID_AGREEMENT", "FAIL", branches
                )
                break
            grids.append(grid)
            if index == 1:
                frames = tuple(visible(g, i, tuple(mapping["tokens"])) for i, g in enumerate(grids))
                projected = projections(s, cast(Any, frames), mapping, context["head"])
                for branch, projection in enumerate(projected):
                    store.write(f"before-{branch}.json", before(projection))
                store.write("seal.json", dict(store.hashes))
            elif index >= 2:
                assert p is not None
                observation = TrustedObservation(
                    visible(grid, 2, tuple(mapping["tokens"])), True, True, True
                )
                result = score(
                    p,
                    read(path, f"before-{index - 2}.json")["candidates"],
                    observation,
                    mapping["tokens"],
                    index - 2,
                )
                store.write(f"score-{index - 2}.json", result)
                branches.append(result)
        store.verify()
        source_check()
        if len(branches) == 2 and calls == 8:
            analysis = conclusion("COMPLETE", "EXACT_CELL_MAPPING_AGREEMENT", "PASS", branches)
        store.check()
    except Contradiction as error:
        analysis = conclusion(
            "STOPPED_INCOMPLETE", "PARTIAL_EXACT_GRID_AGREEMENT", "FAIL", branches
        )
        failure = {"type": type(error).__name__, "message": str(error)}
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        analysis = conclusion("STOPPED_INCOMPLETE", "UNRESOLVED", "INCONCLUSIVE", branches)
    elapsed = clock() - store.started
    if elapsed > SECONDS:
        failure = deadline_failure("closure time bound exceeded", failure)
        analysis = conclusion("STOPPED_INCOMPLETE", "UNRESOLVED", "INCONCLUSIVE", branches)
    finish = {
        "version": VERSION,
        "completion": analysis["completion"],
        "analysis": analysis,
        "failure": failure,
        "attempted_calls": calls,
        "elapsed_seconds": elapsed,
        "external_bytes": external(),
        "files": dict(store.hashes),
    }
    store.write("terminal.json", finish, closure=True)
    # Finalize volatile terminal metadata after persistence; all scientific and
    # returned-object files remain immutable. Closure time failure cannot return PASS.
    closed = clock() - store.started
    if closed > SECONDS and finish["elapsed_seconds"] <= SECONDS:
        analysis = conclusion("STOPPED_INCOMPLETE", "UNRESOLVED", "INCONCLUSIVE", branches)
        finish.update(
            completion="STOPPED_INCOMPLETE",
            analysis=analysis,
            elapsed_seconds=closed,
            failure=deadline_failure("terminal closure crossed time cap", finish["failure"]),
        )
        raw = canonical_json_bytes(finish)
        if store.account(len(raw)) > LIMIT:
            raise OSError(
                "terminal finalization reserve exhausted; packet must remain inconclusive"
            )
        (path / "terminal.json").write_bytes(raw)
    return analysis
