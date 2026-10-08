"""Fixed four-episode exact-synthetic area census; no launch or historical runner."""

from __future__ import annotations

import json
import math
import re
import subprocess
import zipfile
from collections.abc import Callable
from fractions import Fraction as Q
from pathlib import Path
from typing import Any, cast

from epsbench.diagnostics import known_region_area as area
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
    rows,
    visible,
)
from epsbench.diagnostics.restricted_exact_raster import Box
from epsbench.diagnostics.restricted_mask_projection import Projection
from epsbench.schema import Action, Modality, ModalityPermissionSet
from epsbench.utils.canonical import canonical_json_bytes, sha256_bytes

VERSION = "known-region-area-census-v1"
DESIGN = "0005d47e5c40234bf86143504df2eaabdcd5446bfc6185d871838b1696cea871"
SECONDS, LIMIT, REVIEW, RESERVE = 60, 32 * 1024**2, 8 * 1024**2, 1024**2
PRIVILEGED = frozenset({Modality.PRIVILEGED_GENERATION_RECORDS})
SOURCES = (
    *SOURCE_FILES,
    "src/epsbench/diagnostics/known_region_events.py",
    "src/epsbench/diagnostics/known_region_area.py",
    "src/epsbench/diagnostics/known_region_area_census.py",
    "src/epsbench/appearance.py",
    "src/epsbench/utils/seeding.py",
)
PARENTS = (
    (
        (("-11/20", "1/2", "1/4"), ("9/20", "3/4", "7/4")),
        (("-3/5", "9/4", "3/8"), ("4/5", "5/2", "13/8")),
    ),
    (
        (("12/5", "1/4", "2/5"), ("17/5", "1/2", "8/5")),
        (("-9/5", "7/4", "1/3"), ("-3/5", "2", "5/3")),
    ),
)
POSITIONS = ((Q(-5, 8), Q(0), Q(-3, 8), Q(3, 8)), (Q(5, 8), Q(0), Q(-3, 8), Q(3, 8)))
ORDER = tuple(
    (e, v, k)
    for phase in ((0, 1), (2, 3))
    for e in range(4)
    for v in phase
    for k in ("raw", "audit")
)
MANIFEST = canonical_json_bytes(
    {
        "version": VERSION,
        "design_sha256": DESIGN,
        "parents": PARENTS,
        "support": ["13/3", "17/3"],
        "positions": [[str(q) for q in p] for p in POSITIONS],
        "membership": ["C", "C-mirror", "F", "F-mirror"],
        "controls": area.CONTROLS,
        "calibration": {"size": 32, "forward": "-3", "elevation": "1", "fovy": "90", "up_y": "0"},
        "limits": {
            "raw": 16,
            "audit": 16,
            "seconds": SECONDS,
            "inclusive_bytes": LIMIT,
            "review_reserve": REVIEW,
        },
    }
)
INITIAL = {"manifest.json", "context.json", "identities.json"}
ALLOWED = (
    INITIAL
    | {"seal.json", "terminal.json"}
    | {f"before-{e}.json" for e in range(4)}
    | {f"evaluation-{e}.json" for e in range(4)}
    | {f"{e}-{v}-{k}-{suffix}.json" for e, v, k in ORDER for suffix in ("attempt", "return")}
)


def privileged(access: ModalityPermissionSet) -> None:
    if (
        type(access) is not ModalityPermissionSet
        or access.allowed != PRIVILEGED
        or ModalityPermissionSet.model_validate_json(access.model_dump_json()).allowed != PRIVILEGED
    ):
        raise PermissionError("exact typed privileged-generation permission required")


def action(q: Q) -> Action:
    return Action(
        name="lateral_left" if q < 0 else "lateral_right",
        delta_forward=0,
        delta_lateral=float(q),
        delta_yaw=0,
    )


def study(episode: int) -> Study:
    if type(episode) is not int or not 0 <= episode < 4:
        raise ValueError("fixed episode membership required")
    boxes = []
    for lower, upper in PARENTS[episode // 2]:
        lo, hi = tuple(map(Q, lower)), tuple(map(Q, upper))
        if episode % 2:
            lo, hi = (-hi[0], *lo[1:]), (-lo[0], *hi[1:])
        boxes.append(Box(cast(Any, lo), cast(Any, hi)))
    p = POSITIONS[episode % 2]
    return Study(
        MANIFEST,
        DESIGN,
        DESIGN,
        cast(Any, tuple(boxes)),
        (Q(13, 3), Q(17, 3)),
        p[:2],
        action(-p[0]),
        (action(p[2]), action(p[3])),
        p[2:],
    )


def context_check(context: dict[str, Any]) -> None:
    if (
        type(context) is not dict
        or set(context) != {"head", "tree", "sources", "versions"}
        or type(context["versions"]) is not dict
    ):
        raise ValueError("exact source/runtime context required")
    for key, length in (("head", 40), ("tree", 40)):
        if (
            type(context[key]) is not str
            or re.fullmatch(r"[0-9a-f]{" + str(length) + "}", context[key]) is None
        ):
            raise ValueError("exact source identity required")
    if (
        type(context["sources"]) is not dict
        or set(context["sources"]) != set(SOURCES)
        or any(
            type(v) is not str or re.fullmatch(r"[0-9a-f]{64}", v) is None
            for v in context["sources"].values()
        )
    ):
        raise ValueError("complete source-byte closure required")


def verify_source(root: Path, context: dict[str, Any], access: ModalityPermissionSet) -> None:
    privileged(access)
    context_check(context)
    if any(sha256_bytes((root / n).read_bytes()) != h for n, h in context["sources"].items()):
        raise ValueError("source bytes differ")
    actual = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD", "HEAD^{tree}"],
        capture_output=True,
        check=True,
        text=True,
        timeout=5,
    ).stdout.splitlines()
    if actual != [context["head"], context["tree"]]:
        raise ValueError("source head/tree differs")


class Contradiction(ValueError):
    """Complete valid evidence contradicts the declared mapping/qualification."""


def qualify(raw: dict[str, Any], audit: dict[str, Any]) -> tuple[tuple[int, ...], ...]:
    # Only disagreement of otherwise independently admissible evidence is FAIL.
    labels = qualified(raw, {**audit, "labels": raw.get("labels")})
    audit_labels = rows(audit["labels"])
    if labels != audit_labels:
        raise Contradiction("complete producer/reference label disagreement")
    return labels


def clipping(audit: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for support, f in zip(audit["supports"], audit["footprints"], strict=True):
        polygon = tuple(tuple(Q(v) for v in p) for p in f["polygon"])
        result.append(
            {
                "full_centres": support[0],
                "clipped_centres": support[1],
                "footprint_outside_fov": any(abs(v) > 1 for p in polygon for v in p),
                "footprint_touches_fov": any(abs(v) == 1 for p in polygon for v in p),
            }
        )
    return result  # No inference of optical occlusion cause.


def witness(
    context: dict[str, Any], mapping: dict[str, Any], episode: int, view: int, p: Projection | None
) -> dict[str, Any]:
    return {
        "manifest_sha256": sha256_bytes(MANIFEST),
        "head": context["head"],
        "tree": context["tree"],
        "episode_slot": episode,
        "episode": mapping["episode"],
        "tokens": mapping["tokens"],
        "view": view,
        "producer_lateral": str(POSITIONS[episode % 2][view]),
        "audit_lateral": str(POSITIONS[episode % 2][view]),
        "binding_sha256": None if p is None else sha256_bytes(p.binding_bytes()),
    }


def before(ps: tuple[Projection, Projection]) -> dict[str, Any]:
    return {
        "bindings": [p.binding_bytes().hex() for p in ps],
        "features": [p.feature_bytes().hex() for p in ps],
        "candidates": {
            method: [area.control(p, method).hex() for p in ps] for method in area.CONTROLS
        },
    }


def arms(record: dict[str, Any]) -> tuple[tuple[str, tuple[bytes, ...]], ...]:
    return tuple(
        (m, tuple(bytes.fromhex(v) for v in record["candidates"][m])) for m in area.CONTROLS
    )


class Once:
    def __init__(self, observation: TrustedObservation) -> None:
        self.value, self.calls = observation, 0

    def observation(self, index: int) -> TrustedObservation:
        if type(index) is not int or index != 2 or self.calls:
            raise PermissionError("retained target may be fetched exactly once")
        self.calls += 1
        return self.value


def summary(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if len(reports) != 4:
        raise ValueError("all four episodes required, no favorable partial aggregate")
    parents: list[dict[str, Any]] = []
    for g in range(2):
        pair = reports[2 * g : 2 * g + 2]
        controls = {
            m: {
                metric: str(sum((Q(r["arms"][m]["all"][metric]) for r in pair), Q(0)) / 2)
                for metric in ("mse", "mae", "bias", "mean_regret")
            }
            for m in area.CONTROLS
        }
        contrast = any(Q(v) > 0 for r in pair for v in r["action_contrasts"])
        parents.append(
            {
                "contrast": contrast,
                "controls": controls,
                "episodes": [2 * g, 2 * g + 1],
                "fixed_first_mean_regret": str(
                    sum((Q(r["fixed_first_mean_regret"]) for r in pair), Q(0)) / 2
                ),
                "fixed_last_mean_regret": str(
                    sum((Q(r["fixed_last_mean_regret"]) for r in pair), Q(0)) / 2
                ),
            }
        )
    passed = all(
        p["contrast"] and all(Q(c["mean_regret"]) > 0 for c in p["controls"].values())
        for p in parents
    )
    return {
        "parents": parents,
        "all_controls": {
            m: {
                metric: str(sum((Q(p["controls"][m][metric]) for p in parents), Q(0)) / 2)
                for metric in ("mse", "mae", "bias", "mean_regret")
            }
            for m in area.CONTROLS
        },
        "residual": "PASS" if passed else "FAIL",
        "next": "SCIENTIFIC_READINESS_DECISION_ONLY" if passed else "STOP_DECISION_RESIDUAL_ROUTE",
        "scope": "finite development census; no practical margin, population or model authority",
    }


def _read(path: Path, name: str) -> Any:
    raw = (path / name).read_bytes()
    value = json.loads(raw)
    if canonical_json_bytes(value) != raw:
        raise ValueError("canonical retained bytes required")
    return value


def _replay(
    path: Path,
    context: dict[str, Any],
    mappings: list[dict[str, Any]],
    access: ModalityPermissionSet,
    save_evaluation: Callable[[str, bytes], None] | None = None,
) -> dict[str, Any]:
    """Retained-only recomputation, including partial ordered call sequences."""
    privileged(access)
    grids: dict[tuple[int, int], tuple[tuple[int, ...], ...]] = {}
    clips: dict[str, Any] = {}
    projected: list[tuple[Projection, Projection]] = []
    for j in range(0, len(ORDER), 2):
        e, v, _ = ORDER[j]
        p = None
        if v >= 2:
            if not projected:
                projected = [
                    projections(
                        study(i),
                        (
                            visible(grids[i, 0], 0, tuple(mappings[i]["tokens"])),
                            visible(grids[i, 1], 1, tuple(mappings[i]["tokens"])),
                        ),
                        mappings[i],
                        context["head"],
                    )
                    for i in range(4)
                ]
                expected_before = {}
                for i, ps in enumerate(projected):
                    name = f"before-{i}.json"
                    if _read(path, name) != before(ps):
                        raise Contradiction("retained before-only forecast/binding mismatch")
                    expected_before[name] = sha256_bytes((path / name).read_bytes())
                seal = _read(path, "seal.json")
                if seal != expected_before:
                    raise Contradiction("global48 forecast seal mismatch")
            p = projected[e][v - 2]
        for kind in ("raw", "audit"):
            stem = f"{e}-{v}-{kind}"
            if not (path / f"{stem}-attempt.json").exists():
                return {"apparatus": "INCONCLUSIVE", "residual": "INCONCLUSIVE", "aggregate": None}
            if _read(path, f"{stem}-attempt.json") != {
                "kind": kind,
                "arguments": witness(context, mappings[e], e, v, p),
            }:
                raise Contradiction("retained attempted arguments mismatch")
            if not (path / f"{stem}-return.json").exists():
                return {"apparatus": "INCONCLUSIVE", "residual": "INCONCLUSIVE", "aggregate": None}
        raw, audit = (_read(path, f"{e}-{v}-{k}-return.json") for k in ("raw", "audit"))
        grids[e, v] = qualify(raw, audit)
        clips[f"{e}-{v}"] = clipping(audit)
    for e in (0, 2):
        for view in range(4):
            other = 3 if view == 2 else 2 if view == 3 else view
            if grids[e, view] != tuple(tuple(reversed(row)) for row in grids[e + 1, other]):
                raise Contradiction("qualified role-aligned mirrored grids differ")
    reports = []
    for e, ps in enumerate(projected):
        providers = tuple(
            Once(
                TrustedObservation(
                    visible(grids[e, v], 2, tuple(mappings[e]["tokens"])), True, True, True
                )
            )
            for v in (2, 3)
        )
        result = area.evaluate(ps, arms(_read(path, f"before-{e}.json")), providers)
        name = f"evaluation-{e}.json"
        if (path / name).exists() and (path / name).read_bytes() != result.receipt_bytes:
            raise Contradiction("retained area evaluation differs")
        if save_evaluation is not None:
            save_evaluation(name, result.receipt_bytes)
        reports.append(result.report())
    return {
        "apparatus": "PASS",
        "residual": summary(reports)["residual"],
        "aggregate": summary(reports),
        "episodes": reports,
        "clipping": clips,
    }


def inspect(
    path: Path, access: ModalityPermissionSet, source_check: Callable[[], None]
) -> dict[str, Any]:
    privileged(access)
    source_check()
    files = list(path.iterdir())
    if any(not f.is_file() or f.is_symlink() or f.name not in ALLOWED for f in files):
        raise ValueError("closed regular retained file set required")
    finish = _read(path, "terminal.json")
    if (
        type(finish) is not dict
        or set(finish) != {"version", "files", "attempted", "elapsed", "failure", "analysis"}
        or finish["version"] != VERSION
    ):
        raise ValueError("strict terminal vocabulary required")
    actual = {f.name: sha256_bytes(f.read_bytes()) for f in files if f.name != "terminal.json"}
    if (
        actual != finish["files"]
        or type(finish["attempted"]) is not int
        or not 0 <= finish["attempted"] <= 32
    ):
        raise ValueError("terminal file hashes/counts differ")
    attempts = [f"{e}-{v}-{k}-attempt.json" for e, v, k in ORDER]
    if {n for n in actual if n.endswith("-attempt.json")} != set(
        attempts[: finish["attempted"]]
    ) or any(
        n.endswith("-return.json") and n.replace("-return", "-attempt") not in actual
        for n in actual
    ):
        raise ValueError("ordered attempt/return chronology differs")
    elapsed, failure = finish["elapsed"], finish["failure"]
    if (
        type(elapsed) not in (int, float)
        or not math.isfinite(elapsed)
        or elapsed < 0
        or (
            failure is not None
            and (
                type(failure) is not dict
                or set(failure) != {"type", "message"}
                or any(type(v) is not str for v in failure.values())
            )
        )
    ):
        raise ValueError("strict timing/failure evidence required")
    if not INITIAL <= set(actual):
        output = {"apparatus": "INCONCLUSIVE", "residual": "INCONCLUSIVE", "aggregate": None}
    else:
        if (path / "manifest.json").read_bytes() != MANIFEST:
            raise ValueError("prospective manifest changed")
        context, mappings = _read(path, "context.json"), _read(path, "identities.json")
        context_check(context)
        _identities(mappings)
        try:
            output = _replay(path, context, mappings, access)
        except Contradiction as error:
            output = {"apparatus": "FAIL", "residual": "INCONCLUSIVE", "aggregate": None}
            if failure is None or (
                failure["type"] == "Contradiction" and failure["message"] != str(error)
            ):
                raise ValueError("qualified contradiction failure evidence differs") from error
        except (ValueError, TypeError, KeyError, FileNotFoundError):
            output = {"apparatus": "INCONCLUSIVE", "residual": "INCONCLUSIVE", "aggregate": None}
    if failure is not None and failure["type"] != "Contradiction":
        output = {"apparatus": "INCONCLUSIVE", "residual": "INCONCLUSIVE", "aggregate": None}
    if elapsed > SECONDS and (failure is None or failure["type"] != "TimeoutError"):
        raise ValueError("timeout takes operational precedence")
    if (
        output != finish["analysis"]
        or (
            output["apparatus"] == "PASS"
            and (
                failure is not None
                or finish["attempted"] != 32
                or not all(f"evaluation-{e}.json" in actual for e in range(4))
            )
        )
        or (output["apparatus"] == "INCONCLUSIVE" and failure is None)
    ):
        raise ValueError("terminal analysis/failure/completion mismatch")
    return output


def _identities(mappings: Any) -> None:
    if type(mappings) is not list or len(mappings) != 4:
        raise ValueError("four fixed opaque mappings required")
    for m in mappings:
        check_identity(m)
    if (
        len({m["episode"] for m in mappings}) != 4
        or len({t for m in mappings for t in m["tokens"]}) != 12
    ):
        raise ValueError("independent episode-random opaque mappings required")


def archive(
    path: Path, destination: Path, access: ModalityPermissionSet, check: Callable[[], None]
) -> dict[str, str]:
    """Exclusive stdlib ZIP; verify originals and archived bytes without extraction."""
    privileged(access)
    check()
    if (
        destination.resolve().parent != path.resolve().parent
        or destination.resolve() == path.resolve()
    ):
        raise ValueError("archive must be separate sibling of packet")
    files = sorted(path.iterdir())
    if any(not p.is_file() or p.is_symlink() for p in files):
        raise ValueError("regular immutable originals required")
    catalog = {p.name: sha256_bytes(p.read_bytes()) for p in files}
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_STORED) as z:
        for p in files:
            check()
            data = p.read_bytes()
            if sha256_bytes(data) != catalog[p.name]:
                raise ValueError("original changed during archive")
            z.writestr(zipfile.ZipInfo(p.name), data)
    check()
    with zipfile.ZipFile(destination, "r") as z:
        if z.namelist() != [p.name for p in files]:
            raise ValueError("archive membership differs")
        for p in files:
            check()
            if (
                sha256_bytes(z.read(p.name)) != catalog[p.name]
                or sha256_bytes(p.read_bytes()) != catalog[p.name]
            ):
                raise ValueError("archive/original bytes differ")
    check()
    return catalog


def run(
    path: Path,
    access: ModalityPermissionSet,
    adapter: Adapter,
    context: dict[str, Any],
    source_check: Callable[[], None],
    protect: Callable[[Path], None],
    external_bytes: Callable[[], int],
    clock: Callable[[], float],
    started: float,
) -> dict[str, Any]:
    """Already-protected attempt parent; no launch authority conferred by this API."""
    privileged(access)  # Before callback/source/filesystem/provider access.
    context_check(context)
    context = json.loads(canonical_json_bytes(context))
    if path.name != "packet" or type(started) not in (int, float) or not math.isfinite(started):
        raise ValueError("new packet child and first-claim timestamp required")
    source_check()
    owned: dict[str, str] = {}
    owned_bytes, attempted, claimed = 0, 0, False
    failure, analysis = (
        None,
        {"apparatus": "INCONCLUSIVE", "residual": "INCONCLUSIVE", "aggregate": None},
    )

    def check(reserve: int = RESERVE) -> None:
        elapsed = clock() - started
        overhead = external_bytes()
        if not math.isfinite(elapsed) or elapsed < 0 or elapsed > SECONDS:
            raise TimeoutError("whole-operation cooperative deadline exceeded")
        if (
            type(overhead) is not int
            or overhead < 0
            or 2 * owned_bytes + overhead + REVIEW + reserve > LIMIT
        ):
            raise OSError("inclusive packet/archive/staging/log/review ceiling exceeded")

    def write(name: str, data: bytes, terminal: bool = False) -> None:
        nonlocal owned_bytes
        if not terminal:
            check(RESERVE + 2 * len(data))
        overhead = external_bytes()
        if type(overhead) is not int or overhead < 0 or owned_bytes + overhead + len(data) > LIMIT:
            raise OSError("actual inclusive write ceiling exceeded")
        with (path / name).open("xb") as f:
            f.write(data)
            f.flush()
        owned_bytes += len(data)
        owned[name] = sha256_bytes(data)

    def record(name: str, value: Any) -> None:
        write(name, canonical_json_bytes(plain(value)))

    try:
        check()
        path.mkdir(exist_ok=False)
        claimed = True
        protect(path)
        check()
        record("manifest.json", json.loads(MANIFEST))
        record("context.json", context)
        mappings = [identity() for _ in range(4)]
        _identities(mappings)
        record("identities.json", mappings)
        grids: dict[tuple[int, int], tuple[tuple[int, ...], ...]] = {}
        projected: list[tuple[Projection, Projection]] = []
        for j in range(0, len(ORDER), 2):
            e, v, _ = ORDER[j]
            source_check()
            if v >= 2 and not projected:
                projected = [
                    projections(
                        study(i),
                        (
                            visible(grids[i, 0], 0, tuple(mappings[i]["tokens"])),
                            visible(grids[i, 1], 1, tuple(mappings[i]["tokens"])),
                        ),
                        mappings[i],
                        context["head"],
                    )
                    for i in range(4)
                ]
                for i, ps in enumerate(projected):
                    b = before(ps)
                    for _, candidates in arms(b):
                        for projection, c in zip(ps, candidates, strict=True):
                            area.validate(projection, c)
                    record(f"before-{i}.json", b)
                record(
                    "seal.json", {f"before-{i}.json": owned[f"before-{i}.json"] for i in range(4)}
                )
            p = None if v < 2 else projected[e][v - 2]
            w = witness(context, mappings[e], e, v, p)
            returned = []
            for k, call in (("raw", adapter.produce), ("audit", adapter.audit)):
                check(2 * RESERVE)
                record(f"{e}-{v}-{k}-attempt.json", {"kind": k, "arguments": w})
                attempted += 1
                value = plain(call(study(e), json.loads(canonical_json_bytes(w)), check))
                # A bounded returned object is evidence even if the call crossed time.
                data = canonical_json_bytes(value)
                overhead = external_bytes()
                if (
                    type(overhead) is not int
                    or overhead < 0
                    or owned_bytes + overhead + len(data) > LIMIT
                ):
                    raise OSError("returned evidence exceeds actual inclusive ceiling")
                write(f"{e}-{v}-{k}-return.json", data, terminal=True)
                returned.append(value)
                check()
            grids[e, v] = qualify(*returned)
        source_check()
        analysis = _replay(path, context, mappings, access, write)
        check()
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        analysis = {
            "apparatus": "FAIL" if type(error) is Contradiction else "INCONCLUSIVE",
            "residual": "INCONCLUSIVE",
            "aggregate": None,
        }
    elapsed = clock() - started
    if elapsed > SECONDS:
        failure = {
            "type": "TimeoutError",
            "message": "deadline before terminal; initiating " + str(failure),
        }
        analysis = {"apparatus": "INCONCLUSIVE", "residual": "INCONCLUSIVE", "aggregate": None}
    finish = {
        "version": VERSION,
        "files": dict(owned),
        "attempted": attempted,
        "elapsed": elapsed,
        "failure": failure,
        "analysis": analysis,
    }
    outer: dict[str, Any] = {
        "version": VERSION + ":closure",
        "inner": finish,
        "inspection": None,
        "archive_files": None,
        "outer_outcome": "INCONCLUSIVE",
        "failure": None,
    }
    try:
        if not claimed:
            raise OSError("packet not exclusively claimed; originals untouched")
        write("terminal.json", canonical_json_bytes(finish), terminal=True)
        check()
        outer["inspection"] = inspect(path, access, source_check)
        check()
    except Exception as error:
        outer["failure"] = {"type": type(error).__name__, "message": str(error)}
    try:
        if claimed:
            outer["archive_files"] = archive(path, path.parent / "originals.zip", access, check)
            check()
            source_check()
    except Exception as error:
        outer["failure"] = {
            "type": type(error).__name__,
            "message": str(error) + "; initiating " + str(outer["failure"]),
        }
    if (
        outer["failure"] is None
        and outer["inspection"] is not None
        and outer["archive_files"] is not None
    ):
        outer["outer_outcome"] = analysis["apparatus"]
    # Outer closure never rewrites the immutable inner packet/terminal.
    outer["elapsed"] = clock() - started
    if outer["elapsed"] > SECONDS:
        outer["outer_outcome"] = "INCONCLUSIVE"
        outer["failure"] = {
            "type": "TimeoutError",
            "message": "deadline through archive/closure; initiating " + str(outer["failure"]),
        }
    data = canonical_json_bytes(outer)
    # Mandatory final byte callback covers archive/partials plus planned closure bytes.
    try:
        check(len(data))
    except Exception as error:
        outer["outer_outcome"] = "INCONCLUSIVE"
        outer["failure"] = {
            "type": type(error).__name__,
            "message": str(error) + "; initiating " + str(outer["failure"]),
        }
        data = canonical_json_bytes(outer)
    overhead = external_bytes()
    if type(overhead) is not int or overhead < 0 or owned_bytes + overhead + len(data) > LIMIT:
        return {**outer, "outer_outcome": "INCONCLUSIVE", "closure_not_written": True}
    try:
        with (path.parent / "closure.json").open("xb") as f:
            f.write(data)
            f.flush()
    except Exception as error:
        return {
            **outer,
            "outer_outcome": "INCONCLUSIVE",
            "closure_not_written": True,
            "failure": {
                "type": type(error).__name__,
                "message": str(error) + "; initiating " + str(outer["failure"]),
            },
        }
    if clock() - started > SECONDS:
        # Returned dominant outcome records terminal-write overrun; wrapper retains it.
        return {
            **outer,
            "outer_outcome": "INCONCLUSIVE",
            "closure_write_overrun": True,
            "failure": {
                "type": "TimeoutError",
                "message": "closure write crossed deadline; initiating " + str(outer["failure"]),
            },
        }
    return outer
