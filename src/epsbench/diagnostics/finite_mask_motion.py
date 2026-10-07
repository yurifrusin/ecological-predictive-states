"""Frozen finite mask-motion rules and globally sealed two-phase collection.

No production happens at import; real operations require separate admission.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from collections.abc import Callable
from fractions import Fraction as Q
from pathlib import Path
from typing import Any, cast

import numpy as np

from epsbench.diagnostics import mask_development_qualification as shared
from epsbench.diagnostics.boundary_observation import VisibleRaster
from epsbench.diagnostics.causal_region_lifecycle import keys
from epsbench.diagnostics.neutral_observation_target import Candidate
from epsbench.diagnostics.restricted_mask_projection import Projection
from epsbench.diagnostics.visible_forecast_contract import permissions
from epsbench.schema import Action
from epsbench.utils.canonical import canonical_json_bytes, sha256_bytes

VERSION = "finite-mask-motion-v1"
METHODS = ("motion", "persistence", "empty", "reversed")
LIMIT = 32 * 1024 * 1024
REVIEW_RESERVE = 8 * 1024 * 1024
PROCESS_LIMIT = 1024**3
ENVIRONMENT_LIMIT = 2 * 1024**3
SOURCE_FILES = (*shared.SOURCE_FILES, "src/epsbench/diagnostics/finite_mask_motion.py")


def action(step: Q) -> Action:
    return Action(
        name="lateral_left" if step < 0 else "lateral_right",
        delta_forward=0,
        delta_lateral=float(step),
        delta_yaw=0,
    )


def grid(definition: bytes, adoption: bytes) -> tuple[shared.Study, ...]:
    """Pure declarations only; no sampling, identity allocation or qualification."""
    members = []
    for a in (Q(-49, 64), Q(-17, 64), Q(15, 64)):
        for b in (Q(-33, 64), Q(31, 64)):
            for d in (Q(2), Q(3)):
                payload = {
                    "version": shared.VERSION,
                    "definition_sha256": sha256_bytes(definition),
                    "adoption_sha256": sha256_bytes(adoption),
                    "calibration": {
                        "size": 32,
                        "forward": "-3",
                        "elevation": "1",
                        "up_y": "0",
                        "fovy": "90",
                        "clipping": "NONE",
                    },
                    "support": ["2", "2"],
                    "boxes": [
                        {"lower": [str(a), "1/4", "1/5"], "upper": [str(a + 1), "3/4", "7/5"]},
                        {
                            "lower": [str(b), str(d), "1/5"],
                            "upper": [str(b + 1), str(d + Q(1, 4)), "7/5"],
                        },
                    ],
                    "prefix": ["-3/4", "0"],
                    "executed_action": action(Q(3, 4)).model_dump(mode="json"),
                    "announced_actions": [
                        action(q).model_dump(mode="json") for q in (Q(-1, 2), Q(1, 2))
                    ],
                    "steps": ["-1/2", "1/2"],
                    "signals": list(shared.SIGNALS),
                    # Immutable shared Study parser envelope, not cohort authority.
                    "limits": {
                        "producer_calls": 4,
                        "audits": 4,
                        "seconds": 60,
                        "inclusive_bytes": shared.LIMIT,
                        "review_reserve_bytes": shared.REVIEW_RESERVE,
                    },
                }
                members.append(shared.parse(canonical_json_bytes(payload)))
    return tuple(members)


def physical_fingerprint(study: shared.Study) -> str:
    """Common physical schema for available-known comparisons, independent of hashes/IDs.

    Historical declarations must be converted to this same schema before comparison.
    This cannot certify novelty against unrecovered historical members (18 of 21).
    """
    p = json.loads(study.payload)
    return sha256_bytes(
        canonical_json_bytes(
            {
                "schema": "exact-box-geometry-calibration-support-v1",
                "calibration": p["calibration"],
                "support": p["support"],
                "boxes": sorted(p["boxes"], key=canonical_json_bytes),
            }
        )
    )


def membership(definition: bytes, adoption: bytes, exclusions: tuple[str, ...]) -> bytes:
    if (
        type(exclusions) is not tuple
        or len(set(exclusions)) != len(exclusions)
        or any(type(x) is not str or re.fullmatch(r"[0-9a-f]{64}", x) is None for x in exclusions)
    ):
        raise ValueError("canonical common-schema available-known fingerprints required")
    studies = grid(definition, adoption)
    fingerprints = tuple(physical_fingerprint(s) for s in studies)
    if len(set(fingerprints)) != 12 or set(fingerprints) & set(exclusions):
        raise ValueError("duplicate/known physical geometry; no substitution permitted")
    return canonical_json_bytes(
        {
            "version": VERSION,
            "methods": METHODS,
            "studies": [json.loads(s.payload) for s in studies],
            "fingerprints": fingerprints,
            "available_known_exclusions": sorted(exclusions),
            "novelty_limit": "18-of-21; not exhaustive",
            "rule": "shift[-8,8];unique;interior;strict;half-away;symmetric-clear",
            "limits": {
                "raw": 48,
                "audit": 48,
                "forecasts": 96,
                "seconds": 120,
                "inclusive_bytes": LIMIT,
                "review_reserve_bytes": REVIEW_RESERVE,
                "process_peak_bytes": PROCESS_LIMIT,
                "environment_bytes": ENVIRONMENT_LIMIT,
            },
        }
    )


def shift(mask: Any, k: int) -> Any:
    if (
        type(k) is not int
        or not isinstance(mask, np.ndarray)
        or mask.dtype != np.bool_
        or mask.shape != (32, 32)
    ):
        raise ValueError("integer shift and complete Boolean 32x32 mask required")
    out = np.zeros((32, 32), dtype=np.bool_)
    if -32 < k < 32:
        if k >= 0:
            out[:, k:] = mask[:, : 32 - k]
        else:
            out[:, : 32 + k] = mask[:, -k:]
    return out


def round_away(value: Q) -> int:
    if type(value) is not Q:
        raise ValueError("exact rational required")
    q = abs(value) + Q(1, 2)
    return (1 if value >= 0 else -1) * (q.numerator // q.denominator)


def estimate(previous: Any, current: Any) -> dict[str, Any]:
    shift(previous, 0)
    shift(current, 0)
    costs = [int(np.count_nonzero(shift(previous, k) ^ current)) for k in range(-8, 9)]
    best = [k for k, cost in zip(range(-8, 9), costs, strict=True) if cost == min(costs)]
    if not np.any(current):
        reason = "EMPTY_CURRENT"
    elif not np.any(previous):
        reason = "EMPTY_PREVIOUS"
    elif len(best) != 1:
        reason = "TIED"
    elif abs(best[0]) == 8:
        reason = "BOUNDARY"
    elif min(costs) >= costs[8]:
        reason = "NO_IMPROVEMENT"
    else:
        reason = "UNIQUE_INTERIOR_IMPROVEMENT"
    return {
        "costs": costs,
        "minimizers": best,
        "reason": reason,
        "k": best[0] if reason == "UNIQUE_INTERIOR_IMPROVEMENT" else 0,
    }


def forecast(p: Projection, method: str) -> tuple[bytes, dict[str, Any]]:
    if type(p) is not Projection:
        raise ValueError("typed Projection required")
    permissions(p.access)
    source = p.revalidate()
    if method not in METHODS or type(method) is not str or p.availability != (True, True):
        raise ValueError("fixed method and both lawful endpoint flags required")
    if (
        source.shape != (32, 32)
        or len(source.frames) != 2
        or source.executed != ((Q(0), Q(3, 4), Q(0)),)
        or source.announced not in ((Q(0), Q(-1, 2), Q(0)), (Q(0), Q(1, 2), Q(0)))
    ):
        raise ValueError("frozen complete chronology and genuine actions required")
    f = json.loads(p.feature_bytes())
    masks = []
    details = []
    for node in f["nodes"]:
        previous = np.array(node["previous"], dtype=np.bool_)
        current = np.array(node["current"], dtype=np.bool_)
        measured = estimate(previous, current)
        q = source.announced[1] * (-1 if method == "reversed" else 1)
        k = round_away(Q(measured["k"]) * q / source.executed[-1][1])
        masks.append(
            np.zeros((32, 32), dtype=np.bool_)
            if method == "empty"
            else current.copy()
            if method == "persistence"
            else shift(current, k)
        )
        details.append({**measured, "forecast_shift": k if method in ("motion", "reversed") else 0})
    occupancy = np.zeros((32, 32), dtype=np.int32)
    for mask in masks:
        occupancy += mask
    conflict = occupancy >= 2
    known = tuple((name, mask & ~conflict) for name, mask in zip(p.alignment, masks, strict=True))
    saved = p.save(known, np.zeros((32, 32), dtype=np.bool_))
    return saved, {
        "method": method,
        "nodes": details,
        "conflict_pixels": int(np.count_nonzero(conflict)),
    }


def contrast(saved_correct: bytes, saved_reversed: bytes) -> dict[str, Any]:
    a, b = (json.loads(raw)["candidate"] for raw in (saved_correct, saved_reversed))
    if set(a["known"]) != set(b["known"]) or a["new"] != b["new"]:
        raise ValueError("same complete inventory/NEW required")
    counts = {
        name: int(
            np.count_nonzero(
                np.array(a["known"][name], dtype=np.bool_)
                ^ np.array(b["known"][name], dtype=np.bool_)
            )
        )
        for name in a["known"]
    }
    return {
        "known_symmetric_difference": counts,
        "known_symmetric_difference_pixels": sum(counts.values()),
        "identical": saved_correct == saved_reversed,
    }


def verify_forecast(p: Projection, method: str, saved: bytes, details: dict[str, Any]) -> None:
    """Independent padded-domain arithmetic check; does not call estimator/shift/rounder."""
    source = p.revalidate()
    if p.availability != (True, True) or source.shape != (32, 32):
        raise ValueError("complete lawful fixed projection required")
    f = json.loads(p.feature_bytes())
    proposed = []
    checked_nodes = []
    for node in f["nodes"]:
        previous, current = (np.array(node[n], dtype=np.bool_) for n in ("previous", "current"))
        padded = np.pad(previous, ((0, 0), (8, 8)))
        costs = [int(np.count_nonzero(padded[:, 8 - k : 40 - k] != current)) for k in range(-8, 9)]
        minimizing = [i - 8 for i, cost in enumerate(costs) if cost == min(costs)]
        k = 0
        if not current.any():
            reason = "EMPTY_CURRENT"
        elif not previous.any():
            reason = "EMPTY_PREVIOUS"
        elif len(minimizing) > 1:
            reason = "TIED"
        elif minimizing[0] in (-8, 8):
            reason = "BOUNDARY"
        elif min(costs) == costs[8]:
            reason = "NO_IMPROVEMENT"
        else:
            reason, k = "UNIQUE_INTERIOR_IMPROVEMENT", minimizing[0]
        ratio = Q(k) * source.announced[1] / source.executed[-1][1]
        if method == "reversed":
            ratio = -ratio
        n, d = abs(ratio.numerator), ratio.denominator
        rounded = (2 * n + d) // (2 * d) * (-1 if ratio < 0 else 1)
        delta = rounded if method in ("motion", "reversed") else 0
        checked_nodes.append(
            {
                "costs": costs,
                "minimizers": minimizing,
                "reason": reason,
                "k": k,
                "forecast_shift": delta,
            }
        )
        padded_current = np.pad(current, ((0, 0), (32, 32)))
        proposed.append(
            np.zeros((32, 32), dtype=np.bool_)
            if method == "empty"
            else padded_current[:, 32 - delta : 64 - delta]
        )
    count = sum((a.astype(np.int32) for a in proposed), np.zeros((32, 32), dtype=np.int32))
    conflicts = count > 1
    expected = p.save(
        tuple((name, a & ~conflicts) for name, a in zip(p.alignment, proposed, strict=True)),
        np.zeros((32, 32), dtype=np.bool_),
    )
    expected_details = {
        "method": method,
        "nodes": checked_nodes,
        "conflict_pixels": int(conflicts.sum()),
    }
    if expected != saved or expected_details != details:
        raise ValueError("independent exact rule/canonical candidate differs")


class Sink(shared.Sink):
    def __init__(
        self,
        path: Path,
        clock: Callable[[], float],
        external: Callable[[], int],
        process_peak: Callable[[], int],
        environment_bytes: int,
    ) -> None:
        super().__init__(path, clock, external)
        self.process_peak = process_peak
        self.environment_bytes = environment_bytes
        self.peak = 0
        self.last_external: int | None = None

    def check(self) -> None:
        if self.clock() - self.started > 120:
            raise TimeoutError("inclusive cooperative study budget exceeded")
        peak = self.process_peak()
        if (
            type(peak) is not int
            or peak <= 0
            or peak > PROCESS_LIMIT
            or type(self.environment_bytes) is not int
            or not 0 < self.environment_bytes <= ENVIRONMENT_LIMIT
        ):
            raise ValueError("measured process/environment envelope exceeded or unavailable")
        self.peak = max(self.peak, peak)
        external = self.external()
        if (
            type(external) is not int
            or external < 0
            or (2 * (self.bytes + external + shared.LOG_RESERVE) + REVIEW_RESERVE > LIMIT)
        ):
            raise OSError("post-write inclusive accounting unavailable or exceeded")
        self.last_external = external

    def write(self, name: str, raw: bytes, terminal: bool = False) -> None:
        if Path(name).name != name or type(raw) is not bytes:
            raise ValueError("bounded local immutable file required")
        if not terminal:
            self.check()
        external = self.external()
        reserve = 0 if terminal else shared.TERMINAL_RESERVE
        if (
            type(external) is not int
            or external < 0
            or 2 * (self.bytes + len(raw) + external + shared.LOG_RESERVE + reserve)
            + REVIEW_RESERVE
            > LIMIT
        ):
            raise OSError("inclusive originals/archive/cache/wrapper/log/review cap exceeded")
        self.last_external = external
        with (self.path / name).open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        self.bytes += len(raw)
        self.hashes[name] = sha256_bytes(raw)
        if not terminal:
            self.check()  # Post-write expiration cannot qualify a result.


def check_context(context: dict[str, Any], manifest: bytes) -> None:
    keys(context, {"head", "tree", "manifest_sha256", "sources", "versions", "purpose"})
    if (
        context["purpose"] != VERSION
        or context["manifest_sha256"] != sha256_bytes(manifest)
        or type(context["versions"]) is not dict
        or not context["versions"]
    ):
        raise ValueError("exact cohort/runtime binding required")
    for key, size in (("head", 40), ("tree", 40)):
        if (
            type(context[key]) is not str
            or re.fullmatch(r"[0-9a-f]{" + str(size) + "}", context[key]) is None
        ):
            raise ValueError("exact head/tree required")
    values = keys(context["sources"], set(SOURCE_FILES)).values()
    if any(type(v) is not str or re.fullmatch(r"[0-9a-f]{64}", v) is None for v in values):
        raise ValueError("complete source bytes required")


def verify_source(root: Path, context: dict[str, Any], manifest: bytes) -> None:
    check_context(context, manifest)
    if any(sha256_bytes((root / n).read_bytes()) != v for n, v in context["sources"].items()):
        raise ValueError("source bytes changed")
    observed = subprocess.run(
        [
            "git",
            "-c",
            "safe.directory=" + root.resolve().as_posix(),
            "-C",
            str(root),
            "rev-parse",
            "HEAD",
            "HEAD^{tree}",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout.splitlines()
    if observed != [context["head"], context["tree"]]:
        raise ValueError("source head/tree differs")


def group_context(context: dict[str, Any], study: shared.Study) -> dict[str, Any]:
    return {**context, "manifest_sha256": sha256_bytes(study.payload)}


def count_channels(p: Projection, saved: bytes, value: Any) -> list[dict[str, Any]]:
    source = p.revalidate()
    candidate = Candidate.from_bytes(canonical_json_bytes(json.loads(saved)["candidate"]), source)
    rows = []
    for (name, predicted), (_, truth) in zip(
        (*candidate.channels.known, ("NEW", candidate.channels.new)),
        (*value.target.channels.known, ("NEW", value.target.channels.new)),
        strict=True,
    ):
        assert predicted is not None and truth is not None
        tp = int(np.count_nonzero(predicted & truth))
        fp = int(np.count_nonzero(predicted & ~truth))
        fn = int(np.count_nonzero(~predicted & truth))
        rows.append(
            {
                "channel": name,
                "prediction_pixels": tp + fp,
                "target_pixels": tp + fn,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "iou": str(Q(tp, tp + fp + fn)) if tp + fp + fn else None,
            }
        )
    return rows


def summarize(groups: list[dict[str, Any]]) -> dict[str, Any]:
    if len(groups) != 12:
        raise ValueError("complete twelve-group scores required")
    losses = {m: sum((Q(g["losses"][m]) for g in groups), Q(0)) / 12 for m in METHODS}
    flags = {
        "no_prefix_change": all(not g["prefix_change"] for g in groups),
        "all_targets_persistence": all(g["targets_persistence"] for g in groups),
        "no_target_action_contrast": all(not g["target_action_contrast"] for g in groups),
        "all_correct_reversed_identical": all(g["correct_reversed_identical"] for g in groups),
    }
    comparisons = {m: losses["motion"] < losses[m] for m in METHODS if m != "motion"}
    differences = [
        {
            "group": i,
            "paired": {m: str(Q(g["losses"]["motion"]) - Q(g["losses"][m])) for m in comparisons},
            "ties": {m: g["losses"]["motion"] == g["losses"][m] for m in comparisons},
        }
        for i, g in enumerate(groups)
    ]
    return {
        "finite_outcome": "POSITIVE"
        if not any(flags.values()) and all(comparisons.values())
        else "NEGATIVE",
        "losses": {m: str(v) for m, v in losses.items()},
        "flags": flags,
        "strict_comparisons": comparisons,
        "group_differences": differences,
        "groups": groups,
    }


def inspect(
    path: Path, context: dict[str, Any], source_check: Callable[[], None], *, targets: bool = False
) -> dict[str, Any]:
    """Reconstruct prefixes, rule candidates and neutral scores from retained originals."""
    manifest = (path / "manifest.json").read_bytes()
    check_context(context, manifest)
    source_check()
    p = shared.read(path, "manifest.json")
    definition, adoption = (
        (path / "definition.md").read_bytes(),
        (path / "adoption.md").read_bytes(),
    )
    if (
        manifest != membership(definition, adoption, tuple(p["available_known_exclusions"]))
        or shared.read(path, "context.json") != context
    ):
        raise ValueError("fixed canonical membership/context differs")
    seal = shared.read(path, "global-seal.json")
    keys(seal, {"version", "members", "forecasts", "manifest_sha256"})
    if (
        seal["version"] != VERSION
        or seal["manifest_sha256"] != sha256_bytes(manifest)
        or len(seal["forecasts"]) != 96
    ):
        raise ValueError("complete immutable global forecast seal required")
    for name, digest in seal["members"].items():
        if Path(name).name != name or sha256_bytes((path / name).read_bytes()) != digest:
            raise ValueError("sealed before-only original changed")
    expected_forecasts = [
        f"g{g:02d}-b{b}-{m}-candidate.json" for g in range(12) for b in range(2) for m in METHODS
    ]
    if seal["forecasts"] != expected_forecasts or any(
        n not in seal["members"] for n in expected_forecasts
    ):
        raise ValueError("exact 96-member forecast coverage required")
    expected_members = {"manifest.json", "definition.md", "adoption.md", "context.json"}
    for g in range(12):
        expected_members.add(f"g{g:02d}-identity.json")
        for b in range(2):
            expected_members.update(
                f"g{g:02d}-prefix{b}-{suffix}.json" for suffix in ("witness", "raw", "audit")
            )
            expected_members.update(
                f"g{g:02d}-b{b}-{suffix}.json"
                for suffix in ("input", "state", "projection", "candidate")
            )
            expected_members.add(f"g{g:02d}-b{b}-forecast-contrast.json")
            expected_members.update(
                f"g{g:02d}-b{b}-{m}-{suffix}.json"
                for m in METHODS
                for suffix in ("candidate", "rule")
            )
    expected_members.update(
        f"call-{i:02d}-{suffix}.json" for i in range(1, 49) for suffix in ("attempt", "complete")
    )
    if set(seal["members"]) != expected_members:
        raise ValueError("global seal must cover every before-only original and call record")
    phases = ("prefix", "target") if targets else ("prefix",)
    call_index = 0
    for phase in phases:
        for g in range(12):
            for b in range(2):
                for kind in ("raw", "audit"):
                    call_index += 1
                    attempted = shared.read(path, f"call-{call_index:02d}-attempt.json")
                    completed = shared.read(path, f"call-{call_index:02d}-complete.json")
                    expected = {
                        "group": g,
                        "phase": phase,
                        "branch": b,
                        "kind": kind,
                        "witness": shared.read(path, f"g{g:02d}-{phase}{b}-witness.json"),
                        "completed": False,
                        "retained": False,
                    }
                    if attempted != expected or completed != {
                        **expected,
                        "completed": True,
                        "retained": True,
                    }:
                        raise ValueError("exact no-retry before-seal/target chronology required")
    groups = []
    episodes: set[str] = set()
    all_tokens: set[str] = set()
    for g, study in enumerate(grid(definition, adoption)):
        mapping = shared.read(path, f"g{g:02d}-identity.json")
        shared.check_identity(mapping)
        if mapping["episode"] in episodes or set(mapping["tokens"]) & all_tokens:
            raise ValueError("fresh group identity required")
        episodes.add(mapping["episode"])
        all_tokens.update(mapping["tokens"])
        ctx = group_context(context, study)

        def retained(
            phase: str,
            b: int,
            projection: Projection | None = None,
            *,
            g: int = g,
            study: shared.Study = study,
            mapping: dict[str, Any] = mapping,
            ctx: dict[str, Any] = ctx,
        ) -> tuple[VisibleRaster, dict[str, Any]]:
            stem = f"g{g:02d}-{phase}{b}"
            w = shared.read(path, stem + "-witness.json")
            if w != shared.witness(study, mapping, ctx, phase, b, projection):
                raise ValueError("exact group/branch/endpoint witness differs")
            raw, audit = (
                shared.read(path, stem + "-raw.json"),
                shared.read(path, stem + "-audit.json"),
            )
            return shared.visible(
                shared.qualified(raw, audit),
                b if phase == "prefix" else 2,
                tuple(mapping["tokens"]),
            ), w

        frames = cast(
            tuple[VisibleRaster, VisibleRaster], tuple(retained("prefix", b)[0] for b in range(2))
        )
        ps = shared.projections(study, frames, mapping, context["head"])
        branch_reports = []
        known_targets = []
        identical = True
        targets_persistence = True
        for b, projection in enumerate(ps):
            for suffix, data in shared.before_records(projection).items():
                if (path / f"g{g:02d}-b{b}-{suffix}.json").read_bytes() != data:
                    raise ValueError("retained projection/prefix differs")
            saved_methods = {}
            for m in METHODS:
                name = f"g{g:02d}-b{b}-{m}"
                saved = (path / (name + "-candidate.json")).read_bytes()
                details = shared.read(path, name + "-rule.json")
                verify_forecast(projection, m, saved, details)
                saved_methods[m] = saved
            difference = contrast(saved_methods["motion"], saved_methods["reversed"])
            if shared.read(path, f"g{g:02d}-b{b}-forecast-contrast.json") != difference:
                raise ValueError("pre-target correct/reversed differences changed")
            identical &= difference["identical"]
            if not targets:
                continue
            target, observed = retained("target", b, projection)
            expected = shared.witness(study, mapping, ctx, "target", b, projection)
            method_reports = {}
            for m, saved in saved_methods.items():
                value = projection.evaluate(saved, shared.Target(target, observed, expected))
                report = value.report
                if (
                    report["N"] != (len(projection.alignment) + 1) * 1024
                    or report["C"] != report["N"]
                    or report["U"] != 0
                ):
                    raise ValueError("complete neutral denominator required")
                channels = count_channels(projection, saved, value)
                if sum(r["FP"] + r["FN"] for r in channels) != report["E"]:
                    raise ValueError("independent pixel counts disagree with neutral error")
                method_reports[m] = {
                    "N": report["N"],
                    "C": report["C"],
                    "U": report["U"],
                    "E": report["E"],
                    "error": str(Q(report["E"], report["N"])),
                    "observation_exact": report["observation_exact"],
                    "channels": channels,
                    "undefined_iou_count": sum(r["iou"] is None for r in channels),
                    "receipt": json.loads(value.receipt_bytes),
                }
                if m == "persistence":
                    # Compare known masks only; NEW remains scored separately.
                    targets_persistence &= (
                        all(r["FP"] + r["FN"] == 0 for r in channels[:-1])
                        and channels[-1]["FN"] == 0
                    )
                    known_targets.append(
                        [a.tolist() for _, a in value.target.channels.known if a is not None]
                    )
            branch_reports.append(method_reports)
        if targets:
            f = json.loads(ps[0].feature_bytes())
            groups.append(
                {
                    "group": g,
                    "inventory_count": len(ps[0].alignment),
                    "prefix_change": any(n["previous"] != n["current"] for n in f["nodes"]),
                    "targets_persistence": bool(targets_persistence),
                    "target_action_contrast": known_targets[0] != known_targets[1],
                    "correct_reversed_identical": bool(identical),
                    "branches": branch_reports,
                    "losses": {
                        m: str(sum((Q(x[m]["error"]) for x in branch_reports), Q(0)) / 2)
                        for m in METHODS
                    },
                }
            )
    return summarize(groups) if targets else {"seal_valid": True, "forecast_count": 96}


def collect(
    manifest: bytes,
    definition: bytes,
    adoption: bytes,
    context: dict[str, Any],
    output: Path,
    protect: Callable[[Path], None],
    source_check: Callable[[], None],
    adapter: shared.Adapter,
    external: Callable[[], int],
    process_peak: Callable[[], int],
    environment_bytes: int,
    close_external: Callable[[dict[str, Any]], None],
    *,
    allocate: Callable[[], dict[str, Any]] = shared.identity,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """One attempt only. Adapter target operations are reachable only after global seal validation.

    External wrapper must durably retain closure/failure, including failure of final local writes.
    Callbacks are admission evidence, not a claim that this function guarantees host resources.
    """
    started = clock()
    context = json.loads(canonical_json_bytes(context))
    check_context(context, manifest)
    p = json.loads(manifest)
    if manifest != membership(definition, adoption, tuple(p["available_known_exclusions"])):
        raise ValueError("frozen cohort membership differs")
    source_check()
    output.mkdir()
    protect(output)
    sink = Sink(output, clock, external, process_peak, environment_bytes)
    sink.started = started
    calls: list[dict[str, Any]] = []
    failure: dict[str, Any] | None = None
    pending = None
    analysis: dict[str, Any] = {"finite_outcome": "INCONCLUSIVE"}
    target_enabled = False
    studies = grid(definition, adoption)
    mappings: list[dict[str, Any]] = []
    projections = []
    try:
        for name, raw in (
            ("manifest.json", manifest),
            ("definition.md", definition),
            ("adoption.md", adoption),
        ):
            sink.write(name, raw)
        sink.json("context.json", context)

        def frame(
            g: int, phase: str, b: int, projection: Projection | None = None
        ) -> VisibleRaster:
            nonlocal pending
            if phase == "target" and not target_enabled:
                raise PermissionError("target capability unavailable before complete global seal")
            study = studies[g]
            w = shared.witness(
                study, mappings[g], group_context(context, study), phase, b, projection
            )
            stem = f"g{g:02d}-{phase}{b}"
            sink.json(stem + "-witness.json", w)
            returns = []
            for kind, operation in (("raw", adapter.produce), ("audit", adapter.audit)):
                sink.verify()
                source_check()
                if sum(c["kind"] == kind for c in calls) >= 48:
                    raise ValueError("exact operation cap exceeded")
                call = {
                    "group": g,
                    "phase": phase,
                    "branch": b,
                    "kind": kind,
                    "witness": w,
                    "completed": False,
                    "retained": False,
                }
                calls.append(call)
                sink.json(f"call-{len(calls):02d}-attempt.json", call)
                returned = shared.plain(
                    operation(study, json.loads(canonical_json_bytes(w)), sink.check)
                )
                call["completed"] = True
                pending = {"call": call.copy(), "returned": returned}
                sink.json(stem + "-" + kind + ".json", returned)
                call["retained"] = True
                pending = None
                sink.json(f"call-{len(calls):02d}-complete.json", call)
                returns.append(returned)
            return shared.visible(
                shared.qualified(returns[0], returns[1]),
                b if phase == "prefix" else 2,
                tuple(mappings[g]["tokens"]),
            )

        # Prefixes for the entire fixed cohort precede every target.
        for g, study in enumerate(studies):
            mapping = allocate()
            sink.json(f"g{g:02d}-identity.json", mapping)
            shared.check_identity(mapping)
            if any(
                mapping["episode"] == x["episode"] or set(mapping["tokens"]) & set(x["tokens"])
                for x in mappings
            ):
                raise ValueError("fresh distinct group identities required")
            mappings.append(mapping)
            frames = cast(
                tuple[VisibleRaster, VisibleRaster], tuple(frame(g, "prefix", b) for b in range(2))
            )
            ps = shared.projections(study, frames, mapping, context["head"])
            projections.append(ps)
            for b, projection in enumerate(ps):
                for suffix, raw in shared.before_records(projection).items():
                    sink.write(f"g{g:02d}-b{b}-{suffix}.json", raw)
                saved_methods = {}
                for method in METHODS:
                    saved, details = forecast(projection, method)
                    saved_methods[method] = saved
                    sink.write(f"g{g:02d}-b{b}-{method}-candidate.json", saved)
                    sink.json(f"g{g:02d}-b{b}-{method}-rule.json", details)
                sink.json(
                    f"g{g:02d}-b{b}-forecast-contrast.json",
                    contrast(saved_methods["motion"], saved_methods["reversed"]),
                )
        sink.verify()
        sink.json(
            "global-seal.json",
            {
                "version": VERSION,
                "members": dict(sink.hashes),
                "forecasts": [
                    f"g{g:02d}-b{b}-{m}-candidate.json"
                    for g in range(12)
                    for b in range(2)
                    for m in METHODS
                ],
                "manifest_sha256": sha256_bytes(manifest),
            },
        )
        inspect(output, context, source_check)
        target_enabled = True
        for g, ps in enumerate(projections):
            for b, projection in enumerate(ps):
                sink.verify()  # Detect any forecast rewrite before each target access.
                frame(g, "target", b, projection)
        analysis = inspect(output, context, source_check, targets=True)
        source_check()
        sink.json("analysis.json", analysis)
        sink.check()
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        if pending is not None:
            failure["unretained_return"] = pending
    completion = (
        "COMPLETE"
        if failure is None
        and len(calls) == 96
        and all(c["completed"] and c["retained"] for c in calls)
        else "STOPPED_INCOMPLETE"
    )
    if completion != "COMPLETE":
        analysis = {"finite_outcome": "INCONCLUSIVE", "partial_analysis": analysis}
    terminal = {
        "version": VERSION,
        "completion": completion,
        "first_failure": failure,
        "analysis": analysis,
        "calls": calls,
        "files": dict(sink.hashes),
        "elapsed_seconds": clock() - started,
        "process_peak_bytes": sink.peak,
        "environment_bytes": environment_bytes,
        "external_bytes": sink.last_external,
    }
    try:
        sink.json("terminal.json", terminal, True)
        sink.verify()  # Includes time/resources after the final durable local write.
        source_check()
    except Exception as error:
        terminal = {
            **terminal,
            "completion": "STOPPED_INCOMPLETE",
            "first_failure": failure or {"type": type(error).__name__, "message": str(error)},
            "analysis": {"finite_outcome": "INCONCLUSIVE", "partial_analysis": analysis},
            "local_terminal_qualified": False,
        }
    try:
        close_external(terminal)
        sink.check()  # Durable external closure is included in the cooperative envelope.
    except Exception as error:
        # No recursive receipt writes, retries, cleanup or promotion of the stale local file.
        raise RuntimeError("external durable closure/resource evidence unqualified") from error
    return terminal
