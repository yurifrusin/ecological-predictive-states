"""Bounded development aggregation; no review-state, seed, roster or launch automation."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction as Q
from typing import Any

from epsbench.diagnostics.disclosure_controls import METHODS, THRESHOLD, VERSION, policy
from epsbench.diagnostics.visible_forecast_contract import _json, _keys
from epsbench.utils.canonical import canonical_json_bytes

FAMILIES = ("slab", "doorway")
SCORED_METHODS = (*METHODS, "always_abstain", "constant_0", "constant_1")
CAUSES = ("VISIBLE", "COMPLETE_OCCLUSION", "FOV_EXIT", "UNKNOWN")


def _hash(value: str) -> None:
    if type(value) is not str or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("exact SHA256 binding required")


def _integer(value: int, maximum: int) -> None:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError("bounded strict count required")


def _score(data: bytes) -> dict[str, Any]:
    if type(data) is not bytes or len(data) > 131072:
        raise ValueError("bounded score bytes required before JSON parsing")
    p = _json(data)
    _keys(
        p,
        {
            "version",
            "forecast_sha256",
            "future_sha256",
            "optical_sha256",
            "rows",
            "new_counts",
            "new_pixels",
        },
    )
    if p["version"] != VERSION + ":score" or type(p["rows"]) is not list or len(p["rows"]) > 16:
        raise ValueError("bounded complete score required")
    _hash(p["optical_sha256"])
    for key, maximum in (("new_counts", 16), ("new_pixels", 1024)):
        if type(p[key]) is not list or len(p[key]) != 2:
            raise ValueError("two complete new-support counts required")
        for count in p[key]:
            _integer(count, maximum)
    for key in ("forecast_sha256", "future_sha256"):
        if type(p[key]) is not list or len(p[key]) != 2:
            raise ValueError("two forecast/future bindings required")
        for value in p[key]:
            _hash(value)
    for i, row in enumerate(p["rows"]):
        _keys(row, {"row", "remembered", "support", "success", "retrace", "oracle_cost", "methods"})
        if type(row["row"]) is not int or row["row"] != i or type(row["remembered"]) is not bool:
            raise ValueError("complete aligned strict rows required")
        for key in ("success", "retrace"):
            if (
                type(row[key]) is not list
                or len(row[key]) != 2
                or any(type(v) is not bool for v in row[key])
            ):
                raise ValueError("two strict outcome/view flags required")
        if type(row["support"]) is not list or len(row["support"]) != 2:
            raise ValueError("two support counts required")
        for count in row["support"]:
            _integer(count, 1024)
        if row["success"] != [count >= THRESHOLD for count in row["support"]]:
            raise ValueError("task success differs from support")
        oracle = policy(Q(int(row["success"][0])), Q(int(row["success"][1])))
        oracle_cost = 2 if oracle is None else 1
        if type(row["oracle_cost"]) is not int or row["oracle_cost"] != oracle_cost:
            raise ValueError("incorrect oracle cost")
        _keys(row["methods"], set(SCORED_METHODS))
        for name, method in row["methods"].items():
            _keys(method, {"choice", "cost", "regret"})
            choice = method["choice"]
            if choice is not None and (type(choice) is not int or choice not in (0, 1)):
                raise ValueError("exact action/abstention required")
            if name == "always_abstain" and choice is not None:
                raise ValueError("abstention control altered")
            if name in ("zero", "half") and choice is not None:
                raise ValueError("constant probability abstention policy altered")
            if name == "one" and choice != 0:
                raise ValueError("constant one policy tie priority altered")
            if name == "current_support" and row["remembered"] and choice is not None:
                raise ValueError("absent current-support policy altered")
            if name.startswith("constant_") and choice != int(name[-1]):
                raise ValueError("constant action control altered")
            cost = 2 if choice is None else (1 if row["success"][choice] else 5)
            if (
                type(method["cost"]) is not int
                or type(method["regret"]) is not int
                or method["cost"] != cost
                or method["regret"] != cost - oracle_cost
            ):
                raise ValueError("incorrect task cost/regret")
    if canonical_json_bytes(p) != data:
        raise ValueError("canonical score required")
    for arm in range(2):
        new_count, new_pixels = p["new_counts"][arm], p["new_pixels"][arm]
        if bool(new_count) != bool(new_pixels) or new_pixels < new_count:
            raise ValueError("NEW count/pixel support inconsistent")
        if sum(r["support"][arm] for r in p["rows"]) + new_pixels > 1024:
            raise ValueError("future support exceeds raster cap")
        if len(p["rows"]) + new_count > 16:
            raise ValueError("future inventory exceeds cap")
    return p


@dataclass(frozen=True)
class Group:
    """Private adapter evidence assertions; not authenticated qualification credentials."""

    family: str
    ancestry: str
    qualified: bool
    complete: bool
    parity: bool
    contradiction: bool
    score: bytes | None
    causes: tuple[str, ...] | None

    def __post_init__(self) -> None:
        if type(self.family) is not str or self.family not in FAMILIES:
            raise ValueError("declared construction family required")
        _hash(self.ancestry)
        if any(
            type(v) is not bool
            for v in (self.qualified, self.complete, self.parity, self.contradiction)
        ):
            raise ValueError("strict evidence flags required")
        if self.score is None:
            if self.causes is not None or (self.qualified and self.complete):
                raise ValueError("complete qualified groups need a score, including empty rows")
        else:
            p = _score(self.score)
            if type(self.causes) is not tuple or len(self.causes) != len(p["rows"]):
                raise ValueError("all known row causes required")
            for cause, row in zip(self.causes, p["rows"], strict=True):
                if type(cause) is not str or cause not in CAUSES:
                    raise ValueError("closed evaluator cause vocabulary required")
                if cause in ("COMPLETE_OCCLUSION", "FOV_EXIT") and not row["remembered"]:
                    raise ValueError("absence certificate attached to visible row")
                if cause == "VISIBLE" and row["remembered"]:
                    raise ValueError("visible cause attached to absent row")


def _mean(values: list[Q]) -> Q | None:
    return None if not values else sum(values, Q(0)) / len(values)


def summarize(groups: tuple[Group, ...]) -> bytes:
    if type(groups) is not tuple or len(groups) > 24 or any(type(g) is not Group for g in groups):
        raise ValueError("at most 24 exact retained group records required")
    by_family: dict[str, dict[str, list[Group]]] = {f: defaultdict(list) for f in FAMILIES}
    ancestry_family: dict[str, str] = {}
    for g in groups:
        g.__post_init__()
        if g.ancestry in ancestry_family and ancestry_family[g.ancestry] != g.family:
            raise ValueError("same physical ancestry cannot straddle family declarations")
        ancestry_family[g.ancestry] = g.family
        by_family[g.family][g.ancestry].append(g)
    families: dict[str, Any] = {}
    all_optical: set[str] = set()
    crossed: dict[str, int] = defaultdict(int)
    for family in FAMILIES:
        optical: set[str] = set()
        eligible = occluded = contrast = 0
        null_slots = 0
        oracle_means: list[Q] = []
        method_means: dict[str, dict[str, list[Q]]] = {
            m: {"cost": [], "regret": []} for m in SCORED_METHODS
        }
        control_regrets: dict[str, list[Q]] = {m: [] for m in SCORED_METHODS}
        for members in by_family[family].values():
            primary_means: list[Q] = []
            primary_methods: dict[str, dict[str, list[Q]]] = {
                m: {"cost": [], "regret": []} for m in SCORED_METHODS
            }
            has_primary = has_occluded = has_contrast = False
            for g in members:
                if g.score is None or not (g.qualified and g.complete):
                    continue
                p = _score(g.score)
                optical.add(p["optical_sha256"])
                primary = [r for r in p["rows"] if r["remembered"]]
                null_slots += int(not primary)
                if primary:
                    has_primary = True
                    primary_means.append(
                        sum((Q(r["oracle_cost"]) for r in primary), Q(0)) / len(primary)
                    )
                    for m in SCORED_METHODS:
                        for metric in ("cost", "regret"):
                            primary_methods[m][metric].append(
                                sum((Q(r["methods"][m][metric]) for r in primary), Q(0))
                                / len(primary)
                            )
                assert g.causes is not None
                for row, cause in zip(p["rows"], g.causes, strict=True):
                    for retrace in row["retrace"]:
                        crossed[f"{family}:{cause}:{'retrace' if retrace else 'new_view'}"] += 1
                    new_occluded = (
                        row["remembered"]
                        and cause == "COMPLETE_OCCLUSION"
                        and not any(row["retrace"])
                    )
                    has_occluded |= row["remembered"] and cause == "COMPLETE_OCCLUSION"
                    has_contrast |= new_occluded and row["success"][0] != row["success"][1]
                    if new_occluded:
                        for method in SCORED_METHODS:
                            control_regrets[method].append(Q(row["methods"][method]["regret"]))
            eligible += int(has_primary)
            occluded += int(has_occluded)
            contrast += int(has_contrast)
            mean = _mean(primary_means)
            if mean is not None:
                oracle_means.append(mean)
            for m in SCORED_METHODS:
                for metric in ("cost", "regret"):
                    method_mean = _mean(primary_methods[m][metric])
                    if method_mean is not None:
                        method_means[m][metric].append(method_mean)
        cost = _mean(oracle_means)
        families[family] = {
            "slots": sum(len(m) for m in by_family[family].values()),
            "ancestries": len(by_family[family]),
            "eligible": eligible,
            "null_slots": null_slots,
            "complete_occlusion": occluded,
            "new_occlusion_contrast": contrast,
            "distinct_optical": len(optical),
            "oracle_cost": None if cost is None else str(cost),
            "perfect_controls": [
                m for m, rs in control_regrets.items() if rs and all(r == 0 for r in rs)
            ],
            "methods": {
                m: {
                    metric: None if (v := _mean(values)) is None else str(v)
                    for metric, values in metrics.items()
                }
                for m, metrics in method_means.items()
            },
        }
        all_optical |= optical
    total_eligible = sum(f["eligible"] for f in families.values())
    adequate = (
        len(all_optical) >= 20
        and total_eligible >= 16
        and all(
            f["distinct_optical"] >= 10
            and f["eligible"] >= 8
            and f["complete_occlusion"] >= 4
            and f["new_occlusion_contrast"] >= 4
            and f["oracle_cost"] is not None
            and Q(f["oracle_cost"]) <= Q(7, 4)
            for f in families.values()
        )
    )
    if any(not g.parity for g in groups):
        status, reason = "STOP", "INFORMATION_PARITY"
    elif any(g.contradiction for g in groups):
        status, reason = "FAIL", "APPARATUS_CONTRADICTION_PREDICTIVE_INCONCLUSIVE"
    elif len(groups) != 24 or any(
        not (g.complete and g.qualified) or g.score is None for g in groups
    ):
        status, reason = "INCONCLUSIVE", "MISSING_OR_UNQUALIFIED_EVIDENCE"
    elif not adequate:
        status, reason = "FAIL", "TASK_ADEQUACY"
    elif any(f["perfect_controls"] for f in families.values()):
        status, reason = "STOP", "SOLVED_NEW_VIEW_OCCLUSION_STRATUM"
    else:
        status, reason = "PASS", "DEVELOPMENT_TASK_ADEQUACY_ONLY"
    costs = [Q(f["oracle_cost"]) for f in families.values() if f["oracle_cost"] is not None]
    overall = _mean(costs) if len(costs) == 2 else None
    return canonical_json_bytes(
        {
            "version": "oracle-disclosure-report-v1",
            "status": status,
            "reason": reason,
            "families": families,
            "slots": len(groups),
            "distinct_ancestries": len(ancestry_family),
            "distinct_optical": len(all_optical),
            "overall_oracle_cost": None if overall is None else str(overall),
            "crossed_views": dict(crossed),
            "phase_gate_effect": "NONE",
        }
    )
