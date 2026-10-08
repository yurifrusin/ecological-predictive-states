"""Untrained support controls and causal scoring; no producer, model or study roster."""

from __future__ import annotations

import json
from fractions import Fraction as Q
from typing import Any

import numpy as np

from epsbench.diagnostics.a2_reappearance import _motion, _round_away, _shift
from epsbench.diagnostics.bounded_prefix_memory import BoundedPrefixMemory, access_check
from epsbench.diagnostics.restricted_mask_projection import Trust
from epsbench.diagnostics.visible_forecast_contract import CausalInput, TokenFrame
from epsbench.utils.canonical import canonical_json_bytes, sha256_bytes

VERSION = "oracle-disclosure-controls-v1"
THRESHOLD = 16
METHODS = (
    "zero",
    "one",
    "half",
    "current_support",
    "exact_retrace",
    "motion_only",
    "motion_overlap",
)


def checked(memory: BoundedPrefixMemory) -> tuple[CausalInput, dict[str, Any]]:
    if type(memory) is not BoundedPrefixMemory:
        raise TypeError("exact lawful memory required")
    access_check(memory.access)
    source = memory.revalidate()
    if any(c[0] or c[2] or not c[1] for c in (*source.executed, source.announced)):
        raise ValueError("nonzero pure lateral commands required")
    if any(
        max(abs(q.numerator).bit_length(), q.denominator.bit_length()) > 64
        for c in (*source.executed, source.announced)
        for q in c
    ):
        raise ValueError("bounded exact commands required")
    feature_bytes, binding_bytes = memory.feature_bytes(), memory.binding_bytes()
    features, binding = json.loads(feature_bytes), json.loads(binding_bytes)
    positions = _positions(source)
    for i in range(3):
        for j in range(i):
            if positions[i] == positions[j] and (
                features["frames"][i]["masks"] != features["frames"][j]["masks"]
            ):
                raise ValueError("static repeated prefix locations must have identical masks")
    availability = tuple(f["contacts"]["available"] for f in features["frames"])
    rebuilt = BoundedPrefixMemory(
        source,
        memory.access,
        Trust(True, True, True, True),
        availability,
        binding["episode"],
        binding["source_head"],
    )
    if (
        memory.alignment != source.inventory
        or rebuilt.feature_bytes() != feature_bytes
        or rebuilt.binding_bytes() != binding_bytes
    ):
        raise ValueError("feature/alignment/binding does not match owned causal source")
    return source, features


def _positions(source: CausalInput) -> tuple[Q, Q, Q]:
    return Q(0), source.executed[0][1], source.executed[0][1] + source.executed[1][1]


def _value(count: int | None, reason: str, fallback: bool = False) -> dict[str, Any]:
    return {
        "p": "1/2" if fallback else str(int(count is not None and count >= THRESHOLD)),
        "support": count,
        "fallback": fallback,
        "reason": reason,
    }


def predict(memory: BoundedPrefixMemory) -> bytes:
    source, features = checked(memory)
    frames = features["frames"]
    positions = _positions(source)
    future = positions[2] + source.announced[1]
    velocities = [
        _motion(frames, i, [c[1] for c in source.executed]) for i in range(len(memory.alignment))
    ]
    last = [
        max((t for t in range(3) if frames[t]["observed"][i]), default=-1)
        for i in range(len(memory.alignment))
    ]
    rows = []
    for i, seen in enumerate(last):
        row: dict[str, Any] = {
            name: {"p": p, "support": None, "fallback": False, "reason": "CONSTANT"}
            for name, p in (("zero", "0"), ("one", "1"), ("half", "1/2"))
        }
        current = np.asarray(frames[2]["masks"][i], dtype=np.bool_)
        row["current_support"] = _value(int(np.count_nonzero(current)), "CURRENT_SUPPORT")
        retrace = next((t for t, position in enumerate(positions) if position == future), None)
        row["exact_retrace"] = (
            _value(None, "NO_EXACT_VIEW", True)
            if retrace is None
            else _value(int(np.count_nonzero(frames[retrace]["masks"][i])), "EXACT_PREFIX_VIEW")
        )
        velocity, reason = velocities[i]
        selected: list[int] = []
        if velocity is None or seen < 0:
            for method in ("motion_only", "motion_overlap"):
                row[method] = _value(None, reason or "NO_SIGHTING", True)
        else:
            original = np.asarray(frames[seen]["masks"][i], dtype=np.bool_)
            now = _shift(original, _round_away(velocity * (positions[2] - positions[seen])))
            transported = _shift(original, _round_away(velocity * (future - positions[seen])))
            row["motion_only"] = _value(int(np.count_nonzero(transported)), "TRANSPORTED_SUPPORT")
            selected = [
                j
                for j in range(len(last))
                if j != i
                and last[j] == 2
                and np.any(now & np.asarray(frames[2]["masks"][j], dtype=np.bool_))
            ]
            residual = transported.copy()
            blocker_reason = None
            for j in selected:
                blocker_velocity, why = velocities[j]
                if blocker_velocity is None:
                    blocker_reason = "BLOCKER_" + (why or "NO_SIGHTING")
                    break
                blocker = _shift(
                    np.asarray(frames[last[j]]["masks"][j], dtype=np.bool_),
                    _round_away(blocker_velocity * (future - positions[last[j]])),
                )
                residual &= ~blocker
            row["motion_overlap"] = (
                _value(None, blocker_reason, True)
                if blocker_reason is not None
                else _value(int(np.count_nonzero(residual)), "RESIDUAL_SUPPORT")
            )
        rows.append({"methods": row, "selected": selected})
    return canonical_json_bytes(
        {
            "version": VERSION,
            "threshold": THRESHOLD,
            "methods": METHODS,
            "binding_sha256": sha256_bytes(memory.binding_bytes()),
            "feature_sha256": sha256_bytes(memory.feature_bytes()),
            "rows": rows,
        }
    )


def validate(memory: BoundedPrefixMemory, saved: bytes) -> dict[str, Any]:
    checked(memory)  # Denial/causal validation precedes inspection of saved forecasts.
    if type(saved) is not bytes or saved != predict(memory):
        raise ValueError("complete canonical fixed forecast and exact action binding required")
    result: dict[str, Any] = json.loads(saved)
    return result


def matched(negative: BoundedPrefixMemory, positive: BoundedPrefixMemory) -> CausalInput:
    a, af = checked(negative)
    b, bf = checked(positive)
    if a.announced[1] >= 0 or b.announced[1] <= 0 or a.announced[1] != -b.announced[1]:
        raise ValueError("equal magnitude negative/positive query pair required")
    af = {k: v for k, v in af.items() if k != "announced"}
    bf = {k: v for k, v in bf.items() if k != "announced"}
    ab, bb = json.loads(negative.binding_bytes()), json.loads(positive.binding_bytes())
    if af != bf or any(ab[k] != bb[k] for k in ("episode", "source_head", "alignment")):
        raise ValueError("same lawful history, contacts, association and source identity required")
    return a


def policy(negative: Q, positive: Q) -> int | None:
    for p in (negative, positive):
        if type(p) is not Q or not 0 <= p <= 1:
            raise ValueError("exact probability required")
    costs = (Q(2), 5 - 4 * negative, 5 - 4 * positive)
    winner = min(range(3), key=lambda i: costs[i])
    return None if winner == 0 else winner - 1


def optical_digest(memory: BoundedPrefixMemory) -> str:
    _, features = checked(memory)
    frames = features["frames"]
    order = sorted(
        range(len(memory.alignment)),
        key=lambda i: canonical_json_bytes([f["masks"][i] for f in frames]),
    )
    remap = {old: new for new, old in enumerate(order)}
    normalized = []
    for f in frames:
        contacts = f["contacts"]
        pairs = (
            None
            if not contacts["available"]
            else sorted(sorted((remap[a], remap[b])) for a, b in contacts["pairs"])
        )
        normalized.append(
            {
                "masks": [f["masks"][i] for i in order],
                "observed": [f["observed"][i] for i in order],
                "contacts": {"available": contacts["available"], "pairs": pairs},
            }
        )
    return sha256_bytes(canonical_json_bytes({"shape": features["shape"], "frames": normalized}))


def score(
    negative: BoundedPrefixMemory,
    positive: BoundedPrefixMemory,
    saved: tuple[bytes, bytes],
    futures: tuple[TokenFrame, TokenFrame],
    trust: Trust,
) -> bytes:
    """Validate saved causal forecasts before reading trusted future rasters.

    Software binding is not independent evidence of historical wall-clock sealing.
    All association/image trust assertions remain the adapter's qualification duty.
    """
    source = matched(negative, positive)
    if type(saved) is not tuple or len(saved) != 2:
        raise ValueError("two sealed forecasts required")
    forecasts = (validate(negative, saved[0]), validate(positive, saved[1]))
    if type(trust) is not Trust:
        raise ValueError("exact future trust required")
    trust.check()
    if type(futures) is not tuple or len(futures) != 2:
        raise ValueError("two complete future frames required")
    if any(type(f) is not TokenFrame or f.index != 3 or f.shape != source.shape for f in futures):
        raise ValueError("exact index-3 same-shape futures required")
    if len({*source.inventory, *(t for f in futures for t, _ in f.masks)}) > 16:
        raise ValueError("future inventory cap exceeded")
    counts = [dict((t, int(np.count_nonzero(mask))) for t, mask in f.masks) for f in futures]
    positions = _positions(source)
    retrace = [
        positions[2] + m.revalidate().announced[1] in positions for m in (negative, positive)
    ]
    current = dict(source.frames[2].masks)
    rows = []
    for i, name in enumerate(source.inventory):
        supports = [count.get(name, 0) for count in counts]
        success = [support >= THRESHOLD for support in supports]
        oracle = policy(Q(int(success[0])), Q(int(success[1])))
        oracle_cost = 2 if oracle is None else (1 if success[oracle] else 5)
        methods = {}
        for method in METHODS:
            ps = [Q(f["rows"][i]["methods"][method]["p"]) for f in forecasts]
            choice = policy(ps[0], ps[1])
            cost = 2 if choice is None else (1 if success[choice] else 5)
            methods[method] = {"choice": choice, "cost": cost, "regret": cost - oracle_cost}
        methods["always_abstain"] = {"choice": None, "cost": 2, "regret": 2 - oracle_cost}
        for arm in range(2):
            cost = 1 if success[arm] else 5
            methods[f"constant_{arm}"] = {"choice": arm, "cost": cost, "regret": cost - oracle_cost}
        rows.append(
            {
                "row": i,
                "remembered": name not in current,
                "support": supports,
                "success": success,
                "retrace": retrace,
                "oracle_cost": oracle_cost,
                "methods": methods,
            }
        )
    known = set(source.inventory)
    return canonical_json_bytes(
        {
            "version": VERSION + ":score",
            "forecast_sha256": [sha256_bytes(s) for s in saved],
            "future_sha256": [sha256_bytes(canonical_json_bytes(f.payload())) for f in futures],
            "optical_sha256": optical_digest(negative),
            "rows": rows,
            "new_counts": [sum(t not in known for t in count) for count in counts],
            "new_pixels": [sum(v for t, v in count.items() if t not in known) for count in counts],
        }
    )
