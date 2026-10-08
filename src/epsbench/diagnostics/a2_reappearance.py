"""Fixed A2 reappearance diagnostic rules over lawful bounded prefix memory."""

from __future__ import annotations

import json
from fractions import Fraction
from typing import Any

import numpy as np

from epsbench.diagnostics.bounded_prefix_memory import BoundedPrefixMemory, access_check
from epsbench.utils.canonical import canonical_json_bytes, sha256_bytes

VERSION = "a2-reappearance-v1"
METHODS = (
    "motion_overlap",
    "zero",
    "one",
    "half",
    "current_presence",
    "toward_last_view",
    "exact_retrace",
    "motion_only",
)


def _round_away(value: Fraction) -> int:
    magnitude = abs(value)
    rounded = (2 * magnitude.numerator + magnitude.denominator) // (2 * magnitude.denominator)
    return rounded if value >= 0 else -rounded


def _shift(mask: np.ndarray, columns: int) -> np.ndarray:
    result = np.zeros(mask.shape, dtype=np.bool_)
    width = mask.shape[1]
    if columns >= 0:
        if columns < width:
            result[:, columns:] = mask[:, : width - columns]
    elif -columns < width:
        result[:, : width + columns] = mask[:, -columns:]
    return result


def _centroid(mask: np.ndarray) -> Fraction | None:
    columns = np.nonzero(mask)[1]
    return None if len(columns) == 0 else Fraction(int(columns.sum()), len(columns))


def _motion(
    frames: list[dict[str, Any]], index: int, executed: list[Fraction]
) -> tuple[Fraction | None, str | None]:
    sightings = [i for i in range(len(frames)) if frames[i]["observed"][index]]
    if len(sightings) < 2:
        return None, "INSUFFICIENT_HISTORY"
    earlier, later = sightings[-2:]
    denominator = sum(executed[earlier:later], Fraction())
    if denominator == 0:
        return None, "ZERO_COMMAND_DENOMINATOR"
    a = _centroid(np.asarray(frames[earlier]["masks"][index], dtype=np.bool_))
    b = _centroid(np.asarray(frames[later]["masks"][index], dtype=np.bool_))
    if a is None or b is None:
        return None, "EMPTY_SIGHTING"
    return (b - a) / denominator, None


def _distance(a: Fraction, b: Fraction) -> Fraction:
    return abs(a - b)


def predict(memory: BoundedPrefixMemory) -> bytes:
    """Return canonical q rows for every aligned token and fixed method.

    Binding identifiers and feature hashes are emitted as provenance only. They do
    not participate in any predicted value.
    """
    if type(memory) is not BoundedPrefixMemory:
        raise TypeError("BoundedPrefixMemory required")
    access_check(memory.access)
    feature_bytes = memory.feature_bytes()
    binding_bytes = memory.binding_bytes()
    source = memory.revalidate()
    features = json.loads(feature_bytes)
    executed = [command[1] for command in source.executed]
    announced = source.announced[1]
    if any(command[0] != 0 or command[2] != 0 for command in (*source.executed, source.announced)):
        raise ValueError("only pure lateral commands are supported")
    frames: list[dict[str, Any]] = features["frames"]
    if len(frames) != 3 or len(executed) != 2:
        raise ValueError("three frames and two executed commands required")
    positions = [Fraction()]
    for command in executed:
        positions.append(positions[-1] + command)
    current_position = positions[-1]
    future_position = current_position + announced
    last_visible: dict[int, int | None] = {}
    velocities: dict[int, tuple[Fraction | None, str | None]] = {}
    for index in range(len(memory.alignment)):
        seen = [i for i in range(3) if frames[i]["observed"][index]]
        last_visible[index] = seen[-1] if seen else None
        velocities[index] = _motion(frames, index, executed)

    rows: list[dict[str, Any]] = []
    blocker_rows: list[dict[str, Any]] = []
    for index, _token in enumerate(memory.alignment):
        last = last_visible[index]
        present_now = last == 2
        base_mask = (
            np.asarray(frames[last]["masks"][index], dtype=np.bool_)
            if last is not None
            else np.zeros(source.shape, dtype=np.bool_)
        )
        velocity, velocity_reason = velocities[index]
        row_methods: dict[str, dict[str, Any]] = {
            name: {"q": q, "fallback": False, "reason": "FIXED_CONTROL"}
            for name, q in (("zero", "0"), ("one", "1"), ("half", "1/2"))
        }
        row_methods["current_presence"] = {
            "q": "1" if present_now else "0",
            "fallback": False,
            "reason": "CURRENT_MASK",
        }

        last_position = positions[last] if last is not None else current_position
        if last is None:
            toward_q = "1/2"
            toward_reason = "NO_SIGHTING"
        else:
            before_distance = _distance(current_position, last_position)
            after_distance = _distance(future_position, last_position)
            if after_distance < before_distance:
                toward_q, toward_reason = "1", "STRICTLY_CLOSER"
            elif after_distance == before_distance:
                toward_q, toward_reason = "1/2", "EQUAL_DISTANCE"
            else:
                toward_q, toward_reason = "0", "NOT_CLOSER"
        row_methods["toward_last_view"] = {
            "q": toward_q,
            "fallback": toward_q == "1/2",
            "reason": toward_reason,
        }

        retrace_q: str | None = None
        for frame_index, position in enumerate(positions):
            if future_position == position:
                retrace_q = (
                    "1"
                    if frames[frame_index]["observed"][index]
                    and np.any(np.asarray(frames[frame_index]["masks"][index], dtype=np.bool_))
                    else "0"
                )
                break
        row_methods["exact_retrace"] = {
            "q": "1/2" if retrace_q is None else retrace_q,
            "fallback": retrace_q is None,
            "reason": "NO_EXACT_VIEW" if retrace_q is None else "EXACT_PREFIX_VIEW",
        }

        if velocity is None or last is None:
            fallback_reason = velocity_reason or "NO_SIGHTING"
            for name in ("motion_overlap", "motion_only"):
                row_methods[name] = {"q": "1/2", "fallback": True, "reason": fallback_reason}
            blocker_rows.append({"selected": [], "erased_support": []})
        else:
            to_current = _round_away(velocity * (current_position - positions[last]))
            to_future = _round_away(velocity * (future_position - positions[last]))
            current_target = _shift(base_mask, to_current)
            future_target = _shift(base_mask, to_future)
            if not future_target.any():
                only_q = "0"
            else:
                only_q = "1"
            row_methods["motion_only"] = {
                "q": only_q,
                "fallback": False,
                "reason": "TRANSPORTED_SUPPORT" if only_q == "1" else "CROPPED_EMPTY",
            }

            blockers: list[int] = []
            for other in range(len(memory.alignment)):
                if other == index or last_visible[other] != 2:
                    continue
                other_mask = np.asarray(frames[2]["masks"][other], dtype=np.bool_)
                if np.any(current_target & other_mask):
                    blockers.append(other)
            fallback = None
            transported_blockers: list[np.ndarray] = []
            for other in blockers:
                other_velocity, other_reason = velocities[other]
                other_last = last_visible[other]
                if other_velocity is None or other_last is None:
                    fallback = other_reason or "NO_SIGHTING"
                    break
                shift = _round_away(other_velocity * (future_position - positions[other_last]))
                transported_blockers.append(
                    _shift(np.asarray(frames[other_last]["masks"][other], dtype=np.bool_), shift)
                )
            if fallback is not None:
                row_methods["motion_overlap"] = {
                    "q": "1/2",
                    "fallback": True,
                    "reason": f"BLOCKER_{fallback}",
                }
                blocker_rows.append(
                    {
                        "selected": blockers,
                        "erased_support": [
                            *(
                                int(np.count_nonzero(future_target & blocker))
                                for blocker in transported_blockers
                            ),
                            *([None] * (len(blockers) - len(transported_blockers))),
                        ],
                    }
                )
            else:
                residual = future_target.copy()
                erased_support = [
                    int(np.count_nonzero(future_target & blocker))
                    for blocker in transported_blockers
                ]
                for blocker in transported_blockers:
                    residual &= ~blocker
                q = "1" if residual.any() else "0"
                row_methods["motion_overlap"] = {
                    "q": q,
                    "fallback": False,
                    "reason": "RESIDUAL_SUPPORT" if q == "1" else "BLOCKED_SUPPORT",
                }
                blocker_rows.append({"selected": blockers, "erased_support": erased_support})
        rows.append(row_methods)

    payload = {
        "version": VERSION,
        "methods": list(METHODS),
        "binding_sha256": sha256_bytes(binding_bytes),
        "feature_sha256": sha256_bytes(feature_bytes),
        "alignment_count": len(memory.alignment),
        "rows": rows,
        "blockers": blocker_rows,
    }
    return canonical_json_bytes(payload)
