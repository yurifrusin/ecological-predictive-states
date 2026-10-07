"""Prospective finite restricted-mask membership and local receipt checks.

This module does not render or audit geometry.  It only defines a deterministic
membership stream and validates bytes/receipts supplied by a separately bound
caller.  Calling ``build_membership`` is reserved for a later frozen operation.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, cast

from epsbench.diagnostics.causal_region_lifecycle import action_bytes
from epsbench.diagnostics.restricted_mask_projection import Projection, numeric
from epsbench.schema import Action
from epsbench.utils.canonical import canonical_json_bytes, sha256_bytes

VERSION = "restricted-mask-sampler-v2"
UNIT_VERSION = "restricted-mask-unit-v2"
SCALE = 1 << 32
SPLITS = (("train", 64), ("development", 16), ("final", 32))
_DOMAIN = b"epsbench/restricted-mask-sampler-v2\0"
_BOX_KEYS = ("a", "b")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class Box:
    lower: tuple[Fraction, Fraction, Fraction]
    upper: tuple[Fraction, Fraction, Fraction]


@dataclass(frozen=True)
class Unit:
    split: str
    ordinal: int
    box_a: Box
    box_b: Box
    prefix0_lateral: Fraction
    executed_lateral: Fraction
    prefix1_lateral: Fraction
    target_laterals: tuple[Fraction, Fraction]
    fingerprint: str
    payload: bytes


@dataclass(frozen=True)
class QualifiedObservation:
    """A local raw/audit concordance, not proof of producer execution."""

    unit_fingerprint: str
    unit_payload_sha256: str
    episode: str
    phase: str
    branch: int
    source_head: str
    source_tree: str
    context_sha256: str
    frame_index: int
    lateral: Fraction
    labels: tuple[tuple[int, ...], ...]
    raw_sha256: str
    audit_sha256: str


def _fraction(value: Any) -> Fraction:
    if type(value) is not str or len(value) > 64:
        raise ValueError("bounded rational string required")
    result = Fraction(value)
    if str(result) != value:
        raise ValueError("reduced rational spelling required")
    return result


def _exact_keys(value: Any, expected: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected or any(type(k) is not str for k in value):
        raise ValueError("exact object fields required")
    return value


def _grid_bounds(lower: Fraction, upper: Fraction) -> tuple[int, int]:
    lo = (lower.numerator * SCALE + lower.denominator - 1) // lower.denominator
    hi = (upper.numerator * SCALE) // upper.denominator
    if lo > hi:
        raise ValueError("interval has no 2^-32 lattice points")
    return lo, hi


def _on_lattice(value: Fraction) -> bool:
    return (value * SCALE).denominator == 1


def _randbelow(key: bytes, split: str, ordinal: int, component: str, bound: int) -> int:
    if bound <= 0:
        raise ValueError("positive draw bound required")
    limit = ((1 << 256) // bound) * bound
    counter = 0
    while True:
        digest = hashlib.sha256(
            _DOMAIN
            + len(split).to_bytes(2, "big")
            + split.encode("ascii")
            + key
            + ordinal.to_bytes(4, "big")
            + len(component).to_bytes(2, "big")
            + component.encode("ascii")
            + counter.to_bytes(8, "big")
        ).digest()
        value = int.from_bytes(digest, "big")
        if value < limit:
            return value % bound
        counter += 1
        if counter >= 1 << 32:
            raise RuntimeError("counter stream exhausted")


def _draw(
    key: bytes, split: str, ordinal: int, name: str, low: Fraction, high: Fraction
) -> Fraction:
    lo, hi = _grid_bounds(low, high)
    return Fraction(lo + _randbelow(key, split, ordinal, name, hi - lo + 1), SCALE)


def _box_payload(box: Box) -> dict[str, list[str]]:
    return {"lower": [str(v) for v in box.lower], "upper": [str(v) for v in box.upper]}


def _box_from(value: Any) -> Box:
    record = _exact_keys(value, {"lower", "upper"})
    vectors = []
    for name in ("lower", "upper"):
        raw = record[name]
        if type(raw) is not list or len(raw) != 3:
            raise ValueError("three exact coordinates required")
        vectors.append(tuple(_fraction(v) for v in raw))
    lower = cast(tuple[Fraction, Fraction, Fraction], vectors[0])
    upper = cast(tuple[Fraction, Fraction, Fraction], vectors[1])
    if any(a >= b for a, b in zip(lower, upper, strict=True)):
        raise ValueError("positive box extents required")
    extents = tuple(b - a for a, b in zip(lower, upper, strict=True))
    if any(not _on_lattice(value) for value in (*lower, *extents)):
        raise ValueError("box coordinates and extents must lie on the fixed lattice")
    return Box(lower, upper)


def _fingerprint(box_a: Box, box_b: Box) -> str:
    """Hash the physical geometry/calibration only, independent of names or split."""
    geometry = canonical_json_bytes(
        {
            "calibration": {"size": 32, "forward": "-3", "elevation": "1", "fovy": "90"},
            "support_half_extents": ["3", "3"],
            "a": _box_payload(box_a),
            "b": _box_payload(box_b),
        }
    )
    return sha256_bytes(b"epsbench/restricted-mask-geometry-v2\0" + geometry)


def _unit_payload(
    split: str,
    ordinal: int,
    box_a: Box,
    box_b: Box,
    prefix0: Fraction,
    executed: Fraction,
    prefix1: Fraction,
    targets: tuple[Fraction, Fraction],
) -> bytes:
    return canonical_json_bytes(
        {
            "version": UNIT_VERSION,
            "split": split,
            "ordinal": ordinal,
            "support_half_extents": ["3", "3"],
            "boxes": {"a": _box_payload(box_a), "b": _box_payload(box_b)},
            "prefix_laterals": [str(prefix0), str(prefix1)],
            "executed_lateral": str(executed),
            "executed_action": {
                "name": "lateral_right" if executed > 0 else "lateral_left",
                "delta_forward": 0.0,
                "delta_lateral": float(executed),
                "delta_yaw": 0.0,
            },
            "target_laterals": [str(v) for v in targets],
            "target_actions": [
                {
                    "name": "lateral_right" if v > prefix1 else "lateral_left",
                    "delta_forward": 0.0,
                    "delta_lateral": float(v - prefix1),
                    "delta_yaw": 0.0,
                }
                for v in targets
            ],
            "fingerprint": _fingerprint(box_a, box_b),
        }
    )


def validate_unit(payload: bytes) -> Unit:
    """Parse one canonical unit and enforce the adopted rational domain."""
    if type(payload) is not bytes or not 1 <= len(payload) <= 16384:
        raise ValueError("bounded unit bytes required")
    try:
        record = _exact_keys(
            json.loads(payload),
            {
                "version",
                "split",
                "ordinal",
                "support_half_extents",
                "boxes",
                "prefix_laterals",
                "executed_lateral",
                "executed_action",
                "target_laterals",
                "target_actions",
                "fingerprint",
            },
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("valid JSON unit required") from exc
    if (
        record["version"] != UNIT_VERSION
        or type(record["split"]) is not str
        or record["split"] not in {n for n, _ in SPLITS}
    ):
        raise ValueError("fixed sampler version and split required")
    split_count = dict(SPLITS)[record["split"]]
    if type(record["ordinal"]) is not int or not 0 <= record["ordinal"] < split_count:
        raise ValueError("nonnegative exact ordinal required")
    if record["support_half_extents"] != ["3", "3"]:
        raise ValueError("fixed support extent required")
    boxes = _exact_keys(record["boxes"], set(_BOX_KEYS))
    a, b = _box_from(boxes["a"]), _box_from(boxes["b"])
    # Exact proposal intervals, including the strict separated-y and above-support domain.
    intervals = (
        (a.lower[0], Fraction(-3), Fraction(3)),
        (a.lower[1], Fraction(0), Fraction(1)),
        (a.lower[2], Fraction(1, 8), Fraction(1, 2)),
        (a.upper[0] - a.lower[0], Fraction(1, 2), Fraction(3, 2)),
        (a.upper[1] - a.lower[1], Fraction(1, 4), Fraction(3, 4)),
        (a.upper[2] - a.lower[2], Fraction(1, 2), Fraction(3, 2)),
        (b.lower[0], Fraction(-4), Fraction(4)),
        (b.lower[1], Fraction(2), Fraction(3)),
        (b.lower[2], Fraction(1, 8), Fraction(1, 2)),
        (b.upper[0] - b.lower[0], Fraction(1, 2), Fraction(3, 2)),
        (b.upper[1] - b.lower[1], Fraction(1, 4), Fraction(3, 4)),
        (b.upper[2] - b.lower[2], Fraction(1, 2), Fraction(3, 2)),
    )
    if any(not lo <= value <= hi for value, lo, hi in intervals):
        raise ValueError("box outside fixed exact domain")
    if not a.upper[1] < b.lower[1] or a.lower[2] <= 0 or b.lower[2] <= 0:
        raise ValueError("separated boxes above support required")
    if type(record["prefix_laterals"]) is not list or len(record["prefix_laterals"]) != 2:
        raise ValueError("two prefix positions required")
    prefix0, prefix1 = (_fraction(v) for v in record["prefix_laterals"])
    executed = _fraction(record["executed_lateral"])
    if not Fraction(-2) <= prefix0 <= Fraction(2) or executed not in {
        Fraction(-3, 4),
        Fraction(3, 4),
    }:
        raise ValueError("fixed prefix/executed lateral domain required")
    if not _on_lattice(prefix0):
        raise ValueError("prefix position must lie on the fixed lattice")
    if prefix1 != prefix0 + executed:
        raise ValueError("executed displacement must bind both prefixes")
    executed_action = Action.model_validate(record["executed_action"])
    if numeric(executed_action) != (Fraction(0), executed, Fraction(0)):
        raise ValueError("genuine executed Action and rational displacement differ")
    if type(record["target_laterals"]) is not list or len(record["target_laterals"]) != 2:
        raise ValueError("two target positions required")
    targets = tuple(_fraction(v) for v in record["target_laterals"])
    if targets != (prefix1 - Fraction(1, 2), prefix1 + Fraction(1, 2)):
        raise ValueError("fixed opposed non-return targets required")
    for lateral in (prefix0, prefix1, *targets):
        for box in (a, b):
            for x in (box.lower[0], box.upper[0]):
                for y in (box.lower[1], box.upper[1]):
                    for z in (box.lower[2], box.upper[2]):
                        if max(abs((x - lateral) / (y + 3)), abs((z - 1) / (y + 3))) > 8:
                            raise ValueError("exact footprint exceeds the adopted reference bound")
    actions = record["target_actions"]
    if type(actions) is not list or len(actions) != 2:
        raise ValueError("two genuine target Action records required")
    for action, target in zip(actions, targets, strict=True):
        checked = Action.model_validate(action)
        if numeric(checked) != (Fraction(0), target - prefix1, Fraction(0)) or checked.name != (
            "lateral_right" if target > prefix1 else "lateral_left"
        ):
            raise ValueError("target Action and rational endpoint differ")
    fp = _fingerprint(a, b)
    if type(record["fingerprint"]) is not str or record["fingerprint"] != fp:
        raise ValueError("canonical geometry fingerprint required")
    canonical = _unit_payload(
        record["split"], record["ordinal"], a, b, prefix0, executed, prefix1, targets
    )
    if canonical != payload:
        raise ValueError("canonical unit byte serialization required")
    return Unit(
        record["split"], record["ordinal"], a, b, prefix0, executed, prefix1, targets, fp, payload
    )


def build_membership(
    train_key: bytes,
    development_key: bytes,
    final_key: bytes,
    *,
    closed_fingerprints: frozenset[str],
) -> bytes:
    """Build the fixed cohort; collisions stop the cohort with no replacement.

    The caller must supply the reviewed closed-member geometry fingerprints. This
    module cannot establish completeness of that denylist from fingerprints alone.
    """
    keys = (train_key, development_key, final_key)
    if any(type(key) is not bytes or len(key) != 32 for key in keys) or len(set(keys)) != 3:
        raise ValueError("three distinct explicit 32-byte split keys required")
    if (
        type(closed_fingerprints) is not frozenset
        or not closed_fingerprints
        or any(type(fp) is not str or _HEX64.fullmatch(fp) is None for fp in closed_fingerprints)
    ):
        raise ValueError("explicit closed-member fingerprint set required")
    all_seen: set[str] = set()
    split_records = []
    for (split, count), key in zip(SPLITS, keys, strict=True):
        units = []
        for ordinal in range(count):

            def draw(
                name: str,
                lo: Fraction,
                hi: Fraction,
                key: bytes = key,
                split: str = split,
                ordinal: int = ordinal,
            ) -> Fraction:
                return _draw(key, split, ordinal, name, lo, hi)

            a_lower = (
                draw("a.lower.x", Fraction(-3), Fraction(3)),
                draw("a.lower.y", Fraction(0), Fraction(1)),
                draw("a.lower.z", Fraction(1, 8), Fraction(1, 2)),
            )
            a_size = (
                draw("a.size.x", Fraction(1, 2), Fraction(3, 2)),
                draw("a.size.y", Fraction(1, 4), Fraction(3, 4)),
                draw("a.size.z", Fraction(1, 2), Fraction(3, 2)),
            )
            b_lower = (
                draw("b.lower.x", Fraction(-4), Fraction(4)),
                draw("b.lower.y", Fraction(2), Fraction(3)),
                draw("b.lower.z", Fraction(1, 8), Fraction(1, 2)),
            )
            b_size = (
                draw("b.size.x", Fraction(1, 2), Fraction(3, 2)),
                draw("b.size.y", Fraction(1, 4), Fraction(3, 4)),
                draw("b.size.z", Fraction(1, 2), Fraction(3, 2)),
            )
            upper_a = cast(
                tuple[Fraction, Fraction, Fraction],
                tuple(x + d for x, d in zip(a_lower, a_size, strict=True)),
            )
            upper_b = cast(
                tuple[Fraction, Fraction, Fraction],
                tuple(x + d for x, d in zip(b_lower, b_size, strict=True)),
            )
            box_a = Box(a_lower, upper_a)
            box_b = Box(b_lower, upper_b)
            prefix0 = draw("prefix0", Fraction(-2), Fraction(2))
            sign = -1 if _randbelow(key, split, ordinal, "executed.sign", 2) == 0 else 1
            executed = Fraction(3 * sign, 4)
            prefix1 = prefix0 + executed
            targets = (prefix1 - Fraction(1, 2), prefix1 + Fraction(1, 2))
            fp = _fingerprint(box_a, box_b)
            if fp in all_seen or fp in closed_fingerprints:
                raise ValueError("duplicate or closed geometry fingerprint; fixed cohort stopped")
            all_seen.add(fp)
            units.append(
                _unit_payload(split, ordinal, box_a, box_b, prefix0, executed, prefix1, targets)
            )
        split_records.append(
            {
                "name": split,
                "key_sha256": sha256_bytes(key),
                "units": [json.loads(u) for u in units],
            }
        )
    return canonical_json_bytes({"version": VERSION, "splits": split_records})


def _rows(value: Any) -> tuple[tuple[int, ...], ...]:
    if (
        type(value) is not list
        or len(value) != 32
        or any(type(row) is not list or len(row) != 32 for row in value)
    ):
        raise ValueError("complete 32x32 role raster required")
    if any(type(label) is not int or not 0 <= label <= 3 for row in value for label in row):
        raise ValueError("controlled integer role labels required")
    return tuple(tuple(row) for row in value)


def _qualified_labels(raw: Any, audit: Any) -> tuple[tuple[int, ...], ...]:
    raw = _exact_keys(raw, {"labels", "ties"})
    audit = _exact_keys(audit, {"labels", "ties", "boundaries", "supports", "footprints"})
    labels = _rows(raw["labels"])
    if (
        labels != _rows(audit["labels"])
        or raw["ties"] != []
        or audit["ties"] != []
        or audit["boundaries"] != []
    ):
        raise ValueError("raw/audit disagreement, tie or boundary")
    supports, footprints = audit["supports"], audit["footprints"]
    if (
        type(supports) is not list
        or len(supports) != 2
        or type(footprints) is not list
        or len(footprints) != 2
    ):
        raise ValueError("two complete face-domain records required")
    for support, footprint in zip(supports, footprints, strict=True):
        if (
            type(support) is not list
            or len(support) != 3
            or support[2] is not False
            or any(type(v) is not int or v < 0 for v in support[:2])
            or support[1] > support[0]
        ):
            raise ValueError("support boundary or audit contradiction")
        footprint = _exact_keys(footprint, {"polygon", "depths", "domain"})
        polygon, depths = footprint["polygon"], footprint["depths"]
        if (
            footprint["domain"] is not True
            or type(polygon) is not list
            or not 3 <= len(polygon) <= 8
        ):
            raise ValueError("unsupported audited footprint")
        if any(
            type(p) is not list or len(p) != 2 or any(abs(_fraction(q)) > 8 for q in p)
            for p in polygon
        ):
            raise ValueError("footprint coordinate outside certified work bound")
        if (
            type(depths) is not list
            or len(depths) != 2
            or not 0 < _fraction(depths[0]) <= _fraction(depths[1])
        ):
            raise ValueError("positive ordered face depths required")
    return labels


def verify_qualification(
    declaration: dict[str, Any],
    handedraw: dict[str, Any],
    audit: dict[str, Any],
    observed_projection_receipt: Projection | None,
) -> QualifiedObservation:
    """Check supplied receipt concordance and caller claims, never producer provenance.

    A successful return means only that the handed raw and audit receipts agree and
    their caller-supplied context matches the declared unit/action. It does not prove
    that this source executed, that a camera emitted the raster, or that external
    Trust assertions are true.
    """
    declaration = _exact_keys(
        declaration,
        {
            "unit",
            "unit_fingerprint",
            "phase",
            "branch",
            "episode",
            "source_head",
            "source_tree",
            "context_sha256",
            "frame_index",
            "lateral",
            "raw_lateral",
            "audit_lateral",
            "action",
        },
    )
    unit_payload = declaration["unit"]
    if type(unit_payload) is not bytes:
        raise ValueError("caller-bound canonical unit bytes required")
    unit = validate_unit(unit_payload)
    if declaration["unit_fingerprint"] != unit.fingerprint:
        raise ValueError("caller-bound unit fingerprint differs")
    phase, branch = declaration["phase"], declaration["branch"]
    if (
        type(phase) is not str
        or phase not in {"prefix", "target"}
        or type(branch) is not int
        or branch not in (0, 1)
    ):
        raise ValueError("exact phase and branch required")
    episode = declaration["episode"]
    head, context = declaration["source_head"], declaration["context_sha256"]
    tree = declaration["source_tree"]
    if type(episode) is not str or re.fullmatch(r"[0-9a-f]{32}", episode) is None:
        raise ValueError("opaque episode binding required")
    if (
        type(head) is not str
        or re.fullmatch(r"[0-9a-f]{40}", head) is None
        or type(tree) is not str
        or re.fullmatch(r"[0-9a-f]{40}", tree) is None
        or type(context) is not str
        or _HEX64.fullmatch(context) is None
    ):
        raise ValueError("caller-bound source/context digest required")
    frame_index = declaration["frame_index"]
    if type(frame_index) is not int or frame_index != (branch if phase == "prefix" else 2):
        raise ValueError("frame chronology differs")
    lateral, raw_lateral, audit_lateral = (
        _fraction(declaration[k]) for k in ("lateral", "raw_lateral", "audit_lateral")
    )
    if raw_lateral != lateral or audit_lateral != lateral:
        raise ValueError("raw/audit rational lateral claims differ")
    action_value = declaration["action"]
    frozen_unit = json.loads(unit.payload)
    if phase == "prefix":
        expected = (unit.prefix0_lateral, unit.prefix1_lateral)[branch]
        action_ok = (
            action_value is None
            if branch == 0
            else (
                action_bytes(Action.model_validate(action_value))
                == action_bytes(Action.model_validate(frozen_unit["executed_action"]))
            )
        )
        if observed_projection_receipt is not None or lateral != expected or not action_ok:
            raise ValueError("prefix receipt must bind its frame and preceding executed Action")
    else:
        if type(observed_projection_receipt) is not Projection:
            raise ValueError("typed before-only projection required for target receipt")
        projection = observed_projection_receipt
        source = projection.revalidate()
        expected = unit.target_laterals[branch]
        action = Action.model_validate(action_value)
        if (
            projection.cutoff != 1
            or projection.episode != episode
            or projection.source_head != head
            or projection.announced != action_bytes(action)
            or context != sha256_bytes(projection.binding_bytes())
            or action_bytes(action)
            != action_bytes(Action.model_validate(frozen_unit["target_actions"][branch]))
            or lateral != expected
            or len(source.frames) != 2
            or source.shape != (32, 32)
            or source.executed[-1] != (Fraction(0), unit.executed_lateral, Fraction(0))
            or source.announced != numeric(action)
            or tuple(frame.index for frame in source.frames) != (0, 1)
            or frame_index != source.frames[-1].index + 1
        ):
            raise ValueError("target projection/action/frame binding differs")
    labels = _qualified_labels(handedraw, audit)
    return QualifiedObservation(
        unit.fingerprint,
        sha256_bytes(unit.payload),
        episode,
        phase,
        branch,
        head,
        tree,
        context,
        frame_index,
        lateral,
        labels,
        sha256_bytes(canonical_json_bytes(handedraw)),
        sha256_bytes(canonical_json_bytes(audit)),
    )
