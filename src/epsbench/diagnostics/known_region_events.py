"""Model-free known-region event marginals and identical-information encodings."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from fractions import Fraction as Q
from typing import Any

import numpy as np

from epsbench.diagnostics.causal_region_lifecycle import Provider, TrustedObservation
from epsbench.diagnostics.neutral_observation_target import _target
from epsbench.diagnostics.restricted_mask_projection import Projection
from epsbench.diagnostics.visible_forecast_contract import Array, CausalInput, _json, _keys
from epsbench.utils.canonical import canonical_json_bytes, sha256_bytes

VERSION = "known-region-events-v1"
EVENTS = ("appearance", "disappearance", "gained_support", "lost_support")
FIELDS = (
    "nodes",
    "endpoints",
    "executed_action",
    "announced_action",
    "current_union",
    "previous_union",
)
NODE_FIELDS = ("previous", "current", "difference", "status", "first_seen_age", "last_seen_age")
Probability = int | float | Q
Rows = tuple[tuple[str, tuple[Probability, ...]], ...]


def checked(projection: Projection) -> CausalInput:
    if type(projection) is not Projection:
        raise ValueError("exact typed Projection required")
    source = projection.revalidate()  # Typed permissions precede parsing/access.
    if len(source.frames) != 2 or not source.inventory or projection.availability != (True, True):
        raise ValueError(
            "complete episode-start 0/1 prefix, known nodes and both relations required"
        )
    return source


def events(before: Array, after: Array) -> tuple[bool, bool, bool, bool]:
    """Observational support predicates; gain and loss can co-occur."""
    return (
        not bool(np.any(before)) and bool(np.any(after)),
        bool(np.any(before)) and not bool(np.any(after)),
        bool(np.any(after & ~before)),
        bool(np.any(before & ~after)),
    )


def probability(value: Probability) -> Q:
    if type(value) not in (int, float, Q) or (type(value) is float and not math.isfinite(value)):
        raise ValueError("finite exact numeric probability required; Boolean is not a probability")
    result = Q(value)
    if not 0 <= result <= 1:
        raise ValueError("probability outside [0,1]")
    return result


def save(projection: Projection, rows: Rows) -> bytes:
    checked(projection)
    if type(rows) is not tuple or any(type(row) is not tuple or len(row) != 2 for row in rows):
        raise ValueError("typed complete probability rows required")
    if tuple(sorted(name for name, _ in rows)) != projection.alignment:
        raise ValueError("exact unique prefix inventory required")
    values = []
    for name, row in sorted(rows):
        if type(row) is not tuple or len(row) != 4:
            raise ValueError("all four event probabilities required")
        values.append([name, [str(probability(v)) for v in row]])
    return canonical_json_bytes(
        {
            "version": VERSION + ":candidate",
            "binding_sha256": sha256_bytes(projection.binding_bytes()),
            "rows": values,
        }
    )


def validate(projection: Projection, saved: bytes) -> tuple[tuple[Q, ...], ...]:
    checked(projection)
    p = _json(saved)
    _keys(p, {"version", "binding_sha256", "rows"})
    if (
        p["version"] != VERSION + ":candidate"
        or p["binding_sha256"] != sha256_bytes(projection.binding_bytes())
        or type(p["rows"]) is not list
        or len(p["rows"]) != len(projection.alignment)
    ):
        raise ValueError("exact input/action/target binding and coverage required")
    rows = []
    for item in p["rows"]:
        if (
            type(item) is not list
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not list
            or len(item[1]) != 4
            or any(type(v) is not str for v in item[1])
        ):
            raise ValueError("four canonical rational strings required per row")
        try:
            values = tuple(Q(v) for v in item[1])
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError("invalid probability rational") from error
        rows.append((item[0], values))
    canonical = save(projection, tuple(rows))
    if canonical != saved:
        raise ValueError("canonical complete sealed probability candidate required")
    return tuple(row for _, row in rows)


@dataclass(frozen=True)
class Evaluation:
    """Immutable evaluator output; labels/provenance never enter feature encodings."""

    target: tuple[tuple[bool, bool, bool, bool], ...]
    losses: tuple[tuple[Q, ...], ...]
    probabilities: tuple[tuple[Q, ...], ...]
    statuses: tuple[str, ...]
    receipt_bytes: bytes

    def report(self) -> dict[str, Any]:
        n = len(self.target)
        by_event = tuple(sum((r[j] for r in self.losses), Q(0)) / n for j in range(4))
        strata = {}
        for status in ("VISIBLE", "REMEMBERED_ABSENT"):
            indices = [i for i, s in enumerate(self.statuses) if s == status]
            strata[status] = {
                "nodes": len(indices),
                "brier": None
                if not indices
                else str(
                    sum((sum(self.losses[i], Q(0)) for i in indices), Q(0)) / (4 * len(indices))
                ),
            }
        return {
            "version": VERSION,
            "nodes": n,
            "N": 4 * n,
            "C": 4 * n,
            "U": 0,
            "coverage": "1",
            "brier": str(sum(by_event, Q(0)) / 4),
            "event_brier": dict(zip(EVENTS, map(str, by_event), strict=True)),
            "event_frequency": {
                e: str(Q(sum(row[j] for row in self.target), n)) for j, e in enumerate(EVENTS)
            },
            "strata": strata,
            "cause_status": "UNAVAILABLE_UNSUPPORTED",
        }


def evaluate(projection: Projection, saved: bytes, provider: Provider) -> Evaluation:
    source = checked(projection)
    predictions = validate(projection, saved)  # Seal before any evaluator future operation.
    binding_digest = sha256_bytes(projection.binding_bytes())
    statuses = tuple(node["status"] for node in json.loads(projection.feature_bytes())["nodes"])
    observation = provider.observation(2)  # Exactly one fetch; no retries or per-node fetches.
    if (
        type(observation) is not TrustedObservation
        or observation.raster is None
        or observation.complete_image is not True
        or observation.complete_association is not True
        or observation.stable_identity is not True
        or (
            observation.unresolved_labels is not None
            and (
                type(observation.unresolved_labels) is not int or observation.unresolved_labels != 0
            )
        )
        or (
            observation.unresolved_pixels is not None
            and (
                type(observation.unresolved_pixels) is not int or observation.unresolved_pixels != 0
            )
        )
    ):
        raise ValueError("future complete stable association qualification unresolved")
    target = _target(source, observation.raster)  # Unchanged neutral known/NEW construction.
    current = dict(source.frames[1].masks)
    empty = np.zeros(source.shape, dtype=np.bool_)
    label_rows = []
    for name, after in target.channels.known:
        assert after is not None
        label_rows.append(events(current.get(name, empty), after))
    labels = tuple(label_rows)
    losses = tuple(
        tuple((p - int(y)) ** 2 for p, y in zip(row, truth, strict=True))
        for row, truth in zip(predictions, labels, strict=True)
    )
    receipt = canonical_json_bytes(
        {
            "version": VERSION + ":receipt",
            "binding_sha256": binding_digest,
            "candidate_sha256": sha256_bytes(saved),
            "neutral_target_sha256": target.digest,
            "target_index": 2,
            "qualification": "EXTERNAL_ADAPTER_ASSERTIONS_ONLY",
        }
    )
    return Evaluation(labels, losses, predictions, statuses, receipt)


def control(projection: Projection, method: str) -> bytes:
    checked(projection)
    f = json.loads(projection.feature_bytes())
    rows = []
    for name, node in zip(projection.alignment, f["nodes"], strict=True):
        values: tuple[Q, ...]
        if method == "stable":
            values = (Q(0),) * 4
        elif method == "half":
            values = (Q(1, 2),) * 4
        elif method == "replay":
            values = tuple(
                Q(int(v))
                for v in events(
                    np.array(node["previous"], dtype=np.bool_),
                    np.array(node["current"], dtype=np.bool_),
                )
            )
        elif method == "visibility_saturation":
            values = (
                Q(0),
                Q(0),
                Q(int(node["status"] == "VISIBLE")),
                Q(int(node["status"] == "VISIBLE")),
            )
        else:
            raise ValueError("only four fixed untrained controls admitted")
        rows.append((name, values))
    return save(projection, tuple(rows))


def structured(
    projection: Projection, order: tuple[int, ...] | None = None, *, omit_relations: bool = False
) -> bytes:
    checked(projection)
    if type(omit_relations) is not bool:
        raise ValueError("typed ablation flag required")
    n = len(projection.alignment)
    order = tuple(range(n)) if order is None else order
    if (
        type(order) is not tuple
        or any(type(i) is not int for i in order)
        or sorted(order) != list(range(n))
    ):
        raise ValueError("exact complete row permutation required")
    f = json.loads(projection.feature_bytes())
    f["nodes"] = [f["nodes"][i] for i in order]
    inverse = {old: new for new, old in enumerate(order)}
    for endpoint in f["endpoints"]:
        endpoint["pairs"] = (
            None
            if omit_relations
            else sorted([sorted((inverse[a], inverse[b])) for a, b in endpoint["pairs"]])
        )
    return canonical_json_bytes(
        {"version": VERSION + ":structured", "relations_omitted": omit_relations, "features": f}
    )


def unstructured(
    projection: Projection, order: tuple[int, ...] | None = None, *, omit_relations: bool = False
) -> bytes:
    """Positional serialization, no field names/graph object; masks retain their dimensions."""
    p = json.loads(structured(projection, order, omit_relations=omit_relations))
    f = p["features"]
    values = [
        [[row[k] for k in NODE_FIELDS] for row in f["nodes"]],
        [[e["age"], e["available"], e["pairs"]] for e in f["endpoints"]],
        *[f[k] for k in FIELDS[2:]],
    ]
    return canonical_json_bytes(
        {
            "version": VERSION + ":unstructured",
            "relations_omitted": omit_relations,
            "values": values,
        }
    )


def reconstruct(data: bytes) -> bytes:
    """Invert our positional codec to the structured payload without consulting a target."""
    p = _json(data)
    _keys(p, {"version", "relations_omitted", "values"})
    v = p["values"]
    if (
        p["version"] != VERSION + ":unstructured"
        or type(p["relations_omitted"]) is not bool
        or type(v) is not list
        or len(v) != 6
        or type(v[0]) is not list
        or type(v[1]) is not list
        or len(v[1]) != 2
        or any(type(row) is not list or len(row) != 6 for row in v[0])
        or any(type(e) is not list or len(e) != 3 for e in v[1])
        or data != canonical_json_bytes(p)
    ):
        raise ValueError("canonical positional encoding required")
    f = dict(zip(FIELDS, v, strict=True))
    f["nodes"] = [dict(zip(NODE_FIELDS, row, strict=True)) for row in v[0]]
    f["endpoints"] = [dict(zip(("age", "available", "pairs"), e, strict=True)) for e in v[1]]
    return canonical_json_bytes(
        {
            "version": VERSION + ":structured",
            "relations_omitted": p["relations_omitted"],
            "features": f,
        }
    )
