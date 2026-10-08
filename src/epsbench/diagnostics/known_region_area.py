"""Model-free all-known-node image-area prediction and per-target action selection."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction as Q
from typing import Any, cast

import numpy as np

from epsbench.diagnostics.boundary_observation import VisibleRaster
from epsbench.diagnostics.causal_region_lifecycle import Provider, TrustedObservation
from epsbench.diagnostics.known_region_events import checked, probability
from epsbench.diagnostics.neutral_observation_target import _target
from epsbench.diagnostics.restricted_mask_projection import Projection
from epsbench.diagnostics.visible_forecast_contract import Array, CausalInput, Limits, _json, _keys
from epsbench.schema import ModalityPermissionSet
from epsbench.utils.canonical import canonical_json_bytes, sha256_bytes

VERSION = "known-region-area-v1"
CONTROLS = ("zero", "persistence", "replay", "linear_trend", "centroid_transport", "history_lookup")
Rows = tuple[tuple[str, int | float | Q], ...]
Arms = Sequence[tuple[str, Sequence[bytes]]]


def domain(projection: Projection) -> CausalInput:
    source = checked(projection)
    if any(c[0] != 0 or c[2] != 0 for c in (*source.executed, source.announced)):
        raise ValueError("whole executed/announced commands must be pure lateral")
    return source


def save(projection: Projection, rows: Rows) -> bytes:
    """Canonical complete forecast; floats denote their exact IEEE rational value."""
    domain(projection)
    if type(rows) is not tuple or any(type(r) is not tuple or len(r) != 2 for r in rows):
        raise ValueError("complete typed area rows required")
    if (
        any(type(name) is not str for name, _ in rows)
        or tuple(sorted(n for n, _ in rows)) != projection.alignment
    ):
        raise ValueError("exact unique prefix-known inventory required")
    values = [(name, probability(value)) for name, value in sorted(rows)]
    if sum((v for _, v in values), Q(0)) > 1:
        raise ValueError("per-action known area sum exceeds one")
    return canonical_json_bytes(
        {
            "version": VERSION + ":candidate",
            "binding_sha256": sha256_bytes(projection.binding_bytes()),
            "rows": [[n, str(v)] for n, v in values],
        }
    )


def validate(projection: Projection, saved: bytes) -> tuple[Q, ...]:
    domain(projection)  # Typed permission denial precedes parsing.
    p = _json(saved)
    _keys(p, {"version", "binding_sha256", "rows"})
    if (
        p["version"] != VERSION + ":candidate"
        or p["binding_sha256"] != sha256_bytes(projection.binding_bytes())
        or type(p["rows"]) is not list
    ):
        raise ValueError("exact area/action binding required")
    rows = []
    for row in p["rows"]:
        if type(row) is not list or len(row) != 2 or any(type(v) is not str for v in row):
            raise ValueError("canonical name/rational rows required")
        try:
            rows.append((row[0], Q(row[1])))
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError("invalid rational area") from error
    if save(projection, tuple(rows)) != saved:
        raise ValueError("canonical complete area forecast required")
    return tuple(value for _, value in rows)


def _area(mask: Array) -> Q:
    return Q(int(np.count_nonzero(mask)), mask.size)


def _round(value: Q) -> int:
    """Nearest integer, exact half ties away from zero."""
    magnitude = abs(value)
    result = (2 * magnitude.numerator + magnitude.denominator) // (2 * magnitude.denominator)
    return result if value >= 0 else -result


def _transport(previous: Array, current: Array, ratio: Q) -> Q:
    if not np.any(previous) or not np.any(current):
        return _area(current)
    coordinates = [np.argwhere(mask) for mask in (previous, current)]
    displacement = tuple(
        _round(
            ratio
            * (
                Q(int(coordinates[1][:, j].sum()), len(coordinates[1]))
                - Q(int(coordinates[0][:, j].sum()), len(coordinates[0]))
            )
        )
        for j in range(2)
    )
    shifted = coordinates[1] + np.array(displacement)
    valid = np.all((shifted >= 0) & (shifted < np.array(current.shape)), axis=1)
    return Q(int(np.count_nonzero(valid)), current.size)


def control(projection: Projection, method: str) -> bytes:
    source = domain(projection)
    if type(method) is not str or method not in CONTROLS:
        raise ValueError("declared model-free area control required")
    # Predictor reads only observed masks and whole action commands, never provenance.
    previous, current = (dict(frame.masks) for frame in source.frames)
    empty = np.zeros(source.shape, dtype=np.bool_)
    e, a = source.executed[0][1], source.announced[1]
    values = []
    for name in source.inventory:
        p, c = previous.get(name, empty), current.get(name, empty)
        if method == "zero":
            value = Q(0)
        elif method == "replay":
            value = _area(p)
        elif method == "persistence":
            value = _area(c)
        elif method == "linear_trend":
            value = (
                _area(c)
                if e == 0
                else min(Q(1), max(Q(0), _area(c) + a / e * (_area(c) - _area(p))))
            )
        elif method == "history_lookup" and a == 0:
            value = _area(c)
        elif method == "history_lookup" and a == -e:
            value = _area(p)
        else:
            value = _area(c) if a == 0 or e == 0 else _transport(p, c, a / e)
        values.append(value)
    total = sum(values, Q(0))
    if method == "linear_trend" and total > 1:
        values = [v / total for v in values]
    # Transport cannot increase any current mask cardinality; overlap needs no scaling.
    return save(projection, tuple(zip(source.inventory, values, strict=True)))


def _seal(projections: Sequence[Projection], arms: Arms) -> tuple[Any, ...]:
    projections = tuple(projections)
    validated = tuple(domain(p) for p in projections)
    sources = tuple(
        CausalInput.from_bytes(
            s.canonical_bytes(),
            s.permissions,
            Limits(s.limits.max_frames, s.limits.max_pixels, s.limits.max_tokens),
        )
        for s in validated
    )
    arms = tuple((name, tuple(candidates)) for name, candidates in arms)
    if (
        not projections
        or not arms
        or any(type(name) is not str or not name for name, _ in arms)
        or len({name for name, _ in arms}) != len(arms)
    ):
        raise ValueError("nonempty distinct named arms/actions required")
    common = []
    for p, source in zip(projections, sources, strict=True):
        payload = json.loads(source.canonical_bytes())
        del payload["announced"]
        common.append(
            canonical_json_bytes(
                {
                    "prefix": payload,
                    "state": json.loads(p._state),
                    "episode": p.episode,
                    "source_head": p.source_head,
                    "cutoff": p.cutoff,
                    "availability": p.availability,
                }
            )
        )
    if len(set(common)) != 1:
        raise ValueError("common genuine prefix/identity/association/provenance required")
    actions = tuple(s.announced for s in sources)
    if len(set(actions)) != len(actions):
        raise ValueError("duplicate numeric actions denied")
    order = tuple(sorted(range(len(actions)), key=lambda j: actions[j]))
    values = []
    for name, candidates in arms:
        if len(candidates) != len(projections) or any(type(c) is not bytes for c in candidates):
            raise ValueError("all arms must seal exact bytes for every action")
        values.append(
            (
                name,
                tuple(validate(p, c) for p, c in zip(projections, candidates, strict=True)),
                candidates,
            )
        )
    # All metadata and callbacks are captured before future retrieval. Sources own bytes.
    statuses = tuple(node["status"] for node in json.loads(projections[0].feature_bytes())["nodes"])
    bindings = tuple(sha256_bytes(p.binding_bytes()) for p in projections)
    return sources, actions, order, tuple(sorted(values)), statuses, bindings


def _metrics(
    q: tuple[tuple[Q, ...], ...],
    y: tuple[tuple[Q, ...], ...],
    indices: tuple[int, ...],
    order: tuple[int, ...],
) -> dict[str, Any]:
    if not indices:
        return {"nodes": 0, "mse": None, "mae": None, "bias": None, "mean_regret": None}
    k, n = len(y), len(indices)
    errors = tuple(q[a][i] - y[a][i] for a in range(k) for i in indices)
    selected = tuple(next(a for a in order if q[a][i] == max(r[i] for r in q)) for i in indices)
    regrets = tuple(max(r[i] for r in y) - y[a][i] for i, a in zip(indices, selected, strict=True))
    mse = sum((e * e for e in errors), Q(0)) / (k * n)
    regret = sum(regrets, Q(0)) / n
    assert regret <= 1 and regret * regret <= 4 * k * mse
    return {
        "nodes": n,
        "mse": str(mse),
        "mae": str(sum(map(abs, errors), Q(0)) / (k * n)),
        "bias": str(sum(errors, Q(0)) / (k * n)),
        "mean_regret": str(regret),
        "selected_action_indices": selected,
        "per_target_regret": tuple(map(str, regrets)),
        "predicted_ties": tuple(
            tuple(a for a in order if q[a][i] == q[selected[j]][i]) for j, i in enumerate(indices)
        ),
        "regret_bound_squared": str(min(Q(1), 4 * k * mse)),
    }


@dataclass(frozen=True)
class Evaluation:
    """Immutable evidence; report access does not access providers or projections."""

    receipt_bytes: bytes

    def report(self) -> dict[str, Any]:
        return cast(
            dict[str, Any], json.loads(self.receipt_bytes)["report"]
        )  # Fresh mutable presentation only.


def evaluate(
    projections: Sequence[Projection], arms: Arms, providers: Sequence[Provider]
) -> Evaluation:
    """All-arm seal first; one trusted future per action; no partial result on failure."""
    sources, actions, order, sealed, statuses, bindings = _seal(projections, arms)
    providers = tuple(providers)
    if len(providers) != len(sources):
        raise ValueError("one provider per common action required")
    fetches = tuple(provider.observation for provider in providers)
    observations, targets, new = [], [], []
    for source, fetch in zip(sources, fetches, strict=True):
        observation = fetch(2)
        if (
            type(observation) is not TrustedObservation
            or observation.raster is None
            or any(
                v is not True
                for v in (
                    observation.complete_image,
                    observation.complete_association,
                    observation.stable_identity,
                )
            )
            or any(
                v is not None and (type(v) is not int or v != 0)
                for v in (observation.unresolved_labels, observation.unresolved_pixels)
            )
        ):
            raise ValueError("complete trusted future association required")
        target = _target(source, observation.raster)
        targets.append(tuple(_area(mask) for _, mask in target.channels.known if mask is not None))
        assert target.channels.new is not None
        new.append(str(_area(target.channels.new)))
        raster = observation.raster
        observations.append(
            {
                "index": raster.sequence_index,
                "segmentation": raster.segmentation.tolist(),
                "identities": raster.identities,
            }
        )
    y = tuple(targets)
    n = len(statuses)
    report = {
        "scope": (
            "known-node future image area; independent target queries; no simultaneous achievement"
        ),
        "qualification": "external trusted adapter assertions only",
        "actions": [[str(v) for v in actions[a]] for a in order],
        "target_areas": [[str(v) for v in y[a]] for a in order],
        "statuses": statuses,
        "new_area": [new[a] for a in order],
        "action_contrasts": [str(max(r[i] for r in y) - min(r[i] for r in y)) for i in range(n)],
        "optimal_ties": [
            tuple(j for j, a in enumerate(order) if y[a][i] == max(r[i] for r in y))
            for i in range(n)
        ],
        "fixed_first_mean_regret": str(
            sum((max(r[i] for r in y) - y[order[0]][i] for i in range(n)), Q(0)) / n
        ),
        "fixed_last_mean_regret": str(
            sum((max(r[i] for r in y) - y[order[-1]][i] for i in range(n)), Q(0)) / n
        ),
        "arms": {},
    }
    for name, forecasts, _ in sealed:
        # Report selectors in canonical numeric action order, not caller order.
        q = tuple(forecasts[a] for a in order)
        ordered_y = tuple(y[a] for a in order)
        canonical_order = tuple(range(len(order)))
        report["arms"][name] = {
            "all": _metrics(q, ordered_y, tuple(range(n)), canonical_order),
            "strata": {
                status: _metrics(
                    q,
                    ordered_y,
                    tuple(i for i, s in enumerate(statuses) if s == status),
                    canonical_order,
                )
                for status in ("VISIBLE", "REMEMBERED_ABSENT")
            },
        }
    receipt = canonical_json_bytes(
        {
            "version": VERSION + ":evaluation",
            "bindings": [bindings[a] for a in order],
            "candidate_sha256": {
                name: [sha256_bytes(candidates[a]) for a in order] for name, _, candidates in sealed
            },
            "observations": [observations[a] for a in order],
            "report": report,
        }
    )
    return Evaluation(receipt)


@dataclass(frozen=True)
class _Retained:
    observation_bytes: bytes
    shape: tuple[int, int]
    max_tokens: int

    def observation(self, index: int) -> TrustedObservation:
        p = _json(self.observation_bytes)
        _keys(p, {"index", "segmentation", "identities"})
        if p["index"] != index or type(p["index"]) is not int:
            raise ValueError("retained target chronology differs")
        # Fixed source budget/shape precedes retained array expansion.
        rows = p["segmentation"]
        if (
            type(rows) is not list
            or len(rows) != self.shape[0]
            or any(
                type(r) is not list
                or len(r) != self.shape[1]
                or any(type(v) is not int or not 0 <= v <= 2**31 - 1 for v in r)
                for r in rows
            )
        ):
            raise ValueError("retained integer segmentation required")
        if (
            type(p["identities"]) is not list
            or len(p["identities"]) > self.max_tokens
            or any(
                type(r) is not list or len(r) != 2 or type(r[0]) is not int or type(r[1]) is not str
                for r in p["identities"]
            )
        ):
            raise ValueError("retained strict identity pairs required")
        raster = VisibleRaster(
            index, np.array(rows, dtype=np.int32), tuple((r[0], r[1]) for r in p["identities"])
        )
        return TrustedObservation(raster, True, True, True)


def inspect(
    projections: Sequence[Projection], arms: Arms, receipt: bytes, access: ModalityPermissionSet
) -> Evaluation:
    """Retained-only exact replay; no claim that a receipt proves physical qualification."""
    from epsbench.diagnostics.visible_forecast_contract import permissions

    permissions(access)  # Before evidence parsing, including invalid supplied evidence.
    projections = tuple(projections)
    for projection in projections:
        domain(projection)
    arms = tuple((name, tuple(candidates)) for name, candidates in arms)
    sources, _, order, _, _, _ = _seal(projections, arms)
    p = _json(receipt)
    _keys(p, {"version", "bindings", "candidate_sha256", "observations", "report"})
    if (
        p["version"] != VERSION + ":evaluation"
        or type(p["observations"]) is not list
        or len(p["observations"]) != len(projections)
    ):
        raise ValueError("complete retained evaluation required")
    retained = tuple(
        _Retained(canonical_json_bytes(o), sources[a].shape, sources[a].limits.max_tokens)
        for a, o in zip(order, p["observations"], strict=True)
    )
    by_original = tuple(retained[order.index(a)] for a in range(len(order)))
    result = evaluate(projections, arms, by_original)
    if result.receipt_bytes != receipt:
        raise ValueError("retained binding/report/canonical bytes differ")
    return result
