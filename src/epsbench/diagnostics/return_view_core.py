"""Return-view oracle contracts. Geometry, RGB and flow are not rule inputs."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations
from typing import Any

import numpy as np
import numpy.typing as npt

from epsbench.diagnostics.boundary_observation import (
    BoundaryObservation,
    BoundaryObservationView,
    VisibleRaster,
)
from epsbench.schema import Modality, ModalityPermissionSet
from epsbench.utils.canonical import canonical_json_bytes

VERSION = "return-view-v1"
REQUIRED = frozenset(
    {
        Modality.SURFACE_REGIONS,
        Modality.REGION_CORRESPONDENCE,
        Modality.BOUNDARY_STRUCTURE,
        Modality.EXECUTED_ACTION,
    }
)
Array = npt.NDArray[Any]


@dataclass(frozen=True)
class Observation:
    raster: VisibleRaster
    boundary: BoundaryObservation

    def __post_init__(self) -> None:
        if type(self.raster) is not VisibleRaster or type(self.boundary) is not BoundaryObservation:
            raise ValueError("typed observations required")
        expected = observe(self.raster)
        if self.boundary.canonical_bytes() != expected.boundary.canonical_bytes():
            raise ValueError("boundary must derive solely from admitted raster")


class _Provider:
    def __init__(self, raster: VisibleRaster) -> None:
        self.value = raster

    def raster(self, sequence_index: int) -> VisibleRaster:
        if sequence_index != self.value.sequence_index:
            raise PermissionError("unavailable frame")
        return self.value


def observe(raster: VisibleRaster) -> Observation:
    # Avoid recursive constructor revalidation while constructing the derived observation.
    boundary = BoundaryObservationView(
        _Provider(raster), ModalityPermissionSet(allowed=REQUIRED), raster.sequence_index
    ).observe(raster.sequence_index)
    result = object.__new__(Observation)
    object.__setattr__(
        result,
        "raster",
        VisibleRaster(raster.sequence_index, raster.segmentation, raster.identities),
    )
    object.__setattr__(result, "boundary", boundary)
    return result


@dataclass(frozen=True)
class HistoryView:
    observations: tuple[Observation, ...]
    executed: tuple[Fraction, ...]
    announced: Fraction
    permissions: ModalityPermissionSet
    recent_only: bool

    def __post_init__(self) -> None:
        if (
            type(self.permissions) is not ModalityPermissionSet
            or self.permissions.allowed != REQUIRED
        ):
            raise PermissionError("exact typed diagnostic permissions required")
        if type(self.recent_only) is not bool:
            raise ValueError("typed history policy required")
        indices = (1, 2) if self.recent_only else (0, 1, 2)
        if tuple(o.raster.sequence_index for o in self.observations) != indices:
            raise ValueError("complete two-frame recent or three-frame full history required")
        if len(self.executed) != 2 or any(
            type(a) is not Fraction for a in (*self.executed, self.announced)
        ):
            raise ValueError("exact rational pure-lateral commands required")
        shape = self.observations[0].raster.segmentation.shape
        tokens: dict[int, str] = {}
        for o in self.observations:
            Observation(o.raster, o.boundary)
            if o.raster.segmentation.shape != shape:
                raise ValueError("fixed calibration/raster required")
            for label, token in o.raster.identities:
                if label in tokens and tokens[label] != token:
                    raise ValueError("association changed")
                tokens[label] = token
        if len(set(tokens.values())) != len(tokens):
            raise ValueError("association must remain bijective")
        object.__setattr__(self, "observations", tuple(self.observations))
        object.__setattr__(self, "executed", tuple(self.executed))
        object.__setattr__(self, "permissions", self.permissions.model_copy(deep=True))

    def canonical_bytes(self, renaming: dict[str, str] | None = None) -> bytes:
        rename: dict[str | None, str | None] = {}
        rename.update(renaming or {})
        return canonical_json_bytes(
            {
                "domain": VERSION + ":input",
                "recent_only": self.recent_only,
                "executed": [str(a) for a in self.executed],
                "announced": str(self.announced),
                "observations": [
                    {
                        "index": o.raster.sequence_index,
                        "tokens": [
                            [
                                None
                                if v == 0
                                else rename.get(
                                    dict(o.raster.identities)[int(v)],
                                    dict(o.raster.identities)[int(v)],
                                )
                                for v in row
                            ]
                            for row in o.raster.segmentation
                        ],
                        "boundary": [
                            {
                                "axis": e.axis.value,
                                "row": e.row,
                                "column": e.column,
                                "negative": rename.get(
                                    e.negative_surface_id, e.negative_surface_id
                                ),
                                "positive": rename.get(
                                    e.positive_surface_id, e.positive_surface_id
                                ),
                                "ownership": e.ownership,
                            }
                            for e in o.boundary.edges
                        ],
                    }
                    for o in self.observations
                ],
            }
        )


def tokens(view: HistoryView) -> tuple[str, ...]:
    return tuple(sorted({t for o in view.observations for _, t in o.raster.identities}))


def collision_bijection(a: HistoryView, b: HistoryView) -> dict[str, str] | None:
    ta, tb = tokens(a), tokens(b)
    if len(ta) != len(tb) or len(ta) > 3:
        return None
    for order in permutations(ta):
        mapping = dict(zip(tb, order, strict=True))
        if a.canonical_bytes() == b.canonical_bytes(mapping):
            return mapping
    return None


def masks(observation: Observation, wanted: tuple[str, ...]) -> tuple[tuple[str, Array], ...]:
    lookup = {token: label for label, token in observation.raster.identities}
    return tuple(
        (
            token,
            np.frombuffer(
                (observation.raster.segmentation == lookup.get(token, -1)).tobytes(), dtype=np.bool_
            ).reshape(observation.raster.segmentation.shape),
        )
        for token in wanted
    )


@dataclass(frozen=True)
class Forecast:
    rule: str
    masks: tuple[tuple[str, Array | None], ...]

    def __post_init__(self) -> None:
        if self.rule not in {"current-mask-persistence", "exact-action-return-cache"}:
            raise ValueError("fixed rule required")
        if len({t for t, _ in self.masks}) != len(self.masks):
            raise ValueError("unique forecast inventory required")
        snapshot = []
        for token, array in self.masks:
            if array is not None:
                if array.dtype != np.bool_ or array.ndim != 2:
                    raise ValueError("boolean masks required")
                array = np.frombuffer(array.tobytes(), dtype=np.bool_).reshape(array.shape)
            snapshot.append((token, array))
        object.__setattr__(self, "masks", tuple(snapshot))

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(
            {
                "domain": VERSION + ":forecast",
                "rule": self.rule,
                "masks": {t: None if a is None else a.astype(int).tolist() for t, a in self.masks},
            }
        )


def persistence(view: HistoryView) -> Forecast:
    if not view.recent_only:
        raise PermissionError("persistence receives the complete recent window")
    return Forecast("current-mask-persistence", masks(view.observations[-1], tokens(view)))


def return_cache(view: HistoryView) -> Forecast:
    if view.recent_only:
        raise PermissionError("cache requires full history")
    positions = (Fraction(0), view.executed[0], sum(view.executed, Fraction(0)))
    target = positions[-1] + view.announced
    match = next(
        (o for p, o in zip(positions, view.observations, strict=True) if p == target), None
    )
    return Forecast(
        "exact-action-return-cache",
        tuple((t, None) for t in tokens(view)) if match is None else masks(match, tokens(view)),
    )


def score(forecast: Forecast, target: Observation) -> dict[str, int | bool]:
    truth = dict(masks(target, tuple(t for t, _ in forecast.masks)))
    inventory = {t for t, _ in forecast.masks}
    not_observed = len({t for _, t in target.raster.identities} - inventory)
    unknown = sum(a is None for _, a in forecast.masks)
    errors = sum(int(np.count_nonzero(a != truth[t])) for t, a in forecast.masks if a is not None)
    return {
        "unknown": unknown,
        "not_observed_unscored": not_observed,
        "symmetric_difference": errors,
        "exact": not unknown and not errors and not not_observed,
    }
