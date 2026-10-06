"""Raster-only boundary observation v1; no ownership inference or legacy adapter."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
import numpy.typing as npt

from epsbench.schema import BoundaryAxis, Modality, ModalityPermissionSet
from epsbench.utils.canonical import canonical_json_bytes

VERSION = "boundary-observation-v1"
Array = npt.NDArray[Any]


def _index(value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("nonnegative integer sequence index required")


@dataclass(frozen=True)
class VisibleRaster:
    sequence_index: int
    segmentation: Array
    identities: tuple[tuple[int, str], ...]

    def __post_init__(self) -> None:
        _index(self.sequence_index)
        a = self.segmentation
        if not isinstance(a, np.ndarray) or (
            a.dtype != np.int32 or a.ndim != 2 or min(a.shape) < 1 or np.any(a < 0)
        ):
            raise ValueError("nonempty nonnegative int32 segmentation required")
        pairs = tuple((label, token) for label, token in self.identities)
        labels = {int(v) for v in np.unique(a) if v != 0}
        if (
            any(type(label) is not int or type(token) is not str for label, token in pairs)
            or {label for label, _ in pairs} != labels
            or len(pairs) != len(labels)
            or len({token for _, token in pairs}) != len(pairs)
            or any(re.fullmatch(r"surface-[0-9a-f]{16}", token) is None for _, token in pairs)
        ):
            raise ValueError(
                "bijective opaque identities for exactly visible nonzero labels required"
            )
        snapshot = np.frombuffer(a.tobytes(order="C"), dtype=np.int32).reshape(a.shape)
        object.__setattr__(self, "segmentation", snapshot)
        object.__setattr__(self, "identities", tuple(sorted(pairs)))


@dataclass(frozen=True)
class ObservedEdge:
    axis: BoundaryAxis
    row: int
    column: int
    negative_surface_id: str | None
    positive_surface_id: str | None
    ownership: Literal["unknown"] = "unknown"

    def __post_init__(self) -> None:
        _index(self.row)
        _index(self.column)
        tokens = (self.negative_surface_id, self.positive_surface_id)
        if (
            type(self.axis) is not BoundaryAxis
            or self.ownership != "unknown"
            or tokens[0] == tokens[1]
            or any(
                t is not None
                and (type(t) is not str or re.fullmatch(r"surface-[0-9a-f]{16}", t) is None)
                for t in tokens
            )
        ):
            raise ValueError("typed differing visible neighbours and unknown ownership required")


@dataclass(frozen=True)
class BoundaryObservation:
    sequence_index: int
    shape: tuple[int, int]
    edges: tuple[ObservedEdge, ...]

    def __post_init__(self) -> None:
        _index(self.sequence_index)
        shape = tuple(self.shape)
        if len(shape) != 2 or any(type(v) is not int or v < 1 for v in shape):
            raise ValueError("positive two-dimensional shape required")
        edges = tuple(self.edges)
        positions = set()
        for edge in edges:
            if type(edge) is not ObservedEdge:
                raise ValueError("typed observed edges required")
            dr, dc = (0, 1) if edge.axis == BoundaryAxis.HORIZONTAL else (1, 0)
            position = (edge.axis, edge.row, edge.column)
            if edge.row >= shape[0] - dr or edge.column >= shape[1] - dc or position in positions:
                raise ValueError("unique internal lattice edges required")
            positions.add(position)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(
            self, "edges", tuple(sorted(edges, key=lambda e: (e.axis.value, e.row, e.column)))
        )

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(
            {
                "version": VERSION,
                "sequence_index": self.sequence_index,
                "shape": list(self.shape),
                "edges": [
                    {
                        "axis": edge.axis.value,
                        "row": edge.row,
                        "column": edge.column,
                        "negative_surface_id": edge.negative_surface_id,
                        "positive_surface_id": edge.positive_surface_id,
                        "ownership": edge.ownership,
                    }
                    for edge in self.edges
                ],
            }
        )


class RasterProvider(Protocol):
    def raster(self, sequence_index: int) -> VisibleRaster: ...


@dataclass(frozen=True)
class BoundaryObservationView:
    provider: RasterProvider
    permissions: ModalityPermissionSet
    decision_index: int

    def __post_init__(self) -> None:
        _index(self.decision_index)
        if type(self.permissions) is not ModalityPermissionSet:
            raise ValueError("typed permissions required")
        # Own a snapshot of the caller's frozen permission object.
        object.__setattr__(self, "permissions", self.permissions.model_copy(deep=True))

    def observe(self, sequence_index: int) -> BoundaryObservation:
        _index(sequence_index)
        if sequence_index > self.decision_index:
            raise PermissionError("future raster denied before provider access")
        if not self.permissions.permits(Modality.SURFACE_REGIONS):
            raise PermissionError("surface regions denied before provider access")
        raster = self.provider.raster(sequence_index)
        if type(raster) is not VisibleRaster or raster.sequence_index != sequence_index:
            raise ValueError("typed raster chronology differs")
        # Revalidate and own provider data; no oracle boundary payload can be admitted.
        raster = VisibleRaster(sequence_index, raster.segmentation, raster.identities)
        lookup = dict(raster.identities)
        a = raster.segmentation
        h, w = a.shape
        edges = []
        for axis, dr, dc in ((BoundaryAxis.HORIZONTAL, 0, 1), (BoundaryAxis.VERTICAL, 1, 0)):
            for row in range(h - dr):
                for column in range(w - dc):
                    negative = int(a[row, column])
                    positive = int(a[row + dr, column + dc])
                    if negative != positive:
                        edges.append(
                            ObservedEdge(
                                axis, row, column, lookup.get(negative), lookup.get(positive)
                            )
                        )
        return BoundaryObservation(sequence_index, (h, w), tuple(edges))
