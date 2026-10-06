"""Finite synthetic construct checks, not an empirical study or RGB extraction."""

import json
from dataclasses import FrozenInstanceError, fields, replace
from fractions import Fraction

import numpy as np
import pytest

from epsbench.diagnostics.boundary_observation import (
    VERSION,
    BoundaryObservation,
    BoundaryObservationView,
    ObservedEdge,
    VisibleRaster,
)
from epsbench.diagnostics.causal_history_core import Command, CompletedFlow
from epsbench.schema import BoundaryAxis, Modality, ModalityPermissionSet

A = "surface-0000000000000001"
B = "surface-0000000000000002"
C = "surface-0000000000000003"


class Provider:
    def __init__(self, raster: VisibleRaster) -> None:
        self.value = raster
        self.calls = 0

    def raster(self, sequence_index: int) -> VisibleRaster:
        self.calls += 1
        return self.value


def raster(values: list[list[int]], i: int = 0) -> VisibleRaster:
    a = np.array(values, dtype=np.int32)
    return VisibleRaster(
        i, a, tuple((int(v), {1: A, 2: B}[int(v)]) for v in sorted(set(a.flat)) if v)
    )


def view(provider: Provider, decision: int = 0) -> BoundaryObservationView:
    return BoundaryObservationView(
        provider, ModalityPermissionSet(allowed=frozenset({Modality.SURFACE_REGIONS})), decision
    )


def test_hidden_intervention_and_visible_positive_control() -> None:
    # Two evaluator-private continuations: identical declared past evidence, differing oracle owner.
    rgb = np.zeros((1, 2, 3), dtype=np.uint8)
    optical = (raster([[1, 2]], 0), raster([[1, 2]], 1))
    flow = CompletedFlow(
        0,
        Command(Fraction(1)),
        np.zeros((1, 2, 2), dtype=np.int32),
        np.ones((1, 2), dtype=np.uint8),
        np.zeros((1, 2), dtype=np.uint8),
    )
    past = (optical, (rgb.copy(), rgb.copy()), (flow,), (flow.command,))
    variants = ((past, "hidden-left", A), (past, "hidden-right", B))
    assert variants[0][1:] != variants[1][1:]
    assert (
        variants[0][0] is variants[1][0]
    )  # Every declared past array/association/flow/command fixed.
    results = [view(Provider(v[0][0][1]), 1).observe(1) for v in variants]
    assert results[0].canonical_bytes() == results[1].canonical_bytes()
    assert results[0].edges == (ObservedEdge(BoundaryAxis.HORIZONTAL, 0, 0, A, B),)
    assert all(e.ownership == "unknown" for e in results[0].edges)
    changed = view(Provider(raster([[1, 1]], 1)), 1).observe(1)
    assert changed.canonical_bytes() != results[0].canonical_bytes()
    moved = view(Provider(raster([[1, 1, 2]], 1)), 1).observe(1)
    assert moved.edges[0].column == 1


def test_renaming_and_label_permutation_equivariance() -> None:
    original = view(Provider(raster([[1, 2], [0, 1]]))).observe(0)
    renamed = VisibleRaster(0, np.array([[7, 3], [0, 7]], dtype=np.int32), ((3, A), (7, C)))
    result = view(Provider(renamed)).observe(0)
    mapping = {A: C, B: A, None: None}
    assert result.edges == tuple(
        replace(
            e,
            negative_surface_id=mapping[e.negative_surface_id],
            positive_surface_id=mapping[e.positive_surface_id],
        )
        for e in original.edges
    )
    assert all(e.ownership == "unknown" for e in result.edges)


def test_denial_before_fetch_and_chronology() -> None:
    p = Provider(raster([[1]]))
    for i in (1, 9):
        with pytest.raises(PermissionError):
            view(p).observe(i)
    denied = BoundaryObservationView(p, ModalityPermissionSet(allowed=frozenset({Modality.RGB})), 0)
    with pytest.raises(PermissionError):
        denied.observe(0)
    for i in (-1, True):
        with pytest.raises(ValueError):
            view(p).observe(i)
    assert p.calls == 0
    with pytest.raises(ValueError, match="chronology"):
        view(p, 1).observe(1)
    assert p.calls == 1
    assert {f.name for f in fields(VisibleRaster)} == {
        "sequence_index",
        "segmentation",
        "identities",
    }


def test_owned_immutable_payload_and_canonical_serialization() -> None:
    source = np.array([[1, 2]], dtype=np.int32)
    identities = ((2, B), (1, A))
    r = VisibleRaster(0, source, identities)
    output = view(Provider(r)).observe(0)
    payload = output.canonical_bytes()
    source[:] = 0
    assert r.segmentation.tolist() == [[1, 2]]
    with pytest.raises(ValueError):
        r.segmentation.setflags(write=True)
    with pytest.raises(ValueError):
        r.segmentation[0, 0] = 0
    with pytest.raises(FrozenInstanceError):
        output.sequence_index = 8  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        output.edges[0].row = 8  # type: ignore[misc]
    assert payload == output.canonical_bytes()
    assert json.loads(payload) == {
        "version": VERSION,
        "sequence_index": 0,
        "shape": [1, 2],
        "edges": [
            {
                "axis": "horizontal",
                "row": 0,
                "column": 0,
                "negative_surface_id": A,
                "positive_surface_id": B,
                "ownership": "unknown",
            }
        ],
    }
    assert (
        payload == json.dumps(json.loads(payload), sort_keys=True, separators=(",", ":")).encode()
    )


@pytest.mark.parametrize(
    "values,axes",
    [
        ([[0]], []),
        ([[1]], []),
        ([[1, 1]], []),
        ([[0, 1]], [BoundaryAxis.HORIZONTAL]),
        ([[1], [0]], [BoundaryAxis.VERTICAL]),
        (
            [[1, 2], [2, 1]],
            [
                BoundaryAxis.HORIZONTAL,
                BoundaryAxis.HORIZONTAL,
                BoundaryAxis.VERTICAL,
                BoundaryAxis.VERTICAL,
            ],
        ),
    ],
)
def test_internal_degenerate_lattice(values: list[list[int]], axes: list[BoundaryAxis]) -> None:
    result = view(Provider(raster(values))).observe(0)
    assert [e.axis for e in result.edges] == axes
    assert all(e.ownership == "unknown" for e in result.edges)


@pytest.mark.parametrize(
    "array,identities",
    [
        (np.empty((0, 2), dtype=np.int32), ()),
        (np.array([[1]], dtype=np.int64), ((1, A),)),
        (np.array([[-1]], dtype=np.int32), ()),
        (np.array([[1]], dtype=np.int32), ()),
        (np.array([[1]], dtype=np.int32), ((1, "raw-identifier"),)),
        (np.array([[1, 2]], dtype=np.int32), ((1, A), (2, A))),
        (np.array([[1]], dtype=np.int32), ((True, A),)),
    ],
)
def test_invalid_rasters_fail_closed(array: np.ndarray, identities: tuple) -> None:  # type: ignore[type-arg]
    with pytest.raises(ValueError):
        VisibleRaster(0, array, identities)


def test_output_constructor_validation_and_order() -> None:
    h = ObservedEdge(BoundaryAxis.HORIZONTAL, 0, 0, A, None)
    v = ObservedEdge(BoundaryAxis.VERTICAL, 0, 0, A, None)
    assert BoundaryObservation(0, (2, 2), (v, h)).edges == (h, v)
    with pytest.raises(ValueError):
        BoundaryObservation(0, (1, 1), (h,))
    with pytest.raises(ValueError):
        BoundaryObservation(0, (2, 2), (h, h))
    with pytest.raises(ValueError):
        ObservedEdge(BoundaryAxis.HORIZONTAL, 0, 0, A, A)
    with pytest.raises(ValueError):
        ObservedEdge(BoundaryAxis.HORIZONTAL, 0, 0, A, B, "owned")  # type: ignore[arg-type]
