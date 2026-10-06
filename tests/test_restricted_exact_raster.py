"""Hand-derived synthetic rays only; no descriptor membership or reference imports."""

import ast
from dataclasses import replace
from fractions import Fraction as Q
from pathlib import Path
from typing import Any

import pytest

from epsbench.diagnostics.restricted_exact_raster import (
    Box,
    Calibration,
    box_hit,
    nearest,
    support_hit,
)

ZERO = (Q(0), Q(0), Q(0))
BOX = Box((Q(2), Q(2), Q(2)), (Q(3), Q(3), Q(3)))
EXTENT = (Q(1), Q(1))


@pytest.mark.parametrize(
    ("ray", "expected"),
    [
        ((Q(1), Q(1), Q(1)), Q(2)),  # corner
        ((Q(1), Q(1), Q(5, 4)), Q(2)),  # edge
        ((Q(1), Q(5, 4), Q(5, 4)), Q(2)),  # face
        ((Q(1), Q(3, 2), Q(1)), Q(2)),  # single-point tangency
        ((Q(1), Q(3, 2) - Q(1, 1000), Q(1)), Q(2)),
        ((Q(1), Q(3, 2) + Q(1, 1000), Q(1)), None),
        ((Q(-1), Q(-1), Q(-1)), None),  # behind
    ],
)
def test_closed_slabs(ray: tuple[Q, Q, Q], expected: Q | None) -> None:
    assert box_hit(ZERO, ray, BOX) == expected


def test_parallel_inside_and_zero_depth() -> None:
    assert box_hit((Q(2), Q(0), Q(5, 2)), (Q(0), Q(1), Q(0)), BOX) == 2
    assert box_hit((Q(2) - Q(1, 1000), Q(0), Q(5, 2)), (Q(0), Q(1), Q(0)), BOX) is None
    assert box_hit((Q(5, 2), Q(5, 2), Q(5, 2)), (Q(1), Q(0), Q(0)), BOX) == Q(1, 2)
    assert box_hit(BOX.lower, (Q(-1), Q(-1), Q(-1)), BOX) is None
    assert box_hit(BOX.lower, (Q(1), Q(1), Q(1)), BOX) == 1


@pytest.mark.parametrize(
    ("ray", "expected"),
    [
        ((Q(1), Q(1), Q(-1)), Q(1)),
        ((Q(1) - Q(1, 1000), Q(1), Q(-1)), Q(1)),
        ((Q(1) + Q(1, 1000), Q(1), Q(-1)), None),
        ((Q(0), Q(0), Q(0 + 1)), None),
        ((Q(1), Q(0), Q(0)), None),
    ],
)
def test_support_edges(ray: tuple[Q, Q, Q], expected: Q | None) -> None:
    assert support_hit((Q(0), Q(0), Q(1)), ray, EXTENT) == expected
    assert support_hit(ZERO, (Q(0), Q(0), Q(-1)), EXTENT) is None


def test_ordering_miss_and_retained_ties() -> None:
    distant = Box((Q(4), Q(4), Q(4)), (Q(5), Q(5), Q(5)))
    ray = (Q(1), Q(1), Q(1))
    assert nearest(ZERO, ray, (BOX, distant), EXTENT).label == 2
    assert nearest(ZERO, ray, (distant, BOX), EXTENT).label == 3
    tie = nearest(ZERO, ray, (BOX, BOX), EXTENT)
    assert (tie.label, tie.depth, tie.ties) == (0, Q(2), (2, 3))
    miss = nearest(ZERO, (Q(-1), Q(-1), Q(-1)), (BOX, distant), EXTENT)
    assert (miss.label, miss.depth, miss.ties) == (0, None, ())
    floor_box = Box((Q(-1), Q(-1), Q(-1)), (Q(1), Q(1), Q(0)))
    floor_tie = nearest((Q(0), Q(0), Q(1)), (Q(0), Q(0), Q(-1)), (floor_box, BOX), EXTENT)
    assert (floor_tie.label, floor_tie.depth, floor_tie.ties) == (0, Q(1), (1, 2))


def test_camera_orientation_translation_and_reflection() -> None:
    camera = Calibration()
    assert camera.ray(0, 0) == (Q(-31, 32), Q(1), Q(31, 32))
    assert camera.ray(31, 31) == (Q(31, 32), Q(1), Q(-31, 32))
    assert camera.origin(Q(17, 13)) == (Q(17, 13), Q(-3), Q(1))
    ray = (Q(1), Q(1), Q(1))
    shifted = Box((Q(9), Q(9), Q(9)), (Q(10), Q(10), Q(10)))
    assert box_hit((Q(7), Q(7), Q(7)), ray, shifted) == 2
    reflected = Box((Q(-3), Q(2), Q(2)), (Q(-2), Q(3), Q(3)))
    assert box_hit(ZERO, (Q(-1), Q(1), Q(1)), reflected) == 2


@pytest.mark.parametrize(
    "change", [{"width": 16}, {"width": True}, {"up_y": Q(1)}, {"fovy": Q(60)}, {"forward": -3.0}]
)
def test_unsupported_calibration(change: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        replace(Calibration(), **change)


def test_malformed_inputs() -> None:
    with pytest.raises(ValueError):
        Box((Q(0), Q(0), Q(0)), (Q(0), Q(1), Q(1)))
    with pytest.raises(ValueError):
        Box((0, Q(0), Q(0)), (Q(1), Q(1), Q(1)))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        box_hit(ZERO, ZERO, BOX)
    with pytest.raises(ValueError):
        support_hit(ZERO, (Q(1), Q(1), Q(1)), (Q(0), Q(1)))
    with pytest.raises(ValueError):
        Calibration().ray(-1, 0)
    with pytest.raises(ValueError):
        Calibration().origin(0.0)  # type: ignore[arg-type]


def test_source_independence_recipe_and_failure_retention() -> None:
    root = Path(__file__).resolve().parents[1]
    kernel = (root / "src/epsbench/diagnostics/restricted_exact_raster.py").read_text(
        encoding="utf-8-sig"
    )
    imports = [n.module for n in ast.walk(ast.parse(kernel)) if isinstance(n, ast.ImportFrom)]
    assert all(n is None or not n.startswith("epsbench") for n in imports)
    producer = (root / "src/epsbench/diagnostics/restricted_learning_producer.py").read_text(
        encoding="utf-8"
    )
    assert '"src/epsbench/diagnostics/restricted_exact_raster.py"' in producer
    assert '"version": RASTER_VERSION' in producer
    assert producer.index('"ties": produced.ties') < producer.index("result = audit(")
    assert "or result.ties or produced.ties" in producer
    assert "scene.frame" not in producer and "causal_history_fixture import" not in producer


def test_symbolic_raster_loop_without_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    import epsbench.diagnostics.restricted_exact_raster as kernel

    calls: list[tuple[Q, Q, Q]] = []
    checks: list[None] = []

    def forbidden(*args: Any) -> Any:
        raise AssertionError("physical intersections forbidden in symbolic loop")

    def fake(origin: tuple[Q, Q, Q], ray: tuple[Q, Q, Q], *args: Any) -> kernel.Decision:
        assert origin == (Q(17, 13), Q(-3), Q(1))
        ordinal = len(calls)
        calls.append(ray)
        if ordinal == 34:
            return kernel.Decision(0, Q(1), (2, 3))
        return kernel.Decision(ordinal % 3 + 1, Q(1), ())

    monkeypatch.setattr(kernel, "nearest", fake)
    monkeypatch.setattr(kernel, "box_hit", forbidden)
    monkeypatch.setattr(kernel, "support_hit", forbidden)
    result = kernel.raster(Q(17, 13), (BOX, BOX), EXTENT, lambda: checks.append(None))
    assert len(checks) == 32 and len(calls) == 1024
    assert calls[:2] == [(Q(-31, 32), Q(1), Q(31, 32)), (Q(-29, 32), Q(1), Q(31, 32))]
    assert calls[32] == (Q(-31, 32), Q(1), Q(29, 32))
    assert result.ties == ((1, 2, (2, 3)),)
    assert len(result.labels) == 32 and all(len(row) == 32 for row in result.labels)
    assert result.labels[1] == tuple(0 if c == 2 else (32 + c) % 3 + 1 for c in range(32))
