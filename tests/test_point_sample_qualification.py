"""Public analytic fixtures; exact expectations precede instrument comparison."""

from dataclasses import replace
from fractions import Fraction as Q
from typing import Any, cast

import pytest

from epsbench.diagnostics import disclosure_producer as producer
from epsbench.diagnostics import disclosure_reference as reference
from epsbench.diagnostics import point_sample_qualification as point
from epsbench.diagnostics.occupancy_reference import Solid, face_hit, support_hit
from epsbench.diagnostics.restricted_exact_raster import CALIBRATION, Box, box_hit
from epsbench.diagnostics.restricted_exact_raster import support_hit as slab_floor
from epsbench.schema import Modality, ModalityPermissionSet

ACCESS = ModalityPermissionSet(allowed=frozenset({Modality.PRIVILEGED_GENERATION_RECORDS}))


def box(x0: Q, x1: Q, y0: Q = Q(-1), z0: Q = Q(1)) -> Box:
    return Box((x0, y0, z0), (x1, Q(-1, 2), Q(9, 8)))


FACE = box(Q(0), Q(1, 8))
EDGE = box(Q(1, 16), Q(1, 8))
CORNER = box(Q(1, 16), Q(1, 8), z0=Q(17, 16))
TANGENT = box(Q(0), Q(1, 16))
OFF = Box((Q(5), Q(0), Q(1, 4)), (Q(6), Q(1, 2), Q(1, 2)))
OTHER = Box((Q(-6), Q(0), Q(1, 4)), (Q(-5), Q(1, 2), Q(1, 2)))
NEAR = box(Q(-1, 8), Q(1, 8), y0=Q(-2))
CLOSE = box(Q(0), Q(1, 8), y0=Q(-1) + Q(1, 1024))
# At (15,16): d=(1/32,1,1/32), t=2 gives (1/16,-1,17/16).
# Tangency enters y at t=2 exactly when it exits x: closed interval [2,2].
# At (31,16): d=(1/32,1,-31/32), floor t=32/31, (x,y)=(1/31,-61/31).
CASES = (
    ("face", (FACE, OFF), 15, (Q(2), None), None, 2, False),
    ("edge", (EDGE, OFF), 15, (Q(2), None), None, 2, True),
    ("corner", (CORNER, OFF), 15, (Q(2), None), None, 2, True),
    ("tangent", (TANGENT, OFF), 15, (Q(2), None), None, 2, True),
    ("floor-interior", (OFF, OTHER), 31, (None, None), Q(32, 31), 1, False),
    ("floor-edge", (OFF, OTHER), 31, (None, None), Q(32, 31), 1, True),
    ("floor-corner", (OFF, OTHER), 31, (None, None), Q(32, 31), 1, True),
    ("no-hit", (OFF, OTHER), 15, (None, None), None, 0, False),
    ("face-tie", (FACE, FACE), 15, (Q(2), Q(2)), None, 0, False),
    ("edge-tie", (EDGE, TANGENT), 15, (Q(2), Q(2)), None, 0, True),
    ("hidden-tie", (NEAR, FACE, FACE), 15, (Q(1), Q(2), Q(2)), None, 2, False),
    ("close-unequal", (FACE, CLOSE), 15, (Q(2), Q(2) + Q(1, 1024)), None, 2, False),
)


def noop() -> None:
    pass


@pytest.mark.parametrize("name,boxes,row,distances,floor,winner,boundary", CASES)
def test_analytic_gold(
    name: str,
    boxes: tuple[Box, ...],
    row: int,
    distances: tuple[Q | None, ...],
    floor: Q | None,
    winner: int,
    boundary: bool,
) -> None:
    extent = (Q(2), Q(3))
    if name == "floor-edge":
        extent = (Q(1, 31), Q(3))
    if name == "floor-corner":
        extent = (Q(1, 31), Q(61, 31))
    origin, ray = CALIBRATION.origin(Q(0)), CALIBRATION.ray(row, 16)
    # Each algorithm is checked against independently hand-derived distances first.
    for b, expected in zip(boxes, distances, strict=True):
        assert box_hit(origin, ray, b) == expected
        assert face_hit(origin, ray, Solid(b.lower, b.upper))[0] == expected
    assert slab_floor(origin, ray, extent) == floor
    assert support_hit(origin, ray, extent)[0] == floor
    raw = producer.raster(ACCESS, Q(0), boxes, extent, noop)
    audit = reference.audit(
        ACCESS, Q(0), tuple(Solid(b.lower, b.upper) for b in boxes), extent, noop
    )
    assert raw.labels[row][16] == audit.labels[row][16] == winner
    assert ((row, 16) in audit.ties) == (name in ("face-tie", "edge-tie"))
    assert ((row, 16) in audit.boundaries) == boundary
    q = point.qualify(ACCESS, point.VERSION, Q(0), boxes, extent, raw, audit)
    assert q.policy == point.VERSION and len(q.input_sha256) == 64
    assert not q.contradiction
    assert q.qualified == (not raw.ties and not audit.ties)
    if name in ("edge", "corner", "tangent", "floor-edge", "floor-corner"):
        assert q.qualified and not audit.qualified
        assert q.reference is audit and q.produced is raw
        assert any(unit.cause == "UNKNOWN" for unit in audit.units)


def test_permutation_binding_contradiction_and_missing() -> None:
    boxes, extent = (FACE, OFF), (Q(2), Q(3))
    raw = producer.raster(ACCESS, Q(0), boxes, extent, noop)
    audit = reference.audit(
        ACCESS, Q(0), tuple(Solid(b.lower, b.upper) for b in boxes), extent, noop
    )
    q = point.qualify(ACCESS, point.VERSION, Q(0), boxes, extent, raw, audit)
    reversed_raw = producer.raster(ACCESS, Q(0), boxes[::-1], extent, noop)
    reversed_audit = reference.audit(
        ACCESS, Q(0), tuple(Solid(b.lower, b.upper) for b in boxes[::-1]), extent, noop
    )
    other = point.qualify(
        ACCESS, point.VERSION, Q(0), boxes[::-1], extent, reversed_raw, reversed_audit
    )
    assert other.input_sha256 != q.input_sha256
    remap = {0: 0, 1: 1, 2: 3, 3: 2}
    assert tuple(tuple(remap[v] for v in r) for r in raw.labels) == reversed_raw.labels
    assert q.qualified == other.qualified
    changed = replace(raw, labels=((0,) * 32,) * 32)
    assert not point.qualify(ACCESS, point.VERSION, Q(0), boxes, extent, changed, audit).qualified
    for missing in (None, replace(raw, labels=())):
        with pytest.raises(ValueError):
            point.qualify(ACCESS, point.VERSION, Q(0), boxes, extent, cast(Any, missing), audit)


def test_denial_policy_and_domain_before_evidence_reads() -> None:
    bomb = cast(Any, object())
    denied = ModalityPermissionSet(allowed=frozenset({Modality.SURFACE_REGIONS}))
    with pytest.raises(PermissionError):
        point.qualify(denied, bomb, bomb, bomb, bomb, bomb, bomb)
    with pytest.raises(ValueError, match="policy"):
        point.qualify(ACCESS, "legacy", bomb, bomb, bomb, bomb, bomb)
    with pytest.raises(ValueError, match="ahead"):
        bad = box(Q(0), Q(1), y0=Q(-3))
        point.qualify(ACCESS, point.VERSION, Q(0), (bad, OFF), (Q(2), Q(3)), bomb, bomb)


def test_malformed_pixel_diagnostics_rejected() -> None:
    boxes, extent = (FACE, OFF), (Q(2), Q(3))
    raw = producer.raster(ACCESS, Q(0), boxes, extent, noop)
    audit = reference.audit(
        ACCESS, Q(0), tuple(Solid(b.lower, b.upper) for b in boxes), extent, noop
    )
    for bad in (((True, 0),), ((32, 0),), ((0, 0), (0, 0)), ((0,),)):
        with pytest.raises(ValueError):
            point.qualify(
                ACCESS,
                point.VERSION,
                Q(0),
                boxes,
                extent,
                raw,
                replace(audit, boundaries=cast(Any, bad)),
            )
    with pytest.raises(ValueError):
        point.qualify(
            ACCESS, point.VERSION, Q(0), boxes, extent, replace(raw, ties=((0, 0, (2, 2)),)), audit
        )
