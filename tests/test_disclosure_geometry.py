"""Public hand-computed ray fixtures, unrelated to private study constructions."""

from __future__ import annotations

from fractions import Fraction as Q
from typing import Any, cast

import pytest

from epsbench.diagnostics import disclosure_producer as producer
from epsbench.diagnostics import disclosure_reference as reference
from epsbench.diagnostics.occupancy_reference import Solid
from epsbench.diagnostics.restricted_exact_raster import Box
from epsbench.schema import Modality, ModalityPermissionSet

ACCESS = ModalityPermissionSet(allowed=frozenset({Modality.PRIVILEGED_GENERATION_RECORDS}))
FLOOR = (Q(17, 16), Q(19, 16))


def check() -> None:
    pass


def compare(solids: tuple[Solid, ...]) -> reference.Audit:
    boxes = tuple(Box(s.lower, s.upper) for s in solids)
    result = producer.raster(ACCESS, Q(0), boxes, FLOOR, check)
    audit = reference.audit(ACCESS, Q(0), solids, FLOOR, check)
    assert result.labels == audit.labels
    assert [(r, c) for r, c, _ in result.ties] == list(audit.ties)
    return audit


def test_scalar_floor_fov_and_three_box_ray() -> None:
    outside = (
        Solid((Q(6), Q(-1), Q(1, 4)), (Q(7), Q(-1, 2), Q(7, 4))),
        Solid((Q(-7), Q(-1), Q(1, 4)), (Q(-6), Q(-1, 2), Q(7, 4))),
    )
    audit = compare(outside)
    # Ray j=1,k=-11 reaches floor at d=32/11; x=1/11,y=-1/11.
    assert audit.labels[21][16] == 1
    assert audit.labels[15][16] == 0
    assert audit.units[0].cause == "UNKNOWN"
    assert [s.cause for s in audit.units[1:]] == ["FOV_EXIT", "FOV_EXIT"]
    third = Solid((Q(3), Q(0), Q(1, 4)), (Q(4), Q(1), Q(7, 4)))
    three = compare((*outside, third))
    # j=31,k=-1 enters x=3 at d=96/31, before floor and inside other slabs.
    assert three.labels[16][31] == 4
    assert three.qualified  # Frustum x=1 is off pixel centres, not ambiguous raster truth.
    assert three.fov_boundary[2]
    assert three.units[3].cause == "UNKNOWN"


def test_all_unit_single_other_cover_and_union_unknown() -> None:
    front = Solid((Q(-1), Q(-1), Q(1, 4)), (Q(1), Q(-1, 2), Q(7, 4)))
    back = Solid((Q(-1, 4), Q(0), Q(3, 4)), (Q(1, 4), Q(1, 4), Q(5, 4)))
    for solids, target, covering in (((front, back), 2, 2), ((back, front), 1, 3)):
        audit = compare(solids)
        status = audit.units[target]
        assert status.visible == 0 and status.full_support is not None and status.full_support > 0
        assert status.cause == "COMPLETE_OCCLUSION"
        assert status.covering_label == covering
    left = Solid(front.lower, (Q(0), front.upper[1], front.upper[2]))
    right = Solid((Q(0), front.lower[1], front.lower[2]), front.upper)
    union = compare((left, right, back))
    assert union.units[3].visible == 0
    assert union.units[3].cause == "UNKNOWN"
    assert union.units[3].covering_label is None


def test_ties_retained_without_label_order_truth() -> None:
    solid = Solid((Q(-1), Q(-1), Q(1, 4)), (Q(1), Q(-1, 2), Q(7, 4)))
    audit = compare((solid, solid))
    assert audit.ties and not audit.qualified
    assert all(audit.labels[r][c] == 0 for r, c in audit.ties)
    assert all(u.cause == "UNKNOWN" for u in audit.units)


def test_denial_precedes_geometry_and_callback() -> None:
    denied = ModalityPermissionSet(allowed=frozenset({Modality.DEPTH}))
    for function in (producer.raster, reference.audit):
        with pytest.raises(PermissionError):
            function(
                denied,
                cast(Any, object()),
                cast(Any, object()),
                cast(Any, object()),
                lambda: pytest.fail("callback after denied permission"),
            )


def test_caps_exact_types_and_physical_descriptor_order() -> None:
    box = Box((Q(-1), Q(0), Q(1, 4)), (Q(1), Q(1), Q(2)))
    other = Box((Q(2), Q(0), Q(1, 4)), (Q(3), Q(1), Q(2)))
    assert producer.ancestry(ACCESS, (box, other), FLOOR) == producer.ancestry(
        ACCESS, (other, box), FLOOR
    )
    with pytest.raises(ValueError):
        producer.raster(ACCESS, Q(0), (box,), FLOOR, check)
    with pytest.raises(ValueError):
        producer.raster(ACCESS, Q(1, 2**65), (box, other), FLOOR, check)
    with pytest.raises(ValueError):
        reference.audit(ACCESS, cast(Any, 0), (Solid(box.lower, box.upper),) * 2, FLOOR, check)


def test_callback_exception_propagates() -> None:
    box = Box((Q(-1), Q(0), Q(1, 4)), (Q(1), Q(1), Q(2)))

    def stop() -> None:
        raise RuntimeError("cooperative stop")

    with pytest.raises(RuntimeError, match="cooperative stop"):
        producer.raster(ACCESS, Q(0), (box, box), FLOOR, stop)
