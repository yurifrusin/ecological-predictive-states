"""Bounded privileged slab instrumentation; no study geometry or collection entrypoint."""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction as Q

from epsbench.diagnostics.restricted_exact_raster import CALIBRATION, Box, Raster, box_hit
from epsbench.diagnostics.restricted_exact_raster import support_hit as floor_hit
from epsbench.schema import Modality, ModalityPermissionSet
from epsbench.utils.canonical import canonical_json_bytes, sha256_bytes


def access_check(access: ModalityPermissionSet) -> None:
    if (
        type(access) is not ModalityPermissionSet
        or type(access.allowed) is not frozenset
        or access.allowed != frozenset({Modality.PRIVILEGED_GENERATION_RECORDS})
        or any(type(m) is not Modality for m in access.allowed)
    ):
        raise PermissionError("exact typed privileged instrumentation access required")


def rational(value: Q) -> None:
    if (
        type(value) is not Q
        or max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > 64
    ):
        raise ValueError("bounded exact rational required")


def domain(lateral: Q, boxes: tuple[Box, ...], extent: tuple[Q, Q]) -> None:
    rational(lateral)
    if abs(lateral) > 8 or type(boxes) is not tuple or len(boxes) not in (2, 3):
        raise ValueError("bounded lateral and two/three boxes required")
    if type(extent) is not tuple or len(extent) != 2:
        raise ValueError("exact floor extent required")
    for value in extent:
        rational(value)
        if not 0 < value <= 8:
            raise ValueError("bounded positive floor extent required")
    for box in boxes:
        if type(box) is not Box:
            raise ValueError("exact producer Box required")
        Box(box.lower, box.upper)
        for value in (*box.lower, *box.upper):
            rational(value)
            if abs(value) > 8:
                raise ValueError("bounded box coordinates required")
        if box.lower[1] <= -3 or box.lower[2] <= 0:
            raise ValueError("boxes must lie ahead of camera and above floor")


def raster(
    access: ModalityPermissionSet,
    lateral: Q,
    boxes: tuple[Box, ...],
    extent: tuple[Q, Q],
    check: Callable[[], None],
) -> Raster:
    access_check(access)  # Before reading caller geometry or invoking callbacks.
    domain(lateral, boxes, extent)
    origin = CALIBRATION.origin(lateral)
    rows = []
    ties = []
    for row in range(32):
        check()
        labels = []
        for column in range(32):
            ray = CALIBRATION.ray(row, column)
            hits = (floor_hit(origin, ray, extent), *(box_hit(origin, ray, b) for b in boxes))
            finite = [(i + 1, t) for i, t in enumerate(hits) if t is not None]
            if not finite:
                labels.append(0)
                continue
            nearest = min(t for _, t in finite)
            winners = tuple(i for i, t in finite if t == nearest)
            labels.append(winners[0] if len(winners) == 1 else 0)
            if len(winners) > 1:
                ties.append((row, column, winners))
        rows.append(tuple(labels))
    return Raster(tuple(rows), tuple(ties))


def ancestry(access: ModalityPermissionSet, boxes: tuple[Box, ...], extent: tuple[Q, Q]) -> str:
    """Private physical descriptor binding, excluding trajectory/episode/token/run metadata."""
    access_check(access)
    domain(Q(0), boxes, extent)
    # Raw order is not physical ancestry. Sort geometric descriptors, not label identities.
    descriptors = sorted(tuple(str(v) for v in (*b.lower, *b.upper)) for b in boxes)
    return sha256_bytes(
        canonical_json_bytes({"boxes": descriptors, "floor": list(map(str, extent))})
    )
