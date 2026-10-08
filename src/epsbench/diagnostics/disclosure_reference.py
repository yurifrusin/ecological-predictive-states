"""Independent privileged face/hull audit for bounded disclosure instrumentation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction as Q
from math import ceil, floor

from epsbench.diagnostics.occupancy_reference import (
    Footprint,
    Solid,
    continuous_cover,
    face_hit,
    footprint,
    in_hull,
    support_hit,
)
from epsbench.schema import Modality, ModalityPermissionSet

CAUSES = frozenset({"VISIBLE", "COMPLETE_OCCLUSION", "FOV_EXIT", "UNKNOWN"})


def access_check(access: ModalityPermissionSet) -> None:
    if (
        type(access) is not ModalityPermissionSet
        or type(access.allowed) is not frozenset
        or access.allowed != frozenset({Modality.PRIVILEGED_GENERATION_RECORDS})
        or any(type(m) is not Modality for m in access.allowed)
    ):
        raise PermissionError("exact typed privileged audit access required")


def _rational(value: Q) -> None:
    if (
        type(value) is not Q
        or max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > 64
    ):
        raise ValueError("bounded exact rational required")


def _domain(lateral: Q, solids: tuple[Solid, ...], extent: tuple[Q, Q]) -> None:
    _rational(lateral)
    if abs(lateral) > 8 or type(solids) is not tuple or len(solids) not in (2, 3):
        raise ValueError("bounded lateral and two/three solids required")
    if type(extent) is not tuple or len(extent) != 2:
        raise ValueError("exact floor extent required")
    for value in extent:
        _rational(value)
        if not 0 < value <= 8:
            raise ValueError("bounded positive floor extent required")
    for solid in solids:
        if type(solid) is not Solid:
            raise ValueError("exact reference Solid required")
        if type(solid.lower) is not tuple or type(solid.upper) is not tuple:
            raise ValueError("tuple reference coordinates required")
        if len(solid.lower) != 3 or len(solid.upper) != 3:
            raise ValueError("three reference coordinates required")
        Solid(solid.lower, solid.upper)
        for value in (*solid.lower, *solid.upper):
            _rational(value)
            if abs(value) > 8:
                raise ValueError("bounded solid coordinates required")
        if solid.lower[1] <= -3 or solid.lower[2] <= 0:
            raise ValueError("solids must lie ahead of camera and above floor")


@dataclass(frozen=True)
class UnitStatus:
    label: int
    visible: int
    full_support: int | None
    clipped_support: int | None
    cause: str
    covering_label: int | None


@dataclass(frozen=True)
class Audit:
    labels: tuple[tuple[int, ...], ...]
    ties: tuple[tuple[int, int], ...]
    boundaries: tuple[tuple[int, int], ...]
    footprints: tuple[Footprint, ...]
    footprint_boundary: tuple[bool, ...]
    fov_boundary: tuple[bool, ...]
    units: tuple[UnitStatus, ...]

    @property
    def qualified(self) -> bool:
        return not (self.ties or self.boundaries)


def audit(
    access: ModalityPermissionSet,
    lateral: Q,
    solids: tuple[Solid, ...],
    extent: tuple[Q, Q],
    check: Callable[[], None],
) -> Audit:
    access_check(access)  # Independent permission boundary; no producer import.
    _domain(lateral, solids, extent)
    origin = (lateral, Q(-3), Q(1))
    footprints = tuple(footprint(origin, s) for s in solids)
    supports = []
    for f in footprints:
        if not f.domain:
            raise ValueError("positive complete reference footprint required")
        if max(abs(v) for p in f.polygon for v in p) > 8:
            raise ValueError("bounded reference footprint required")
        xs, ys = [p[0] for p in f.polygon], [p[1] for p in f.polygon]
        r0, r1 = ceil((31 - 32 * max(ys)) / 2), floor((31 - 32 * min(ys)) / 2)
        c0, c1 = ceil((31 + 32 * min(xs)) / 2), floor((31 + 32 * max(xs)) / 2)
        full = clipped = 0
        boundary = False
        for r in range(r0, r1 + 1):
            check()
            for c in range(c0, c1 + 1):
                inside, edge = in_hull((Q(2 * c - 31, 32), Q(31 - 2 * r, 32)), f.polygon)
                if inside:
                    full += 1
                    clipped += int(0 <= r < 32 and 0 <= c < 32)
                    boundary |= edge
        supports.append((full, clipped, boundary))
    rows = []
    ties = []
    boundaries = []
    for r in range(32):
        check()
        row = []
        for c in range(32):
            ray = (Q(2 * c - 31, 32), Q(1), Q(31 - 2 * r, 32))
            hits = [support_hit(origin, ray, extent), *(face_hit(origin, ray, s) for s in solids)]
            finite = [(t, i + 1, edge) for i, (t, edge) in enumerate(hits) if t is not None]
            if not finite:
                row.append(0)
                continue
            distance = min(t for t, _, _ in finite)
            winners = [(label, edge) for t, label, edge in finite if t == distance]
            row.append(winners[0][0] if len(winners) == 1 else 0)
            if len(winners) > 1:
                ties.append((r, c))
            if any(edge for _, edge in winners):
                boundaries.append((r, c))
        rows.append(tuple(row))
    labels = tuple(rows)
    fov_boundary = tuple(any(v in (Q(-1), Q(1)) for p in f.polygon for v in p) for f in footprints)
    # Floor has no bounded forward footprint/cause certificate in this interface.
    units = [UnitStatus(1, sum(row.count(1) for row in labels), None, None, "UNKNOWN", None)]
    for i, f in enumerate(footprints):
        label = i + 2
        visible = sum(row.count(label) for row in labels)
        full, clipped, boundary = supports[i]
        xs, ys = [p[0] for p in f.polygon], [p[1] for p in f.polygon]
        outside = max(xs) < -1 or min(xs) > 1 or max(ys) < -1 or min(ys) > 1
        inside = min(xs) > -1 and max(xs) < 1 and min(ys) > -1 and max(ys) < 1
        covering = None
        cause = "UNKNOWN"
        if not (ties or boundaries or boundary or fov_boundary[i]):
            if visible:
                cause = "VISIBLE"  # Sampled presence, not continuous unoccluded extent.
            elif outside:
                cause = "FOV_EXIT"
            elif inside and full > 0:
                covering = next(
                    (
                        j + 2
                        for j, s in enumerate(solids)
                        if j != i and continuous_cover(origin, f, s)
                    ),
                    None,
                )
                if covering is not None:
                    cause = "COMPLETE_OCCLUSION"
        units.append(UnitStatus(label, visible, full, clipped, cause, covering))
    return Audit(
        labels,
        tuple(ties),
        tuple(boundaries),
        footprints,
        tuple(boundary for _, _, boundary in supports),
        fov_boundary,
        tuple(units),
    )
