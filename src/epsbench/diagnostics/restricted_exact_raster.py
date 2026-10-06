"""Privileged restricted producer: exact closed rational slabs, never reference labels."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction as Q

VERSION = "restricted-exact-slab-v1"
Point = tuple[Q, Q, Q]


def _point(value: Point) -> None:
    if type(value) is not tuple or len(value) != 3 or any(type(v) is not Q for v in value):
        raise ValueError("three exact rational coordinates required")


@dataclass(frozen=True)
class Box:
    lower: Point
    upper: Point

    def __post_init__(self) -> None:
        _point(self.lower)
        _point(self.upper)
        if any(a >= b for a, b in zip(self.lower, self.upper, strict=True)):
            raise ValueError("strict nonempty rational box required")


@dataclass(frozen=True)
class Calibration:
    width: int = 32
    height: int = 32
    forward: Q = Q(-3)
    elevation: Q = Q(1)
    up_y: Q = Q(0)
    fovy: Q = Q(90)

    def __post_init__(self) -> None:
        if (
            type(self.width) is not int
            or type(self.height) is not int
            or (self.width, self.height) != (32, 32)
            or any(type(v) is not Q for v in (self.forward, self.elevation, self.up_y, self.fovy))
            or (self.forward, self.elevation, self.up_y, self.fovy) != (Q(-3), Q(1), Q(0), Q(90))
        ):
            raise ValueError("only fixed untilted square 32x32 90-degree calibration supported")

    def origin(self, lateral: Q) -> Point:
        if type(lateral) is not Q:
            raise ValueError("exact rational lateral position required")
        return lateral, self.forward, self.elevation

    def ray(self, row: int, column: int) -> Point:
        if any(type(v) is not int or not 0 <= v < 32 for v in (row, column)):
            raise ValueError("bounded integer row and column required")
        return Q(2 * column + 1, 32) - 1, Q(1), 1 - Q(2 * row + 1, 32)


def _ray(origin: Point, ray: Point) -> None:
    _point(origin)
    _point(ray)
    if ray == (Q(0), Q(0), Q(0)):
        raise ValueError("nonzero rational ray required")


def box_hit(origin: Point, ray: Point, box: Box) -> Q | None:
    """Intersect closed slabs; use first strictly positive entry or inside-origin exit."""
    _ray(origin, ray)
    if type(box) is not Box:
        raise ValueError("exact producer box required")
    near: Q | None = None
    far: Q | None = None
    for axis in range(3):
        if ray[axis] == 0:
            if not box.lower[axis] <= origin[axis] <= box.upper[axis]:
                return None
            continue
        a = (box.lower[axis] - origin[axis]) / ray[axis]
        b = (box.upper[axis] - origin[axis]) / ray[axis]
        lo, hi = min(a, b), max(a, b)
        near = lo if near is None else max(near, lo)
        far = hi if far is None else min(far, hi)
        if near > far:
            return None
    # A nonzero ray and finite box always supply at least one finite slab.
    assert near is not None and far is not None
    hit = near if near > 0 else far
    return hit if hit > 0 else None


def support_hit(origin: Point, ray: Point, extent: tuple[Q, Q]) -> Q | None:
    _ray(origin, ray)
    if (
        type(extent) is not tuple
        or len(extent) != 2
        or any(type(v) is not Q or v <= 0 for v in extent)
    ):
        raise ValueError("positive rational support half extents required")
    if ray[2] == 0:
        return None
    hit = -origin[2] / ray[2]
    if hit <= 0:
        return None
    if any(abs(origin[i] + hit * ray[i]) > extent[i] for i in range(2)):
        return None
    return hit


@dataclass(frozen=True)
class Decision:
    label: int
    depth: Q | None
    ties: tuple[int, ...]


def nearest(origin: Point, ray: Point, boxes: tuple[Box, Box], extent: tuple[Q, Q]) -> Decision:
    if type(boxes) is not tuple or len(boxes) != 2:
        raise ValueError("exactly two producer boxes required")
    hits = (support_hit(origin, ray, extent), *(box_hit(origin, ray, b) for b in boxes))
    finite = [(i + 1, t) for i, t in enumerate(hits) if t is not None]
    if not finite:
        return Decision(0, None, ())
    depth = min(t for _, t in finite)
    winners = tuple(i for i, t in finite if t == depth)
    # A tie is retained as missing truth, never resolved by label ordering.
    return Decision(
        winners[0] if len(winners) == 1 else 0, depth, winners if len(winners) > 1 else ()
    )


@dataclass(frozen=True)
class Raster:
    labels: tuple[tuple[int, ...], ...]
    ties: tuple[tuple[int, int, tuple[int, ...]], ...]


CALIBRATION = Calibration()


def raster(
    lateral: Q,
    boxes: tuple[Box, Box],
    extent: tuple[Q, Q],
    check: Callable[[], None],
    calibration: Calibration = CALIBRATION,
) -> Raster:
    if type(calibration) is not Calibration:
        raise ValueError("supported exact calibration required")
    origin = calibration.origin(lateral)
    rows: list[tuple[int, ...]] = []
    ties = []
    for row in range(32):
        check()
        labels = []
        for column in range(32):
            decision = nearest(origin, calibration.ray(row, column), boxes, extent)
            labels.append(decision.label)
            if decision.ties:
                ties.append((row, column, decision.ties))
        rows.append(tuple(labels))
    return Raster(tuple(rows), tuple(ties))
