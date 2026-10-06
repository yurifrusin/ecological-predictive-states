"""Privileged scalar rational geometry; independent of producer and predictor."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction as Q
from itertools import product
from math import ceil, floor

Point = tuple[Q, Q, Q]
Point2 = tuple[Q, Q]
Check = Callable[[], None]


@dataclass(frozen=True)
class Solid:
    lower: Point
    upper: Point

    def __post_init__(self) -> None:
        if any(type(v) is not Q for v in (*self.lower, *self.upper)) or any(
            a >= b for a, b in zip(self.lower, self.upper, strict=True)
        ):
            raise ValueError("strict rational nonempty box required")


def face_hit(origin: Point, ray: Point, solid: Solid) -> tuple[Q | None, bool]:
    hits: dict[Q, bool] = {}
    for axis in range(3):
        if ray[axis] == 0:
            continue
        for plane in (solid.lower[axis], solid.upper[axis]):
            t = (plane - origin[axis]) / ray[axis]
            p = tuple(origin[i] + t * ray[i] for i in range(3))
            others = [i for i in range(3) if i != axis]
            if t > 0 and all(solid.lower[i] <= p[i] <= solid.upper[i] for i in others):
                boundary = any(p[i] in (solid.lower[i], solid.upper[i]) for i in others)
                hits[t] = hits.get(t, False) or boundary
    return (None, False) if not hits else (min(hits), hits[min(hits)])


def support_hit(origin: Point, ray: Point, extent: tuple[Q, Q]) -> tuple[Q | None, bool]:
    if ray[2] == 0:
        return None, False
    t = -origin[2] / ray[2]
    x, y = origin[0] + t * ray[0], origin[1] + t * ray[1]
    if t <= 0 or abs(x) > extent[0] or abs(y) > extent[1]:
        return None, False
    return t, abs(x) == extent[0] or abs(y) == extent[1]


def cross(a: Point2, b: Point2, c: Point2) -> Q:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def hull(points: tuple[Point2, ...]) -> tuple[Point2, ...]:
    ordered = sorted(set(points))
    halves: list[list[Point2]] = []
    for values in (ordered, list(reversed(ordered))):
        half: list[Point2] = []
        for p in values:
            while len(half) > 1 and cross(half[-2], half[-1], p) <= 0:
                half.pop()
            half.append(p)
        halves.append(half)
    return tuple(halves[0][:-1] + halves[1][:-1])


def in_hull(p: Point2, polygon: tuple[Point2, ...]) -> tuple[bool, bool]:
    values = [cross(a, polygon[(i + 1) % len(polygon)], p) for i, a in enumerate(polygon)]
    return bool(values) and all(v >= 0 for v in values), any(v == 0 for v in values)


@dataclass(frozen=True)
class Footprint:
    polygon: tuple[Point2, ...]
    depths: tuple[Q, Q] | None
    domain: bool

    def __post_init__(self) -> None:
        if (
            type(self.domain) is not bool
            or type(self.polygon) is not tuple
            or any(
                type(p) is not tuple or len(p) != 2 or any(type(v) is not Q for v in p)
                for p in self.polygon
            )
        ):
            raise ValueError("strict rational footprint required")
        if self.domain:
            if (
                len(self.polygon) < 3
                or type(self.depths) is not tuple
                or len(self.depths) != 2
                or any(type(v) is not Q for v in self.depths)
                or not 0 < self.depths[0] <= self.depths[1]
            ):
                raise ValueError("positive complete footprint required")
        elif self.polygon or self.depths is not None:
            raise ValueError("unknown domain must not fabricate extent")


def footprint(origin: Point, solid: Solid) -> Footprint:
    vertices = tuple(product(*zip(solid.lower, solid.upper, strict=True)))
    depths = tuple(v[1] - origin[1] for v in vertices)
    if min(depths) <= 0:
        return Footprint((), None, False)
    points = tuple(
        ((v[0] - origin[0]) / d, (v[2] - origin[2]) / d)
        for v, d in zip(vertices, depths, strict=True)
    )
    polygon = hull(points)
    return Footprint(polygon, (min(depths), max(depths)), len(polygon) >= 3)


def continuous_cover(origin: Point, target: Footprint, foreground: Solid) -> bool:
    if not target.domain or target.depths is None:
        return False
    d = foreground.lower[1] - origin[1]
    return 0 < d < target.depths[0] and all(
        foreground.lower[0] < origin[0] + p[0] * d < foreground.upper[0]
        and foreground.lower[2] < origin[2] + p[1] * d < foreground.upper[2]
        for p in target.polygon
    )


@dataclass(frozen=True)
class Audit:
    labels: tuple[tuple[int, ...], ...]
    ties: tuple[tuple[int, int], ...]
    boundaries: tuple[tuple[int, int], ...]
    # One target-only footprint count per box, including centres outside image.
    supports: tuple[tuple[int, int, bool], ...]
    footprints: tuple[Footprint, ...]


def audit(
    origin: Point, solids: tuple[Solid, Solid], extent: tuple[Q, Q], size: int, check: Check
) -> Audit:
    if type(size) is not int or not 1 <= size <= 32 or len(solids) != 2:
        raise ValueError("bounded square raster/two boxes required")
    footprints = tuple(footprint(origin, s) for s in solids)
    supports: list[tuple[int, int, bool]] = []
    for f in footprints:
        if not f.domain:
            supports.append((0, 0, True))
            continue
        # Prospective table is bounded; refuse an unbounded centre enumeration.
        bound = max(abs(v) for p in f.polygon for v in p)
        if bound > 8:
            raise ValueError("rational footprint work bound exceeded")
        full = clipped = 0
        boundary = False
        min_x, max_x = min(p[0] for p in f.polygon), max(p[0] for p in f.polygon)
        min_y, max_y = min(p[1] for p in f.polygon), max(p[1] for p in f.polygon)
        r0, r1 = ceil((size - 1 - size * max_y) / 2), floor((size - 1 - size * min_y) / 2)
        c0, c1 = ceil((size - 1 + size * min_x) / 2), floor((size - 1 + size * max_x) / 2)
        for r in range(r0, r1 + 1):
            check()
            for c in range(c0, c1 + 1):
                p = (Q(2 * c + 1 - size, size), Q(size - 2 * r - 1, size))
                inside, edge = in_hull(p, f.polygon)
                if inside:
                    full += 1
                    clipped += int(0 <= r < size and 0 <= c < size)
                    boundary |= edge
        supports.append((full, clipped, boundary))
    rows = []
    ties = []
    boundaries = []
    for r in range(size):
        check()
        row = []
        for c in range(size):
            ray = (Q(2 * c + 1 - size, size), Q(1), Q(size - 2 * r - 1, size))
            candidates = [
                support_hit(origin, ray, extent),
                *(face_hit(origin, ray, s) for s in solids),
            ]
            finite = [(t, i + 1, edge) for i, (t, edge) in enumerate(candidates) if t is not None]
            if not finite:
                row.append(0)
                continue
            nearest = min(t for t, _, _ in finite)
            winners = [(i, edge) for t, i, edge in finite if t == nearest]
            row.append(winners[0][0])
            if len(winners) > 1:
                ties.append((r, c))
            if any(edge for _, edge in winners):
                boundaries.append((r, c))
        rows.append(tuple(row))
    return Audit(tuple(rows), tuple(ties), tuple(boundaries), tuple(supports), footprints)


@dataclass(frozen=True)
class Status:
    cause: str
    flags: tuple[str, ...]
    full_centres: int
    clipped_centres: int
    visible_centres: int
    footprint: Footprint
    boundary: bool
    tie: bool

    def __post_init__(self) -> None:
        allowed = {
            "UNKNOWN_DOMAIN",
            "UNKNOWN",
            "COMPLETE_FRAME_LOSS",
            "PARTIAL_FRAME_LOSS",
            "COMPLETE_IN_FRUSTUM_OCCLUSION",
            "PARTIAL_OCCLUSION",
            "VISIBLE",
        }
        if (
            type(self.cause) is not str
            or type(self.flags) is not tuple
            or type(self.boundary) is not bool
            or type(self.tie) is not bool
            or type(self.footprint) is not Footprint
        ):
            raise ValueError("strict status record types required")
        if self.cause not in allowed or any(
            f not in {"FRAME_LOSS", "OCCLUSION", "MIXED_FRAME_LOSS_AND_OCCLUSION"}
            for f in self.flags
        ):
            raise ValueError("closed cause/flag vocabulary required")
        if any(
            type(v) is not int or v < 0
            for v in (self.full_centres, self.clipped_centres, self.visible_centres)
        ):
            raise ValueError("strict centre counts required")
        if self.clipped_centres > self.full_centres or (
            self.cause != "UNKNOWN_DOMAIN" and self.visible_centres > self.clipped_centres
        ):
            raise ValueError("status support counts inconsistent")


def status(
    result: Audit, raw_label: int, observed: bool, origin: Point, foreground: Solid
) -> Status:
    if type(raw_label) is not int or raw_label not in (1, 2, 3) or type(observed) is not bool:
        raise ValueError("strict observed raw status label required")
    visible = sum(row.count(raw_label) for row in result.labels)
    if raw_label == 1:
        return Status(
            "UNKNOWN_DOMAIN",
            (),
            0,
            0,
            visible,
            Footprint((), None, False),
            False,
            bool(result.ties),
        )
    f = result.footprints[raw_label - 2]
    full, clipped, boundary = result.supports[raw_label - 2]
    tie = bool(result.ties)
    flags: tuple[str, ...]
    if not f.domain:
        cause, flags = "UNKNOWN_DOMAIN", ()
    else:
        xs, ys = [p[0] for p in f.polygon], [p[1] for p in f.polygon]
        outside = max(xs) < -1 or min(xs) > 1 or max(ys) < -1 or min(ys) > 1
        inside = min(xs) > -1 and max(xs) < 1 and min(ys) > -1 and max(ys) < 1
        boundary |= any(v in (Q(-1), Q(1)) for v in (*xs, *ys))
        clipping = not inside
        hidden = clipped > visible
        flags = tuple(
            [
                *(["FRAME_LOSS"] if clipping else []),
                *(["OCCLUSION"] if hidden else []),
                *(["MIXED_FRAME_LOSS_AND_OCCLUSION"] if clipping and hidden else []),
            ]
        )
        if tie or boundary or not observed:
            cause = "UNKNOWN"
        elif outside:
            cause = "COMPLETE_FRAME_LOSS"
        elif clipping:
            cause = "PARTIAL_FRAME_LOSS"
        elif (
            visible == 0 and full > 0 and raw_label == 3 and continuous_cover(origin, f, foreground)
        ):
            cause = "COMPLETE_IN_FRUSTUM_OCCLUSION"
        elif visible > 0 and hidden:
            cause = "PARTIAL_OCCLUSION"
        elif (
            visible > 0
            and visible == clipped
            and (raw_label == 2 or disjoint(f.polygon, result.footprints[0].polygon))
        ):
            cause = "VISIBLE"
        else:
            cause = "UNKNOWN"
    return Status(cause, flags, full, clipped, visible, f, boundary, tie)


def disjoint(a: tuple[Point2, ...], b: tuple[Point2, ...]) -> bool:
    """Strict separating axis proof; no finite-sample unoccluded inference."""
    if not a or not b:
        return False
    for polygon in (a, b):
        for i, p in enumerate(polygon):
            q = polygon[(i + 1) % len(polygon)]
            axis = (p[1] - q[1], q[0] - p[0])
            av = [v[0] * axis[0] + v[1] * axis[1] for v in a]
            bv = [v[0] * axis[0] + v[1] * axis[1] for v in b]
            if max(av) < min(bv) or max(bv) < min(av):
                return True
    return False
