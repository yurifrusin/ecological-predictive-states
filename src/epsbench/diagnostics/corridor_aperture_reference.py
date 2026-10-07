"""Independent face-plane/rectangle truth, restricted to one nine-box domain."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from itertools import product
from typing import Literal

from epsbench.diagnostics.corridor_aperture import BOXES, NAMES, Box, Point, index
from epsbench.diagnostics.corridor_aperture_camera import camera_map, validate_camera

COORDINATE_TOLERANCE = Q(1, 10**12)
PROJECTION_TOLERANCE = Q(1, 500000)
IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class CompiledBox:
    name: str
    raw_id: int
    kind: str
    position: tuple[float, ...]
    half_size: tuple[float, ...]
    rotation: tuple[float, ...]


@dataclass(frozen=True)
class SceneCamera:
    pos: tuple[float, float, float]
    forward: tuple[float, float, float]
    up: tuple[float, float, float]
    frustum_near: float
    frustum_far: float
    frustum_top: float
    frustum_bottom: float
    frustum_center: float
    frustum_width: float
    orthographic: int


@dataclass(frozen=True)
class DrawDomain:
    sequence_index: int
    boxes: tuple[CompiledBox, ...]
    camera_position: tuple[float, ...]
    camera_rotation: tuple[float, ...]
    fovy: float
    projection: tuple[float, ...]  # OpenGL column-major retained float32 matrix.
    modelview: tuple[float, ...]
    near: float
    far: float
    extent: float
    model_znear: float
    model_zfar: float
    scene_cameras: tuple[SceneCamera, ...]
    draw_boxes: tuple[CompiledBox, ...]


def rational(value: float) -> Q:
    if type(value) is not float:
        raise ValueError("finite floating captured fact required")
    try:
        return Q.from_float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError("finite floating captured fact required") from exc


def vector(value: tuple[float, ...], length: int) -> None:
    if type(value) is not tuple or len(value) != length:
        raise ValueError("exact immutable floating vector required")
    for element in value:
        rational(element)


def modelview(q: int) -> tuple[float, ...]:
    return (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, float(q), 1.0)


def domain_boxes(domain: DrawDomain) -> tuple[Box, ...]:
    """No arbitrary solids, orientation, calibration or callable acceptance."""
    if type(domain) is not DrawDomain:
        raise ValueError("typed fixed draw domain required")
    index(domain.sequence_index)
    if type(domain.boxes) is not tuple or len(domain.boxes) != 9:
        raise ValueError("nine compiled boxes required")
    if any(type(b) is not CompiledBox for b in domain.boxes) or (
        tuple(b.name for b in domain.boxes) != NAMES
    ):
        raise ValueError("exact compiled membership/order required")
    raw_ids = tuple(b.raw_id for b in domain.boxes)
    if any(type(v) is not int or not 0 <= v < 2**31 for v in raw_ids) or len(set(raw_ids)) != 9:
        raise ValueError("unique nonnegative compiled IDs required")
    result = []
    for actual, ideal in zip(domain.boxes, BOXES, strict=True):
        vector(actual.position, 3)
        vector(actual.half_size, 3)
        vector(actual.rotation, 9)
        if type(actual) is not CompiledBox or actual.kind != "box" or actual.rotation != IDENTITY:
            raise ValueError("axis-aligned compiled box required")
        if len(actual.position) != 3 or len(actual.half_size) != 3:
            raise ValueError("compiled coordinate shape differs")
        center = tuple(map(rational, actual.position))
        half = tuple(map(rational, actual.half_size))
        if any(v <= 0 for v in half):
            raise ValueError("positive actual thickness required")
        lower = tuple(c - h for c, h in zip(center, half, strict=True))
        upper = tuple(c + h for c, h in zip(center, half, strict=True))
        for a, b in zip((*lower, *upper), (*ideal.lower, *ideal.upper), strict=True):
            if abs(a - b) > COORDINATE_TOLERANCE:
                raise ValueError("compiled realization outside fixed bounded domain")
        result.append(Box(actual.name, lower, upper))  # type: ignore[arg-type]
    if type(domain.draw_boxes) is not tuple or len(domain.draw_boxes) != 9:
        raise ValueError("nine retained actual draw boxes required")
    drawn = []
    for actual, compiled in zip(domain.draw_boxes, domain.boxes, strict=True):
        if (
            type(actual) is not CompiledBox
            or type(actual.raw_id) is not int
            or actual.name != compiled.name
            or (
                actual.raw_id != compiled.raw_id
                or actual.kind != "box"
                or actual.rotation != IDENTITY
            )
        ):
            raise ValueError("actual drawn membership/orientation differs")
        vector(actual.position, 3)
        vector(actual.half_size, 3)
        vector(actual.rotation, 9)
        if len(actual.position) != 3 or len(actual.half_size) != 3:
            raise ValueError("actual drawn coordinate shape differs")
        for x, y in zip(
            (*actual.position, *actual.half_size),
            (*compiled.position, *compiled.half_size),
            strict=True,
        ):
            if abs(rational(x) - rational(y)) > PROJECTION_TOLERANCE:
                raise ValueError("draw realization outside fixed coordinate bound")
        lo = tuple(
            rational(c) - rational(h)
            for c, h in zip(actual.position, actual.half_size, strict=True)
        )
        hi = tuple(
            rational(c) + rational(h)
            for c, h in zip(actual.position, actual.half_size, strict=True)
        )
        if any(a >= b for a, b in zip(lo, hi, strict=True)):
            raise ValueError("drawn solid has nonpositive thickness")
        drawn.append(Box(actual.name, lo, hi))  # type: ignore[arg-type]
    vector(domain.camera_position, 3)
    vector(domain.camera_rotation, 9)
    vector(domain.modelview, 16)
    vector(domain.projection, 16)
    validate_camera(domain)
    if len(domain.projection) != 16:
        raise ValueError("complete retained projection required")
    p = tuple(map(rational, domain.projection))
    # No skew, offset, orthographic or stereo ray domain.
    for i in (1, 2, 3, 4, 6, 7, 8, 9, 12, 13, 15):
        if p[i] != 0:
            raise ValueError("unsupported draw ray projection")
    if (
        p[11] != -1
        or abs(p[0] - 1) > PROJECTION_TOLERANCE
        or (abs(p[5] - Q(4, 3)) > PROJECTION_TOLERANCE)
    ):
        raise ValueError("actual sample convention differs")
    # fovy is checked independently from matrix; both must agree with this domain.
    import math

    rational(domain.fovy)
    if abs(rational(math.tan(math.radians(domain.fovy) / 2)) - Q(3, 4)) > (COORDINATE_TOLERANCE):
        raise ValueError("compiled FOV differs")
    near, far, extent = map(rational, (domain.near, domain.far, domain.extent))
    if not 0 < near < far or extent <= 0:
        raise ValueError("invalid clipping/extent")
    if abs(rational(domain.model_znear) - Q(1, 100)) > COORDINATE_TOLERANCE or (
        abs(rational(domain.model_zfar) - 30) > COORDINATE_TOLERANCE
    ):
        raise ValueError("model clipping declaration differs")
    # SDK draw frusta can store float32 rounded near/far; bounded ratio comparison.
    if abs(near / extent - Q(1, 100)) > PROJECTION_TOLERANCE or (
        abs(far / extent - 30) > PROJECTION_TOLERANCE * 30
    ):
        raise ValueError("draw clipping disagrees with model extent")
    validate_frusta(domain)
    return tuple(drawn)


def clip_planes(domain: DrawDomain) -> tuple[Q, Q]:
    """Actual reverse-Z/zero-to-one projection planes; no model-only clipping proof."""
    a, b = rational(domain.projection[10]), rational(domain.projection[14])
    if a <= 0 or b <= 0:
        raise ValueError("unsupported reverse-Z projection depth coefficients")
    return b / (a + 1), b / a


def validate_frusta(domain: DrawDomain) -> None:
    if type(domain.scene_cameras) is not tuple or len(domain.scene_cameras) != 2:
        raise ValueError("two retained native scene cameras required")

    # Pinned SDK setView uses mjv_averageCamera for mono, then viewport aspect.
    def mean(name: str) -> Q:
        return sum((rational(getattr(c, name)) for c in domain.scene_cameras), Q(0)) / 2

    n, f = mean("frustum_near"), mean("frustum_far")
    top, bottom = mean("frustum_top"), mean("frustum_bottom")
    halfwidth = mean("frustum_width")
    if halfwidth == 0:
        halfwidth = Q(2, 3) * (top - bottom)
    if (
        not 0 < n < f
        or top <= 0
        or bottom != -top
        or mean("frustum_center") != 0
        or (halfwidth <= 0)
    ):
        raise ValueError("unsupported actual draw frustum")
    expected = {0: n / halfwidth, 5: 2 * n / (top - bottom), 10: n / (f - n), 14: f * n / (f - n)}
    for i, value in expected.items():
        if abs(rational(domain.projection[i]) - value) > PROJECTION_TOLERANCE * abs(value):
            raise ValueError("retained projection and actual draw frustum disagree")
    projected_near, projected_far = clip_planes(domain)
    for a, b in (
        (n, rational(domain.near)),
        (f, rational(domain.far)),
        (projected_near, n),
        (projected_far, f),
    ):
        if abs(a - b) > PROJECTION_TOLERANCE * abs(b):
            raise ValueError("actual frustum/projection/model clipping disagree")


@dataclass(frozen=True)
class Hit:
    status: Literal["HIT", "CLEAR", "TIE", "BOUNDARY", "CLIPPED"]
    names: tuple[str, ...]
    forward_depth: Q | None


def first_hit(origin: Point, direction: Point, boxes: tuple[Box, ...], near: Q, far: Q) -> Hit:
    """Enumerate all six face planes independently; no slab intersection reuse."""
    if (
        not (0 < near < far)
        or type(origin) is not tuple
        or type(direction) is not tuple
        or len(origin) != 3
        or len(direction) != 3
        or any(type(v) is not Q for v in (*origin, *direction))
        or not any(direction)
    ):
        raise ValueError("positive forward-depth ray/clipping required")
    candidates: list[tuple[Q, str, bool]] = []
    for b in boxes:
        for axis in range(3):
            if direction[axis] == 0:
                continue
            for plane in (b.lower[axis], b.upper[axis]):
                t = (plane - origin[axis]) / direction[axis]
                if t <= 0:
                    continue
                point = tuple(o + t * d for o, d in zip(origin, direction, strict=True))
                others = tuple(i for i in range(3) if i != axis)
                if all(b.lower[i] <= point[i] <= b.upper[i] for i in others):
                    edge = any(point[i] in (b.lower[i], b.upper[i]) for i in others)
                    candidates.append((t, b.name, edge))
    if not candidates:
        return Hit("CLEAR", (), None)
    t = min(v[0] for v in candidates)
    nearest = [v for v in candidates if v[0] == t]
    names = tuple(sorted({v[1] for v in nearest}))
    if not near < t < far:
        return Hit("CLIPPED", names, t)
    if len(names) != 1:
        return Hit("TIE", names, t)
    if any(v[2] for v in nearest):
        return Hit("BOUNDARY", names, t)
    return Hit("HIT", names, t)


def sample_ray(domain: DrawDomain, row: int, column: int) -> tuple[Point, Point]:
    if (
        type(row) is not int
        or type(column) is not int
        or not 0 <= row < 96
        or not 0 <= column < 128
    ):
        raise ValueError("fixed raster sample index required")
    p = tuple(map(rational, domain.projection))
    return camera_map(domain.modelview).ray(
        Q(2 * column + 1 - 128, 128) / p[0], Q(95 - 2 * row, 96) / p[5]
    )


def target_cause(domain: DrawDomain) -> str:
    """Whole-volume clearances using actual bounded compiled values, not ID zero."""
    boxes = domain_boxes(domain)
    by_name = {b.name: b for b in boxes}
    target = by_name["target"]
    walls = boxes[3:8]
    if len({(b.lower[0], b.upper[0]) for b in walls}) != 1:
        raise ValueError("right slab planes disagree")
    camera = camera_map(domain.modelview)
    origin = camera.origin
    ys, zs, horizontal, vertical, depths = [], [], [], [], []
    for x, y, z in product(*zip(target.lower, target.upper, strict=True)):
        view = camera.view((x, y, z))
        depth = -view[2]
        depths.append(depth)
        if depth <= 0:
            raise ValueError("target not wholly ahead")
        horizontal.append(view[0] / depth)
        vertical.append(view[1] / depth)
        for b in (walls[0].lower[0], walls[0].upper[0]):
            if x <= origin[0]:
                raise ValueError("positive target/slab denominator required")
            fraction = (b - origin[0]) / (x - origin[0])
            if not 0 < fraction < 1:
                raise ValueError("camera/slab/target ordering clearance missing")
            ys.append(origin[1] + fraction * (y - origin[1]))
            zs.append(origin[2] + fraction * (z - origin[2]))
    p = tuple(map(rational, domain.projection))
    if (
        max(abs(x * p[0]) for x in horizontal) >= 1
        or (max(abs(z * p[5]) for z in vertical) >= 1)
        or not clip_planes(domain)[0] < min(depths) <= max(depths) < clip_planes(domain)[1]
    ):
        raise ValueError("target clipping/frame clearance missing")
    # Convex half-space separation of every origin-to-target segment.
    if not by_name["floor"].upper[2] < min(origin[2], target.lower[2]) or (
        by_name["left"].upper[0] >= min(origin[0], target.lower[0])
        or max(origin[1], target.upper[1]) >= by_name["end"].lower[1]
    ):
        raise ValueError("competing blocker clearance missing")
    if not by_name["right_bottom"].upper[2] < min(zs) <= max(zs) < (by_name["right_top"].lower[2]):
        raise ValueError("target aperture height clearance missing")
    pier = by_name["right_pier"]
    if domain.sequence_index == 1:
        if not pier.lower[1] < min(ys) <= max(ys) < pier.upper[1]:
            raise ValueError("complete finite-thickness pier cover missing")
        return "COMPLETE_PIER_OCCLUSION_IN_FRAME"
    lo, hi = (
        (by_name["right_front"].upper[1], pier.lower[1])
        if domain.sequence_index == 0
        else (pier.upper[1], by_name["right_back"].lower[1])
    )
    if not lo < min(ys) <= max(ys) < hi:
        raise ValueError("full aperture clearance missing")
    return "UNOBSTRUCTED_TARGET_IN_FRAME"
