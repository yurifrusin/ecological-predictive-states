"""Fixed v2 camera admission and exact realised affine rays; privileged only."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction as Q
from typing import TYPE_CHECKING, Any, cast

from epsbench.diagnostics.corridor_aperture import POSES, Point

if TYPE_CHECKING:
    from epsbench.diagnostics.corridor_aperture_reference import DrawDomain

POLICY = "aperture-camera-numerical-v2"
B64 = 64 * Q(1, 2**52)
B32 = 4 * Q(1, 2**23) + B64
ROTATION = (1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0)
Matrix = tuple[Point, Point, Point]


def point(values: tuple[float, ...]) -> Point:
    if (
        type(values) is not tuple
        or len(values) != 3
        or any(type(v) is not float or not math.isfinite(v) for v in values)
    ):
        raise ValueError("finite immutable floating camera vector required")
    return cast(Point, tuple(Q.from_float(v) for v in values))


def dot(a: Point, b: Point) -> Q:
    return sum((x * y for x, y in zip(a, b, strict=True)), Q(0))


def multiply(a: Matrix, b: Point) -> Point:
    return cast(Point, tuple(dot(row, b) for row in a))


def transpose(a: Matrix) -> Matrix:
    return cast(Matrix, tuple(tuple(a[j][i] for j in range(3)) for i in range(3)))


def cross(a: Point, b: Point) -> Point:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def determinant(a: Matrix) -> Q:
    return dot(a[0], cross(a[1], a[2]))


def inverse(a: Matrix) -> Matrix:
    det = determinant(a)
    if det == 0:
        raise ValueError("singular actual camera matrix")
    cofactor = (cross(a[1], a[2]), cross(a[2], a[0]), cross(a[0], a[1]))
    return cast(Matrix, tuple(tuple(v / det for v in row) for row in transpose(cofactor)))


def norm(a: Matrix) -> Q:
    return max(sum(map(abs, row), Q(0)) for row in a)


class NumericalCameraError(ValueError):
    """A numerical rejection with the compact checks observed before it."""

    def __init__(self, reason: str, report: dict[str, Any]) -> None:
        self.report = report
        super().__init__(reason)


def measure(residual: Q, budget: Q, name: str, report: dict[str, Any] | None) -> Q:
    passed = residual <= budget
    if report is not None:
        report.setdefault("checks", {})[name] = {
            "residual": str(residual),
            "budget": str(budget),
            "status": "ADMITTED" if passed else "REJECTED",
        }
    if not passed:
        raise NumericalCameraError(name + " outside numerical camera allowance", report or {})
    return residual


def close(
    actual: tuple[Q, ...],
    expected: tuple[Q, ...],
    budget: Q,
    name: str,
    report: dict[str, Any] | None = None,
) -> Q:
    return measure(
        max(abs(x - y) for x, y in zip(actual, expected, strict=True)), budget, name, report
    )


def rigid(a: Matrix, budget: Q, name: str, report: dict[str, Any]) -> None:
    allowance = 6 * budget + 9 * budget * budget
    det = determinant(a)
    measure(abs(det - 1), allowance, name + " determinant", report)
    if det <= 0:
        raise NumericalCameraError("reflected or non-rigid camera matrix", report)
    columns = transpose(a)
    residual = max(
        abs(dot(x, y) - int(i == j)) for i, x in enumerate(columns) for j, y in enumerate(columns)
    )
    measure(residual, allowance, name + " orthogonality", report)


def rows(values: tuple[float, ...]) -> Matrix:
    if type(values) is not tuple or len(values) != 9:
        raise ValueError("complete immutable camera rotation required")
    return cast(Matrix, tuple(point(values[i : i + 3]) for i in (0, 3, 6)))


def flatten(a: Matrix) -> tuple[Q, ...]:
    return tuple(v for row in a for v in row)


def compiled(position: tuple[float, ...], rotation: tuple[float, ...], q: int) -> dict[str, Any]:
    report: dict[str, Any] = {"policy": POLICY, "checks": {}}
    p, r = point(position), rows(rotation)
    rp = close(p, (Q(0), Q(q), Q(1)), B64 * max(1, q), "compiled position", report)
    rr = close(flatten(r), tuple(map(Q.from_float, ROTATION)), B64, "compiled rotation", report)
    rigid(r, B64, "compiled", report)
    return report | {
        "policy": POLICY,
        "position_residual": str(rp),
        "rotation_residual": str(rr),
        "position_budget": str(B64 * max(1, q)),
        "rotation_budget": str(B64),
    }


@dataclass(frozen=True)
class CameraMap:
    linear: Matrix
    translation: Point
    inverse_linear: Matrix
    origin: Point

    def view(self, world: Point) -> Point:
        return cast(
            Point,
            tuple(
                x + y for x, y in zip(multiply(self.linear, world), self.translation, strict=True)
            ),
        )

    def ray(self, x: Q, y: Q) -> tuple[Point, Point]:
        return self.origin, multiply(self.inverse_linear, (x, y, Q(-1)))


def camera_map(modelview: tuple[float, ...], report: dict[str, Any] | None = None) -> CameraMap:
    if type(modelview) is not tuple or len(modelview) != 16:
        raise ValueError("complete retained camera affine matrix required")
    for value in modelview:
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("finite actual camera matrix required")
    if tuple(modelview[i] for i in (3, 7, 11, 15)) != (0.0, 0.0, 0.0, 1.0):
        raise ValueError("exact affine camera structure required")
    # OpenGL column-major storage; affine rows are [i,i+4,i+8].
    a = cast(Matrix, tuple(point(tuple(modelview[i + 4 * j] for j in range(3))) for i in range(3)))
    t = point(modelview[12:15])
    inv = inverse(a)
    measure(norm(a) * norm(inv), (1 + 3 * B32) / (1 - 3 * B32), "draw condition", report)
    origin = multiply(inv, cast(Point, tuple(-v for v in t)))
    return CameraMap(a, t, inv, origin)


def unit(v: Point) -> Point:
    length = math.sqrt(float(dot(v, v)))
    if not math.isfinite(length) or length <= 0:
        raise ValueError("degenerate scene camera axis")
    return cast(Point, tuple(Q.from_float(float(x) / length) for x in v))


def validate_camera(domain: DrawDomain) -> dict[str, Any]:
    q = POSES[domain.sequence_index]
    report = compiled(domain.camera_position, domain.camera_rotation, q)
    p = point(domain.camera_position)
    r = rows(domain.camera_rotation)
    actual = camera_map(domain.modelview, report)
    expected_a = transpose(r)
    expected_t = cast(Point, tuple(-v for v in multiply(expected_a, p)))
    nominal_a = rows((1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0))
    nominal_t = (Q(0), Q(-1), Q(q))
    for expected, name in ((nominal_a, "draw nominal basis"), (expected_a, "draw/compiled basis")):
        report[name] = str(close(flatten(actual.linear), flatten(expected), B32, name, report))
    for expected_translation, name in (
        (nominal_t, "draw nominal translation"),
        (expected_t, "draw/compiled translation"),
    ):
        report[name] = str(
            close(actual.translation, expected_translation, B32 * max(1, q), name, report)
        )
    rigid(actual.linear, B32, "draw", report)
    close(actual.origin, p, B32 * max(1, q), "draw/compiled origin", report)
    close(actual.origin, (Q(0), Q(q), Q(1)), B32 * max(1, q), "draw nominal origin", report)
    if type(domain.scene_cameras) is not tuple or len(domain.scene_cameras) != 2:
        raise ValueError("two actual scene cameras required")
    forwards, ups, positions = ([], [], [])
    for eye_index, cam in enumerate(domain.scene_cameras):
        from epsbench.diagnostics.corridor_aperture_reference import SceneCamera

        if (
            type(cam) is not SceneCamera
            or type(cam.orthographic) is not int
            or cam.orthographic != 0
        ):
            raise ValueError("typed perspective scene camera required")
        cp, forward, up = (point(cam.pos), point(cam.forward), point(cam.up))
        measure(abs(cp[0]), Q(1, 10), f"scene{eye_index} eye x", report)
        close(cp[1:], p[1:], B32 * max(1, q), f"scene{eye_index}/compiled eye", report)
        close(
            forward,
            cast(Point, tuple(-r[i][2] for i in range(3))),
            B32,
            f"scene{eye_index}/compiled forward",
            report,
        )
        close(
            up,
            cast(Point, tuple(r[i][1] for i in range(3))),
            B32,
            f"scene{eye_index}/compiled up",
            report,
        )
        close(forward, (Q(0), Q(1), Q(0)), B32, f"scene{eye_index} nominal forward", report)
        close(up, (Q(0), Q(0), Q(1)), B32, f"scene{eye_index} nominal up", report)
        bound = 6 * B32 + 9 * B32 * B32
        measure(
            max(abs(dot(forward, forward) - 1), abs(dot(up, up) - 1), abs(dot(forward, up))),
            bound,
            f"scene{eye_index} orthogonality",
            report,
        )
        forwards.append(forward)
        ups.append(up)
        positions.append(cp)

    def mean(vectors: list[Point]) -> Point:
        return cast(Point, tuple((vectors[0][i] + vectors[1][i]) / 2 for i in range(3)))

    eye = mean(positions)
    close(eye, p, B32 * max(1, q), "scene mono/compiled origin", report)
    close(eye, (Q(0), Q(q), Q(1)), B32 * max(1, q), "scene nominal origin", report)
    forward = unit(mean(forwards))
    right = unit(cross(forward, mean(ups)))
    up = unit(cross(right, forward))
    scene_a = (right, up, cast(Point, tuple(-v for v in forward)))
    scene_t = cast(Point, tuple(-v for v in multiply(scene_a, eye)))
    report["draw/scene basis"] = str(
        close(flatten(actual.linear), flatten(scene_a), B32, "draw/scene basis", report)
    )
    report["draw/scene translation"] = str(
        close(actual.translation, scene_t, B32 * max(1, q), "draw/scene translation", report)
    )
    report["draw_basis_budget"] = str(B32)
    report["draw_translation_budget"] = str(B32 * max(1, q))
    return report
