"""Public analytic camera adversaries; no native or complete-raster enumeration."""

from __future__ import annotations

import ast
import json
import math
from dataclasses import replace
from fractions import Fraction as Q
from pathlib import Path
from typing import Any, cast

import pytest

from epsbench.diagnostics.corridor_aperture import VERSION, Box, Point, configuration
from epsbench.diagnostics.corridor_aperture_camera import (
    B32,
    B64,
    POLICY,
    camera_map,
    compiled,
    validate_camera,
)
from epsbench.diagnostics.corridor_aperture_capture import SourceBinding, _camera_record
from epsbench.diagnostics.corridor_aperture_reference import (
    domain_boxes,
    first_hit,
    sample_ray,
    target_cause,
)
from tests.test_corridor_aperture import CAM_ROT, domain


def changed(values: tuple[float, ...], slot: int, value: float) -> tuple[float, ...]:
    return (*values[:slot], value, *values[slot + 1 :])


def test_declared_v3_retains_camera_budget_configuration_matches_executable_policy() -> None:
    config = configuration()
    assert VERSION.endswith("-v3")
    assert config["camera_numerical_policy"] == POLICY
    assert Q(config["camera_compiled_component_allowance"]) == B64
    assert Q(config["camera_draw_component_allowance"]) == B32


def test_compiled_allowance_accepts_boundary_and_preserves_raw_equality() -> None:
    d = domain(0)
    p = (float(B64 * 2), 2.0, 1.0)
    r = changed(CAM_ROT, 1, float(B64))
    assert compiled(p, r, 2)["rotation_residual"] == str(B64)
    assert validate_camera(replace(d, camera_position=p, camera_rotation=r))
    source = SourceBinding(
        "a" * 40,
        "b" * 40,
        __import__(
            "epsbench.diagnostics.corridor_aperture", fromlist=["config_root"]
        ).config_root(),
    )
    record = json.loads(_camera_record(0, p, r, source, "mujoco:fake"))
    assert record["observed"]["position_hex"][0] == p[0].hex()
    assert record["component_equal"]["position"][0] is False
    assert record["numerical_camera"]["status"] == "NUMERICALLY_EQUIVALENT"
    with pytest.raises(ValueError, match="compiled position outside"):
        compiled((math.nextafter(p[0], math.inf), 2.0, 1.0), r, 2)
    with pytest.raises(ValueError, match="compiled rotation outside"):
        compiled(p, changed(r, 1, math.nextafter(float(B64), math.inf)), 2)


@pytest.mark.parametrize(
    "slot,value", [(0, -1.0), (0, 1.01), (1, 0.01), (0, float("nan")), (0, float("inf"))]
)
def test_wrong_reflected_scaled_sheared_or_nonfinite_compiled_basis_denied(
    slot: int, value: float
) -> None:
    with pytest.raises(ValueError):
        domain_boxes(replace(domain(0), camera_rotation=changed(CAM_ROT, slot, value)))


def test_tiny_nonorthogonality_is_admitted_without_orthogonalizing() -> None:
    rotation = changed(CAM_ROT, 0, 1.0 + float(B64))
    d = replace(domain(0), camera_rotation=rotation)
    domain_boxes(d)
    assert d.camera_rotation[0] == 1.0 + float(B64)


@pytest.mark.parametrize(
    "slot,value", [(0, -1.0), (0, 1.01), (4, 0.01), (3, 1e-20), (0, float("nan")), (0, 0.0)]
)
def test_wrong_draw_basis_or_affine_structure_denied(slot: int, value: float) -> None:
    d = domain(0)
    with pytest.raises(ValueError):
        domain_boxes(replace(d, modelview=changed(d.modelview, slot, value)))


def test_draw_allowance_boundary_and_downstream_disagreement_denied() -> None:
    d = domain(0)
    good = replace(d, modelview=changed(d.modelview, 4, float(B32)))
    domain_boxes(good)
    bad = replace(d, modelview=changed(d.modelview, 4, math.nextafter(float(B32), math.inf)))
    with pytest.raises(ValueError, match="draw nominal basis outside"):
        domain_boxes(bad)
    # Each operand is nominal-close; their mutual difference exceeds the same stage budget.
    c = replace(d.scene_cameras[0], forward=(float(B32), 1.0, 0.0))
    disagreement = replace(d, scene_cameras=(c, c), modelview=changed(d.modelview, 4, float(B32)))
    with pytest.raises(ValueError, match="draw/scene basis outside"):
        domain_boxes(disagreement)


@pytest.mark.parametrize(
    "field,value",
    [
        ("up", (0.0, 1.0, 0.0)),
        ("forward", (0.0, 1.0, 0.01)),
        ("pos", (0.0, 2.01, 1.0)),
        ("orthographic", True),
    ],
)
def test_scene_camera_fields_cannot_move_rejection_downstream(field: str, value: Any) -> None:
    d = domain(0)
    c = replace(d.scene_cameras[0], **{field: value})
    with pytest.raises(ValueError):
        domain_boxes(replace(d, scene_cameras=(c, c)))


def test_nominal_and_realised_ray_camera_depth_identity_and_clipping() -> None:
    d = domain(0)
    origin, direction = sample_ray(d, 47, 88)
    assert origin == (Q(0), Q(2), Q(1))
    assert direction == (Q(49, 128), Q(1), Q(1, 96) / Q.from_float(4 / 3))
    # A distinct rational affine map tests column-major extraction, translation and inverse.
    m = (0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, -3.0, -4.0, 1.0)
    actual = camera_map(m)
    origin, direction = actual.ray(Q(1, 4), Q(1, 8))
    s = Q(5)
    world = cast(Point, tuple(o + s * v for o, v in zip(origin, direction, strict=True)))
    assert actual.view(world) == (s / 4, s / 8, -s)
    target = Box("analytic", (-Q(20), -Q(20), -Q(2)), (Q(20), Q(20), Q(-1)))
    hit = first_hit(origin, direction, (target,), Q(1), Q(10))
    assert hit.status == "HIT" and hit.forward_depth == 5
    assert first_hit(origin, direction, (target,), Q(5), Q(10)).status == "CLIPPED"


def test_exact_general_ray_boundary_and_tie_are_not_tolerance_decisions() -> None:
    b = Box("box", (Q(0), Q(0), Q(0)), (Q(1), Q(1), Q(1)))
    origin = (Q(1, 2), Q(1, 2), Q(3))
    direction = (Q(0), Q(0), Q(-1))
    other = replace(b, name="other")
    assert first_hit(origin, direction, (b, other), Q(1), Q(5)).status == "TIE"
    assert first_hit((Q(0), Q(1, 2), Q(3)), direction, (b,), Q(1), Q(5)).status == "BOUNDARY"
    with pytest.raises(ValueError):
        first_hit(origin, (Q(0), Q(0), Q(0)), (b,), Q(1), Q(5))


@pytest.mark.parametrize("i", [0, 1, 2])
def test_realised_origin_and_corners_keep_strict_volume_clearances(i: int) -> None:
    d = domain(i)
    t = changed(d.modelview, 12, float(B32 / 4))
    real = replace(d, modelview=t)
    assert camera_map(t).origin[0] != 0
    assert sample_ray(real, 47, 88)[0] != sample_ray(d, 47, 88)[0]
    assert target_cause(real) == target_cause(d)
    # Moving an actual pier into an opening cannot be excused by camera allowance.
    pier = replace(d.draw_boxes[6], position=(2.0, 7.0, 1.0))
    with pytest.raises(ValueError):
        target_cause(replace(real, draw_boxes=(*d.draw_boxes[:6], pier, *d.draw_boxes[7:])))


def test_near_zero_ray_clearance_is_exact_unknown() -> None:
    b = Box("tiny", (Q(0), Q(0), Q(0)), (Q(1), Q(1), Q(1)))
    direction = (Q(0), Q(0), Q(-1))
    edge = first_hit((Q(0), Q(1, 2), Q(3)), direction, (b,), Q(1), Q(5))
    interior = first_hit((Q(1, 2**100), Q(1, 2), Q(3)), direction, (b,), Q(1), Q(5))
    assert edge.status == "BOUNDARY" and interior.status == "HIT"


@pytest.mark.parametrize(
    "old_purpose,old_root",
    [
        (
            "corridor_aperture_native_v1",
            "663156ea79a70422201c8edc7715f817903058b0192d5b60e31c6d78b94fc911",
        ),
        (
            "corridor_aperture_native_v2",
            "52041034d876cbfa70c3644ba6667c1049b9abfb1dae326bdad0d9cc9e4cb9e2",
        ),
    ],
)
def test_historical_roots_and_runtime_purposes_are_distinct(
    old_purpose: str, old_root: str
) -> None:
    from pydantic import ValidationError

    from epsbench.diagnostics import corridor_aperture_runtime as runtime

    facts = dict(
        source_head="a" * 40,
        source_tree="b" * 40,
        configuration_root=old_root,
        purpose=old_purpose,
        image="sha256:" + "d" * 64,
        manifest_sha256="e" * 64,
    )
    with pytest.raises(ValidationError):
        runtime.ApertureRuntimeBinding.model_validate(facts)
    assert runtime.PURPOSE == "corridor_aperture_native_v3"


def test_ci_routes_v3_branch_to_source_only_job_and_three_exclusions() -> None:
    import yaml

    workflows = yaml.safe_load((Path(__file__).parents[1] / ".github/workflows/ci.yml").read_text())
    branch = "codex/aperture-float-declarations-20261007"
    jobs = workflows["jobs"]
    for job in ("corridor-aperture-source", "quality", "qualify-wgl", "qualify-osmesa"):
        assert branch in jobs[job]["if"]
    tree = ast.parse(
        (
            Path(__file__).parents[1] / "src/epsbench/diagnostics/corridor_aperture_capture.py"
        ).read_text()
    )
    evaluate = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate_frame"
    )
    inversions = [
        n
        for n in ast.walk(evaluate)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "camera_map"
    ]
    assert len(inversions) == 1


def test_target_budget_analytic_arithmetic_before_any_raster_outcome() -> None:
    g = Q(4, 10**6) + Q(1, 10**12)
    E = 25 * B32
    d = Q(79, 20) - g
    x = Q(61, 20) + g
    z = Q(1, 10) + g
    assert (Q(101, 5) + 3 * g) * B32 < E
    assert max(
        64 * E * (d + x) / (d * (d - E)) + 64 * Q(1, 500000) * (x + E) / (d - E),
        64 * E * (d + z) / (d * (d - E)) + 48 * Q(1, 500000) * (z + E) / (d - E),
    ) < Q(1, 1024)


def test_direct_origin_cap_and_rejection_report_keep_measured_failure() -> None:
    from epsbench.diagnostics.corridor_aperture_camera import NumericalCameraError

    d = domain(0)
    pose = (float(2 * B64), 2.0, 1.0)
    m = changed(changed(d.modelview, 4, -float(B32)), 12, -float(B64))
    with pytest.raises(NumericalCameraError, match="draw nominal origin") as rejected:
        validate_camera(replace(d, camera_position=pose, modelview=m))
    checks = rejected.value.report["checks"]
    assert checks["draw/compiled origin"]["status"] == "ADMITTED"
    failure = checks["draw nominal origin"]
    assert failure["status"] == "REJECTED"
    assert Q(failure["residual"]) > Q(failure["budget"])
    assert checks["draw condition"]["status"] == "ADMITTED"


def test_success_report_covers_each_scene_origin_rigidity_and_condition_check() -> None:
    checks = validate_camera(domain(0))["checks"]
    required = (
        "compiled determinant",
        "compiled orthogonality",
        "draw determinant",
        "draw orthogonality",
        "draw condition",
        "draw nominal origin",
        "draw/compiled origin",
        "scene0/compiled eye",
        "scene1/compiled eye",
        "scene0 orthogonality",
        "scene1 orthogonality",
        "scene nominal origin",
        "scene mono/compiled origin",
        "draw/scene basis",
        "draw/scene translation",
    )
    for name in required:
        assert checks[name]["status"] == "ADMITTED"
        assert Q(checks[name]["residual"]) <= Q(checks[name]["budget"])


def test_camera_shapes_and_rational_ray_types_fail_closed() -> None:
    with pytest.raises(ValueError):
        compiled((0.0, 2.0), CAM_ROT, 2)
    with pytest.raises(ValueError):
        compiled((0.0, 2.0, 1.0), (1.0,) * 8, 2)
    with pytest.raises(ValueError):
        first_hit(cast(Point, (Q(0), Q(0))), (Q(0), Q(1), Q(0)), (), Q(1), Q(10))
