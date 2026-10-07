"""Public declarative representation and prevalidation retention adversaries."""

import ast
import json
import math
import platform
import struct
from dataclasses import replace
from fractions import Fraction as Q
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from epsbench.diagnostics import corridor_aperture_capture as capture
from epsbench.diagnostics.corridor_aperture import config_root
from epsbench.diagnostics.corridor_aperture_reference import (
    MODEL_ZNEAR,
    clipping_declarations,
    domain_boxes,
)
from tests.test_corridor_aperture import ALL, domain, fake_pair, fake_progress


def source() -> capture.SourceBinding:
    return capture.SourceBinding("a" * 40, "b" * 40, config_root())


def test_canonical_binary32_declarations_are_exact() -> None:
    d = domain(0)
    assert d.model_znear == float(MODEL_ZNEAR)
    assert d.model_znear != 0.01
    assert clipping_declarations(d)["status"] == "ADMITTED"
    domain_boxes(d)


@pytest.mark.parametrize("field", ["model_znear", "model_zfar"])
@pytest.mark.parametrize("toward", [-math.inf, math.inf])
def test_adjacent_binary32_declarations_rejected(field: str, toward: float) -> None:
    d = domain(0)
    bad = float(np.nextafter(np.float32(getattr(d, field)), np.float32(toward)))
    report = clipping_declarations(replace(d, **cast(dict[str, Any], {field: bad})))
    assert report["status"] == "REJECTED"
    check = report["checks"][field]
    assert check["observed"]["hex"] == bad.hex()
    assert check["residual"] != "0" and check["budget"] == "0"
    with pytest.raises(ValueError, match="clipping declaration"):
        domain_boxes(replace(d, **cast(dict[str, Any], {field: bad})))


@pytest.mark.parametrize("bad", [0.01, 0.0, -1.0, 30.0, math.nan, math.inf, 1, True, "0.01"])
def test_bad_declaration_is_not_rounded_into_agreement(bad: Any) -> None:
    assert clipping_declarations(replace(domain(0), model_znear=bad))["status"] == "REJECTED"


@pytest.mark.parametrize("precision", ["binary64", "Binary32", None, 32])
def test_unsupported_precision_fails_closed(precision: Any) -> None:
    d = replace(domain(0), model_clip_precision=cast(str, precision))
    assert clipping_declarations(d)["status"] == "REJECTED"
    with pytest.raises(ValueError, match="precision"):
        domain_boxes(d)


def test_compact_record_precedes_rejection_and_keeps_missing_operands() -> None:
    d = replace(domain(0), model_znear=0.01)
    saved: dict[str, bytes] = {}
    capture._retain_domain_record(saved.__setitem__, d, source(), {"geom_xpos": "float64"})
    with pytest.raises(ValueError, match="clipping declaration"):
        domain_boxes(d)
    raw = json.loads(saved["camera-numerical-0.json"])
    assert raw["camera"]["status"] == "NUMERICALLY_EQUIVALENT"
    assert raw["domain_status"] == "UNVALIDATED"
    assert raw["clipping_declarations"]["status"] == "REJECTED"
    assert (
        raw["clipping_declarations"]["checks"]["model_znear"]["observed"]["hex"]
        == d.model_znear.hex()
    )
    assert len(raw["compiled_domain"]["boxes"]) == 9
    assert raw["compiled_domain"]["fovy"]["value"] == d.fovy
    assert raw["source"]["head"] == source().head
    assert len(saved["camera-numerical-0.json"]) < 32768
    nonfinite = json.loads(
        capture._domain_numerical_record(replace(d, model_znear=math.nan), source(), {})
    )
    operand = nonfinite["clipping_declarations"]["checks"]["model_znear"]["observed"]
    assert operand == {"value": None, "hex": "nan", "type": "float"}


def test_record_cap_and_sink_failure_are_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    saved = {"earlier": b"preserved"}
    monkeypatch.setattr(capture, "CAMERA_RECORD_LIMIT", 1)
    with pytest.raises(capture.RetentionFailure):
        capture._retain_domain_record(saved.__setitem__, domain(0), source(), {})
    assert saved == {"earlier": b"preserved"}
    monkeypatch.setattr(capture, "CAMERA_RECORD_LIMIT", 32768)

    def broken(name: str, data: bytes) -> None:
        raise OSError("sink closed")

    with pytest.raises(capture.RetentionFailure) as error:
        capture._retain_domain_record(broken, domain(0), source(), {})
    assert isinstance(error.value.__cause__, OSError)


def test_failed_fake_lifecycle_retains_declarations_and_prior_pair() -> None:
    saved: dict[str, bytes] = {}
    d = replace(domain(0), model_znear=0.01)

    class Renderer:
        closed = False

        def close(self) -> None:
            self.closed = True

    renderer = Renderer()

    def paired(observer: Any) -> Any:
        fake_progress(observer)
        return fake_pair()

    def build(pair: Any) -> capture.Frame:
        capture._retain_domain_record(saved.__setitem__, d, source(), {})
        domain_boxes(d)
        raise AssertionError("invalid declaration escaped")

    with pytest.raises(ValueError, match="clipping declaration"):
        capture.retained_capture(renderer, paired, build, ALL, saved.__setitem__, 0, source())
    assert renderer.closed
    assert "view-0-pair_complete.json" in saved
    assert "camera-numerical-0.json" in saved
    assert "view-0-native_failure.json" in saved
    assert "view-0-verified_frame.json" not in saved
    tree = ast.parse(
        (
            Path(__file__).parents[1] / "src/epsbench/diagnostics/corridor_aperture_capture.py"
        ).read_text()
    )
    adapter = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NativeAdapter"
    )
    calls = [
        (n.func.id, n.lineno)
        for n in ast.walk(adapter)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    ]
    assert next(line for name, line in calls if name == "_retain_domain_record") < next(
        line for name, line in calls if name == "Frame"
    )


@pytest.mark.parametrize(
    "old_root",
    [
        "663156ea79a70422201c8edc7715f817903058b0192d5b60e31c6d78b94fc911",
        "52041034d876cbfa70c3644ba6667c1049b9abfb1dae326bdad0d9cc9e4cb9e2",
    ],
)
def test_old_config_root_rejects_even_with_v3_purpose(
    old_root: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from epsbench.diagnostics import corridor_aperture_runtime as runtime
    from epsbench.diagnostics.corridor_aperture import encode

    binding = runtime.ApertureRuntimeBinding(
        source_head="a" * 40,
        source_tree="b" * 40,
        configuration_root=old_root,
        purpose="corridor_aperture_native_v3",
        image="sha256:" + "d" * 64,
        manifest_sha256="e" * 64,
    )
    monkeypatch.setenv("EPS_APERTURE_RUNTIME", "docker_candidate_v1")
    monkeypatch.setenv("EPS_APERTURE_BINDING", encode(binding.model_dump()).decode())
    monkeypatch.setenv("EPS_APERTURE_IMAGE", binding.image)
    for name in (
        "WSL_INTEROP",
        "WSL_DISTRO_NAME",
        "EPS_A1_RUNTIME",
        "EPS_CAUSAL_RUNTIME",
        "EPS_PAIRED_APPEARANCE_RUNTIME",
        "EPS_RENDERER_DISCRIMINATOR_RUNTIME",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    assert runtime.aperture_rejection() == "aperture_configuration_or_image_mismatch"


def rounded(value: Q) -> Q:
    return Q.from_float(struct.unpack("<f", struct.pack("<f", float(value)))[0])


def changed_projection(d: Any, a: Q, b: Q) -> Any:
    p = list(d.projection)
    p[10], p[14] = float(a), float(b)
    return replace(d, projection=tuple(p))


@pytest.mark.parametrize("scale", [Q(1, 2), Q(1), Q(10), Q(100)])
@pytest.mark.parametrize("intermediate", [False, True])
def test_independent_public_rounding_families_within_derived_envelope(
    scale: Q, intermediate: bool
) -> None:
    from epsbench.diagnostics.corridor_aperture_reference import (
        depth_envelope,
        depth_projection_report,
        target_cause,
    )

    n, f = rounded(MODEL_ZNEAR * scale), rounded(30 * scale)
    d = domain(0)
    camera = replace(
        d.scene_cameras[0],
        frustum_near=float(n),
        frustum_far=float(f),
        frustum_top=float(rounded(3 * n / 4)),
        frustum_bottom=-float(rounded(3 * n / 4)),
    )
    d = replace(d, near=float(n), far=float(f), extent=float(scale), scene_cameras=(camera, camera))
    if intermediate:
        denominator = rounded(f - n)
        c = -rounded(rounded(f + n) / denominator)
        ordinary_b = -rounded(2 * rounded(f * n) / denominator)
        a, b = -(c + 1) / 2, -ordinary_b / 2
    else:
        a, b = rounded(n / (f - n)), rounded(f * n / (f - n))
    d = changed_projection(d, a, b)
    env = depth_envelope(d)
    assert abs(a - env.a) <= env.eps_a
    assert env.eps_b >= Q(1, 500000) * env.b
    assert env.near_guard >= Q(1, 500000) * env.near
    assert depth_projection_report(d)["status"] == "ADMITTED"
    assert target_cause(d) == "UNOBSTRUCTED_TARGET_IN_FRAME"


def test_amended_p10_boundary_and_just_over_budget_are_reported() -> None:
    from epsbench.diagnostics.corridor_aperture_reference import (
        depth_envelope,
        depth_projection_report,
    )

    d = domain(0)
    env = depth_envelope(d)
    # Retained values are binary64: choose the greatest value below the exact cap.
    upper = env.a + env.eps_a
    at = float(upper)
    if Q.from_float(at) > upper:
        at = math.nextafter(at, -math.inf)
    admitted = changed_projection(d, Q.from_float(at), env.b)
    assert depth_projection_report(admitted)["checks"]["p10"]["status"] == "ADMITTED"
    rejected = changed_projection(d, Q.from_float(math.nextafter(at, math.inf)), env.b)
    report = depth_projection_report(rejected)
    check = report["checks"]["p10"]
    assert check["status"] == "REJECTED"
    assert Q(check["residual"]) > Q(check["budget"])
    with pytest.raises(ValueError, match="projection"):
        domain_boxes(rejected)


@pytest.mark.parametrize(
    "near,far", [(2.0**-31, 1.0), (1.0, 2.0**31), (0.1, 300.0), (1.0, 3.0), (1.0, 2.0**24)]
)
def test_depth_operand_and_conditioning_domain_rejects(near: float, far: float) -> None:
    from epsbench.diagnostics.corridor_aperture_reference import depth_envelope

    d = domain(0)
    camera = replace(d.scene_cameras[0], frustum_near=near, frustum_far=far)
    with pytest.raises(ValueError):
        depth_envelope(replace(d, scene_cameras=(camera, camera)))


def test_eye_mismatch_and_positive_endpoint_guard() -> None:
    from epsbench.diagnostics import corridor_aperture_reference as reference

    d = domain(0)
    other = replace(d.scene_cameras[1], frustum_far=301.0)
    with pytest.raises(ValueError, match="identical eye"):
        reference.depth_envelope(replace(d, scene_cameras=(d.scene_cameras[0], other)))


def test_target_uncertainty_margin_is_strict_and_rejection_retained() -> None:
    from epsbench.diagnostics.corridor_aperture_reference import (
        depth_envelope,
        depth_projection_report,
    )

    d = domain(0)
    env = depth_envelope(d)
    assert env.delta_far > 0
    # Move only the fake retained target toward the far plane, leaving raw report useful.
    boxes = list(d.draw_boxes)
    boxes[-1] = replace(
        boxes[-1], position=(3.025, float(env.far + 2 - env.delta_far / 2) - 0.05, 1.0)
    )
    bad = replace(d, draw_boxes=tuple(boxes))
    report = depth_projection_report(bad)
    assert report["checks"]["target far margin"]["status"] == "REJECTED"
    retained = json.loads(capture._domain_numerical_record(bad, source(), {}))
    assert retained["depth_projection"]["checks"]["target far margin"]["status"] == "REJECTED"


def test_positive_endpoint_and_underflow_guard_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    from epsbench.diagnostics import corridor_aperture_reference as reference

    d = domain(0)
    env = reference.depth_envelope(d)
    c = (env.far + env.near) / (env.far - env.near)
    monkeypatch.setattr(reference, "ROUNDING_ETA", 2 * env.a / c)
    with pytest.raises(ValueError, match="positive propagated"):
        reference.depth_envelope(d)


@pytest.mark.parametrize("side", ["near", "far"])
def test_actual_target_guard_at_boundary_and_just_inside(
    side: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from epsbench.diagnostics import corridor_aperture_reference as reference

    d = domain(0)
    env = reference.depth_envelope(d)
    near_margin, far_margin = reference.target_depth_margins(d, *reference.clip_planes(d))
    margin = near_margin if side == "near" else far_margin
    field = "delta_near" if side == "near" else "delta_far"
    monkeypatch.setattr(
        reference,
        "depth_envelope",
        lambda domain: replace(env, **cast(dict[str, Any], {field: margin})),
    )
    with pytest.raises(ValueError, match="uncertainty clearance"):
        reference.target_cause(d)
    monkeypatch.setattr(
        reference,
        "depth_envelope",
        lambda domain: replace(env, **cast(dict[str, Any], {field: margin - Q(1, 10**12)})),
    )
    assert reference.target_cause(d) == "UNOBSTRUCTED_TARGET_IN_FRAME"


def test_new_configuration_binds_declarations_and_depth_policy() -> None:
    from epsbench.diagnostics.corridor_aperture import configuration
    from epsbench.diagnostics.corridor_aperture_reference import DEPTH_POLICY

    config = configuration()
    assert config["model_clipping_declaration_precision"] == "binary32"
    assert config["model_clipping_declarations"] == [str(MODEL_ZNEAR), "30"]
    assert config["depth_projection_policy"] == DEPTH_POLICY
    assert config["depth_operand_window"] == [str(Q(1, 2**30)), str(2**30)]
