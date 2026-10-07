"""Public selected-ray and fake-provider checks; no full qualification enumeration."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import replace
from fractions import Fraction as Q
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from epsbench.diagnostics.corridor_aperture import (
    BOXES,
    NAMES,
    config_root,
    configuration,
    encode,
    opaque_mapping,
    scene_xml,
)
from epsbench.diagnostics.corridor_aperture_capture import (
    CAMERA_RECORD_LIMIT,
    EvidenceKind,
    Frame,
    NativeAdapter,
    ObservedState,
    Projection,
    RetentionFailure,
    SourceBinding,
    _camera_record,
    _retain_camera_record,
    collect,
    disposition,
    ecological_observation,
    evaluate_frame,
    privileged_frame,
    retain_three,
    retained_capture,
)
from epsbench.diagnostics.corridor_aperture_reference import (
    IDENTITY,
    CompiledBox,
    DrawDomain,
    SceneCamera,
    clip_planes,
    domain_boxes,
    first_hit,
    modelview,
    sample_ray,
    target_cause,
)
from epsbench.schema import ModalityPermissionSet

ALL = ModalityPermissionSet.all_modalities()
ECO = ModalityPermissionSet.ecological_only()
DENIED = ModalityPermissionSet(allowed=frozenset())
CAM_ROT = (1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0)


def domain(i: int) -> DrawDomain:
    q = (2, 4, 6)[i]
    boxes = tuple(
        CompiledBox(
            b.name,
            j,
            "box",
            tuple(float((a + z) / 2) for a, z in zip(b.lower, b.upper, strict=True)),
            tuple(float((z - a) / 2) for a, z in zip(b.lower, b.upper, strict=True)),
            IDENTITY,
        )
        for j, b in enumerate(BOXES)
    )
    n, f = 0.1, 300.0
    projection = (
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        4 / 3,
        0.0,
        0.0,
        0.0,
        0.0,
        n / (f - n),
        -1.0,
        0.0,
        0.0,
        f * n / (f - n),
        0.0,
    )
    camera = SceneCamera(
        (0.0, float(q), 1.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), n, f, 0.075, -0.075, 0.0, 0.0, 0
    )
    import math

    return DrawDomain(
        i,
        boxes,
        (0.0, float(q), 1.0),
        CAM_ROT,
        math.degrees(2 * math.atan(0.75)),
        projection,
        modelview(q),
        n,
        f,
        10.0,
        0.01,
        30.0,
        (camera, camera),
        boxes,
    )


def frame(i: int, visible: tuple[int, ...] = ()) -> Frame:
    raw: Any = np.full((96, 128), -1, dtype=np.int32)
    for col, value in enumerate(visible):
        raw[0, col] = value
    return Frame(
        EvidenceKind.SYNTHETIC,
        domain(i),
        raw,
        np.ones((96, 128), dtype=np.float32),
        np.zeros((96, 128, 3), dtype=np.uint8),
        np.zeros((96, 128), dtype=np.float32),
        encode({"fake": True}),
        encode({"synthetic_runtime": "public"}),
        config_root(),
        SourceBinding("0" * 40, "1" * 40, config_root()),
    )


def test_fixed_membership_xml_identity_and_thickness() -> None:
    xml = ET.fromstring(scene_xml())
    geoms = xml.findall("./worldbody/geom")
    assert tuple(g.attrib["name"] for g in geoms) == NAMES
    assert len(geoms) == 9
    assert all(all(float(x) > 0 for x in g.attrib["size"].split()) for g in geoms)
    assert xml.find("./visual/quality").attrib["offsamples"] == "0"  # type: ignore[union-attr]
    assert config_root() == config_root()
    assert json.loads(encode(configuration())) == configuration()
    assert BOXES[-1].upper[0] - BOXES[-1].lower[0] == Q(1, 20)


@pytest.mark.parametrize("i", [0, 1, 2])
def test_strict_whole_volume_clearance(i: int) -> None:
    d = domain(i)
    assert len(domain_boxes(d)) == 9
    assert target_cause(d) == (
        "COMPLETE_PIER_OCCLUSION_IN_FRAME" if i == 1 else "UNOBSTRUCTED_TARGET_IN_FRAME"
    )
    near, far = clip_planes(d)
    assert near < Q(79, 20) < Q(161, 20) < far


@pytest.mark.parametrize("i,row,column,face", [(0, 47, 88, "near_y"), (2, 47, 112, "near_x")])
def test_hand_rays_whole_box_not_same_face(i: int, row: int, column: int, face: str) -> None:
    d = domain(i)
    origin, direction = sample_ray(d, row, column)
    hit = first_hit(origin, direction, domain_boxes(d), *clip_planes(d))
    assert hit.status == "HIT" and hit.names == ("target",)
    assert hit.forward_depth is not None
    point = tuple(o + hit.forward_depth * v for o, v in zip(origin, direction, strict=True))
    target = domain_boxes(d)[-1]
    axis = 1 if face == "near_y" else 0
    assert point[axis] == target.lower[axis]
    assert all(target.lower[a] < point[a] < target.upper[a] for a in range(3) if a != axis)


def test_selected_hidden_target_ray_hits_pier_first() -> None:
    d = domain(1)
    direction = (Q(1, 2), Q(1), Q(0))  # Target centre at q4, wall crossing y8.
    actual = first_hit((Q(0), Q(4), Q(1)), direction, domain_boxes(d), *clip_planes(d))
    solo = first_hit((Q(0), Q(4), Q(1)), direction, (domain_boxes(d)[-1],), *clip_planes(d))
    assert actual.names == ("right_pier",) and actual.status == "HIT"
    assert solo.names == ("target",) and solo.status == "HIT"
    assert actual.forward_depth < solo.forward_depth  # type: ignore[operator]


@pytest.mark.parametrize(
    "field,value",
    [
        ("extent", 0.0),
        ("near", 100.0),
        ("far", float("nan")),
        ("model_znear", 0.02),
        ("camera_position", (0.0, 4.0, 1.0)),
        ("camera_rotation", IDENTITY),
        ("fovy", 60.0),
    ],
)
def test_malformed_calibration_denies(field: str, value: Any) -> None:
    with pytest.raises(ValueError):
        target_cause(replace(domain(0), **{field: value}))


def test_exact_camera_mismatch_is_retained_before_validator_rejects() -> None:
    source = SourceBinding("a" * 40, "b" * 40, config_root())
    observed = (0.0, 2.0000000000000004, 1.0)
    payload = _camera_record(0, observed, CAM_ROT, source, "mujoco:fake-3.12.0")
    retained: list[tuple[str, bytes]] = []
    _retain_camera_record(lambda name, data: retained.append((name, data)), payload, 0)
    assert len(payload) <= CAMERA_RECORD_LIMIT
    assert retained == [("camera-pose-0.json", payload)]
    record = json.loads(payload)
    assert record["observed"]["position_hex"][1] == observed[1].hex()
    assert record["component_equal"]["position"] == [True, False, True]
    with pytest.raises(ValueError, match="compiled fixed camera pose differs"):
        domain_boxes(replace(domain(0), camera_position=observed))


@pytest.mark.parametrize(
    "slot,value", [(0, 2.0), (5, 2.0), (8, 0.001), (10, 0.3), (14, 99.0), (15, 1.0)]
)
def test_projection_and_clipping_coefficients_bound(slot: int, value: float) -> None:
    d = domain(0)
    p = list(d.projection)
    p[slot] = value
    with pytest.raises(ValueError):
        domain_boxes(replace(d, projection=tuple(p)))


def test_actual_frustum_mismatch_denied() -> None:
    d = domain(0)
    c = replace(d.scene_cameras[0], frustum_far=5.0)
    with pytest.raises(ValueError):
        domain_boxes(replace(d, scene_cameras=(c, c)))


def test_foreign_geometry_caps_partial_cover_and_boundary_are_not_success() -> None:
    d = domain(0)
    b = replace(d.boxes[-1], half_size=(0.025, 0.05, 0.5))
    with pytest.raises(ValueError):
        domain_boxes(replace(d, boxes=(*d.boxes[:-1], b)))
    draw = replace(d.draw_boxes[6], position=(2.0, 7.0, 1.0))
    with pytest.raises(ValueError):
        target_cause(replace(d, draw_boxes=(*d.draw_boxes[:6], draw, *d.draw_boxes[7:])))
    t = BOXES[-1]
    origin = (Q(0), Q(2), Q(1))
    # Exact edge/cap intersection is explicit unknown, never selected face evidence.
    point = (t.lower[0], t.lower[1], t.lower[2])
    direction = tuple((v - o) / (point[1] - origin[1]) for v, o in zip(point, origin, strict=True))
    hit = first_hit(origin, direction, (t,), Q(1, 100), Q(30))  # type: ignore[arg-type]
    assert hit.status == "BOUNDARY"
    assert first_hit(origin, (Q(3, 8), Q(1), Q(0)), (t,), Q(9), Q(30)).status == "CLIPPED"


@pytest.mark.parametrize("bad", [-1, True, 2**31])
def test_raw_id_bounds_before_encoding(bad: int) -> None:
    with pytest.raises(ValueError):
        opaque_mapping(b"p" * 32, (bad, *range(1, 9)))


def test_nonce_remaps_roles_and_missing_randomness_not_allocated() -> None:
    a = opaque_mapping(b"a" * 32, tuple(range(9)))
    b = opaque_mapping(b"b" * 32, tuple(range(9)))
    assert {t for _, t in a}.isdisjoint(t for _, t in b)
    assert a == opaque_mapping(b"a" * 32, tuple(range(9)))
    with pytest.raises(ValueError):
        opaque_mapping(b"", tuple(range(9)))


def test_observed_projection_privilege_inventory_and_renaming() -> None:
    mapping = opaque_mapping(b"a" * 32, tuple(range(9)))
    with pytest.raises(PermissionError):
        Projection(mapping, ECO)
    p = Projection(mapping, ALL)
    s0 = p.observe(frame(0, (8, 0)), ALL)
    s1 = p.observe(frame(1, (0,)), ALL)
    s2 = p.observe(frame(2, (8, 0)), ALL)
    assert len(s0.observed_ever) == 2 and s1.observed_ever == s0.observed_ever == s2.observed_ever
    assert len(s1.observation.identities) == 1
    text = s2.canonical_bytes().decode()
    for forbidden in ("raw_id", "depth", "target", "camera", "geometry"):
        assert forbidden not in text
    assert len(json.loads(text)["inventory"]) == 2
    renamed = Projection(opaque_mapping(b"b" * 32, tuple(range(9))), ALL).observe(
        frame(0, (8, 0)), ALL
    )
    assert np.array_equal(renamed.observation.segmentation, s0.observation.segmentation)
    assert renamed.observed_ever != s0.observed_ever
    with pytest.raises(ValueError):
        p.observe(frame(2), ALL)
    with pytest.raises(PermissionError):
        Projection(mapping, ALL).observe(frame(0), ECO)


def test_denial_before_provider_access_and_future_denial() -> None:
    class Bomb:
        def frame(self, sequence_index: int) -> Frame:
            raise AssertionError("provider accessed")

        def observation(self, sequence_index: int) -> ObservedState:
            raise AssertionError("provider accessed")

    with pytest.raises(PermissionError):
        privileged_frame(Bomb(), ECO, 0)
    with pytest.raises(PermissionError):
        ecological_observation(Bomb(), DENIED, 0, 0)
    with pytest.raises(PermissionError):
        ecological_observation(Bomb(), ECO, 1, 0)
    with pytest.raises(PermissionError):
        NativeAdapter(ALL)
    with pytest.raises(PermissionError):
        NativeAdapter(ECO, enabled=True)
    with pytest.raises(PermissionError):
        collect()
    with pytest.raises(PermissionError):
        evaluate_frame(frame(0), ALL)  # Denies synthetic before SDK/full-raster audit.


def test_cross_frame_named_raw_membership_is_bound_not_only_id_set() -> None:
    p = Projection(opaque_mapping(b"a" * 32, tuple(range(9))), ALL)
    p.observe(frame(0, (8,)), ALL)
    d = domain(1)
    boxes = (replace(d.boxes[0], raw_id=8), *d.boxes[1:-1], replace(d.boxes[-1], raw_id=0))
    drawn = (
        replace(d.draw_boxes[0], raw_id=8),
        *d.draw_boxes[1:-1],
        replace(d.draw_boxes[-1], raw_id=0),
    )
    altered = replace(frame(1), domain=replace(d, boxes=boxes, draw_boxes=drawn))
    with pytest.raises(ValueError, match="cross-frame"):
        p.observe(altered, ALL)


def test_snapshots_and_operational_identity_separation() -> None:
    f = frame(0, (8,))
    with pytest.raises(ValueError):
        f.raw_labels[0, 0] = 0
    assert (
        replace(f, operational=encode({"runtime": "other"})).scientific_bytes()
        == f.scientific_bytes()
    )
    with pytest.raises(ValueError):
        replace(f, configuration_root="0" * 64)
    with pytest.raises(ValueError):
        replace(f, paired_state=b'{"fake": true}')


def test_complete_three_view_fake_retention_and_ecological_read() -> None:
    class Fake:
        def frame(self, sequence_index: int) -> Frame:
            return frame(sequence_index, (0,))

    originals: dict[str, bytes] = {}
    roots = retain_three(Fake(), ALL, originals.__setitem__)
    assert len(set(roots)) == 3 and len(originals) == 6
    for i in range(3):
        saved = json.loads(originals[f"view-{i}.json"])
        assert saved["kind"] == "SYNTHETIC_SOURCE_ONLY"
        assert saved["domain"]["sequence_index"] == i
        assert set(saved["arrays"]) == {"raw_labels", "depth", "native_rgb", "native_depth"}
    state = Projection(opaque_mapping(b"c" * 32, tuple(range(9))), ALL).observe(frame(0, (0,)), ALL)

    class Public:
        def observation(self, sequence_index: int) -> ObservedState:
            return state

    assert ecological_observation(Public(), ECO, 0, 0).canonical_bytes() == state.canonical_bytes()


def test_finite_adapter_retains_partial_and_both_faults() -> None:
    class Fake:
        def frame(self, sequence_index: int) -> Frame:
            if sequence_index == 1:
                raise ValueError("public fake initiating failure")
            return frame(sequence_index)

    saved: dict[str, bytes] = {}
    with pytest.raises(ValueError, match="initiating"):
        retain_three(Fake(), ALL, saved.__setitem__)
    assert set(saved) == {"view-0.json", "view-0-operational.json", "failure.json"}
    assert json.loads(saved["failure.json"])["completed"] == 1

    def bad_sink(name: str, payload: bytes) -> None:
        raise OSError("public fake retention failure")

    with pytest.raises(RuntimeError) as caught:
        retain_three(Fake(), ALL, bad_sink)
    assert isinstance(caught.value.__cause__, ExceptionGroup)
    assert len(caught.value.__cause__.exceptions) == 2


@pytest.mark.parametrize("unknown,other", [(1, 0), (0, 1), (1, 1)])
def test_valid_target_contradiction_precedes_unrelated_uncertainty(
    unknown: int, other: int
) -> None:
    facts = dict(target_mismatch=1, target_unknown=unknown, other_mismatch=other, target_support=0)
    assert disposition(facts, True) == "FAIL"
    facts["target_mismatch"] = 0
    assert disposition(facts, True) == "INCONCLUSIVE"
    facts["target_support"] = 1
    assert disposition(facts, False) == "FAIL"


def fake_pair() -> Any:
    f = frame(0)
    return SimpleNamespace(
        raw_geom_segmentation=f.raw_labels,
        depth=f.depth,
        native_id_rgb=f.native_rgb,
        native_depth_pre_metric=f.native_depth,
        stable_state={"fake": True},
        operational_state={"fake": True},
        near=0.1,
        far=300.0,
    )


def fake_progress(observer: Any, *, read_failure: bool = False) -> None:
    for label in (
        "draw_input",
        "draw_attempt",
        "draw_complete",
        "draw_output",
        "read_input",
        "read_attempt",
        "read_complete",
        "read_output",
    ):
        if label == "read_complete":
            if read_failure:
                raise ValueError("native read did not return")
            value: Any = (bytes(96 * 128 * 3), bytes(96 * 128 * 4))
        elif label.endswith("input") or label.endswith("output"):
            value = b'{"fake":true}'
        else:
            value = None
        observer(label, value)


@pytest.mark.parametrize("failure", ["read", "validation", "domain", "verify", "close"])
def test_current_view_progress_pair_and_close_failures_retained(failure: str) -> None:
    saved: dict[str, bytes] = {}

    class Renderer:
        closed = 0

        def close(self) -> None:
            self.closed += 1
            if failure == "close":
                raise OSError("close failed")

    def capture(observer: Any) -> Any:
        fake_progress(observer, read_failure=failure == "read")
        if failure == "validation":
            raise ValueError("read state validation failed")
        return fake_pair()

    def build(pair: Any) -> Frame:
        if failure in ("domain", "verify"):
            raise ValueError(failure + " failed")
        return frame(0)

    renderer = Renderer()
    with pytest.raises((ValueError, OSError)):
        retained_capture(renderer, capture, build, ALL, saved.__setitem__, 0, frame(0).source)
    assert renderer.closed == 1
    assert "view-0-native_failure.json" in saved
    assert ("view-0-read_complete.json" in saved) == (failure != "read")
    assert ("view-0-pair_complete.json" in saved) == (failure in ("domain", "verify", "close"))
    if failure != "read":
        raw = json.loads(saved["view-0-read_complete.json"])
        assert raw["orientation"] == "native_bottom_up" and raw["certified_frame"] is False
    if failure == "close":
        assert "view-0-verified_frame.json" in saved
        assert (
            json.loads(saved["view-0-native_failure.json"])["errors"][0]["role"] == "renderer_close"
        )


def test_progress_retention_failure_stops_capture_and_keeps_all_errors() -> None:
    calls: list[str] = []

    class Renderer:
        def close(self) -> None:
            raise OSError("cleanup failed")

    def sink(name: str, payload: bytes) -> None:
        calls.append(name)
        if name.endswith("read_complete.json") or name.endswith("native_failure.json"):
            raise OSError("retention failed")

    def capture(observer: Any) -> Any:
        fake_progress(observer)
        raise AssertionError("capture continued after retention failure")

    with pytest.raises(ExceptionGroup) as caught:
        retained_capture(Renderer(), capture, lambda p: frame(0), ALL, sink, 0, frame(0).source)
    assert len(caught.value.exceptions) == 3
    assert isinstance(caught.value.exceptions[0], RetentionFailure)
    assert isinstance(caught.value.exceptions[0].__cause__, OSError)
    assert "view-0-read_output.json" not in calls and "view-0-pair_complete.json" not in calls


def test_progress_sink_denied_before_fake_access_and_native_hook_static() -> None:
    import ast
    from pathlib import Path

    with pytest.raises(PermissionError):
        retained_capture(
            None, lambda p: None, lambda p: frame(0), ECO, lambda n, b: None, 0, frame(0).source
        )
    source = Path(__file__).parents[1] / "src/epsbench/diagnostics/corridor_aperture_capture.py"
    tree = ast.parse(source.read_text())
    assert any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "CanonicalPairedRenderer"
        and any(k.arg == "progress_observer" for k in n.keywords)
        for n in ast.walk(tree)
    )
    adapter = NativeAdapter(
        ALL, enabled=True, expected_source=frame(0).source, retain=lambda n, b: None
    )
    adapter._failed = True
    with pytest.raises(PermissionError):
        adapter.frame(0)  # Denied before Git or SDK import.


def test_initiating_close_and_failure_sink_errors_remain_distinct() -> None:
    saved: dict[str, bytes] = {}

    class Renderer:
        def close(self) -> None:
            raise OSError("cleanup failure")

    def capture(observer: Any) -> Any:
        fake_progress(observer)
        return fake_pair()

    def build(pair: Any) -> Frame:
        raise ValueError("initiating domain failure")

    def sink(name: str, payload: bytes) -> None:
        if name.endswith("native_failure.json"):
            saved[name] = payload
            raise OSError("failure record retention failure")
        saved[name] = payload

    with pytest.raises(ExceptionGroup) as caught:
        retained_capture(Renderer(), capture, build, ALL, sink, 0, frame(0).source)
    assert [type(e) for e in caught.value.exceptions] == [ValueError, OSError, RetentionFailure]
    errors = json.loads(saved["view-0-native_failure.json"])["errors"]
    assert [e["role"] for e in errors] == ["initiating", "renderer_close"]
    assert "view-0-read_complete.json" in saved and "view-0-pair_complete.json" in saved
