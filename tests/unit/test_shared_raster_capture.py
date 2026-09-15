from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from epsbench.diagnostics import shared_raster_capture as subject


def rgb_for(segids: np.ndarray) -> np.ndarray:
    ids = np.asarray(segids, dtype=np.uint32)
    return np.stack((ids & 255, (ids >> 8) & 255, (ids >> 16) & 255), axis=-1).astype(np.uint8)


def scene_map() -> list[dict[str, int]]:
    return [
        {"segid_plus_one": 1, "objid": 7, "objtype": 5},
        {"segid_plus_one": 258, "objid": 9, "objtype": 5},
    ]


def fake_native_state(_renderer: object) -> dict[str, object]:
    return {
        "rect": [0, 0, 2, 2],
        "scene_flags": [1, 1],
        "ngeom": 1,
        "scene_map": scene_map()[:1],
        "near": 0.01,
        "far": 50.0,
        "extent": 1.0,
        "readPixelFormat": 6407,
        "readDepthMap": 1,
        "read_framebuffer_binding": 5,
        "read_buffer": 1029,
        "pack_alignment": 1,
        "pack_row_length": 0,
        "pack_skip_rows": 0,
        "pack_skip_pixels": 0,
        "pixel_pack_buffer_binding": 0,
        "clip_origin": 36001,
        "clip_depth_mode": 37727,
        "framewidth": 0.0,
        "stereo": 0,
        "stereo_none": 0,
        "rnd_depth": False,
        "segment_enabled": True,
        "idcolor_enabled": True,
        "gl_rgb": 6407,
        "depth_zerofar": 1,
        "actual_current_context": 77,
        "expected_current_context": 77,
        "mjr_currentBuffer": 1,
        "framebuffer_offscreen": 1,
        "offSamples": 0,
        "offFBO": 5,
        "offFBO_r": 0,
        "offWidth": 2,
        "offHeight": 2,
        "offscreen_attachments": {"offFBO": {"object_name": 5}},
        "query_bindings_restored": True,
        "draw_framebuffer_binding": 5,
        "draw_buffer": 1029,
        "scene_cameras": [
            {
                "pos": [0.0, 0.0, 0.0],
                "forward": [0.0, 0.0, -1.0],
                "up": [0.0, 1.0, 0.0],
                "frustum_near": 0.01,
                "frustum_far": 50.0,
                "frustum_top": 1.0,
                "frustum_bottom": -1.0,
                "frustum_center": 0.0,
                "frustum_width": 1.0,
            }
        ],
        "projection_matrix_float32": [0.0] * 16,
        "modelview_matrix_float32": [0.0] * 16,
    }


def fake_context() -> dict[str, object]:
    return {
        "actual_offsamples": 0,
        "sample_buffers": 0,
        "samples": 0,
        "color_storage_dimensions": [2, 2],
        "depth_storage_dimensions": [2, 2],
        "attachment_format": "RGB8",
        "attachment_component_type": "UNSIGNED_NORMALIZED",
        "read_buffer": 1029,
        "osmesa_context_identity": 77,
        "mjr_off_width": 2,
        "mjr_off_height": 2,
        "offscreen_attachments": {"offFBO": {"object_name": 5}},
    }


def test_plan_is_exact_finite_contract() -> None:
    value = subject.plan()
    assert [a["name"] for a in value["attempts"]] == [a.name for a in subject.fixed_attempts()]
    assert value["limits"] == {
        "batch_attempts": 8,
        "contexts": 32,
        "ordinary_pose_endpoints": 64,
        "ordinary_modality_renders": 192,
        "counterfactual_segmentation_renders": 32,
        "native_render_calls": 224,
        "native_readbacks": 224,
        "paired_readbacks": 64,
    }
    assert value["review_profile"] == "DUAL_REVIEW"
    assert value["phase_gate_effect"] == "NONE"


def test_decode_little_channel_order_and_background() -> None:
    rgb = rgb_for(np.array([[0, 1], [258, 0]], dtype=np.uint32))
    segid, objid, objtype = subject.decode_id_colors(rgb, scene_map())
    assert segid.tolist() == [[0, 1], [258, 0]]
    assert objid.tolist() == [[-1, 7], [9, -1]]
    assert objtype.tolist() == [[-1, 5], [5, -1]]


@pytest.mark.parametrize(
    "mapping",
    [
        [{"segid_plus_one": 0, "objid": 1, "objtype": 1}],
        [{"segid_plus_one": 1, "objid": -1, "objtype": 1}],
        [
            {"segid_plus_one": 1, "objid": 1, "objtype": 1},
            {"segid_plus_one": 1, "objid": 2, "objtype": 1},
        ],
        [{"segid_plus_one": 0x1000000, "objid": 1, "objtype": 1}],
    ],
)
def test_decode_rejects_invalid_maps(mapping: list[dict[str, int]]) -> None:
    with pytest.raises(subject.SharedRasterFailure, match="invalid, duplicate, or out-of-range"):
        subject.decode_id_colors(np.zeros((1, 1, 3), dtype=np.uint8), mapping)


def test_decode_rejects_unmapped_nonzero_and_bad_buffer() -> None:
    with pytest.raises(subject.SharedRasterFailure, match="unmapped"):
        subject.decode_id_colors(rgb_for(np.array([[2]], dtype=np.uint32)), scene_map())
    with pytest.raises(subject.SharedRasterFailure, match="shape/dtype/contiguity"):
        subject.decode_id_colors(np.zeros((2, 2, 3), dtype=np.float32), scene_map())
    view = np.zeros((2, 4, 3), dtype=np.uint8)[:, ::2]
    with pytest.raises(subject.SharedRasterFailure, match="contiguity"):
        subject.decode_id_colors(view, scene_map())


def test_depth_conversion_matches_sdk_precision_order() -> None:
    raw = np.ascontiguousarray(np.array([[0.2, 0.7]], dtype=np.float32))
    c, d = subject.depth_coefficients(0.01, 50.0)
    expected = (d / (raw.astype(np.float64) + c)).astype(np.float32)
    assert np.array_equal(subject.convert_native_depth(raw, 0.01, 50.0), expected)


def test_depth_conversion_rejects_invalid_projection_and_buffer() -> None:
    with pytest.raises(subject.SharedRasterFailure, match="near/far"):
        subject.convert_native_depth(np.zeros((1, 1), dtype=np.float32), 2.0, 1.0)
    with pytest.raises(subject.SharedRasterFailure, match="rank"):
        subject.convert_native_depth(np.zeros((1, 1, 1), dtype=np.float32), 0.1, 1.0)


def test_asymmetric_orientation_golden_values() -> None:
    raw_bottom_first = np.ascontiguousarray(
        np.array([[0.0, 0.25, 0.5], [0.75, 1.0, 0.125]], dtype=np.float32)
    )
    oriented = np.ascontiguousarray(np.flipud(raw_bottom_first))
    actual = subject.convert_native_depth(oriented, 1.0, 5.0)
    expected = np.array(
        [[1.25, 1.0, 3.3333333333333335], [5.0, 2.5, 1.6666666666666667]], dtype=np.float32
    )
    assert np.array_equal(actual, expected)


def test_strict_gl_integer_allows_only_named_zero_padding() -> None:
    assert subject._strict_gl_integer(np.array([36001]), "x") == 36001
    assert (
        subject._strict_gl_integer(np.array([36001, 0]), "clip", allow_zero_padding=True) == 36001
    )
    with pytest.raises(subject.SharedRasterFailure):
        subject._strict_gl_integer(np.array([36001, 2]), "clip", allow_zero_padding=True)
    with pytest.raises(subject.SharedRasterFailure):
        subject._strict_gl_integer(np.array([1, 0]), "ordinary")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"framewidth": 1.0}, "framewidth"),
        ({"stereo": 1}, "stereo"),
        ({"rnd_depth": True}, "depth-redraw"),
        ({"segment_enabled": False}, "SEGMENT"),
        ({"readPixelFormat": 1}, "GL_RGB"),
        ({"readDepthMap": 0}, "mjDEPTH_ZEROFAR"),
        ({"pack_alignment": 4}, "packing/PBO"),
        ({"pixel_pack_buffer_binding": 7}, "packing/PBO"),
        ({"actual_current_context": 78}, "current context"),
        ({"mjr_currentBuffer": 0}, "currentBuffer"),
        ({"offSamples": 4}, "sampled"),
        ({"offFBO_r": 9}, "resolve"),
        ({"read_framebuffer_binding": 6}, "offFBO"),
        ({"draw_framebuffer_binding": 6}, "offFBO"),
        ({"read_buffer": 8}, "read buffer"),
        ({"offWidth": 3}, "dimensions"),
        ({"offscreen_attachments": {}}, "attachment"),
        ({"scene_cameras": []}, "camera"),
        ({"projection_matrix_float32": [0.0]}, "projection"),
    ],
)
def test_strict_paired_draw_state_rejects_absolute_constraint_drift(
    changes: dict[str, object], message: str
) -> None:
    state = fake_native_state(object())
    state.update(changes)
    with pytest.raises(subject.SharedRasterFailure, match=message):
        subject.validate_paired_draw_state(
            state,
            expected_context=77,
            expected_read_framebuffer=5,
            expected_read_buffer=1029,
            expected_context_facts=fake_context(),
        )


def test_strict_draw_state_accepts_legitimate_first_draw_clip_transition() -> None:
    constructor_clip = {"clip_origin": 0, "clip_depth_mode": 0}
    draw_state = fake_native_state(object())
    assert (draw_state["clip_origin"], draw_state["clip_depth_mode"]) != (
        constructor_clip["clip_origin"],
        constructor_clip["clip_depth_mode"],
    )
    subject.validate_paired_draw_state(
        draw_state,
        expected_context=77,
        expected_read_framebuffer=5,
        expected_read_buffer=1029,
        expected_context_facts=fake_context(),
    )


def test_permissions_deny_before_any_path_access(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    touched = False
    original = Path.read_text

    def fail_read(self: Path, *args: Any, **kwargs: Any) -> str:
        nonlocal touched
        touched = True
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_read)
    for permission in (
        subject.PairedEvidencePermission(False, True, True),
        subject.PairedEvidencePermission(True, False, True),
    ):
        with pytest.raises(PermissionError):
            subject.PairedArtifactAccess(tmp_path, permission)
    assert not touched


def test_metadata_needs_extra_instrumentation_permission_before_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    access = subject.PairedArtifactAccess(
        tmp_path, subject.PairedEvidencePermission(True, True, False)
    )
    monkeypatch.setattr(Path, "read_text", lambda *a, **k: pytest.fail("opened"))
    with pytest.raises(PermissionError, match="metadata"):
        access.metadata(0, 0)


def test_ledger_rejects_unrelated_initialized_root(tmp_path: Path) -> None:
    root = tmp_path / "study"
    (root / "ledger").mkdir(parents=True)
    (root / "ledger" / "revision-0000.json").write_text("{}")
    with pytest.raises(subject.SharedRasterFailure, match="ledger absent"):
        subject.validate_ledger(root)


def test_immutable_plan_and_hash_chain(tmp_path: Path) -> None:
    root = tmp_path / "study"
    subject.initialise_ledger(root, {"head": "abc"})
    subject.validate_ledger(root)
    path = root / "shared-raster-ledger" / "revision-0000.json"
    record = json.loads(path.read_text())
    record["plan"]["seed"] = 9
    path.write_text(json.dumps(record))
    with pytest.raises(subject.SharedRasterFailure, match="plan"):
        subject.validate_ledger(root)


def test_terminal_failure_cannot_reserve_again(tmp_path: Path) -> None:
    root = tmp_path / "study"
    subject.initialise_ledger(root, {"head": "abc"})
    first = subject.fixed_attempts()[0]
    subject.append_revision(root, "reserved", first)
    subject.append_revision(root, "failed", first, {"why": "test"})
    with pytest.raises(subject.SharedRasterFailure, match="cannot reserve"):
        subject.append_revision(root, "reserved", first)
    with pytest.raises(subject.SharedRasterFailure, match="permanently stops"):
        subject.validate_ledger(root)


def test_prefix_rejects_receipt_tampering(tmp_path: Path) -> None:
    root = tmp_path / "study"
    subject.initialise_ledger(root, {"head": "abc"})
    first = subject.fixed_attempts()[0]
    subject.append_revision(root, "reserved", first)
    receipt = root / "shared-raster-receipts" / f"{first.name}.json"
    receipt.parent.mkdir()
    receipt.write_text("{}")
    subject.append_revision(root, "complete", first, {"receipt_sha256": "0" * 64})
    records = subject.validate_ledger(root)
    with pytest.raises(subject.SharedRasterFailure, match="receipt binding"):
        subject.validate_completed_prefix(root, records, {"head": "abc"})


class FakeFlag:
    mjRND_SEGMENT = 0
    mjRND_IDCOLOR = 1


class FakeGeom:
    segid = 0
    objid = 7
    objtype = 5


class FakeRenderer:
    width = 2
    height = 2

    def __init__(self, module: Any):
        self.module = module
        self._rect, self._mjr_context = object(), object()
        self._scene = SimpleNamespace(flags=np.array([True, True]), ngeom=1, geoms=[FakeGeom()])
        self._model = SimpleNamespace(
            vis=SimpleNamespace(map=SimpleNamespace(znear=0.01, zfar=50.0)),
            stat=SimpleNamespace(extent=1.0),
        )
        self._depth_rendering = True
        self._segmentation_rendering = False

    def render(self) -> np.ndarray:
        if self._depth_rendering:
            out = np.empty((2, 2), dtype=np.float32)
            self.module.mjr_render(self._rect, self._scene, self._mjr_context)
            self.module.mjr_readPixels(None, out, self._rect, self._mjr_context)
            c, d = subject.depth_coefficients(0.01, 50.0)
            out[:] = (d / (out.astype(np.float64) + c)).astype(np.float32)
            out[:] = np.flipud(out)
            return out
        out = np.empty((2, 2, 3), dtype=np.uint8)
        self.module.mjr_render(self._rect, self._scene, self._mjr_context)
        self.module.mjr_readPixels(out, None, self._rect, self._mjr_context)
        image = np.flipud(out)
        return np.stack(
            (np.where(image[:, :, 0] == 0, -1, 7), np.where(image[:, :, 0] == 0, -1, 5)), axis=-1
        ).astype(np.int32)


def test_native_depth_hook_rewrites_color_argument_once_and_preserves_buffers(
    tmp_path: Path,
) -> None:
    calls: list[tuple[object, object]] = []
    module = SimpleNamespace(mjtRndFlag=FakeFlag)

    def native_render(rect: object, scene: object, context: object) -> None:
        return None

    def native_read(rgb: object, depth: object, rect: object, context: object) -> None:
        calls.append((rgb, depth))
        if isinstance(rgb, np.ndarray):
            rgb[:] = rgb_for(np.array([[0, 1], [1, 1]], dtype=np.uint32))
        if isinstance(depth, np.ndarray):
            depth[:] = np.array([[0.2, 0.3], [0.4, 0.5]], dtype=np.float32)

    module.mjr_render, module.mjr_readPixels = native_render, native_read
    attempt = subject.fixed_attempts()[0]
    owner = subject.SharedRasterRendererAdapter(
        module, lambda *a: {}, attempt, tmp_path, native_state_observer=fake_native_state
    )
    owner.contexts = [fake_context()]
    renderer = FakeRenderer(module)
    subject._PRISTINE[id(module)] = (FakeRenderer, native_render, native_read)
    proxy = subject._RendererProxy(owner, renderer, 0)
    proxy._last_update = {
        "data_identity": 1,
        "camera": -1,
        "scene_option_identity": None,
        "geomgroup": None,
    }
    owner.expected_modalities = ["depth", "segmentation"]
    depth = proxy.render()
    renderer._depth_rendering, renderer._segmentation_rendering = False, True
    segmentation = proxy.render()
    assert len(calls) == 2
    assert isinstance(calls[0][0], np.ndarray) and calls[0][1] is not None
    saved_raw = np.load(tmp_path / "episode-000000/frame-0/native_depth_pre_metric.npy")
    assert np.array_equal(saved_raw, np.array([[0.2, 0.3], [0.4, 0.5]], dtype=np.float32))
    assert np.array_equal(
        depth, np.load(tmp_path / "episode-000000/frame-0/canonical_sdk_depth.npy")
    )
    assert np.array_equal(
        segmentation, np.load(tmp_path / "episode-000000/frame-0/canonical_sdk_segmentation.npy")
    )
    assert module.mjr_render is native_render and module.mjr_readPixels is native_read


def test_hook_rejects_state_drift_before_native_call(tmp_path: Path) -> None:
    module = SimpleNamespace(
        mjtRndFlag=FakeFlag, mjr_render=lambda *a: None, mjr_readPixels=lambda *a: None
    )
    attempt = subject.fixed_attempts()[0]
    owner = subject.SharedRasterRendererAdapter(
        module, lambda *a: {}, attempt, tmp_path, native_state_observer=fake_native_state
    )
    owner.contexts = [fake_context()]
    renderer = FakeRenderer(module)
    subject._PRISTINE[id(module)] = (FakeRenderer, module.mjr_render, module.mjr_readPixels)
    proxy = subject._RendererProxy(owner, renderer, 0)
    proxy._last_update = {
        "data_identity": 1,
        "camera": -1,
        "scene_option_identity": None,
        "geomgroup": None,
    }
    owner.expected_modalities = ["rgb"]
    renderer._depth_rendering = True
    with pytest.raises(subject.SharedRasterFailure, match="sequence"):
        proxy.render()


def test_depth_flags_rejected_before_original_render_and_hooks_restored(tmp_path: Path) -> None:
    render_calls = 0
    module = SimpleNamespace(mjtRndFlag=FakeFlag)

    def native_render(*args: object) -> None:
        nonlocal render_calls
        render_calls += 1

    def native_read(*args: object) -> None:
        return None

    module.mjr_render, module.mjr_readPixels = native_render, native_read
    owner = subject.SharedRasterRendererAdapter(
        module,
        lambda *a: {},
        subject.fixed_attempts()[0],
        tmp_path,
        native_state_observer=fake_native_state,
    )
    owner.contexts = [fake_context()]
    renderer = FakeRenderer(module)
    renderer._scene.flags[:] = False
    subject._PRISTINE[id(module)] = (FakeRenderer, native_render, native_read)
    proxy = subject._RendererProxy(owner, renderer, 0)
    proxy._last_update = {
        "data_identity": 1,
        "camera": -1,
        "scene_option_identity": None,
        "geomgroup": None,
    }
    owner.expected_modalities = ["depth"]
    with pytest.raises(subject.SharedRasterFailure, match="flags"):
        proxy.render()
    assert render_calls == 0
    assert module.mjr_render is native_render and module.mjr_readPixels is native_read


def test_native_read_failure_is_retained_and_hooks_restored(tmp_path: Path) -> None:
    module = SimpleNamespace(mjtRndFlag=FakeFlag)

    def native_render(*args: object) -> None:
        return None

    def native_read(*args: object) -> None:
        raise RuntimeError("native failure")

    module.mjr_render, module.mjr_readPixels = native_render, native_read
    owner = subject.SharedRasterRendererAdapter(
        module,
        lambda *a: {},
        subject.fixed_attempts()[0],
        tmp_path,
        native_state_observer=fake_native_state,
    )
    owner.contexts = [fake_context()]
    renderer = FakeRenderer(module)
    subject._PRISTINE[id(module)] = (FakeRenderer, native_render, native_read)
    proxy = subject._RendererProxy(owner, renderer, 0)
    proxy._last_update = {
        "data_identity": 1,
        "camera": -1,
        "scene_option_identity": None,
        "geomgroup": None,
    }
    owner.expected_modalities = ["depth"]
    with pytest.raises(RuntimeError, match="native failure"):
        proxy.render()
    assert owner.native_events[-1]["error"] == {"type": "RuntimeError", "message": "native failure"}
    assert module.mjr_render is native_render and module.mjr_readPixels is native_read
