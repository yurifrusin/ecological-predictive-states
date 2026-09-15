"""CPU-only native-contract tests: every native operation uses an explicit stub."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import mujoco
import numpy as np
import pytest

from epsbench.sim.canonical_paired import (
    CanonicalPairedCaptureError,
    CanonicalPairedRenderer,
    SceneMapEntry,
    convert_native_depth,
    decode_id_colors,
    validate_saved_state,
)


def rgb_for(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    values = values.astype(np.uint32)
    return np.stack((values & 255, (values >> 8) & 255, (values >> 16) & 255), axis=-1).astype(
        np.uint8
    )


def fake_renderer() -> Any:
    flags = np.zeros(int(mujoco.mjtRndFlag.mjNRNDFLAG), dtype=np.uint8)
    geoms = [
        SimpleNamespace(segid=segid, objid=objid, objtype=5) for segid, objid in ((0, 7), (257, 9))
    ]
    return SimpleNamespace(
        width=2,
        height=2,
        _width=2,
        _height=2,
        _depth_rendering=False,
        _segmentation_rendering=False,
        _scene=SimpleNamespace(flags=flags, ngeom=2, geoms=geoms),
        _rect=SimpleNamespace(left=0, bottom=0, width=2, height=2),
        _mjr_context=SimpleNamespace(offSamples=0, offFBO=5, offFBO_r=0, currentBuffer=1),
        _gl_context=SimpleNamespace(make_current=lambda: None),
        _model=SimpleNamespace(
            vis=SimpleNamespace(map=SimpleNamespace(znear=0.01, zfar=20.0)),
            stat=SimpleNamespace(extent=2.5),
        ),
    )


def fake_state(renderer: Any) -> dict[str, Any]:
    camera = {
        "pos": [0.0, 0.0, 0.0],
        "forward": [0.0, 1.0, 0.0],
        "up": [0.0, 0.0, 1.0],
        "frustum_near": float(np.float32(0.025)),
        "frustum_far": 50.0,
        "frustum_top": 0.02,
        "frustum_bottom": -0.02,
        "frustum_center": 0.0,
        "frustum_width": 0.0,
        "orthographic": 0,
    }
    runtime = {
        "actual_backend": "osmesa",
        "context_module": "mujoco.osmesa",
        "requested_offsamples": 0,
        "actual_offsamples": 0,
        "model_offsamples": 0,
        "width": 2,
        "height": 2,
        "sample_buffers": 0,
        "samples": 0,
        "mjr_off_width": 2,
        "mjr_off_height": 2,
        "color_storage_dimensions": [2, 2],
        "depth_storage_dimensions": [2, 2],
        "attachment_format": 32856,
        "attachment_component_type": 35863,
        "gl_samples": 0,
        "gl_vendor": "synthetic-test",
        "gl_renderer": "synthetic-test",
        "gl_version": "synthetic-test",
    }
    mapping = [
        {"segid_plus_one": int(g.segid) + 1, "objid": int(g.objid), "objtype": 5}
        for g in renderer._scene.geoms
    ]
    geometry = [
        {
            "type": 6,
            "objid": item["objid"],
            "objtype": 5,
            "segid": item["segid_plus_one"] - 1,
            "category": 1,
            "dataid": -1,
            "pos": [0.0, 0.0, 0.0],
            "mat": np.eye(3).reshape(-1).tolist(),
            "size": [1.0, 1.0, 1.0],
        }
        for item in mapping
    ]
    return {
        "actual_current_context": 77,
        "expected_current_context": 77,
        "rect": [0, 0, 2, 2],
        "scene_flags": renderer._scene.flags.tolist(),
        "ngeom": 2,
        "scene_map": mapping,
        "scene_geometry": geometry,
        "scene_cameras": [copy.deepcopy(camera), copy.deepcopy(camera)],
        "readPixelFormat": 6407,
        "gl_rgb": 6407,
        "readDepthMap": 1,
        "depth_zerofar": 1,
        "pack_alignment": 1,
        "pack_row_length": 0,
        "pack_skip_rows": 0,
        "pack_skip_pixels": 0,
        "pixel_pack_buffer_binding": 0,
        "read_framebuffer_binding": 5,
        "draw_framebuffer_binding": 5,
        "read_buffer": 36064,
        "draw_buffer": 36064,
        "framewidth": 0.0,
        "stereo": 0,
        "stereo_none": 0,
        "rnd_depth": False,
        "segment_enabled": True,
        "idcolor_enabled": True,
        "query_bindings_restored": True,
        "mjr_currentBuffer": 1,
        "framebuffer_offscreen": 1,
        "near": 0.025,
        "far": 50.0,
        "extent": 2.5,
        "projection_matrix_float32": np.eye(4).reshape(-1).tolist(),
        "modelview_matrix_float32": np.eye(4).reshape(-1).tolist(),
        "clip_origin": 36001,
        "clip_depth_mode": 37727,
        "offSamples": 0,
        "offFBO": 5,
        "offFBO_r": 0,
        "offWidth": 2,
        "offHeight": 2,
        "context_runtime": runtime,
        "offscreen_attachments": {
            "offFBO": {
                "present": True,
                "framebuffer": 5,
                "color0": {
                    "object_name": 11,
                    "object_type": 36161,
                    "component_type": 35863,
                    "internal_format": 32856,
                    "samples": 0,
                },
                "depth": {
                    "object_name": 12,
                    "object_type": 36161,
                    "component_type": 5126,
                    "internal_format": 36013,
                    "samples": 0,
                },
                "draw_framebuffer_samples": 0,
            },
            "offFBO_r": {"present": False},
        },
    }


def capture_fake(
    *, state: Any = fake_state, render: Any = None, read: Any = None
) -> tuple[Any, Any]:
    renderer = fake_renderer()

    def default_read(color: np.ndarray, depth: np.ndarray, *_args: object) -> None:
        color[:] = rgb_for(np.array([[1, 0], [258, 1]], dtype=np.uint32))
        depth[:] = np.array([[0.2, 0.3], [0.6, 0.7]], dtype=np.float32)

    result = CanonicalPairedRenderer(
        renderer,
        state_observer=state,
        native_render=render or (lambda *_args: None),
        native_read_pixels=read or default_read,
        restore_buffer=lambda *_args: None,
    ).capture()
    return renderer, result


def test_one_draw_dual_read_common_flip_and_no_raw_mutation() -> None:
    calls: list[str] = []
    raw_ids = rgb_for(np.array([[1, 0], [258, 1]], dtype=np.uint32))
    raw_depth = np.array([[0.2, 0.3], [0.6, 0.7]], dtype=np.float32)

    def draw(*_args: object) -> None:
        calls.append("draw")

    def read(color: np.ndarray, depth: np.ndarray, *_args: object) -> None:
        calls.append("dual_read")
        assert color.shape == (2, 2, 3) and depth.shape == (2, 2)
        color[:] = raw_ids
        depth[:] = raw_depth

    renderer, result = capture_fake(render=draw, read=read)
    assert calls == ["draw", "dual_read"]
    assert np.array_equal(result.native_id_rgb, np.flipud(raw_ids))
    assert np.array_equal(result.native_depth_pre_metric, np.flipud(raw_depth))
    assert result.raw_geom_segmentation.tolist() == [[9, 7], [7, -1]]
    result.depth[:] = 0
    assert np.array_equal(result.native_depth_pre_metric, np.flipud(raw_depth))
    assert not np.any(renderer._scene.flags)
    assert "framebuffer" not in result.stable_state["offscreen_attachments"]["offFBO"]
    assert "object_name" not in result.stable_state["offscreen_attachments"]["offFBO"]["color0"]
    assert (
        result.operational_state["draw_output"]["offscreen_attachments"]["offFBO"]["framebuffer"]
        == 5
    )


@pytest.mark.parametrize(
    "native_depth",
    [
        np.array([[0.0, 1.0], [0.5, np.nextafter(np.float32(1), np.float32(0))]], dtype=np.float32),
        np.array([[0.2, 0.3], [0.6, 0.7]], dtype=np.float32),
    ],
)
def test_depth_conversion_matches_installed_sdk_with_native_calls_stubbed(
    monkeypatch: pytest.MonkeyPatch,
    native_depth: np.ndarray,
) -> None:
    from mujoco.rendering.classic.renderer import Renderer

    renderer = fake_renderer()
    renderer._depth_rendering = True

    def read(color: object, depth: np.ndarray, *_args: object) -> None:
        assert color is None
        depth[:] = native_depth

    monkeypatch.setattr(mujoco, "mjr_render", lambda *_args: None)
    monkeypatch.setattr(mujoco, "mjr_readPixels", read)
    expected = Renderer.render(renderer)
    actual = convert_native_depth(np.ascontiguousarray(np.flipud(native_depth)), 0.025, 50.0)
    assert actual.dtype == expected.dtype == np.float32
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("phase", ["draw", "read", "observe_after_draw", "observe_after_read"])
def test_flags_and_buffer_restore_on_all_native_failures(phase: str) -> None:
    renderer = fake_renderer()
    calls = [0]
    restored: list[int] = []

    def observe(value: Any) -> dict[str, Any]:
        calls[0] += 1
        if (phase == "observe_after_draw" and calls[0] == 2) or (
            phase == "observe_after_read" and calls[0] == 4
        ):
            raise RuntimeError("injected failure")
        return fake_state(value)

    def fail(*_args: object) -> None:
        raise RuntimeError("injected failure")

    with pytest.raises(RuntimeError, match="injected failure"):
        CanonicalPairedRenderer(
            renderer,
            state_observer=observe,
            native_render=fail if phase == "draw" else (lambda *_args: None),
            native_read_pixels=fail if phase == "read" else (lambda *_args: None),
            restore_buffer=lambda selected, _ctx: restored.append(selected),
        ).capture()
    assert not np.any(renderer._scene.flags)
    assert restored == [1]


@pytest.mark.parametrize(
    "path,value",
    [
        (("actual_current_context",), None),
        (("actual_current_context",), 0),
        (("actual_current_context",), 78),
        (("pack_alignment",), 4),
        (("pack_row_length",), 2),
        (("pack_skip_rows",), 1),
        (("pack_skip_pixels",), 1),
        (("pixel_pack_buffer_binding",), 1),
        (("read_framebuffer_binding",), 6),
        (("draw_framebuffer_binding",), 6),
        (("read_buffer",), 1029),
        (("draw_buffer",), 1029),
        (("readPixelFormat",), 6408),
        (("readDepthMap",), 0),
        (("rnd_depth",), True),
        (("offSamples",), 4),
        (("offFBO_r",), 2),
        (("offWidth",), 3),
        (("segment_enabled",), False),
        (("idcolor_enabled",), False),
        (("offscreen_attachments", "offFBO", "color0", "samples"), 4),
        (("offscreen_attachments", "offFBO", "depth", "internal_format"), 35056),
        (("offscreen_attachments", "offFBO", "depth", "object_name"), 0),
        (("context_runtime", "depth_storage_dimensions"), [3, 2]),
    ],
)
def test_absolute_state_mismatches_fail_before_any_draw(
    path: tuple[str, ...], value: object
) -> None:
    draws: list[object] = []

    def observe(renderer: Any) -> dict[str, Any]:
        result = fake_state(renderer)
        target = result
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        return result

    with pytest.raises(CanonicalPairedCaptureError):
        capture_fake(state=observe, render=lambda *_args: draws.append(True))
    assert draws == []


@pytest.mark.parametrize("phase", [2, 3, 4])
@pytest.mark.parametrize(
    "key", ["attachment_identity", "camera", "geometry", "mapping", "projection"]
)
def test_drift_is_rejected_at_each_boundary(phase: int, key: str) -> None:
    calls = [0]

    def observe(renderer: Any) -> dict[str, Any]:
        calls[0] += 1
        result = fake_state(renderer)
        if calls[0] >= phase:
            if key == "attachment_identity":
                result["offscreen_attachments"]["offFBO"]["color0"]["object_name"] = 44
            elif key == "camera":
                result["scene_cameras"][0]["pos"][0] = 1.0
            elif key == "geometry":
                result["scene_geometry"][0]["pos"][0] = 1.0
            elif key == "mapping":
                result["scene_map"][0]["objid"] = 100
            elif key == "projection":
                result["projection_matrix_float32"][0] = 2.0
        return result

    if phase == 2 and key == "projection":
        # The draw itself legitimately installs the current scene projection.
        capture_fake(state=observe)
    else:
        with pytest.raises(CanonicalPairedCaptureError):
            capture_fake(state=observe)


def test_saved_state_rejects_volatile_handles_and_missing_fields() -> None:
    _, result = capture_fake()
    value = dict(result.stable_state)
    validate_saved_state(value, 2, 2)
    value["actual_current_context"] = 77
    with pytest.raises(CanonicalPairedCaptureError):
        validate_saved_state(value, 2, 2)
    value.pop("actual_current_context")
    value.pop("scene_cameras")
    with pytest.raises(CanonicalPairedCaptureError):
        validate_saved_state(value, 2, 2)


@pytest.mark.parametrize(
    "mapping",
    [
        [SceneMapEntry(1, 7, 5), SceneMapEntry(1, 9, 5)],
        [SceneMapEntry(0, 7, 5)],
        [SceneMapEntry(0x1000000, 7, 5)],
        [SceneMapEntry(1, -1, 5)],
        [SceneMapEntry(1, 7, 1)],
        [SceneMapEntry(1, 2**31, 5)],
    ],
)
def test_invalid_maps_are_rejected(mapping: list[SceneMapEntry]) -> None:
    with pytest.raises(CanonicalPairedCaptureError):
        decode_id_colors(np.zeros((1, 1, 3), dtype=np.uint8), mapping)


def test_unknown_id_and_wrong_dtype_are_rejected() -> None:
    mapping = [SceneMapEntry(1, 7, 5)]
    with pytest.raises(CanonicalPairedCaptureError, match="unmapped"):
        decode_id_colors(rgb_for(np.array([[258]], dtype=np.uint32)), mapping)
    with pytest.raises(CanonicalPairedCaptureError, match="shape or dtype"):
        decode_id_colors(np.zeros((1, 1, 3), dtype=np.float32), mapping)


@pytest.mark.parametrize("value", [np.nan, np.inf, -0.1, 1.1])
def test_native_depth_rejects_invalid_values(value: float) -> None:
    with pytest.raises(CanonicalPairedCaptureError):
        convert_native_depth(np.array([[value]], dtype=np.float32), 0.025, 50.0)
