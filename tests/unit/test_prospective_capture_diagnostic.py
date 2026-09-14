from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import epsbench.diagnostics.capture as capture_module
import epsbench.diagnostics.mujoco_runner as runner_module
from epsbench.diagnostics.capture import (
    BASELINE_CELLS,
    CaptureCell,
    CapturedFrame,
    CaptureResult,
    DiagnosticFailure,
    decode_segmentation_rgb,
    init_capture_ledger,
    run_next_capture_cell,
)


def _frame(raw: int = 42) -> CapturedFrame:
    encoded = np.array([[[4, 0, 0]], [[9, 1, 2]]], dtype=np.uint8)
    pairs = decode_segmentation_rgb(encoded, {3: (42, 5), 131336: (7, 4)}, flip_vertical=True)
    return CapturedFrame(
        np.zeros((2, 1, 3), np.uint8),
        np.zeros((2, 1), np.float32),
        encoded,
        pairs,
        np.where(pairs[..., 1] == 5, pairs[..., 0], -1).astype(np.int32),
        {3: (42, 5), 131336: (7, 4)},
    )


def _result(cell: CaptureCell) -> CaptureResult:
    return CaptureResult(
        _frame(),
        _frame(),
        {
            "requested_offsamples": cell.requested_offsamples,
            "actual_offsamples": cell.requested_offsamples,
            "attachment_format": "fmt",
            "attachment_component_type": "type",
            "gl_vendor": "v",
            "gl_renderer": "r",
            "gl_version": "x",
            "gl_samples": 4,
            "model_sha256": "a",
            "xml_source_sha256": "b",
            "package_sha256": "c",
            "package_version": "3.12.0",
            "binary_sha256": "d",
            "renderer_py_sha256": "e",
            "backend": cell.backend,
            "actual_backend": cell.backend,
            "host": "test",
        },
    )


def _inputs() -> dict[str, Path]:
    return {name: Path(name) for name in capture_module.ORIGINAL_SHA256}


def _install_originals(monkeypatch: pytest.MonkeyPatch) -> None:
    original = _frame().raw_geom_ids
    monkeypatch.setattr(
        capture_module,
        "_verify_original_inputs",
        lambda _: {
            "wgl_before": original,
            "wgl_after": original,
            "osmesa_before": original,
            "osmesa_after": original,
        },
    )


def test_decode_packs_nonzero_green_blue_and_flips_asymmetric_rows() -> None:
    encoded = np.array([[[4, 0, 0]], [[9, 1, 2]]], dtype=np.uint8)
    decoded = decode_segmentation_rgb(encoded, {3: (42, 5), 131336: (7, 4)}, flip_vertical=True)
    assert decoded.tolist() == [[[7, 4]], [[42, 5]]]


def test_full_four_success_and_resume_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_originals(monkeypatch)
    output = tmp_path / "out"
    init_capture_ledger(output, inputs=_inputs())
    seen: list[CaptureCell] = []
    for _ in range(4):
        assert (
            run_next_capture_cell(
                output,
                inputs=_inputs(),
                capture=lambda cell: seen.append(cell) or _result(cell),
                geom_objtype=5,
            )
            == seen[-1]
        )
    assert seen == list(capture_module.EXPECTED_CELLS)
    with pytest.raises(DiagnosticFailure, match="complete"):
        run_next_capture_cell(output, inputs=_inputs(), capture=_result, geom_objtype=5)


def test_baseline_failures_and_reserved_state_stop_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_originals(monkeypatch)
    output = tmp_path / "out"
    init_capture_ledger(output, inputs=_inputs())
    with pytest.raises(DiagnosticFailure, match="failed"):
        run_next_capture_cell(
            output,
            inputs=_inputs(),
            capture=lambda _: (_ for _ in ()).throw(ValueError("boom")),
            geom_objtype=5,
        )
    assert json.loads((output / "ledger.json").read_text())["cells"][0]["state"] == "failed"
    with pytest.raises(DiagnosticFailure, match="permanently"):
        run_next_capture_cell(output, inputs=_inputs(), capture=_result, geom_objtype=5)

    second = tmp_path / "reserved"
    init_capture_ledger(second, inputs=_inputs())
    ledger = json.loads((second / "ledger.json").read_text())
    ledger["cells"][0]["state"] = "reserved"
    (second / "ledger.json").write_text(json.dumps(ledger))
    with pytest.raises(DiagnosticFailure, match="permanently"):
        run_next_capture_cell(second, inputs=_inputs(), capture=_result, geom_objtype=5)


def test_postcapture_validation_retains_all_arrays_and_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_originals(monkeypatch)
    output = tmp_path / "out"
    init_capture_ledger(output, inputs=_inputs())
    bad = _result(capture_module.EXPECTED_CELLS[0])
    bad_frame = CapturedFrame(
        bad.before.rgb,
        bad.before.depth,
        bad.before.encoded_rgb,
        bad.before.decoded_pairs,
        np.full((2, 1), 9, np.int32),
        bad.before.segid_to_object_map,
    )
    with pytest.raises(DiagnosticFailure, match="postcapture"):
        run_next_capture_cell(
            output,
            inputs=_inputs(),
            capture=lambda _: CaptureResult(bad_frame, bad.after, bad.provenance),
            geom_objtype=5,
        )
    cell = output / BASELINE_CELLS[0].name
    assert (cell / "before_encoded_rgb.npy").exists() and (cell / "after_depth.npy").exists()
    assert json.loads((cell / "receipt.json").read_text())["observations_retained"]


def test_rejects_expanded_ledger_and_existing_cell_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_originals(monkeypatch)
    output = tmp_path / "out"
    init_capture_ledger(output, inputs=_inputs())
    ledger = json.loads((output / "ledger.json").read_text())
    ledger["cells"].append(dict(ledger["cells"][0]))
    (output / "ledger.json").write_text(json.dumps(ledger))
    with pytest.raises(DiagnosticFailure, match="structure"):
        run_next_capture_cell(output, inputs=_inputs(), capture=_result, geom_objtype=5)


def test_original_hash_mismatch_fails_before_ledger(tmp_path: Path) -> None:
    with pytest.raises(DiagnosticFailure, match="scope"):
        init_capture_ledger(tmp_path / "out", inputs={})


def test_gl_inspection_uses_renderbuffer_query_and_restores_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    import types

    from epsbench.diagnostics.gl_provenance import inspect_mujoco_offscreen_attachments

    bindings = {1: 71, 2: 72, 3: 73}
    gl = types.SimpleNamespace(
        GL_READ_FRAMEBUFFER_BINDING=1,
        GL_DRAW_FRAMEBUFFER_BINDING=2,
        GL_RENDERBUFFER_BINDING=3,
        GL_READ_FRAMEBUFFER=4,
        GL_DRAW_FRAMEBUFFER=5,
        GL_RENDERBUFFER=6,
        GL_COLOR_ATTACHMENT0=7,
        GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE=8,
        GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME=9,
        GL_FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE=10,
        GL_RENDERBUFFER_INTERNAL_FORMAT=11,
        GL_RENDERBUFFER_SAMPLES=12,
        GL_SAMPLES=13,
        GL_VENDOR=14,
        GL_RENDERER=15,
        GL_VERSION=16,
    )
    gl.glGetIntegerv = lambda key: bindings[key] if key in bindings else 4
    gl.glBindFramebuffer = lambda target, value: bindings.__setitem__(
        1 if target == 4 else 2, value
    )
    gl.glBindRenderbuffer = lambda target, value: bindings.__setitem__(3, value)
    gl.glGetFramebufferAttachmentParameteriv = lambda *_: (
        6 if _[-1] == 8 else 99 if _[-1] == 9 else 10
    )
    calls: list[int] = []
    gl.glGetRenderbufferParameteriv = lambda _, key: calls.append(key) or (88 if key == 11 else 4)
    gl.glGetString = lambda _: b"fake"
    package = types.ModuleType("OpenGL")
    package.GL = gl
    monkeypatch.setitem(sys.modules, "OpenGL", package)
    result = inspect_mujoco_offscreen_attachments(
        types.SimpleNamespace(_mjr_context=types.SimpleNamespace(offFBO=101, offFBO_r=0))
    )
    assert result["attachment_format"] == 88 and calls == [11, 12]
    assert bindings == {1: 71, 2: 72, 3: 73}


def test_observed_backend_mismatch_fails_permanently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_originals(monkeypatch)
    output = tmp_path / "out"
    init_capture_ledger(output, inputs=_inputs())
    expected = capture_module.EXPECTED_CELLS[0]
    result = _result(expected)
    bad = dict(result.provenance)
    bad["actual_backend"] = "osmesa"
    with pytest.raises(DiagnosticFailure, match="postcapture"):
        run_next_capture_cell(
            output,
            inputs=_inputs(),
            capture=lambda _: CaptureResult(result.before, result.after, bad),
            geom_objtype=5,
        )
    assert json.loads((output / "ledger.json").read_text())["cells"][0]["state"] == "failed"
    with pytest.raises(DiagnosticFailure, match="permanently"):
        run_next_capture_cell(output, inputs=_inputs(), capture=_result, geom_objtype=5)


def test_input_preflight_failure_is_reserved_failed_and_never_calls_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_originals(monkeypatch)
    output = tmp_path / "out"
    init_capture_ledger(output, inputs=_inputs())
    monkeypatch.setattr(
        capture_module,
        "_verify_original_inputs",
        lambda _: (_ for _ in ()).throw(DiagnosticFailure("bad fixed input")),
    )
    with pytest.raises(DiagnosticFailure, match="capture failed"):
        run_next_capture_cell(
            output,
            inputs=_inputs(),
            capture=lambda _: pytest.fail("callback must not run"),
            geom_objtype=5,
        )
    assert json.loads((output / "ledger.json").read_text())["cells"][0]["state"] == "failed"


def test_partial_frame_keeps_rgb_depth_encoded_and_pairs_on_map_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_originals(monkeypatch)
    output = tmp_path / "out"
    init_capture_ledger(output, inputs=_inputs())
    partial = capture_module.PartialCapturedFrame(
        rgb=np.array([[[1, 2, 3]]], np.uint8),
        depth=np.array([[2.5]], np.float32),
        encoded_rgb=np.array([[[4, 0, 0]]], np.uint8),
        decoded_pairs=np.array([[[42, 5]]], np.int32),
        segid_to_object_map=None,
    )
    failure = capture_module.PartialCaptureFailure(
        "map failure",
        partial_frame=partial,
        partial_frame_name="after",
        stage="segmentation_scene_map",
    )
    with pytest.raises(DiagnosticFailure, match="capture failed"):
        run_next_capture_cell(
            output,
            inputs=_inputs(),
            capture=lambda _: (_ for _ in ()).throw(failure),
            geom_objtype=5,
        )
    cell = output / BASELINE_CELLS[0].name
    assert np.array_equal(np.load(cell / "partial_after_rgb.npy"), partial.rgb)
    assert np.array_equal(np.load(cell / "partial_after_depth.npy"), partial.depth)
    assert np.array_equal(np.load(cell / "partial_after_encoded_rgb.npy"), partial.encoded_rgb)
    assert np.array_equal(np.load(cell / "partial_after_decoded_pairs.npy"), partial.decoded_pairs)
    assert not (cell / "partial_after_segid_map.json").exists()


class _FaultScene:
    def __init__(self, renderer: _FaultRenderer) -> None:
        self.renderer = renderer
        self.ngeom = 1

    @property
    def geoms(self) -> list[object]:
        self.renderer.raise_if_requested("segmentation_scene_map")
        import types

        return [types.SimpleNamespace(segid=3, objid=42, objtype=5)]


class _FaultRenderer:
    def __init__(self, fault_stage: str | None, fault_frame: int = 1) -> None:
        self.fault_stage = fault_stage
        self.fault_frame = fault_frame
        self.current_frame = 1
        self.mode = "rgb"
        self.height = 1
        self.width = 1
        self.scene = _FaultScene(self)
        self.calls: list[str] = []
        self.closed = False
        self._mjr_context = type("Context", (), {"offSamples": 4})()

    def raise_if_requested(self, stage: str) -> None:
        self.calls.append(stage)
        if self.fault_stage == stage and self.current_frame == self.fault_frame:
            raise RuntimeError(f"{stage} failed")

    def update_scene(self, _data: object, *, camera: int) -> None:
        del camera
        self.raise_if_requested(f"{self.mode}_scene_update")

    def render(self, *, out: np.ndarray | None = None) -> np.ndarray:
        stage = (
            "segmentation_readback_before_decoded_return"
            if self.mode == "segmentation"
            else f"{self.mode}_readback"
        )
        self.raise_if_requested(stage)
        if self.mode == "rgb":
            return np.array([[[1, 2, 3]]], dtype=np.uint8)
        if self.mode == "depth":
            return np.array([[2.5]], dtype=np.float32)
        assert out is not None
        out[...] = np.array([[[4, 0, 0]]], dtype=np.uint8)
        return np.array([[[42, 5]]], dtype=np.int32)

    def enable_depth_rendering(self) -> None:
        self.raise_if_requested("depth_enable")
        self.mode = "depth"

    def disable_depth_rendering(self) -> None:
        self.raise_if_requested("depth_disable")
        self.mode = "rgb"

    def enable_segmentation_rendering(self) -> None:
        self.raise_if_requested("segmentation_enable")
        self.mode = "segmentation"

    def disable_segmentation_rendering(self) -> None:
        self.raise_if_requested("segmentation_disable")
        self.mode = "rgb"
        self.current_frame += 1

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize(
    ("stage", "has_rgb", "has_depth", "has_segmentation", "has_mapping"),
    [
        ("rgb_scene_update", False, False, False, False),
        ("rgb_readback", False, False, False, False),
        ("depth_enable", True, False, False, False),
        ("depth_scene_update", True, False, False, False),
        ("depth_readback", True, False, False, False),
        ("depth_disable", True, True, False, False),
        ("segmentation_enable", True, True, False, False),
        ("segmentation_scene_update", True, True, False, False),
        ("segmentation_readback_before_decoded_return", True, True, False, False),
        ("segmentation_scene_map", True, True, True, False),
        ("segmentation_disable", True, True, True, True),
    ],
)
def test_frame_failure_retains_only_observations_acquired_before_stage(
    stage: str,
    has_rgb: bool,
    has_depth: bool,
    has_segmentation: bool,
    has_mapping: bool,
) -> None:
    import types

    renderer = _FaultRenderer(stage)
    mujoco = types.SimpleNamespace(
        mj_forward=lambda _model, _data: None,
        mjtObj=types.SimpleNamespace(mjOBJ_GEOM=5),
    )
    model = types.SimpleNamespace(cam_pos=np.zeros((1, 3), dtype=np.float64))

    with pytest.raises(runner_module._FrameFailure) as raised:
        runner_module._frame(mujoco, model, object(), renderer, 0, 1.5)

    failure = raised.value
    partial = failure.observations
    assert failure.stage == stage
    assert (partial.rgb is not None) is has_rgb
    assert (partial.depth is not None) is has_depth
    assert (partial.encoded_rgb is not None) is has_segmentation
    assert (partial.decoded_pairs is not None) is has_segmentation
    assert (partial.segid_to_object_map is not None) is has_mapping
    if has_rgb:
        assert np.array_equal(partial.rgb, np.array([[[1, 2, 3]]], dtype=np.uint8))
    if has_depth:
        assert np.array_equal(partial.depth, np.array([[2.5]], dtype=np.float32))
    if has_segmentation:
        assert np.array_equal(partial.encoded_rgb, np.array([[[4, 0, 0]]], dtype=np.uint8))
        assert np.array_equal(partial.decoded_pairs, np.array([[[42, 5]]], dtype=np.int32))
    if has_mapping:
        assert partial.segid_to_object_map == {3: (42, 5)}


@pytest.mark.parametrize(("fault_frame", "frame_name"), [(1, "before"), (2, "after")])
def test_capture_runner_propagates_actual_frame_failure(
    fault_frame: int, frame_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    import types

    renderer = _FaultRenderer("depth_disable", fault_frame=fault_frame)
    model = types.SimpleNamespace(
        cam_pos=np.zeros((1, 3), dtype=np.float64),
        vis=types.SimpleNamespace(quality=types.SimpleNamespace(offsamples=0)),
    )
    data = types.SimpleNamespace(
        cam_xpos=np.zeros((1, 3), dtype=np.float64),
        cam_xmat=np.eye(3, dtype=np.float64).reshape(1, 9),
    )
    contract = types.SimpleNamespace(
        raw_geom_ids=(42,),
        raw_geom_world_positions=((0.0, 0.0, 0.0),),
        raw_geom_compiled_sizes=((1.0, 1.0, 1.0),),
        raw_geom_types=(5,),
        raw_geom_world_rotations_row_major=((1.0, 0.0, 0.0),),
        camera_field_of_view_degrees=55.0,
        camera_world_position=(0.0, 0.0, 0.0),
        camera_world_rotation_row_major=tuple(np.eye(3).reshape(-1)),
    )
    expected_contract = runner_module._contract_mapping(contract)
    fake_mujoco = types.ModuleType("mujoco")
    fake_mujoco.mjtObj = types.SimpleNamespace(mjOBJ_CAMERA=6, mjOBJ_GEOM=5)
    fake_mujoco.MjModel = types.SimpleNamespace(from_xml_string=lambda _xml, _assets: model)
    fake_mujoco.MjData = lambda _model: data
    fake_mujoco.mj_name2id = lambda *_args: 0
    fake_mujoco.mj_forward = lambda _model, _data: None
    fake_mujoco.mj_saveModel = lambda _model, path, _buffer: Path(path).write_bytes(b"compiled")
    fake_mujoco.Renderer = lambda _model, *, height, width: renderer
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)

    import epsbench.sim.compiled as compiled_module
    import epsbench.sim.corridor as corridor_module

    monkeypatch.setattr(compiled_module, "extract_compiled_scene_contract", lambda *_args: contract)
    monkeypatch.setattr(corridor_module, "build_corridor_scene_xml", lambda *_args: "<xml/>")
    config = types.SimpleNamespace(render=types.SimpleNamespace(height=1, width=1))
    geometry = types.SimpleNamespace(
        camera_before_forward_position=1.0,
        camera_after_forward_position=2.0,
    )
    appearance = types.SimpleNamespace(asset_bytes={})
    camera = {
        "camera_world_position": (0.0, 0.0, 0.0),
        "camera_world_rotation_row_major": tuple(np.eye(3).reshape(-1)),
    }

    with pytest.raises(capture_module.PartialCaptureFailure) as raised:
        runner_module.capture_corridor_transition(
            config,
            geometry,
            appearance,
            expected_contract,
            camera,
            camera,
            requested_offsamples=4,
            backend="wgl",
        )

    failure = raised.value
    assert failure.stage == "depth_disable"
    assert failure.partial_frame_name == frame_name
    assert (failure.before is not None) is (fault_frame == 2)
    assert failure.partial_frame is not None
    assert np.array_equal(failure.partial_frame.rgb, np.array([[[1, 2, 3]]], dtype=np.uint8))
    assert np.array_equal(failure.partial_frame.depth, np.array([[2.5]], dtype=np.float32))
    assert failure.partial_frame.encoded_rgb is None
    assert failure.partial_frame.decoded_pairs is None
    assert failure.partial_frame.segid_to_object_map is None
    assert renderer.closed
