from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import epsbench.diagnostics.capture as capture_module
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
