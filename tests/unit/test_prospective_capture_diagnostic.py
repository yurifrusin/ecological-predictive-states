from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from epsbench.diagnostics.capture import (
    BASELINE_CELLS,
    EXPECTED_CELLS,
    CaptureCell,
    CaptureResult,
    DiagnosticFailure,
    decode_segmentation_rgb,
    run_bounded_capture_matrix,
    validate_capture_plan,
)


def _encoded() -> np.ndarray:
    # Segids are intentionally unrelated to geom ids: 3 -> 42 and 8 -> 7.
    return np.array([[[4, 0, 0], [9, 0, 0]]], dtype=np.uint8)


def _result(*, requested: int, controls_match: bool = True) -> CaptureResult:
    return CaptureResult(
        decoded_raw_ids=np.array([[42, 7]], dtype=np.int32),
        encoded_rgb=_encoded(),
        controls_match_original=controls_match,
        provenance={
            "requested_offsamples": requested,
            "actual_offsamples": requested,
            "attachment_format": "GL_RGB",
            "attachment_component_type": "GL_UNSIGNED_BYTE",
            "gl_vendor": "fake",
            "gl_renderer": "fake",
            "gl_version": "fake",
            "model_sha256": "a",
            "xml_source_sha256": "b",
            "package_sha256": "c",
            "binary_sha256": "d",
            "segid_to_object_map": {3: 42, 8: 7},
            "readback_requires_vertical_flip": False,
        },
    )


def test_decode_uses_recorded_nontrivial_segid_map_and_flip() -> None:
    decoded = decode_segmentation_rgb(_encoded(), {3: 42, 8: 7}, flip_vertical=False)
    assert decoded.tolist() == [[42, 7]]
    flipped = decode_segmentation_rgb(
        np.vstack([_encoded(), _encoded()]), {3: 42, 8: 7}, flip_vertical=True
    )
    assert flipped.shape == (2, 2)


def test_plan_is_finite_and_ordered() -> None:
    validate_capture_plan(EXPECTED_CELLS)
    with pytest.raises(DiagnosticFailure, match="exactly"):
        validate_capture_plan((CaptureCell("wgl", 0),))


def test_baseline_failure_blocks_intervention_and_preserves_partial_output(tmp_path: Path) -> None:
    calls: list[CaptureCell] = []

    def capture(cell: CaptureCell) -> CaptureResult:
        calls.append(cell)
        return _result(
            requested=cell.requested_offsamples, controls_match=cell != BASELINE_CELLS[0]
        )

    with pytest.raises(DiagnosticFailure, match="baseline control mismatch"):
        run_bounded_capture_matrix(tmp_path / "ledger", capture)
    assert calls == [BASELINE_CELLS[0]]
    assert (tmp_path / "ledger" / BASELINE_CELLS[0].name / "receipt.json").exists()


def test_no_overwrite_capture_exception_and_decode_mismatch_fail_closed(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    with pytest.raises(DiagnosticFailure, match="reuse"):
        run_bounded_capture_matrix(ledger, lambda _: _result(requested=4))

    def exploding(_: CaptureCell) -> CaptureResult:
        raise ValueError("no context")

    with pytest.raises(DiagnosticFailure, match="capture failed"):
        run_bounded_capture_matrix(tmp_path / "failed", exploding)
    assert (
        "ValueError" in (tmp_path / "failed" / BASELINE_CELLS[0].name / "receipt.json").read_text()
    )

    def mismatched(cell: CaptureCell) -> CaptureResult:
        result = _result(requested=cell.requested_offsamples)
        return CaptureResult(
            decoded_raw_ids=np.array([[7, 42]], dtype=np.int32),
            encoded_rgb=result.encoded_rgb,
            controls_match_original=True,
            provenance=result.provenance,
        )

    with pytest.raises(DiagnosticFailure, match="does not equal"):
        run_bounded_capture_matrix(tmp_path / "bad", mismatched)
