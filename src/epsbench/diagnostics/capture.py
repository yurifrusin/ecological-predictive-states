"""Bounded prospective capture helpers; no renderer or context is created here."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt


class DiagnosticFailure(RuntimeError):
    """A predeclared diagnostic stop condition."""


@dataclass(frozen=True)
class CaptureCell:
    backend: str
    requested_offsamples: int

    @property
    def name(self) -> str:
        return f"{self.backend}-offsamples-{self.requested_offsamples}"


BASELINE_CELLS = (CaptureCell("wgl", 4), CaptureCell("osmesa", 4))
INTERVENTION_CELLS = (CaptureCell("wgl", 0), CaptureCell("osmesa", 0))
EXPECTED_CELLS = BASELINE_CELLS + INTERVENTION_CELLS


@dataclass(frozen=True)
class CaptureResult:
    """Facts returned by one already-authorised capture callback."""

    decoded_raw_ids: npt.NDArray[np.int32]
    encoded_rgb: npt.NDArray[np.uint8]
    controls_match_original: bool
    provenance: Mapping[str, object]


def decode_segmentation_rgb(
    encoded_rgb: npt.NDArray[np.uint8],
    segid_to_raw_geom_id: Mapping[int, int],
    *,
    flip_vertical: bool,
) -> npt.NDArray[np.int32]:
    """Decode RGB id colours using the recorded scene map, never raw-id==segid."""
    if encoded_rgb.ndim != 3 or encoded_rgb.shape[2] != 3:
        raise DiagnosticFailure("encoded segmentation must be HxWx3 uint8")
    if encoded_rgb.dtype != np.uint8:
        raise DiagnosticFailure("encoded segmentation must have dtype uint8")
    packed = (
        encoded_rgb[..., 0].astype(np.int64)
        + 256 * encoded_rgb[..., 1].astype(np.int64)
        + 65536 * encoded_rgb[..., 2].astype(np.int64)
    )
    segids = packed - 1
    decoded = np.full(segids.shape, -1, dtype=np.int32)
    for segid, raw_geom_id in segid_to_raw_geom_id.items():
        decoded[segids == segid] = np.int32(raw_geom_id)
    return np.flipud(decoded) if flip_vertical else decoded


def require_capture_provenance(provenance: Mapping[str, object]) -> None:
    """Reject unavailable material provenance rather than silently guessing."""
    required = {
        "requested_offsamples",
        "actual_offsamples",
        "attachment_format",
        "attachment_component_type",
        "gl_vendor",
        "gl_renderer",
        "gl_version",
        "model_sha256",
        "xml_source_sha256",
        "package_sha256",
        "binary_sha256",
        "segid_to_object_map",
    }
    missing = sorted(name for name in required if provenance.get(name) is None)
    if missing:
        raise DiagnosticFailure("material capture provenance unavailable: " + ", ".join(missing))


def validate_capture_plan(cells: tuple[CaptureCell, ...]) -> None:
    if cells != EXPECTED_CELLS:
        raise DiagnosticFailure(
            "capture plan must be exactly one WGL/OSMesa pair at 4 followed by one pair at 0"
        )


def _new_cell_directory(output_root: Path, cell: CaptureCell) -> Path:
    path = output_root / cell.name
    if path.exists():
        raise DiagnosticFailure(f"refusing to overwrite existing cell output: {path}")
    path.mkdir(parents=True)
    return path


def _write_receipt(directory: Path, payload: Mapping[str, object]) -> None:
    (directory / "receipt.json").write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _persist_cell(directory: Path, cell: CaptureCell, result: CaptureResult) -> None:
    np.save(directory / "decoded_raw_ids.npy", result.decoded_raw_ids, allow_pickle=False)
    np.save(directory / "encoded_segmentation_rgb.npy", result.encoded_rgb, allow_pickle=False)
    _write_receipt(
        directory,
        {
            "cell": {
                "backend": cell.backend,
                "requested_offsamples": cell.requested_offsamples,
            },
            "controls_match_original": result.controls_match_original,
            "provenance": dict(result.provenance),
        },
    )


def run_bounded_capture_matrix(
    output_root: Path,
    capture: Callable[[CaptureCell], CaptureResult],
) -> tuple[CaptureCell, ...]:
    """Run the fixed four-cell matrix once, retaining partial failure receipts.

    The callback is the only code permitted to create a context. It must perform
    all original-evidence and compiled-scene checks before capture.
    """
    validate_capture_plan(EXPECTED_CELLS)
    if output_root.exists():
        raise DiagnosticFailure(f"refusing to reuse diagnostic ledger root: {output_root}")
    output_root.mkdir(parents=True)
    completed: list[CaptureCell] = []
    for cell in EXPECTED_CELLS:
        if cell.requested_offsamples == 0 and len(completed) != len(BASELINE_CELLS):
            raise DiagnosticFailure("intervention blocked until both baseline controls complete")
        directory = _new_cell_directory(output_root, cell)
        try:
            result = capture(cell)
        except Exception as exc:
            _write_receipt(
                directory,
                {
                    "cell": {
                        "backend": cell.backend,
                        "requested_offsamples": cell.requested_offsamples,
                    },
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            )
            raise DiagnosticFailure(f"capture failed for {cell.name}") from exc
        require_capture_provenance(result.provenance)
        if result.provenance["requested_offsamples"] != cell.requested_offsamples:
            raise DiagnosticFailure("callback provenance requested_offsamples does not match cell")
        decoded_from_rgb = decode_segmentation_rgb(
            result.encoded_rgb,
            {
                int(segid): int(raw_id)
                for segid, raw_id in dict(result.provenance["segid_to_object_map"]).items()
            },
            flip_vertical=bool(result.provenance.get("readback_requires_vertical_flip")),
        )
        if not np.array_equal(decoded_from_rgb, result.decoded_raw_ids):
            raise DiagnosticFailure(
                "encoded RGB decode does not equal returned segmentation output"
            )
        _persist_cell(directory, cell, result)
        completed.append(cell)
        if cell in BASELINE_CELLS and not result.controls_match_original:
            raise DiagnosticFailure(
                f"baseline control mismatch for {cell.name}; intervention blocked"
            )
    return tuple(completed)
