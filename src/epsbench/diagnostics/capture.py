"""Finite persistent prospective capture ledger; no graphics context is created here."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

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
ARRAY_KEYS = ("wgl_before", "wgl_after", "osmesa_before", "osmesa_after")

# Exact original episode-0 evidence hashes, bound by the reviewed coordinate audit.
ORIGINAL_SHA256 = {
    "preflight_wgl": "4eef1efc8f3f0daca0dd7274553dfde685be30bc2154937858ce85e056c849fb",
    "generation_wgl": "6c756ac0b252dfacc72170a423206edfcbb21b09a0b66455c645fa8254223abc",
    "preflight_osmesa": "d5794b552558c5067edd37007f6623f904d24662cc803e309c744e75686453d8",
    "generation_osmesa": "726b7eedfb659b70c2f47458c2007b919b6b02093bb7368f77af04e87cfa1876",
    "wgl_manifest": "3f0edd1a514d8907c083124bc4572fc9ba5da13d9ed70e1ae651ef40a912dfa9",
    "osmesa_manifest": "8a4e8b47b7ac6eb302632e5b234e9155d7a54f6dadb349dc27367aa620106779",
    "wgl_config": "a1aa4f3e65b279d0c5ab916bc8fbc8abc61cebeb6e48b34b24be7306458730e5",
    "osmesa_config": "a1aa4f3e65b279d0c5ab916bc8fbc8abc61cebeb6e48b34b24be7306458730e5",
    "wgl_instrumentation": "11862e2b0a9a95140ebe9e5caeb5c00e8833a44d67f6011fe80f9661a258d27f",
    "osmesa_instrumentation": "e640c36d42e3cd9a4faf75ecd9f6a0151feacc267ead8d8032be7d6e5c2b9456",
    "wgl_before": "b5974ac705cd42a42e25abbbe8ed57cd3fec4218eb72b4d623c64fe81459b4d8",
    "wgl_after": "8308f8028b9b0b32ad294baf1ff4bcb6315cbd77ed5bbddc27bb94b35a58af9a",
    "osmesa_before": "6bcb47a40bd671cb8ace39753d2243d75727605dfafdf5d77172663bee956039",
    "osmesa_after": "859ce504f08fc4bcb0ec493dedc13d81620c682201943866dee13a406caa4415",
}


@dataclass(frozen=True)
class CapturedFrame:
    rgb: npt.NDArray[np.uint8]
    depth: npt.NDArray[np.float32]
    encoded_rgb: npt.NDArray[np.uint8]
    decoded_pairs: npt.NDArray[np.int32]
    raw_geom_ids: npt.NDArray[np.int32]
    segid_to_object_map: Mapping[int, tuple[int, int]]


@dataclass(frozen=True)
class CaptureResult:
    before: CapturedFrame
    after: CapturedFrame
    provenance: Mapping[str, object]


class PartialCaptureFailure(DiagnosticFailure):
    """A capture failure that carries observations made before the failure."""

    def __init__(
        self,
        message: str,
        partial: CaptureResult | None = None,
        *,
        before: CapturedFrame | None = None,
        encoded_rgb: npt.NDArray[np.uint8] | None = None,
        stage: str | None = None,
    ) -> None:
        super().__init__(message)
        self.partial = partial
        self.before = before
        self.encoded_rgb = encoded_rgb
        self.stage = stage


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_write(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosticFailure(f"cannot read ledger record: {path}") from exc
    if not isinstance(value, dict):
        raise DiagnosticFailure(f"ledger record is not an object: {path}")
    return value


def _array(path: Path) -> npt.NDArray[np.int32]:
    try:
        loaded = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise DiagnosticFailure(f"cannot load trusted original array: {path}") from exc
    if loaded.dtype != np.int32 or loaded.ndim != 2:
        raise DiagnosticFailure(f"trusted original array must be HxW int32: {path}")
    return cast(npt.NDArray[np.int32], loaded)


def validate_capture_plan(cells: tuple[CaptureCell, ...]) -> None:
    if cells != EXPECTED_CELLS:
        raise DiagnosticFailure("capture plan must be exactly WGL/OSMesa at 4 then WGL/OSMesa at 0")


def decode_segmentation_rgb(
    encoded_rgb: npt.NDArray[np.uint8],
    segid_to_object_map: Mapping[int, tuple[int, int]],
    *,
    flip_vertical: bool,
) -> npt.NDArray[np.int32]:
    """Use MuJoCo's R + G*256 + B*65536 (segid + 1) convention."""
    if encoded_rgb.dtype != np.uint8 or encoded_rgb.ndim != 3 or encoded_rgb.shape[2] != 3:
        raise DiagnosticFailure("encoded segmentation must be HxWx3 uint8")
    packed = (
        encoded_rgb[..., 0].astype(np.int64)
        + 256 * encoded_rgb[..., 1].astype(np.int64)
        + 65536 * encoded_rgb[..., 2].astype(np.int64)
    )
    pairs = np.full((*packed.shape, 2), -1, dtype=np.int32)
    for segid, pair in segid_to_object_map.items():
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise DiagnosticFailure("segid map entries must be (objid, objtype) pairs")
        pairs[packed == int(segid) + 1] = (int(pair[0]), int(pair[1]))
    return np.flipud(pairs) if flip_vertical else pairs


def raw_geom_ids(decoded_pairs: npt.NDArray[np.int32], geom_objtype: int) -> npt.NDArray[np.int32]:
    if decoded_pairs.dtype != np.int32 or decoded_pairs.ndim != 3 or decoded_pairs.shape[2] != 2:
        raise DiagnosticFailure("returned segmentation must be HxWx2 int32")
    return np.where(decoded_pairs[..., 1] == geom_objtype, decoded_pairs[..., 0], -1).astype(
        np.int32
    )


def require_capture_provenance(provenance: Mapping[str, object], cell: CaptureCell) -> None:
    required = {
        "requested_offsamples",
        "actual_offsamples",
        "attachment_format",
        "attachment_component_type",
        "gl_vendor",
        "gl_renderer",
        "gl_version",
        "gl_samples",
        "model_sha256",
        "xml_source_sha256",
        "package_sha256",
        "binary_sha256",
        "renderer_py_sha256",
        "backend",
        "host",
    }
    missing = sorted(key for key in required if provenance.get(key) in (None, ""))
    if missing:
        raise DiagnosticFailure("material capture provenance unavailable: " + ", ".join(missing))
    if provenance["requested_offsamples"] != cell.requested_offsamples:
        raise DiagnosticFailure("recorded requested_offsamples differs from reserved cell")
    if provenance["backend"] != cell.backend:
        raise DiagnosticFailure("actual backend differs from reserved cell")


def _persist_frame(directory: Path, name: str, frame: CapturedFrame) -> None:
    prefix = directory / name
    np.save(prefix.with_name(prefix.name + "_rgb.npy"), frame.rgb, allow_pickle=False)
    np.save(prefix.with_name(prefix.name + "_depth.npy"), frame.depth, allow_pickle=False)
    np.save(
        prefix.with_name(prefix.name + "_encoded_rgb.npy"), frame.encoded_rgb, allow_pickle=False
    )
    np.save(
        prefix.with_name(prefix.name + "_decoded_pairs.npy"),
        frame.decoded_pairs,
        allow_pickle=False,
    )
    np.save(
        prefix.with_name(prefix.name + "_raw_geom_ids.npy"), frame.raw_geom_ids, allow_pickle=False
    )
    _json_write(
        prefix.with_name(prefix.name + "_segid_map.json"),
        {
            str(segid): [int(pair[0]), int(pair[1])]
            for segid, pair in sorted(frame.segid_to_object_map.items())
        },
    )


def _persist_observations(directory: Path, result: CaptureResult) -> None:
    _persist_frame(directory, "before", result.before)
    _persist_frame(directory, "after", result.after)
    _json_write(directory / "provenance.json", dict(result.provenance))


def _validate_frame(frame: CapturedFrame, *, geom_objtype: int) -> None:
    decoded = decode_segmentation_rgb(
        frame.encoded_rgb, frame.segid_to_object_map, flip_vertical=True
    )
    if not np.array_equal(decoded, frame.decoded_pairs):
        raise DiagnosticFailure("encoded RGB decode does not equal returned object-id/type output")
    expected_raw = raw_geom_ids(frame.decoded_pairs, geom_objtype)
    if not np.array_equal(expected_raw, frame.raw_geom_ids):
        raise DiagnosticFailure("raw geom IDs differ from returned object-id/type output")


def _ledger_path(output_root: Path) -> Path:
    return output_root / "ledger.json"


def _verify_original_inputs(inputs: Mapping[str, Path]) -> dict[str, npt.NDArray[np.int32]]:
    """Verify receipt -> manifest -> fixed source/config/array packet before every attempt."""
    if set(inputs) != set(ORIGINAL_SHA256):
        raise DiagnosticFailure("input scope must be the complete fixed original episode-0 packet")
    for name, item in inputs.items():
        if not item.is_file() or _file_sha256(item) != ORIGINAL_SHA256[name]:
            raise DiagnosticFailure(f"fixed original evidence hash mismatch: {name}")
    for backend in ("wgl", "osmesa"):
        receipt = _json_read(inputs[f"generation_{backend}"])
        datasets = receipt.get("datasets")
        if not isinstance(datasets, list) or not any(
            isinstance(item, dict)
            and item.get("scene") == "corridor"
            and item.get("manifest_sha256") == ORIGINAL_SHA256[f"{backend}_manifest"]
            for item in datasets
        ):
            raise DiagnosticFailure(f"generation receipt does not bind {backend} corridor manifest")
        if (
            receipt.get("head") != "d7b7ce8f04426e5869ef6e063f97c421fd10fffc"
            or receipt.get("tree") != "8ef8f5552750e322c01538fdf7be30a1855c0a63"
        ):
            raise DiagnosticFailure("original receipt source binding differs")
        config = _json_read(inputs[f"{backend}_config"])
        seed = (
            _json_read(inputs[f"{backend}_instrumentation"])
            .get("generation_seeds", {})
            .get("episode_seed")
        )
        if (
            config.get("seed"),
            config.get("scene_family"),
            config.get("render", {}).get("width"),
            config.get("render", {}).get("height"),
            config.get("camera", {}).get("field_of_view_degrees"),
            config.get("appearance", {}).get("profile_id"),
            seed,
        ) != (1729, "corridor", 160, 120, 55.0, "legacy_solid_base_v1", 1703363364368450807):
            raise DiagnosticFailure("fixed config or seed differs from original episode")
    arrays = {name: _array(inputs[name]) for name in ARRAY_KEYS}
    if {array.shape for array in arrays.values()} != {(120, 160)}:
        raise DiagnosticFailure("fixed original arrays must all be 120x160")
    return arrays


def init_capture_ledger(output_root: Path, *, inputs: Mapping[str, Path]) -> None:
    """Create a fresh ledger only after exact original evidence binding succeeds."""
    validate_capture_plan(EXPECTED_CELLS)
    if output_root.exists():
        raise DiagnosticFailure(f"refusing to reuse diagnostic ledger root: {output_root}")
    _verify_original_inputs(inputs)
    output_root.mkdir(parents=True)
    _json_write(
        _ledger_path(output_root),
        {
            "schema_version": "prospective_capture_ledger_v3",
            "cells": [
                {
                    "name": cell.name,
                    "backend": cell.backend,
                    "requested_offsamples": cell.requested_offsamples,
                    "state": "pending",
                }
                for cell in EXPECTED_CELLS
            ],
            "fixed_original_sha256": ORIGINAL_SHA256,
        },
    )


def _load_ledger(output_root: Path) -> dict[str, Any]:
    if not output_root.is_dir():
        raise DiagnosticFailure(f"diagnostic ledger does not exist: {output_root}")
    ledger = _json_read(_ledger_path(output_root))
    cells = ledger.get("cells")
    expected = [(cell.name, cell.backend, cell.requested_offsamples) for cell in EXPECTED_CELLS]
    if (
        ledger.get("schema_version") != "prospective_capture_ledger_v3"
        or ledger.get("fixed_original_sha256") != ORIGINAL_SHA256
        or not isinstance(cells, list)
        or len(cells) != 4
    ):
        raise DiagnosticFailure("invalid prospective ledger structure")
    actual = [
        (item.get("name"), item.get("backend"), item.get("requested_offsamples"))
        if isinstance(item, dict)
        else None
        for item in cells
    ]
    states = [item.get("state") if isinstance(item, dict) else None for item in cells]
    if actual != expected or any(
        state not in {"pending", "reserved", "failed", "complete"} for state in states
    ):
        raise DiagnosticFailure("ledger order, scope, or state is invalid")
    if any(state in {"reserved", "failed"} for state in states):
        raise DiagnosticFailure(
            "ledger records a reserved/interrupted or failed attempt; matrix is permanently stopped"
        )
    completed = states.index("pending") if "pending" in states else 4
    if states != ["complete"] * completed + ["pending"] * (4 - completed):
        raise DiagnosticFailure("ledger completion order is invalid")
    return ledger


def _next_reserved_cell(ledger: Mapping[str, Any]) -> CaptureCell:
    cells = ledger["cells"]
    for index, item in enumerate(cells):
        if item["state"] == "pending":
            return EXPECTED_CELLS[index]
    raise DiagnosticFailure("finite capture matrix is already complete")


def _set_cell_state(ledger: dict[str, Any], cell: CaptureCell, state: str, **extra: object) -> None:
    for item in ledger["cells"]:
        if item["name"] == cell.name:
            item["state"] = state
            item.update(extra)
            return
    raise DiagnosticFailure("reserved cell is not in fixed matrix")


def run_next_capture_cell(
    output_root: Path,
    *,
    inputs: Mapping[str, Path],
    capture: Callable[[CaptureCell], CaptureResult],
    geom_objtype: int,
) -> CaptureCell:
    """Reserve one cell; any failed or interrupted reservation stops the matrix."""
    ledger = _load_ledger(output_root)
    originals = _verify_original_inputs(inputs)
    cell = _next_reserved_cell(ledger)
    completed = tuple(
        CaptureCell(str(item["backend"]), int(item["requested_offsamples"]))
        for item in ledger["cells"]
        if item.get("state") == "complete"
    )
    if (
        cell.requested_offsamples == 0
        and tuple(c for c in completed if c in BASELINE_CELLS) != BASELINE_CELLS
    ):
        raise DiagnosticFailure("intervention blocked until both baseline controls complete")
    cell_dir = output_root / cell.name
    if cell_dir.exists():
        raise DiagnosticFailure(f"refusing to overwrite existing cell output: {cell_dir}")
    cell_dir.mkdir()
    _set_cell_state(ledger, cell, "reserved")
    _json_write(_ledger_path(output_root), ledger)
    try:
        result = capture(cell)
    except Exception as exc:
        partial = exc.partial if isinstance(exc, PartialCaptureFailure) else None
        if partial is not None:
            _persist_observations(cell_dir, partial)
        if isinstance(exc, PartialCaptureFailure) and exc.before is not None:
            _persist_frame(cell_dir, "partial_before", exc.before)
        if isinstance(exc, PartialCaptureFailure) and exc.encoded_rgb is not None:
            np.save(cell_dir / "partial_encoded_rgb.npy", exc.encoded_rgb, allow_pickle=False)
        _set_cell_state(
            ledger, cell, "failed", error_type=type(exc).__name__, error_message=str(exc)
        )
        _json_write(_ledger_path(output_root), ledger)
        _json_write(
            cell_dir / "receipt.json",
            {
                "cell": cell.name,
                "state": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "partial_observations_retained": partial is not None,
                "partial_stage": exc.stage if isinstance(exc, PartialCaptureFailure) else None,
                "partial_encoded_rgb_retained": isinstance(exc, PartialCaptureFailure)
                and exc.encoded_rgb is not None,
            },
        )
        raise DiagnosticFailure(f"capture failed for {cell.name}") from exc
    _persist_observations(cell_dir, result)
    try:
        require_capture_provenance(result.provenance, cell)
        _validate_frame(result.before, geom_objtype=geom_objtype)
        _validate_frame(result.after, geom_objtype=geom_objtype)
        if cell in BASELINE_CELLS:
            if not np.array_equal(result.before.raw_geom_ids, originals[f"{cell.backend}_before"]):
                raise DiagnosticFailure(
                    "baseline before frame differs from trusted original decoded array"
                )
            if not np.array_equal(result.after.raw_geom_ids, originals[f"{cell.backend}_after"]):
                raise DiagnosticFailure(
                    "baseline after frame differs from trusted original decoded array"
                )
    except Exception as exc:
        _set_cell_state(
            ledger, cell, "failed", error_type=type(exc).__name__, error_message=str(exc)
        )
        _json_write(_ledger_path(output_root), ledger)
        _json_write(
            cell_dir / "receipt.json",
            {
                "cell": cell.name,
                "state": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "observations_retained": True,
            },
        )
        raise DiagnosticFailure(f"postcapture validation failed for {cell.name}") from exc
    _set_cell_state(ledger, cell, "complete")
    _json_write(_ledger_path(output_root), ledger)
    _json_write(
        cell_dir / "receipt.json",
        {
            "cell": cell.name,
            "state": "complete",
            "baseline_exact_match": cell not in BASELINE_CELLS or True,
        },
    )
    return cell
