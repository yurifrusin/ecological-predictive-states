"""Guarded CPU fixtures for saved public topology and finite control comparisons."""

from __future__ import annotations

import builtins
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from epsbench.data.loader import DatasetLoader, PermissionDeniedError
from epsbench.diagnostics.topology_eight_assessment import (
    CELL_IDS,
    PROTOCOL,
    PublicTopologyInputs,
    _direction_relation,
    _privileged_assessment,
    compare_study,
    derive_public_artifacts,
    load_public_inputs,
    validate_public_artifacts,
)
from epsbench.schema import ModalityPermissionSet, SurfaceReference
from epsbench.utils.canonical import canonical_json_bytes


def inputs(
    *, reverse: bool = False, supported: bool = True, identity: str = "1"
) -> PublicTopologyInputs:
    before = np.asarray([[1, 1, 1]], dtype=np.int32)
    after = np.asarray([[1, 0, 1]], dtype=np.int32)
    vectors = np.zeros((1, 3, 2), dtype=np.int32)
    validity = np.asarray([[1, 0, 1]] if supported else [[0, 0, 0]], dtype=np.uint8)
    reasons = np.where(validity == 1, 0, 4).astype(np.uint8)
    invalid = np.zeros((1, 3), dtype=np.uint8)
    return PublicTopologyInputs(
        (after, before) if reverse else (before, after),
        (SurfaceReference(surface_id="surface-" + identity * 16, segmentation_label=1),),
        (vectors.copy(), vectors.copy()),
        (invalid, validity) if reverse else (validity, invalid),
        (np.full((1, 3), 4, dtype=np.uint8), reasons)
        if reverse
        else (reasons, np.full((1, 3), 4, dtype=np.uint8)),
        (invalid.copy(), invalid.copy()),
        "0" * 64,
    )


def private(identity: str = "1", alternate: bool = False, reverse: bool = False) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "background_surface_id": "surface-" + identity * 16,
        "semantic_to_opaque": {
            "background_surface": "surface-" + identity * 16,
            "support_surface": "surface-" + "2" * 16,
            "occluding_surface": "surface-" + "3" * 16,
        },
        "rgb_hashes": [("b" if alternate else "a") * 64] * 2,
        "native_hashes": [["0" * 64, "0" * 64]] * 2,
        "producer_states": [{}, {}],
        "camera_states": [
            {
                "frame_index": f,
                "camera_world_position": [0.0, 0.0, 0.0],
                "camera_world_rotation_row_major": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            }
            for f in (0, 1)
        ],
        "action": {
            "name": "lateral_left" if reverse else "lateral_right",
            "delta_forward": 0.0,
            "delta_lateral": -1.0 if reverse else 1.0,
            "delta_yaw": 0.0,
        },
        "raw_world_coordinates": {},
        "raw_geom_ids": {},
        "endpoint_reconstruction": "passed",
    }


def study(root: Path, *, supported: bool = True) -> None:
    for ordinal, cell in enumerate(CELL_IDS):
        derive_public_artifacts(
            inputs(reverse=ordinal >= 4, supported=supported), root / "exploratory" / cell
        )
        dataset = root / "datasets" / cell
        dataset.mkdir(parents=True)
        (dataset / "deterministic.bin").write_bytes(b"same")
        (dataset / "run.json").write_text(str(ordinal))
        (root / "privileged").mkdir(exist_ok=True)
        (root / "privileged" / f"{cell}.json").write_bytes(
            canonical_json_bytes(private(alternate=ordinal in (2, 3, 6, 7), reverse=ordinal >= 4))
        )


def test_complete_saved_maps_support_and_independent_corruption_rejection(tmp_path: Path) -> None:
    output = tmp_path / "public"
    annotation = derive_public_artifacts(inputs(), output)
    assert annotation == validate_public_artifacts(output)
    assert [s.forward_count for s in annotation.supports] == [1, 1]
    assert all(s.backward_count == 0 for s in annotation.supports)
    assert len(list(output.glob("*.npy"))) == 12
    corrupted = np.asarray([[1, 0, 0]], dtype=np.int32)
    np.save(output / "components_1.npy", corrupted)
    with pytest.raises(ValueError, match="bytes differ"):
        validate_public_artifacts(output)


def test_zero_support_pairs_and_indeterminate_preserved(tmp_path: Path) -> None:
    annotation = derive_public_artifacts(inputs(supported=False), tmp_path / "public")
    assert len(annotation.supports) == 2
    assert all(s.forward_count == s.backward_count == 0 and not s.edge for s in annotation.supports)
    assert annotation.status == "indeterminate"
    assert len(annotation.events) == 3


def test_permission_denied_before_artifact_or_graphics_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = object.__new__(DatasetLoader)
    loader.permissions = ModalityPermissionSet.ecological_only()

    def sentinel(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("privileged artifact/graphics was accessed")

    monkeypatch.setattr(loader, "_transition", sentinel)
    monkeypatch.setattr(loader, "_instrumentation", sentinel)
    monkeypatch.setattr(builtins, "open", sentinel)
    for operation in (
        lambda: loader.read_rgb(0, 0),
        lambda: loader.read_depth(0, 0),
        lambda: loader.read_camera_world_transform(0, 0),
        lambda: loader.read_raw_mujoco_geom_ids(0),
        lambda: loader.read_canonical_paired_output(0, 0),
        lambda: _privileged_assessment(loader, inputs()),
    ):
        with pytest.raises(PermissionDeniedError):
            operation()
    loader.permissions = ModalityPermissionSet.all_modalities()
    with pytest.raises(PermissionDeniedError, match="exactly ecological"):
        load_public_inputs(loader)


def test_public_import_and_derivation_never_import_renderer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original_import = builtins.__import__

    def guarded(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith(("mujoco", "OpenGL", "glfw", "epsbench.sim")):
            raise AssertionError(f"renderer graph imported: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    importlib.reload(sys.modules["epsbench.diagnostics.topology_eight_assessment"])
    derive_public_artifacts(inputs(), tmp_path / "public")


def test_exact_eight_positive_and_negative_controls(tmp_path: Path) -> None:
    study(tmp_path)
    report = compare_study(tmp_path)
    assert report["hypothesis_positive_bounded"] is True
    assert report["phase_gate_effect"] == "NONE"
    assert all(
        pair["directional_counts_exchange_equal_descriptive"] for pair in report["direction_pairs"]
    )
    (tmp_path / "datasets" / CELL_IDS[1] / "extra.bin").write_bytes(b"extra")
    report = compare_study(tmp_path)
    assert report["hypothesis_positive_bounded"] is False
    assert report["repeat_pairs"][0]["dataset_equal_except_run_json"] is False
    with pytest.raises(ValueError, match="exact ordered"):
        compare_study(tmp_path, CELL_IDS[:-1])


def test_valid_negative_completes_and_rgb_control_disagreement_retained(tmp_path: Path) -> None:
    study(tmp_path, supported=False)
    report = compare_study(tmp_path)
    assert report["hypothesis_positive_bounded"] is False
    assert len(report["target_cells"]) == 8
    assert all(cell["overall_capability"] == "indeterminate" for cell in report["target_cells"])
    path = tmp_path / "privileged" / f"{CELL_IDS[2]}.json"
    payload = json.loads(path.read_bytes())
    payload["rgb_hashes"] = ["a" * 64] * 2
    path.write_bytes(canonical_json_bytes(payload))
    assert compare_study(tmp_path)["appearance_pairs"][0]["rgb_changed_both_endpoints"] is False


def test_reversal_uses_masks_after_surface_correspondence_not_frame_ids(tmp_path: Path) -> None:
    forward = derive_public_artifacts(inputs(), tmp_path / "forward")
    reverse = derive_public_artifacts(inputs(reverse=True, identity="4"), tmp_path / "reverse")
    result = _direction_relation(forward, reverse, private(), private(identity="4"))
    assert result["components_exchange_equal"] is True
    assert result["events_exchange_equal"] is True
    assert result["support_edges_exchange_equal"] is True
    assert forward.portable_graph_sha256 != reverse.portable_graph_sha256
