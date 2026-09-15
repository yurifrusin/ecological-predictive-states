from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from epsbench.diagnostics import renderer_readback
from epsbench.diagnostics.renderer_readback import (
    GL_ENUMS,
    InstrumentedStack,
    append_revision,
    fixed_cells,
    initialise_ledger,
    inverse_reverse_z,
    validate_ledger,
)
from epsbench.diagnostics.revision_capture import ArtifactWriter, RevisionCaptureFailure


class FakeGL:
    def __init__(self, *, matrix_nan: bool = False):
        self.matrix_nan = matrix_nan

    def glIsEnabled(self, enum: int) -> bool:
        assert enum == GL_ENUMS["multisample"]
        return True

    def glGetIntegerv(self, enum: int) -> object:
        values: dict[int, object] = {
            GL_ENUMS["sample_buffers"]: 1,
            GL_ENUMS["samples"]: 4,
            GL_ENUMS["read_framebuffer_binding"]: 10,
            GL_ENUMS["draw_framebuffer_binding"]: 10,
            GL_ENUMS["read_buffer"]: 0x8CE0,
            GL_ENUMS["draw_buffer"]: 0x8CE0,
            GL_ENUMS["clip_depth_mode"]: 0x935F,
            GL_ENUMS["clip_origin"]: 0x8CA1,
            GL_ENUMS["depth_func"]: 0x0204,
            GL_ENUMS["viewport"]: [0, 0, 160, 120],
            GL_ENUMS["subpixel_bits"]: 8,
        }
        return values[enum]

    def glGetDoublev(self, enum: int) -> object:
        if enum == GL_ENUMS["depth_range"]:
            return [0.0, 1.0]
        value = np.eye(4, dtype=np.float64).reshape(-1, order="F")
        if self.matrix_nan:
            value[0] = np.nan
        return value

    def glGetFloatv(self, enum: int) -> object:
        return np.asarray(self.glGetDoublev(enum), dtype=np.float32)

    def glGetMultisamplefv(self, enum: int, index: int) -> object:
        assert enum == GL_ENUMS["sample_position"]
        return [0.125 + index * 0.25, 0.375]

    def glGetError(self) -> int:
        return 0


class FakeBase:
    role = "primary"
    width = 160
    height = 120

    def __init__(self, *, fail_read: bool = False):
        self.fail_read = fail_read
        self.native_render_count = 0
        self.mujoco = SimpleNamespace()
        self.renderer = SimpleNamespace(
            _rect=object(),
            _scene=SimpleNamespace(
                flags=np.array([1, 0], dtype=np.uint8),
                frustum_near=0.1,
                frustum_far=100.0,
                frustum_top=1.0,
                frustum_bottom=-1.0,
                frustum_center=0.0,
                frustum_width=1.0,
                camera=[
                    SimpleNamespace(
                        pos=np.zeros(3),
                        forward=np.array([0.0, 0.0, -1.0]),
                        up=np.array([0.0, 1.0, 0.0]),
                        frustum_near=0.1,
                        frustum_far=100.0,
                        frustum_top=1.0,
                        frustum_bottom=-1.0,
                        frustum_center=0.0,
                        frustum_width=1.0,
                    )
                ],
            ),
            _mjr_context=SimpleNamespace(readDepthMap=1, offSamples=4),
        )
        self.renderer.scene = self.renderer._scene

        def render(rect: object, scene: object, context: object) -> None:
            self.native_render_count += 1

        def read(rgb: object, depth: object, rect: object, context: object) -> None:
            if rgb is not None:
                np.asarray(rgb).reshape(-1)[0] = 7
            if depth is not None:
                np.asarray(depth)[:] = np.float32(0.5)
            if self.fail_read:
                raise RuntimeError("read failed")

        self.mujoco.mjr_render = render
        self.mujoco.mjr_readPixels = read
        renderer_readback.bind_sdk_entrypoints(self.mujoco, require_native=False)

    def make_current(self) -> None:
        pass

    def set_pose(self, pose_name: str) -> dict[str, object]:
        return {}

    def update_scene(self) -> None:
        pass

    def render_rgb(self) -> np.ndarray:
        out = np.zeros((120, 160, 3), np.uint8)
        self.mujoco.mjr_render(
            self.renderer._rect, self.renderer._scene, self.renderer._mjr_context
        )
        self.mujoco.mjr_readPixels(out, None, self.renderer._rect, self.renderer._mjr_context)
        return out

    def enable_depth(self) -> None:
        pass

    def render_depth(self) -> np.ndarray:
        out = np.zeros((120, 160), np.float32)
        self.mujoco.mjr_render(
            self.renderer._rect, self.renderer._scene, self.renderer._mjr_context
        )
        self.mujoco.mjr_readPixels(None, out, self.renderer._rect, self.renderer._mjr_context)
        return inverse_reverse_z(out, 0.01, 10.0)

    def disable_depth(self) -> None:
        pass

    def enable_segmentation(self) -> None:
        pass

    def scene_map(self) -> dict[int, tuple[int, int]]:
        return {0: (1, 5)}

    def render_segmentation(self, out: np.ndarray) -> np.ndarray:
        self.mujoco.mjr_render(
            self.renderer._rect, self.renderer._scene, self.renderer._mjr_context
        )
        self.mujoco.mjr_readPixels(out, None, self.renderer._rect, self.renderer._mjr_context)
        return np.zeros((120, 160, 2), np.int32)

    def disable_segmentation(self) -> None:
        pass

    def provenance(self) -> dict[str, object]:
        return {}

    def close(self) -> None:
        pass


def fixed_plan() -> dict[str, object]:
    return {
        "review_profile": "DUAL_REVIEW",
        "evidence_class": "PUBLIC_REPOSITORY_ONLY",
        "phase_gate_effect": "NONE",
        "archive_sha256": renderer_readback.ARCHIVE_SHA256,
        "manifest_sha256": renderer_readback.MANIFEST_SHA256,
        "source_head": "1" * 40,
        "source_tree": "2" * 40,
        "dependency_lock_sha256": "3" * 64,
        "sdk_version": renderer_readback.SDK_VERSION,
        "sdk_renderer_sha256": renderer_readback.SDK_RENDERER_SHA256,
        "root_seed": 1729,
        "render": {"width": 160, "height": 120},
        "host": renderer_readback.STUDY_HOST,
        "runtimes": ["windows", "wsl"],
        "config_sha256": {
            "corridor": "4" * 64,
            "single_occluder": "5" * 64,
        },
        "cells": [cell.name for cell in fixed_cells()],
        "maximum_renderer_contexts": 8,
        "maximum_pose_endpoints": 16,
        "maximum_modality_render_calls": 48,
        "maximum_sdk_readbacks": 48,
    }


def test_fixed_schedule_and_call_budget() -> None:
    cells = fixed_cells()
    assert len(cells) == 8
    assert [(c.family, c.episode_index, c.backend, c.policy) for c in cells] == [
        ("corridor", 1, "wgl", "joint4"),
        ("corridor", 1, "osmesa", "joint4"),
        ("corridor", 1, "wgl", "joint0"),
        ("corridor", 1, "osmesa", "joint0"),
        ("single_occluder", 0, "wgl", "joint4"),
        ("single_occluder", 0, "osmesa", "joint4"),
        ("single_occluder", 0, "wgl", "joint0"),
        ("single_occluder", 0, "osmesa", "joint0"),
    ]
    assert len(cells) * 2 * 3 == 48


def test_scoped_hooks_capture_raw_depth_and_restore_originals(tmp_path: Path) -> None:
    base = FakeBase()
    render, read = base.mujoco.mjr_render, base.mujoco.mjr_readPixels
    stack = InstrumentedStack(base, ArtifactWriter(tmp_path), FakeGL(), 1)
    stack.set_pose("before")
    result = stack.render_depth()
    assert result.shape == (120, 160)
    assert base.mujoco.mjr_render is render
    assert base.mujoco.mjr_readPixels is read
    event = stack.observations[0]
    assert event["render_calls"] == event["readback_calls"] == 1
    assert event["originals_restored"] is True
    assert event["post_render_state"]["stage"] == "post_render"
    assert (
        event["raw_depth_window_copy_identity"]["bitwise_identical_and_argument_unmodified"] is True
    )
    raw = np.load(tmp_path / "readback-before-depth-raw-depth_window.npy", allow_pickle=False)
    assert raw.dtype == np.float32 and np.all(raw == np.float32(0.5))


def test_failure_preserves_observation_and_restores_originals(tmp_path: Path) -> None:
    base = FakeBase(fail_read=True)
    render, read = base.mujoco.mjr_render, base.mujoco.mjr_readPixels
    stack = InstrumentedStack(base, ArtifactWriter(tmp_path), FakeGL(), 1)
    stack.set_pose("after")
    with pytest.raises(RuntimeError, match="read failed"):
        stack.render_rgb()
    assert base.mujoco.mjr_render is render
    assert base.mujoco.mjr_readPixels is read
    assert stack.observations[0]["render_calls"] == 1
    assert stack.observations[0]["readback_calls"] == 1
    observation = json.loads((tmp_path / "readback-after-rgb-observation.json").read_text("utf-8"))
    assert observation["raw_color"]["complete"] is False
    assert np.load(tmp_path / "readback-after-rgb-raw-color.npy", allow_pickle=False).flat[0] == 7


def test_nonfinite_matrix_is_retained_before_rejection(tmp_path: Path) -> None:
    base = FakeBase()
    stack = InstrumentedStack(base, ArtifactWriter(tmp_path), FakeGL(matrix_nan=True), 1)
    stack.set_pose("before")
    with pytest.raises(RevisionCaptureFailure, match="projection_matrix"):
        stack.render_rgb()
    retained = np.load(
        tmp_path / "readback-before-rgb-projection_matrix-gl-double.npy",
        allow_pickle=False,
    )
    assert np.isnan(retained[0])


def test_reservation_and_failure_permanently_block_retry(tmp_path: Path) -> None:
    root = tmp_path / "probe"
    initialise_ledger(root, fixed_plan())
    cell = fixed_cells()[0]
    append_revision(root, "reserved", cell)
    append_revision(root, "failed", cell, {"reason": "synthetic"})
    with pytest.raises(RevisionCaptureFailure, match="permanently stops"):
        validate_ledger(root)
    with pytest.raises(RevisionCaptureFailure, match="cannot continue"):
        append_revision(root, "reserved", cell)


def test_reverse_z_inverse_roundtrip_uses_sdk_precision_order() -> None:
    near, far = 0.01, 50.0
    metric = np.array([near, 0.1, 1.0, 10.0, far], dtype=np.float32)
    zfar, znear = np.float32(far), np.float32(near)
    c = np.float32(-0.5) * (-(zfar + znear) / (zfar - znear)) - np.float32(0.5)
    d = np.float32(-0.5) * (-(np.float32(2) * zfar * znear) / (zfar - znear))
    raw = (d / metric.astype(np.float64) - c).astype(np.float32)
    reconstructed = inverse_reverse_z(raw, near, far)
    assert np.allclose(reconstructed, metric, rtol=2e-6, atol=1e-7)


def test_matrix_layout_is_explicit_and_finite(tmp_path: Path) -> None:
    base = FakeBase()
    stack = InstrumentedStack(base, ArtifactWriter(tmp_path), FakeGL(), 1)
    stack.set_pose("before")
    stack.render_rgb()
    state = stack.observations[0]["post_render_state"]
    projection = state["queries"]["projection_matrix"]
    assert projection["layout"] == "GL column-major flat16"
    assert projection["finite_and_float_cast_consistent"] is True
    matrix = np.load(tmp_path / projection["math_row_column"]["path"], allow_pickle=False)
    assert matrix.shape == (4, 4) and np.array_equal(matrix, np.eye(4))


def _npy_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    return stream.getvalue()


def test_reference_archive_binds_every_declared_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"fixed"
    entries = [
        {
            "path": "data/value.bin",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    ]
    manifest = json.dumps({"entries": entries}, separators=(",", ":")).encode()
    archive_path = tmp_path / "reference.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("data/value.bin", payload)
        archive.writestr("evidence-manifest.json", manifest)
    monkeypatch.setattr(renderer_readback, "ARCHIVE_BYTES", archive_path.stat().st_size)
    monkeypatch.setattr(
        renderer_readback, "ARCHIVE_SHA256", hashlib.sha256(archive_path.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(renderer_readback, "MANIFEST_SHA256", hashlib.sha256(manifest).hexdigest())
    assert renderer_readback.verify_reference_archive(archive_path) == {
        "data/value.bin": entries[0]["sha256"]
    }


def test_exact_selected_output_comparison_detects_perturbation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = fixed_cells()[0]
    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    old_root = f"study/cells/{renderer_readback._reference_cell_name(cell)}"
    archive_path = tmp_path / "selected.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for pose in ("before", "after"):
            arrays = {
                "rgb": np.zeros((2, 3, 3), np.uint8),
                "depth": np.ones((2, 3), np.float32),
                "encoded-rgb": np.zeros((2, 3, 3), np.uint8),
                "decoded-pairs": np.zeros((2, 3, 2), np.int32),
                "raw-geom-ids": np.zeros((2, 3), np.int32),
            }
            for suffix, value in arrays.items():
                (cell_dir / f"{pose}-{suffix}.npy").write_bytes(_npy_bytes(value))
                archive.writestr(f"{old_root}/{pose}-{suffix}.npy", _npy_bytes(value))
            mapping = b"[]"
            (cell_dir / f"{pose}-segid-map.json").write_bytes(mapping)
            archive.writestr(f"{old_root}/{pose}-segid-map.json", mapping)
    monkeypatch.setattr(renderer_readback, "verify_reference_archive", lambda path: {})
    assert len(renderer_readback.exact_reference_comparison(archive_path, cell, cell_dir)) == 12
    np.save(cell_dir / "before-depth.npy", np.zeros((2, 3), np.float32), allow_pickle=False)
    with pytest.raises(RevisionCaptureFailure, match="differs from fixed PR26"):
        renderer_readback.exact_reference_comparison(archive_path, cell, cell_dir)
