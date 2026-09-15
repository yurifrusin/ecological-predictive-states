from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from epsbench.diagnostics.revision_capture import (
    ArtifactWriter,
    RevisionCaptureFailure,
    StudyCell,
    append_revision,
    canonical_json_bytes,
    fixed_cells,
    initialise_ledger,
    load_cell_result,
    publish_analysis_result,
    publish_bytes,
    validate_ledger,
    verify_handoff_records,
)
from epsbench.diagnostics.revision_runner import (
    PreparedEpisode,
    StageFailure,
    capture_cell,
    run_attempt,
)


class FakeStack:
    width = 3
    height = 2

    def __init__(
        self,
        samples: int,
        role: str,
        backend: str = "wgl",
        fail: str | set[str] | None = None,
    ):
        self.samples = samples
        self.role = role
        self.backend = backend
        self.fail = fail
        self.calls: list[str] = []
        self.pose = "before"

    def hit(self, stage: str) -> None:
        self.calls.append(stage)
        if self.fail == stage or (isinstance(self.fail, set) and stage in self.fail):
            raise RuntimeError(stage)

    def make_current(self) -> None:
        self.hit("make_current")

    def set_pose(self, pose_name: str) -> dict[str, object]:
        self.hit(f"set_pose:{pose_name}")
        self.pose = pose_name
        return pose_facts()[pose_name]

    def update_scene(self) -> None:
        self.hit(f"update:{self.pose}")

    def render_rgb(self) -> np.ndarray:
        self.hit(f"rgb:{self.pose}")
        return np.full((2, 3, 3), 7, np.uint8)

    def enable_depth(self) -> None:
        self.hit("depth_enable")

    def render_depth(self) -> np.ndarray:
        self.hit(f"depth:{self.pose}")
        return np.full((2, 3), 2.0, np.float32)

    def disable_depth(self) -> None:
        self.hit("depth_disable")

    def enable_segmentation(self) -> None:
        self.hit("seg_enable")

    def scene_map(self) -> dict[int, tuple[int, int]]:
        self.hit("scene_map")
        return {0: (3, 5)}

    def render_segmentation(self, out: np.ndarray) -> np.ndarray:
        out[..., 0] = 1
        self.hit(f"seg:{self.pose}")
        return np.full((2, 3, 2), (3, 5), np.int32)

    def disable_segmentation(self) -> None:
        self.hit("seg_disable")

    def provenance(self) -> dict[str, object]:
        self.hit("provenance")
        return {
            "role": self.role,
            "requested_offsamples": self.samples,
            "actual_offsamples": self.samples,
            "actual_backend": self.backend,
            "gl_vendor": "v",
            "gl_renderer": "r",
            "gl_version": "g",
            "offscreen_attachments": {
                "offFBO": {
                    "present": True,
                    "draw_framebuffer_samples": self.samples,
                    "color0": {"samples": self.samples},
                },
                "offFBO_r": {"present": self.samples > 0, "draw_framebuffer_samples": 0},
            },
            "model_stat_extent": 2.0,
            "model_vis_map_znear": 0.01,
            "model_vis_map_zfar": 50.0,
            "model_mjb_sha256": "a" * 64,
            "python_version": "3",
            "mujoco_version": "3",
            "numpy_version": "2",
            "pyopengl_version": "3",
            "glfw_version": "2",
            "package_sha256": "b" * 64,
            "binary_sha256": "c" * 64,
            "renderer_py_sha256": "d" * 64,
        }

    def close(self) -> None:
        self.hit("close")


def pose_facts() -> dict[str, dict[str, object]]:
    return {
        "before": {
            "camera_position": [0.0, 0.0, 1.0],
            "camera_rotation": np.eye(3).tolist(),
            "fovy_degrees": 55.0,
            "state_sha256": "e" * 64,
        },
        "after": {
            "camera_position": [0.0, 1.0, 1.0],
            "camera_rotation": np.eye(3).tolist(),
            "fovy_degrees": 55.0,
            "state_sha256": "f" * 64,
        },
    }


def prepared() -> PreparedEpisode:
    return PreparedEpisode(
        episode_seed=2**61 + 3,
        source={"head": "1" * 40, "tree": "2" * 40},
        config={"family": "corridor", "root_seed": 1729},
        model={
            "normalized_mjb_sha256": "3" * 64,
            "semantic_facts_sha256": "4" * 64,
            "actual_models": {"primary": "a" * 64, "segmentation": "a" * 64},
        },
        geometry_facts={"finite_planes": [], "oriented_boxes": []},
        pose_facts=pose_facts(),
    )


def test_fixed_matrix_exact_order_and_budget() -> None:
    cells = fixed_cells()
    assert len(cells) == 48
    assert cells[0].name.endswith("wgl-joint4")
    assert cells[1].name.endswith("osmesa-joint4")
    assert cells[2].name.endswith("wgl-hybrid")
    assert cells[-1].name.endswith("osmesa-joint0")
    assert sum(2 if cell.policy == "hybrid" else 1 for cell in cells) == 64
    assert len(cells) * 2 * 3 == 288


def test_ledger_is_hash_chained_and_stopped_tail_blocks(tmp_path: Path) -> None:
    root = tmp_path / "study"
    initialise_ledger(root, {"archive_sha256": "a" * 64})
    cell = fixed_cells()[0]
    append_revision(root, "reserved", cell)
    with pytest.raises(RevisionCaptureFailure, match="permanently stops"):
        validate_ledger(root)
    append_revision(root, "failed", cell, {"reason": "synthetic"})
    with pytest.raises(RevisionCaptureFailure, match="permanently stops"):
        validate_ledger(root)
    with pytest.raises(RevisionCaptureFailure, match="stopped ledger"):
        append_revision(root, "reserved", cell)


def test_writer_refuses_replacement_and_loads_arrays(tmp_path: Path) -> None:
    writer = ArtifactWriter(tmp_path)
    ref = writer.array("x.npy", np.array([[1]], dtype=">i4"), validated=True)
    assert ref["dtype"] == "<i4"
    with pytest.raises(FileExistsError):
        writer.array("x.npy", np.array([[2]], dtype=np.int32))


@pytest.mark.parametrize(
    "policy,expected_stacks",
    [
        ("joint4", 1),
        ("hybrid", 2),
        ("joint0", 1),
    ],
)
def test_capture_policies_and_result_interface(
    tmp_path: Path,
    policy: str,
    expected_stacks: int,
) -> None:
    made: list[FakeStack] = []

    def factory(samples: int, role: str) -> FakeStack:
        stack = FakeStack(samples, role)
        made.append(stack)
        return stack

    cell = StudyCell(0, "corridor", 0, "wgl", policy)
    receipt = capture_cell(cell, prepared(), factory, tmp_path, geom_objtype=5)
    assert len(made) == expected_stacks
    assert receipt["state"] == "complete"
    loaded = load_cell_result(tmp_path)
    assert loaded["identity"]["episode_seed"] == 2**61 + 3
    assert len(loaded["poses"]) == 2
    for pose in loaded["poses"]:
        assert pose["modalities"]["rgb"].shape == (2, 3, 3)
        assert pose["modalities"]["depth"].dtype == np.float32
        seg = pose["modalities"]["segmentation"]
        assert seg["encoded_rgb"].shape == (2, 3, 3)
        assert seg["decoded_pairs"].shape == (2, 3, 2)
        assert seg["raw_geom_ids"].shape == (2, 3)
        assert seg["segid_map"] == [[0, 3, 5]]
        assert seg["readback_requires_vertical_flip"] is True
    for stack in made:
        assert stack.calls.count("make_current") >= len(stack.calls) // 2


def test_decoder_failure_retains_mutated_caller_buffer(tmp_path: Path) -> None:
    stack: FakeStack | None = None

    def factory(samples: int, role: str) -> FakeStack:
        nonlocal stack
        stack = FakeStack(samples, role, fail="seg:before")
        return stack

    cell = StudyCell(0, "corridor", 0, "wgl", "joint4")
    with pytest.raises(StageFailure):
        capture_cell(cell, prepared(), factory, tmp_path, geom_objtype=5)
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    encoded = receipt["poses"][0]["modalities"]["segmentation"]["encoded_rgb"]
    assert encoded["complete"] is False
    retained = np.load(tmp_path / encoded["path"], allow_pickle=False)
    assert np.all(retained[..., 0] == 1)
    assert receipt["state"] == "failed"
    assert stack is not None and "close" in stack.calls


@pytest.mark.parametrize(
    "fail_stage",
    [
        "provenance",
        "set_pose:before",
        "rgb:before",
        "depth:before",
        "scene_map",
        "seg:before",
        "set_pose:after",
        "rgb:after",
        "close",
    ],
)
def test_each_failure_stage_publishes_partial_receipt(
    tmp_path: Path,
    fail_stage: str,
) -> None:
    made: list[FakeStack] = []

    def factory(samples: int, role: str) -> FakeStack:
        stack = FakeStack(samples, role, fail=fail_stage)
        made.append(stack)
        return stack

    cell = StudyCell(0, "corridor", 0, "wgl", "joint4")
    with pytest.raises(StageFailure):
        capture_cell(cell, prepared(), factory, tmp_path, geom_objtype=5)
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["state"] == "failed"
    assert receipt["stage_events"]
    if fail_stage in {"set_pose:after", "rgb:after", "close"}:
        assert len(receipt["poses"]) == 2 or fail_stage == "close"
        assert (tmp_path / "before-rgb.npy").exists()


def test_wrong_backend_is_integrity_failure_with_cleanup(tmp_path: Path) -> None:
    made: list[FakeStack] = []

    def factory(samples: int, role: str) -> FakeStack:
        stack = FakeStack(samples, role, backend="osmesa")
        made.append(stack)
        return stack

    with pytest.raises(StageFailure):
        capture_cell(
            StudyCell(0, "corridor", 0, "wgl", "joint4"),
            prepared(),
            factory,
            tmp_path,
            geom_objtype=5,
        )
    assert "close" in made[0].calls


def test_second_context_failure_retains_and_cleans_primary(tmp_path: Path) -> None:
    primary = FakeStack(4, "primary")

    def factory(samples: int, role: str) -> FakeStack:
        if role == "segmentation":
            raise RuntimeError("second context")
        return primary

    with pytest.raises(StageFailure):
        capture_cell(
            StudyCell(0, "corridor", 0, "wgl", "hybrid"),
            prepared(),
            factory,
            tmp_path,
            geom_objtype=5,
        )
    assert "close" in primary.calls


def test_preparation_failure_is_durably_failed_and_locked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "study"
    initialise_ledger(root, {"archive_sha256": "fixed"})
    monkeypatch.setattr(
        "epsbench.diagnostics.revision_runner.verify_input_archive",
        lambda path: {"archive_sha256": "fixed"},
    )
    monkeypatch.setattr(
        "epsbench.diagnostics.revision_runner.verify_handoff_records",
        lambda root: {"windows_root": "C:\\study", "wsl_root": "/mnt/c/study"},
    )

    def fail_prepare() -> tuple[PreparedEpisode, Any]:
        raise RuntimeError("CPU preparation failed")

    with pytest.raises(RuntimeError, match="CPU preparation failed"):
        run_attempt(
            root,
            tmp_path / "packet.zip",
            fixed_cells()[0],
            fail_prepare,
            geom_objtype=5,
        )
    assert (root / "capture-attempt.lock").is_file()
    records = validate_ledger(root, allow_stopped_tail=True)
    assert records[-1]["event"] == "failed"
    with pytest.raises(RevisionCaptureFailure, match="permanently stops"):
        validate_ledger(root)


def test_wrong_live_fbo_samples_fail_before_capture(tmp_path: Path) -> None:
    made: list[FakeStack] = []

    def factory(samples: int, role: str) -> FakeStack:
        stack = FakeStack(samples, role)
        original = stack.provenance

        def bad_provenance() -> dict[str, object]:
            value = original()
            attachments = value["offscreen_attachments"]
            assert isinstance(attachments, dict)
            main = attachments["offFBO"]
            assert isinstance(main, dict)
            main["draw_framebuffer_samples"] = samples + 1
            return value

        stack.provenance = bad_provenance  # type: ignore[method-assign]
        made.append(stack)
        return stack

    with pytest.raises(StageFailure):
        capture_cell(
            StudyCell(0, "corridor", 0, "wgl", "joint4"),
            prepared(),
            factory,
            tmp_path,
            geom_objtype=5,
        )
    assert "rgb:before" not in made[0].calls
    assert "close" in made[0].calls


def test_handoff_requires_cross_checked_tokens_and_roots(tmp_path: Path) -> None:
    directory = tmp_path / "handoff"
    sources = {
        "windows": {"token": "win", "observed_root": r"C:\study"},
        "wsl": {"token": "linux", "observed_root": "/mnt/c/study"},
    }
    for runtime, values in sources.items():
        publish_bytes(
            directory / f"{runtime}.json",
            canonical_json_bytes(
                {
                    "schema": "revision_capture_handoff/v1",
                    "runtime": runtime,
                    **values,
                }
            ),
        )
        peer = "wsl" if runtime == "windows" else "windows"
        publish_bytes(
            directory / f"verified-{runtime}.json",
            canonical_json_bytes(
                {
                    "schema": "revision_capture_handoff_verification/v1",
                    "runtime": runtime,
                    "own_token": values["token"],
                    "peer_token": sources[peer]["token"],
                    "expected_peer_root": sources[peer]["observed_root"],
                }
            ),
        )
    result = verify_handoff_records(tmp_path)
    assert result["windows_root"] == r"C:\study"
    (directory / "verified-wsl.json").write_text("{}")
    with pytest.raises(RevisionCaptureFailure, match="invalid handoff"):
        verify_handoff_records(tmp_path)


def test_foreign_ledger_entry_and_gap_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "foreign"
    initialise_ledger(root, {})
    (root / "ledger" / "note.txt").write_text("foreign", encoding="utf-8")
    with pytest.raises(RevisionCaptureFailure, match="foreign ledger"):
        validate_ledger(root)
    root2 = tmp_path / "gap"
    initialise_ledger(root2, {})
    publish_bytes(
        root2 / "ledger" / "revision-0002.json",
        canonical_json_bytes({"schema": "wrong"}),
    )
    with pytest.raises(RevisionCaptureFailure, match="gap"):
        validate_ledger(root2)


def test_cleanup_exception_never_replaces_original_failure(tmp_path: Path) -> None:
    def factory(samples: int, role: str) -> FakeStack:
        return FakeStack(samples, role, fail={"rgb:before", "close"})

    with pytest.raises(StageFailure) as caught:
        capture_cell(
            StudyCell(0, "corridor", 0, "wgl", "joint4"),
            prepared(),
            factory,
            tmp_path,
            geom_objtype=5,
        )
    assert caught.value.stage == "before.rgb_readback"
    receipt = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["failure"]["stage"] == "before.rgb_readback"
    cleanup = [event for event in receipt["stage_events"] if event["stage"] == "primary.cleanup"]
    assert cleanup[-1]["state"] == "failed"


def test_analysis_publication_separates_nan_arrays_from_finite_json(
    tmp_path: Path,
) -> None:
    reference = publish_analysis_result(
        tmp_path,
        {
            "summary": {"count": 0, "mean": None},
            "analytic_no_hit_depth": np.array([[np.nan]], dtype=np.float64),
        },
    )
    report = json.loads((tmp_path / reference["path"]).read_text(encoding="utf-8"))
    array_ref = report["analytic_no_hit_depth"]
    assert "NaN" not in (tmp_path / reference["path"]).read_text(encoding="utf-8")
    value = np.load(tmp_path / array_ref["path"], allow_pickle=False)
    assert np.isnan(value[0, 0])
    with pytest.raises(FileExistsError):
        publish_analysis_result(tmp_path, {"again": np.array([1])})
