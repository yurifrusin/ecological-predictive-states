from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from epsbench.diagnostics.revision_capture import (
    ArtifactWriter,
    AttemptLock,
    RevisionCaptureFailure,
    StudyCell,
    append_revision,
    canonical_json_bytes,
    fixed_cells,
    initialise_ledger,
    load_cell_result,
    publish_analysis_result,
    publish_bytes,
    translate_study_root,
    validate_ledger,
    verify_handoff_records,
)
from epsbench.diagnostics.revision_mujoco import _study_depth_attachments
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

        def attachment(samples: int, name: int, *, depth: bool = False) -> dict[str, int]:
            return {
                "object_type": 36161,
                "object_name": name,
                "component_type": 5126 if depth else 35863,
                "internal_format": 36013 if depth else 32856,
                "samples": samples,
            }

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
                    "color0": attachment(self.samples, 1),
                    "depth": attachment(self.samples, 2, depth=True),
                },
                "offFBO_r": {
                    "present": self.samples > 0,
                    "draw_framebuffer_samples": 0,
                    **(
                        {
                            "color0": attachment(0, 3),
                            "depth": attachment(0, 4, depth=True),
                        }
                        if self.samples > 0
                        else {}
                    ),
                },
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


@pytest.mark.parametrize("malformed", [np.nan, np.inf, -np.inf])
def test_nonfinite_depth_is_retained_before_schema_failure(
    tmp_path: Path, malformed: float
) -> None:
    def factory(samples: int, role: str) -> FakeStack:
        stack = FakeStack(samples, role)
        stack.render_depth = lambda: np.full(  # type: ignore[method-assign]
            (2, 3), malformed, np.float32
        )
        return stack

    with pytest.raises(StageFailure) as caught:
        capture_cell(
            StudyCell(0, "corridor", 0, "wgl", "joint4"),
            prepared(),
            factory,
            tmp_path,
            geom_objtype=5,
        )
    assert caught.value.stage == "before.depth_schema"
    receipt_text = (tmp_path / "receipt.json").read_text(encoding="utf-8")
    assert "NaN" not in receipt_text and "Infinity" not in receipt_text
    receipt = json.loads(receipt_text)
    assert receipt["poses"][0]["modalities"]["rgb"]["validated"] is True
    depth_ref = receipt["poses"][0]["modalities"]["depth"]
    assert depth_ref["complete"] is False and depth_ref["validated"] is False
    retained = np.load(tmp_path / depth_ref["path"], allow_pickle=False)
    assert np.all(np.isfinite(retained) == np.isfinite(malformed))


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
    receipt = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["renderers"][0]["revision_validation"]["status"] == "validated"


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


@pytest.mark.parametrize(
    "attachment,field,value",
    [
        ("color0", "object_type", 0),
        ("color0", "internal_format", 1),
        ("color0", "component_type", True),
        ("depth", "internal_format", 32856),
        ("depth", "component_type", 35863),
    ],
)
def test_unsupported_attachment_facts_fail_closed_and_are_retained(
    tmp_path: Path, attachment: str, field: str, value: object
) -> None:
    def factory(samples: int, role: str) -> FakeStack:
        stack = FakeStack(samples, role)
        original = stack.provenance

        def invalid() -> dict[str, object]:
            provenance = original()
            records = provenance["offscreen_attachments"]
            assert isinstance(records, dict)
            main = records["offFBO"]
            assert isinstance(main, dict)
            facts = main[attachment]
            assert isinstance(facts, dict)
            facts[field] = value
            return provenance

        stack.provenance = invalid  # type: ignore[method-assign]
        return stack

    with pytest.raises(StageFailure) as caught:
        capture_cell(
            StudyCell(0, "corridor", 0, "wgl", "joint4"),
            prepared(),
            factory,
            tmp_path,
            geom_objtype=5,
        )
    assert caught.value.stage == "primary.provenance_validation"
    renderer = json.loads((tmp_path / "receipt.json").read_text())["renderers"][0]
    assert renderer["offscreen_attachments"]["offFBO"][attachment][field] == value
    assert renderer["revision_validation"]["status"] == "invalid"


def test_invalid_provenance_is_retained_as_strict_json(tmp_path: Path) -> None:
    def factory(samples: int, role: str) -> FakeStack:
        stack = FakeStack(samples, role)
        original = stack.provenance

        def invalid() -> dict[str, object]:
            value = original()
            value.pop("gl_vendor")
            value["invalid_observation"] = np.nan
            return value

        stack.provenance = invalid  # type: ignore[method-assign]
        return stack

    with pytest.raises(StageFailure):
        capture_cell(
            StudyCell(0, "corridor", 0, "wgl", "joint4"),
            prepared(),
            factory,
            tmp_path,
            geom_objtype=5,
        )
    text = (tmp_path / "receipt.json").read_text(encoding="utf-8")
    assert "NaN" not in text
    renderer = json.loads(text)["renderers"][0]
    assert renderer["revision_validation"]["status"] == "invalid"
    assert renderer["invalid_observation"] == {"revision_nonfinite_float": "nan"}


def test_model_hash_mismatch_provenance_is_retained(tmp_path: Path) -> None:
    def factory(samples: int, role: str) -> FakeStack:
        stack = FakeStack(samples, role)
        original = stack.provenance

        def mismatched() -> dict[str, object]:
            value = original()
            value["model_mjb_sha256"] = "9" * 64
            return value

        stack.provenance = mismatched  # type: ignore[method-assign]
        return stack

    with pytest.raises(StageFailure):
        capture_cell(
            StudyCell(0, "corridor", 0, "wgl", "joint4"),
            prepared(),
            factory,
            tmp_path,
            geom_objtype=5,
        )
    renderer = json.loads((tmp_path / "receipt.json").read_text())["renderers"][0]
    assert renderer["model_mjb_sha256"] == "9" * 64
    assert renderer["revision_validation"]["status"] == "invalid"


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
                    "host": "DESKTOP-TPUQMNG",
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
                    "host": "DESKTOP-TPUQMNG",
                    "own_token": values["token"],
                    "peer_token": sources[peer]["token"],
                    "expected_peer_root": sources[peer]["observed_root"],
                    "observed_root": values["observed_root"],
                }
            ),
        )

    def translate(value: str, runtime: str) -> str:
        return sources["wsl" if runtime == "windows" else "windows"]["observed_root"]

    result = verify_handoff_records(
        tmp_path,
        current_runtime="wsl",
        current_host="DESKTOP-TPUQMNG",
        current_root="/mnt/c/study",
        translator=translate,
    )
    assert result["windows_root"] == r"C:\study"
    (directory / "verified-wsl.json").write_text("{}")
    with pytest.raises(RevisionCaptureFailure, match="invalid handoff"):
        verify_handoff_records(
            tmp_path,
            current_runtime="wsl",
            current_host="DESKTOP-TPUQMNG",
            current_root="/mnt/c/study",
            translator=translate,
        )


def test_handoff_rejects_wrong_runtime_host_root_and_translation(tmp_path: Path) -> None:
    directory = tmp_path / "handoff"
    sources = {
        "windows": {"token": "win", "observed_root": r"C:\study"},
        "wsl": {"token": "linux", "observed_root": "/mnt/c/study"},
    }
    for runtime, values in sources.items():
        peer = "wsl" if runtime == "windows" else "windows"
        publish_bytes(
            directory / f"{runtime}.json",
            canonical_json_bytes(
                {
                    "schema": "revision_capture_handoff/v1",
                    "runtime": runtime,
                    "host": "DESKTOP-TPUQMNG",
                    **values,
                }
            ),
        )
        publish_bytes(
            directory / f"verified-{runtime}.json",
            canonical_json_bytes(
                {
                    "schema": "revision_capture_handoff_verification/v1",
                    "runtime": runtime,
                    "host": "DESKTOP-TPUQMNG",
                    "own_token": values["token"],
                    "peer_token": sources[peer]["token"],
                    "expected_peer_root": sources[peer]["observed_root"],
                    "observed_root": values["observed_root"],
                }
            ),
        )
    valid = dict(
        current_runtime="wsl",
        current_host="DESKTOP-TPUQMNG",
        current_root="/mnt/c/study",
        translator=lambda value, runtime: sources["wsl" if runtime == "windows" else "windows"][
            "observed_root"
        ],
    )
    for override in (
        {"current_runtime": "windows"},
        {"current_host": "OTHER"},
        {"current_root": "/mnt/c/other"},
        {"translator": lambda value, runtime: "/mnt/c/wrong"},
    ):
        with pytest.raises(RevisionCaptureFailure):
            verify_handoff_records(tmp_path, **(valid | override))


@pytest.mark.parametrize(
    "executing_runtime,source_runtime,expected_executable,expected_flag,translated",
    [
        ("windows", "windows", "wsl.exe", "-u", "/mnt/c/study"),
        ("windows", "wsl", "wsl.exe", "-w", r"C:\study"),
        ("wsl", "windows", "wslpath", "-u", "/mnt/c/study"),
        ("wsl", "wsl", "wslpath", "-w", r"C:\study"),
    ],
)
def test_translation_uses_executing_runtime_and_direction(
    monkeypatch: pytest.MonkeyPatch,
    executing_runtime: str,
    source_runtime: str,
    expected_executable: str,
    expected_flag: str,
    translated: str,
) -> None:
    seen: list[object] = []

    def check_output(command: list[str], **options: object) -> str:
        seen.extend((command, options))
        return translated + "\n"

    monkeypatch.setattr("subprocess.check_output", check_output)
    source = r"C:\study" if source_runtime == "windows" else "/mnt/c/study"
    assert (
        translate_study_root(source, source_runtime, executing_runtime=executing_runtime)
        == translated
    )
    command = seen[0]
    assert isinstance(command, list)
    assert command[0] == expected_executable and expected_flag in command
    options = seen[1]
    assert isinstance(options, dict)
    assert options["timeout"] == 10
    assert options["stderr"] is not None


def test_depth_attachment_inspection_restores_gl_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeGL:
        GL_READ_FRAMEBUFFER_BINDING = 1
        GL_DRAW_FRAMEBUFFER_BINDING = 2
        GL_RENDERBUFFER_BINDING = 3
        GL_READ_FRAMEBUFFER = 4
        GL_DRAW_FRAMEBUFFER = 5
        GL_DEPTH_ATTACHMENT = 6
        GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE = 7
        GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME = 8
        GL_FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE = 9
        GL_RENDERBUFFER = 36161
        GL_RENDERBUFFER_INTERNAL_FORMAT = 10
        GL_RENDERBUFFER_SAMPLES = 11

        def __init__(self) -> None:
            self.read = 101
            self.draw = 102
            self.renderbuffer = 103

        def glGetIntegerv(self, parameter: int) -> int:
            return {
                self.GL_READ_FRAMEBUFFER_BINDING: self.read,
                self.GL_DRAW_FRAMEBUFFER_BINDING: self.draw,
                self.GL_RENDERBUFFER_BINDING: self.renderbuffer,
            }[parameter]

        def glBindFramebuffer(self, target: int, value: int) -> None:
            if target == self.GL_READ_FRAMEBUFFER:
                self.read = value
            else:
                self.draw = value

        def glGetFramebufferAttachmentParameteriv(
            self, target: int, attachment: int, parameter: int
        ) -> int:
            assert target == self.GL_DRAW_FRAMEBUFFER
            assert attachment == self.GL_DEPTH_ATTACHMENT
            if parameter == self.GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE:
                return self.GL_RENDERBUFFER
            if parameter == self.GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME:
                return {10: 2, 20: 4}[self.draw]
            return 5126

        def glBindRenderbuffer(self, target: int, value: int) -> None:
            assert target == self.GL_RENDERBUFFER
            self.renderbuffer = value

        def glGetRenderbufferParameteriv(self, target: int, parameter: int) -> int:
            assert target == self.GL_RENDERBUFFER
            if parameter == self.GL_RENDERBUFFER_INTERNAL_FORMAT:
                return 36013
            return {2: 4, 4: 0}[self.renderbuffer]

    gl = FakeGL()
    monkeypatch.setitem(sys.modules, "OpenGL", SimpleNamespace(GL=gl))
    renderer = SimpleNamespace(_mjr_context=SimpleNamespace(offFBO=10, offFBO_r=20))
    records = _study_depth_attachments(renderer)
    assert records["offFBO"]["samples"] == 4
    assert records["offFBO_r"]["samples"] == 0
    assert (gl.read, gl.draw, gl.renderbuffer) == (101, 102, 103)


def test_attempt_lock_closes_before_successful_unlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    terminal = tmp_path / "terminal.json"
    terminal.write_text("{}", encoding="utf-8")
    lock = AttemptLock(tmp_path)
    lock.acquire()
    closed = False
    original_close = os.close
    original_unlink = Path.unlink

    def observed_close(fd: int) -> None:
        nonlocal closed
        original_close(fd)
        closed = True

    def observed_unlink(path: Path, *args: object, **kwargs: object) -> None:
        assert closed
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "close", observed_close)
    monkeypatch.setattr(Path, "unlink", observed_unlink)
    lock.release_after_success(terminal)
    assert not lock.path.exists()


def test_attempt_lock_retains_replacement_after_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    terminal = tmp_path / "terminal.json"
    terminal.write_text("{}", encoding="utf-8")
    tampered = AttemptLock(tmp_path)
    tampered.acquire()
    original_close = os.close

    def replace_after_close(fd: int) -> None:
        original_close(fd)
        tampered.path.unlink()
        tampered.path.write_text("foreign", encoding="utf-8")

    monkeypatch.setattr(os, "close", replace_after_close)
    with pytest.raises(RevisionCaptureFailure, match="identity changed"):
        tampered.release_after_success(terminal)
    assert tampered.path.read_text(encoding="utf-8") == "foreign"


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
