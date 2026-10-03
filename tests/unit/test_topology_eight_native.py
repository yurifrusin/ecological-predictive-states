"""Guarded CPU tests: synthetic SDK only; no actual context or capture."""

from __future__ import annotations

import copy
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from epsbench.diagnostics import topology_eight_native as n


def runtime() -> dict[str, Any]:
    return dict(
        actual_backend="osmesa",
        context_module="mujoco.osmesa",
        requested_offsamples=0,
        actual_offsamples=0,
        model_offsamples=0,
        width=160,
        height=120,
        sample_buffers=0,
        samples=0,
        mjr_off_width=160,
        mjr_off_height=120,
        color_storage_dimensions=[160, 120],
        depth_storage_dimensions=[160, 120],
        attachment_format=32856,
        attachment_component_type=35863,
        gl_samples=0,
        gl_vendor="synthetic",
        gl_renderer="synthetic llvmpipe",
        gl_version="synthetic",
    )


def fixture(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, list[str], dict[str, Any]]:
    calls: list[str] = []
    facts = runtime()
    facts.update(
        osmesa_context_identity=77,
        offscreen_attachments={"offFBO": {"framebuffer": 5}, "offFBO_r": {"present": False}},
    )

    def construct(model: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append("construct")
        return SimpleNamespace(
            width=160,
            height=120,
            _model=model,
            _rect=SimpleNamespace(width=160, height=120),
            _mjr_context=object(),
            _scene=SimpleNamespace(flags=[False, False]),
            update_scene=lambda *a, **k: None,
            close=lambda: calls.append("close"),
        )

    def draw(*args: Any) -> None:
        calls.append("draw")

    def read(color: Any, depth: Any, *args: Any) -> None:
        calls.append("read")
        color[:] = 1
        if depth is not None:
            depth[:] = 0.5

    module = SimpleNamespace(
        Renderer=construct,
        mjr_render=draw,
        mjr_readPixels=read,
        mjtRndFlag=SimpleNamespace(mjRND_SEGMENT=0, mjRND_IDCOLOR=1),
    )

    def observe(renderer: Any) -> dict[str, Any]:
        return dict(
            context_runtime=runtime(),
            actual_current_context=77,
            expected_current_context=77,
            offscreen_attachments=copy.deepcopy(facts["offscreen_attachments"]),
            read_framebuffer_binding=5,
            draw_framebuffer_binding=5,
            scene_map=[],
            segment_enabled=renderer._scene.flags[0],
            idcolor_enabled=renderer._scene.flags[1],
        )

    monkeypatch.setattr(n, "bind_pristine_sdk", lambda m: None)
    monkeypatch.setattr(n, "observe_shared_context", lambda *a: copy.deepcopy(facts))
    monkeypatch.setattr(n, "observe_canonical_paired_state", observe)
    monkeypatch.setattr(n, "stable_state", lambda s: s)
    monkeypatch.setattr(n, "validate_saved_state", lambda *a: None)
    monkeypatch.setattr(n, "counterfactual_hash", lambda *a: "synthetic-counterfactual")
    return module, calls, facts


def construct(module: Any) -> Any:
    return module.Renderer(
        SimpleNamespace(vis=SimpleNamespace(quality=SimpleNamespace(offsamples=0))),
        height=120,
        width=160,
    )


def update(renderer: Any, modality: str) -> None:
    renderer._scene.flags[:] = [modality != "rgb"] * 2
    renderer.update_scene(
        object(),
        camera="camera",
        scene_option=SimpleNamespace(
            geomgroup=[1, 0 if modality == "counterfactual_segmentation" else 1]
        ),
    )


def draw(module: Any, renderer: Any) -> None:
    module.mjr_render(renderer._rect, renderer._scene, renderer._mjr_context)


def read(module: Any, renderer: Any, paired: bool = False) -> None:
    module.mjr_readPixels(
        np.empty((120, 160, 3), dtype=np.uint8),
        np.empty((120, 160), dtype=np.float32) if paired else None,
        renderer._rect,
        renderer._mjr_context,
    )


def test_exact_six_pairs_one_context_and_hooks_restored(monkeypatch: pytest.MonkeyPatch) -> None:
    module, calls, _ = fixture(monkeypatch)
    originals = module.Renderer, module.mjr_render, module.mjr_readPixels
    monitor = n.NativeMonitor(module)
    with monitor:
        renderer = construct(module)
        for modality in n.MODALITIES:
            update(renderer, modality)
            draw(module, renderer)
            read(module, renderer, modality == "canonical_pair")
    monitor.validate_complete()
    assert calls == ["construct"] + ["draw", "read"] * 6 + ["close"]
    assert (module.Renderer, module.mjr_render, module.mjr_readPixels) == originals
    receipt = monitor.receipt()
    assert [e["modality"] for e in receipt["native_events"]] == list(n.MODALITIES)
    assert [e["frame_index"] for e in receipt["native_events"]] == [0] * 3 + [1] * 3
    assert receipt["failed_calls"] == []
    json.dumps(receipt)


@pytest.mark.parametrize(
    "bad",
    [
        "second_constructor",
        "read_without_draw",
        "wrong_modality",
        "second_draw",
        "hidden_depth",
        "wrong_destination",
        "foreign_context",
        "excess_pairs",
    ],
)
def test_rejection_precedes_native_call_and_latches(
    monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    module, calls, _ = fixture(monkeypatch)
    monitor = n.NativeMonitor(module)
    with monitor:
        renderer = construct(module)
        update(renderer, "rgb")
        if bad in {"second_draw", "hidden_depth", "wrong_destination"}:
            draw(module, renderer)
        if bad == "wrong_modality":
            renderer._scene.flags[:] = [True, True]
        if bad == "excess_pairs":
            for modality in n.MODALITIES:
                update(renderer, modality)
                draw(module, renderer)
                read(module, renderer, modality == "canonical_pair")
        before = list(calls)
        with pytest.raises(n.NativeCaptureFailure):
            if bad == "second_constructor":
                construct(module)
            elif bad == "read_without_draw":
                read(module, renderer)
            elif bad == "hidden_depth":
                read(module, renderer, True)
            elif bad == "wrong_destination":
                module.mjr_readPixels(
                    np.empty((2, 2, 3), dtype=np.uint8), None, renderer._rect, renderer._mjr_context
                )
            elif bad == "foreign_context":
                module.mjr_render(renderer._rect, renderer._scene, object())
            else:
                draw(module, renderer)
        assert calls == before
        with pytest.raises(n.NativeCaptureFailure, match="permanently stopped"):
            construct(module)
        assert calls == before and monitor.failed_calls
    assert monitor.failed and monitor.restored
    with pytest.raises(n.NativeCaptureFailure):
        monitor.validate_complete()


def test_constructor_failure_is_retained_and_no_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    module, calls, _ = fixture(monkeypatch)

    def fail(*args: Any, **kwargs: Any) -> None:
        calls.append("failed-construct")
        raise RuntimeError("synthetic constructor failure")

    module.Renderer = fail
    monitor = n.NativeMonitor(module)
    with monitor:
        with pytest.raises(RuntimeError, match="synthetic constructor"):
            construct(module)
        with pytest.raises(n.NativeCaptureFailure):
            construct(module)
    assert calls == ["failed-construct"]
    assert monitor.constructor_attempts[0]["status"] == "failed"
    assert monitor.restored and monitor.failed


def test_other_thread_latches_before_constructor(monkeypatch: pytest.MonkeyPatch) -> None:
    module, calls, _ = fixture(monkeypatch)
    errors: list[BaseException] = []
    monitor = n.NativeMonitor(module)

    def worker() -> None:
        try:
            construct(module)
        except BaseException as error:
            errors.append(error)

    with monitor:
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        with pytest.raises(n.NativeCaptureFailure):
            construct(module)
    assert errors and calls == [] and monitor.failed and monitor.restored


@pytest.mark.parametrize(
    "key,value",
    [
        ("actual_backend", "egl"),
        ("gl_renderer", "hardware"),
        ("sample_buffers", 1),
        ("width", 320),
        ("actual_offsamples", 4),
    ],
)
def test_bad_context_closed_and_retained(
    monkeypatch: pytest.MonkeyPatch, key: str, value: Any
) -> None:
    module, calls, facts = fixture(monkeypatch)
    facts[key] = value
    monitor = n.NativeMonitor(module)
    with pytest.raises(n.NativeCaptureFailure):
        with monitor:
            construct(module)
    assert calls == ["construct", "close"]
    assert monitor.constructor_attempts[0]["status"] == "failed"
    assert monitor.restored and monitor.failed


def test_runtime_prefix_drift_closes_before_draw(monkeypatch: pytest.MonkeyPatch) -> None:
    module, calls, _ = fixture(monkeypatch)
    baseline = runtime()
    baseline["gl_version"] = "different"
    monitor = n.NativeMonitor(module, baseline)
    with pytest.raises(n.NativeCaptureFailure, match="completed prefix"):
        with monitor:
            construct(module)
    assert calls == ["construct", "close"] and monitor.restored


def test_state_drift_rejected_before_read(monkeypatch: pytest.MonkeyPatch) -> None:
    module, calls, _ = fixture(monkeypatch)
    monitor = n.NativeMonitor(module)
    with monitor:
        renderer = construct(module)
        update(renderer, "rgb")
        draw(module, renderer)
        observe = n.observe_canonical_paired_state

        def drift(renderer: Any) -> Any:
            state = dict(observe(renderer))
            state["actual_current_context"] = 88
            return state

        monkeypatch.setattr(n, "observe_canonical_paired_state", drift)
        with pytest.raises(n.NativeCaptureFailure, match="provenance drift"):
            read(module, renderer)
    assert calls == ["construct", "draw", "close"] and monitor.failed


def test_native_draw_failure_accounted_and_hooks_restored(monkeypatch: pytest.MonkeyPatch) -> None:
    module, calls, _ = fixture(monkeypatch)

    def fail(*args: Any) -> None:
        calls.append("draw-failed")
        raise RuntimeError("synthetic draw failure")

    module.mjr_render = fail
    monitor = n.NativeMonitor(module)
    with pytest.raises(RuntimeError, match="synthetic draw"):
        with monitor:
            renderer = construct(module)
            update(renderer, "rgb")
            draw(module, renderer)
    assert calls == ["construct", "draw-failed", "close"]
    assert monitor.events[0]["render_calls"] == 1 and monitor.events[0]["readback_calls"] == 0
    assert monitor.failed_calls and monitor.restored


def test_capture_rejects_nonmembership_without_graphics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        n, "prepare_graphics_environment", lambda: pytest.fail("graphics preparation reached")
    )
    with pytest.raises(n.NativeCaptureFailure, match="membership"):
        n.capture_cell({}, tmp_path / "dataset")
    assert not (tmp_path / "dataset").exists()


def test_eight_cells_count_actual_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    module, calls, _ = fixture(monkeypatch)
    baseline: dict[str, object] | None = None
    contexts = renders = reads = 0
    for _ in range(8):
        monitor = n.NativeMonitor(module, baseline)
        with monitor:
            renderer = construct(module)
            for modality in n.MODALITIES:
                update(renderer, modality)
                draw(module, renderer)
                read(module, renderer, modality == "canonical_pair")
        monitor.validate_complete()
        baseline = n.context_runtime_binding(monitor.contexts[0])
        contexts += len(monitor.contexts)
        renders += sum(e["render_calls"] for e in monitor.events)
        reads += sum(e["readback_calls"] for e in monitor.events)
    assert (contexts, renders, reads) == (8, 48, 48)
    assert calls.count("construct") == calls.count("close") == 8


def test_environment_is_set_before_import_and_drift_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(n, "sys", SimpleNamespace(modules={}))
    monkeypatch.setattr(n, "_GRAPHICS_PREPARED", False)
    for key in ("MUJOCO_GL", "PYOPENGL_PLATFORM"):
        monkeypatch.delenv(key, raising=False)
    n.prepare_graphics_environment()
    assert os.environ["MUJOCO_GL"] == os.environ["PYOPENGL_PLATFORM"] == "osmesa"
    monkeypatch.setattr(n, "sys", SimpleNamespace(modules={"mujoco": object()}))
    n.prepare_graphics_environment()
    monkeypatch.delenv("MUJOCO_GL")
    with pytest.raises(n.NativeCaptureFailure, match="backend environment"):
        n.prepare_graphics_environment()


def test_preimported_graphics_rejected_without_env_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(n, "sys", SimpleNamespace(modules={"mujoco": object()}))
    monkeypatch.setattr(n, "_GRAPHICS_PREPARED", False)
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    with pytest.raises(n.NativeCaptureFailure, match="graphics imported"):
        n.prepare_graphics_environment()
    assert "MUJOCO_GL" not in os.environ


def test_update_failure_stops_future_native_access(monkeypatch: pytest.MonkeyPatch) -> None:
    module, calls, _ = fixture(monkeypatch)
    monitor = n.NativeMonitor(module)
    with monitor:
        renderer = construct(module)

        def fail(*args: Any, **kwargs: Any) -> None:
            calls.append("update-failed")
            raise RuntimeError("synthetic update failure")

        renderer._renderer.update_scene = fail
        with pytest.raises(RuntimeError, match="synthetic update"):
            update(renderer, "rgb")
        with pytest.raises(n.NativeCaptureFailure):
            update(renderer, "rgb")
    assert calls == ["construct", "update-failed", "close"]
    assert monitor.failed_calls[0]["operation"] == "update_scene"


def test_cleanup_failure_is_retained_and_restores_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    module, calls, _ = fixture(monkeypatch)
    originals = module.Renderer, module.mjr_render, module.mjr_readPixels
    monitor = n.NativeMonitor(module)
    with pytest.raises(RuntimeError, match="synthetic close"):
        with monitor:
            renderer = construct(module)

            def fail() -> None:
                calls.append("close-failed")
                raise RuntimeError("synthetic close failure")

            renderer._renderer.close = fail
    assert (module.Renderer, module.mjr_render, module.mjr_readPixels) == originals
    assert monitor.failed and not monitor.cleanup_complete and monitor.restored
    assert monitor.failed_calls[-1]["operation"] == "cleanup"


@pytest.mark.parametrize("fail", [False, True])
def test_adapter_uses_fixed_producer_arguments_and_retains_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fail: bool
) -> None:
    import sys

    module, calls, _ = fixture(monkeypatch)
    source = Path(n.__file__).resolve().parents[3]
    membership = json.loads(
        (source / "docs/protocols/topology-eight-transition-v1-membership.json").read_bytes()
    )
    cell = membership["records"][0]
    producer_arguments: list[Any] = []

    def generate(config: Any, episodes: int, output: Path, **kwargs: Any) -> None:
        producer_arguments.append((config, episodes, output, kwargs))
        renderer = construct(module)
        if fail:
            raise RuntimeError("synthetic producer failure")
        for modality in n.MODALITIES:
            update(renderer, modality)
            draw(module, renderer)
            read(module, renderer, modality == "canonical_pair")

    monkeypatch.setattr(n, "prepare_graphics_environment", lambda: None)
    monkeypatch.setitem(sys.modules, "mujoco", module)
    monkeypatch.setitem(
        sys.modules, "epsbench.data.generate", SimpleNamespace(generate_dataset=generate)
    )
    monkeypatch.setitem(
        sys.modules,
        "epsbench.appearance",
        SimpleNamespace(
            load_appearance_registry=lambda path: "synthetic-appearance",
            load_evaluation_seed_registry=lambda path: "synthetic-seeds",
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "epsbench.sim.canonical_paired",
        SimpleNamespace(
            require_supported_runtime=lambda backend: calls.append("runtime-" + backend)
        ),
    )
    monkeypatch.setattr(n, "validate_saved_native", lambda dataset, receipt: runtime())
    if fail:
        with pytest.raises(RuntimeError, match="synthetic producer") as raised:
            n.capture_cell(cell, tmp_path / "dataset", source=source)
        receipt = raised.value.native_receipt  # type: ignore[attr-defined]
        assert receipt["failed"] and receipt["constructor_restored"]
        assert receipt["cleanup_complete"]
    else:
        receipt = n.capture_cell(cell, tmp_path / "dataset", source=source)
        assert receipt["runtime_binding"] == runtime()
        assert len(receipt["native_events"]) == 6
    config, episodes, output, kwargs = producer_arguments[0]
    assert config.model_dump(mode="json") == cell["resolved_configuration"]
    assert episodes == 1 and output == tmp_path / "dataset"
    assert kwargs == {
        "appearance_registry": "synthetic-appearance",
        "seed_registry": "synthetic-seeds",
        "component_topology": False,
        "capture_mode": "canonical_paired",
    }
    assert calls[0] == "runtime-osmesa"
    assert not output.exists()


@pytest.mark.parametrize("renderer_name", ["softpipe", "software", "swrast"])
def test_other_software_renderers_rejected_before_progress(
    monkeypatch: pytest.MonkeyPatch, renderer_name: str
) -> None:
    module, calls, facts = fixture(monkeypatch)
    facts["gl_renderer"] = renderer_name
    monitor = n.NativeMonitor(module)
    with monitor:
        with pytest.raises(n.NativeCaptureFailure, match="llvmpipe"):
            construct(module)
        assert calls == ["construct", "close"]
        assert monitor.contexts == [] and monitor.events == [] and monitor.proxies == []
        assert monitor.constructor_attempts[0]["status"] == "failed"
        assert monitor.constructor_attempts[0]["close_complete"] is True
        with pytest.raises(n.NativeCaptureFailure, match="permanently stopped"):
            construct(module)
        assert calls == ["construct", "close"]
    assert monitor.failed and monitor.restored
    with pytest.raises(n.NativeCaptureFailure):
        monitor.validate_complete()
