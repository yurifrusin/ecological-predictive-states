from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import epsbench.diagnostics.osmesa_joint0_qualification as q


class FakeNative:
    def __init__(self, *, fail: bool = False, duplicate: bool = False) -> None:
        self.fail = fail
        self.duplicate = duplicate
        self.mjr_render = self.native_render
        self.mjr_readPixels = self.native_read
        self.Renderer = self.construct
        self.closed = 0

    def native_render(self, rect: object, scene: object, context: object) -> None:
        if self.fail:
            raise RuntimeError("native failure")

    def native_read(self, rgb: object, depth: object, rect: object, context: object) -> None:
        return None

    def construct(self, model: object, *args: object, **kwargs: object) -> FakeRenderer:
        return FakeRenderer(self)


class FakeRenderer:
    def __init__(self, sdk: FakeNative) -> None:
        self.sdk = sdk
        self._rect, self._scene, self._mjr_context = object(), object(), object()
        self._depth_rendering = False
        self._segmentation_rendering = False
        self.width, self.height = 160, 120

    def update_scene(self, data: object, *args: object, **kwargs: object) -> None:
        return None

    def render(self, *args: object, **kwargs: object) -> np.ndarray:
        self.sdk.mjr_render(self._rect, self._scene, self._mjr_context)
        if self.sdk.duplicate:
            self.sdk.mjr_render(self._rect, self._scene, self._mjr_context)
        if self._depth_rendering:
            self.sdk.mjr_readPixels(
                None, np.zeros((1,), dtype=np.float32), self._rect, self._mjr_context
            )
        else:
            self.sdk.mjr_readPixels(
                np.zeros((1,), dtype=np.uint8), None, self._rect, self._mjr_context
            )
        return np.zeros((1,), dtype=np.uint8)

    def enable_depth_rendering(self) -> None:
        self._depth_rendering = True

    def disable_depth_rendering(self) -> None:
        self._depth_rendering = False

    def enable_segmentation_rendering(self) -> None:
        self._segmentation_rendering = True

    def disable_segmentation_rendering(self) -> None:
        self._segmentation_rendering = False

    def close(self) -> None:
        self.sdk.closed += 1


def model() -> object:
    return SimpleNamespace(vis=SimpleNamespace(quality=SimpleNamespace(offsamples=4)))


def bind(fake: FakeNative) -> None:
    q._PRISTINE[id(fake)] = (fake.Renderer, fake.mjr_render, fake.mjr_readPixels)


def observe(renderer: object, model: object, width: int, height: int) -> dict[str, object]:
    assert model.vis.quality.offsamples == 0
    return {
        "actual_backend": "osmesa",
        "context_module": "mujoco.osmesa",
        "requested_offsamples": 0,
        "actual_offsamples": 0,
        "model_offsamples": 0,
        "width": width,
        "height": height,
        "sample_buffers": 0,
        "samples": 0,
        "mjr_off_width": 160,
        "mjr_off_height": 120,
        "color_storage_dimensions": [160, 120],
        "depth_storage_dimensions": [160, 120],
        "attachment_format": 1,
        "attachment_component_type": 2,
        "gl_samples": 0,
        "gl_vendor": "Mesa",
        "gl_renderer": "llvmpipe",
        "gl_version": "4.5",
    }


def one_render(proxy: object, modality: str, *, counterfactual: bool = False) -> None:
    proxy.update_scene(
        object(), scene_option=SimpleNamespace(geomgroup=np.array([1, 0 if counterfactual else 1]))
    )
    if modality == "depth":
        proxy.enable_depth_rendering()
    elif "segmentation" in modality:
        proxy.enable_segmentation_rendering()
    proxy.render()
    if modality == "depth":
        proxy.disable_depth_rendering()
    elif "segmentation" in modality:
        proxy.disable_segmentation_rendering()


def test_fixed_plan_and_call_budget() -> None:
    attempts = q.fixed_attempts()
    assert [a.expected_native_calls for a in attempts] == [32] * 4 + [24] * 4
    assert q.plan()["limits"] == {
        "batch_attempts": 8,
        "contexts": 32,
        "ordinary_pose_endpoints": 64,
        "ordinary_modality_renders": 192,
        "counterfactual_segmentation_renders": 32,
        "native_render_calls": 224,
        "native_readbacks": 224,
    }


@pytest.mark.parametrize("family", ["single_occluder", "corridor"])
def test_adapter_four_contexts_sequence_and_restoration(family: str) -> None:
    fake = FakeNative()
    bind(fake)
    attempt = next(a for a in q.fixed_attempts() if a.family == family)
    with q.RendererAdapter(fake, observe, attempt) as adapter:
        for _episode in range(4):
            proxy = fake.Renderer(model(), width=160, height=120)
            for _pose in range(2):
                one_render(proxy, "rgb")
                one_render(proxy, "depth")
                one_render(proxy, "segmentation")
                if family == "single_occluder":
                    one_render(proxy, "counterfactual_segmentation", counterfactual=True)
            proxy.close()
    adapter.validate_complete()
    assert fake.Renderer is q._PRISTINE[id(fake)][0]
    assert fake.closed == 4


def test_duplicate_native_call_rejected_before_second_original_and_restored() -> None:
    fake = FakeNative(duplicate=True)
    bind(fake)
    with q.RendererAdapter(fake, observe, q.fixed_attempts()[0]) as adapter:
        proxy = fake.Renderer(model(), width=160, height=120)
        proxy.update_scene(object())
        with pytest.raises(q.QualificationFailure, match="duplicate"):
            proxy.render()
    assert adapter.native_events[0]["render_calls"] == 1
    assert adapter.native_events[0]["restored"] is True


def test_native_failure_retained_and_hooks_restored() -> None:
    fake = FakeNative(fail=True)
    bind(fake)
    with q.RendererAdapter(fake, observe, q.fixed_attempts()[0]) as adapter:
        proxy = fake.Renderer(model(), width=160, height=120)
        proxy.update_scene(object())
        with pytest.raises(RuntimeError, match="native failure"):
            proxy.render()
    event = adapter.native_events[0]
    assert event["error"] == {"type": "RuntimeError", "message": "native failure"}
    assert event["restored"] is True


def test_wrong_thread_and_reentrancy_rejected_before_native() -> None:
    fake = FakeNative()
    bind(fake)
    with q.RendererAdapter(fake, observe, q.fixed_attempts()[0]) as adapter:
        proxy = fake.Renderer(model(), width=160, height=120)
        proxy.update_scene(object())
        proxy._rendering = True
        with pytest.raises(q.QualificationFailure, match="thread/reentrancy"):
            proxy.render()
        proxy._rendering = False
        errors: list[BaseException] = []
        thread = threading.Thread(target=lambda: _capture(errors, proxy.render))
        thread.start()
        thread.join()
    assert isinstance(errors[0], q.QualificationFailure)
    assert adapter.native_events == []


def _capture(errors: list[BaseException], operation: object) -> None:
    try:
        operation()
    except BaseException as exc:
        errors.append(exc)


def test_observation_failure_closes_constructed_renderer() -> None:
    fake = FakeNative()
    bind(fake)
    with q.RendererAdapter(
        fake, lambda *args: (_ for _ in ()).throw(ValueError("facts")), q.fixed_attempts()[0]
    ):
        with pytest.raises(ValueError, match="facts"):
            fake.Renderer(model(), width=160, height=120)
    assert fake.closed == 1


def test_fifth_context_rejected_without_constructor_call() -> None:
    fake = FakeNative()
    bind(fake)
    with q.RendererAdapter(fake, observe, q.fixed_attempts()[0]):
        proxies = [fake.Renderer(model(), width=160, height=120) for _ in range(4)]
        with pytest.raises(q.QualificationFailure, match="fifth"):
            fake.Renderer(model(), width=160, height=120)
        for proxy in proxies:
            proxy.close()
    assert fake.closed == 4


def test_ledger_chain_reservation_completion_and_no_retry(tmp_path: Path) -> None:
    q.initialise_ledger(tmp_path / "run", {"binding": "x"})
    attempt = q.fixed_attempts()[0]
    q.append_revision(tmp_path / "run", "reserved", attempt)
    with pytest.raises(q.QualificationFailure, match="permanently stops"):
        q.validate_ledger(tmp_path / "run")
    q.append_revision(tmp_path / "run", "failed", attempt, {"reason": "x"})
    with pytest.raises(q.QualificationFailure, match="stopped"):
        q.append_revision(tmp_path / "run", "reserved", q.fixed_attempts()[1])


def test_ledger_plan_and_chain_falsification_rejected(tmp_path: Path) -> None:
    root = tmp_path / "run"
    first = q.initialise_ledger(root, {"binding": "x"})
    value = json.loads(first.read_text())
    value["plan"]["seed"] = 1
    first.write_text(json.dumps(value))
    with pytest.raises(q.QualificationFailure):
        q.validate_ledger(root)


def test_repeat_identity_is_bitwise_and_excludes_only_run_json(tmp_path: Path) -> None:
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    np.save(left / "x.npy", np.array([0.0], dtype=np.float32), allow_pickle=False)
    np.save(right / "x.npy", np.array([-0.0], dtype=np.float32), allow_pickle=False)
    (left / "run.json").write_text("left")
    (right / "run.json").write_text("right")
    with pytest.raises(q.QualificationFailure, match="content bytes"):
        q.compare_repeat_artifacts(left, right)


def test_counterfactual_requires_geomgroup_evidence() -> None:
    fake = FakeNative()
    bind(fake)
    with q.RendererAdapter(fake, observe, q.fixed_attempts()[0]) as adapter:
        proxy = fake.Renderer(model(), width=160, height=120)
        one_render(proxy, "rgb")
        one_render(proxy, "depth")
        proxy.enable_segmentation_rendering()
        proxy.update_scene(object(), scene_option=SimpleNamespace(geomgroup=np.array([1, 1])))
        proxy.render()
        proxy.update_scene(object(), scene_option=SimpleNamespace(geomgroup=np.array([1, 0])))
        proxy.render()
        proxy.close()
    assert [x["modality"] for x in adapter.native_events][-2:] == [
        "segmentation",
        "counterfactual_segmentation",
    ]


def test_wrong_modality_rejected_before_native_call() -> None:
    fake = FakeNative()
    bind(fake)
    with q.RendererAdapter(fake, observe, q.fixed_attempts()[0]) as adapter:
        proxy = fake.Renderer(model(), width=160, height=120)
        proxy.enable_depth_rendering()
        proxy.update_scene(object())
        with pytest.raises(q.QualificationFailure, match="modality sequence"):
            proxy.render()
        proxy.close()
    assert adapter.native_events == []


def test_context_runtime_drift_closes_context_before_render() -> None:
    fake = FakeNative()
    bind(fake)

    def drift(renderer: object, model: object, width: int, height: int) -> dict[str, object]:
        facts = observe(renderer, model, width, height)
        facts["gl_renderer"] = "different"
        return facts

    baseline_model = model()
    baseline_model.vis.quality.offsamples = 0
    baseline = q.context_runtime_binding(observe(object(), baseline_model, 160, 120))
    with q.RendererAdapter(fake, drift, q.fixed_attempts()[0], baseline):
        with pytest.raises(q.QualificationFailure, match="runtime differs"):
            fake.Renderer(model(), width=160, height=120)
    assert fake.closed == 1


def test_artifact_manifest_detects_post_capture_mutation(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    artifact = dataset / "value.bin"
    artifact.write_bytes(b"before")
    recorded = q.artifact_manifest(dataset)
    artifact.write_bytes(b"after")
    assert q.artifact_manifest(dataset) != recorded


def test_rehashed_failed_then_reserved_history_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "run"
    q.initialise_ledger(root, {"binding": "x"})
    first = q.fixed_attempts()[0]
    q.append_revision(root, "reserved", first)
    q.append_revision(root, "failed", first)
    second = q.fixed_attempts()[1]
    third = {
        "schema": q.LEDGER_SCHEMA,
        "revision": 3,
        "predecessor_sha256": q.digest_file(root / "ledger/revision-0002.json"),
        "event": "reserved",
        "attempt_ordinal": second.ordinal,
        "attempt_name": second.name,
        "details": {},
    }
    q.publish_bytes(root / "ledger/revision-0003.json", q.canonical(third))
    with pytest.raises(q.QualificationFailure, match="failed qualification event"):
        q.validate_ledger(root, allow_stopped=True)


def test_receipt_artifact_binding_rejects_post_capture_change(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    artifact = dataset / "value.bin"
    artifact.write_bytes(b"original")
    receipt_binding = q.artifact_manifest(dataset)
    artifact.write_bytes(b"altered")
    with pytest.raises(q.QualificationFailure, match="differs from receipt"):
        q.validate_artifact_binding(dataset, receipt_binding)


def test_source_binding_mutation_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    expected = {
        "head": "a",
        "tree": "b",
        "lock_sha256": q.LOCK_SHA256,
        "config_sha256": {},
        "registry_sha256": "c",
        "seed_registry_sha256": "d",
    }
    changed = dict(expected)
    changed["head"] = "different"
    monkeypatch.setattr(q, "git_binding", lambda *args, **kwargs: changed)
    with pytest.raises(q.QualificationFailure, match="binding changed"):
        q.validate_bindings(tmp_path, expected)


def test_ecological_permission_denial_precedes_transition_io() -> None:
    from epsbench.data import DatasetLoader, PermissionDeniedError
    from epsbench.schema import ModalityPermissionSet

    loader = DatasetLoader.__new__(DatasetLoader)
    loader.permissions = ModalityPermissionSet.ecological_only()
    calls: list[str] = []

    def protected(*args: object) -> object:
        calls.append("transition")
        raise AssertionError("protected I/O reached")

    loader._transition = protected
    with pytest.raises(PermissionDeniedError):
        loader.read_depth(0, 0)
    assert calls == []


def test_validate_plan_rejects_any_mutation() -> None:
    changed = q.plan()
    changed["quality"] = {"offsamples": 4}
    with pytest.raises(q.QualificationFailure):
        q.validate_plan(changed)


def test_first_bind_rejects_preexisting_public_wrapper() -> None:
    original = object()
    classic = SimpleNamespace(Renderer=original)
    native_render, native_read = object(), object()
    extension = SimpleNamespace(mjr_render=native_render, mjr_readPixels=native_read)
    public = SimpleNamespace(
        Renderer=lambda: None, mjr_render=native_render, mjr_readPixels=native_read
    )
    with pytest.raises(q.QualificationFailure, match="installed classic SDK class"):
        q.verify_pristine_entrypoints(public, classic, extension)
