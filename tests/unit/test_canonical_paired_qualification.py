"""CPU-only finite-runner checks; no context or real graphics is constructed."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mujoco
import numpy as np
import pytest

from epsbench.diagnostics import canonical_paired_qualification as q
from epsbench.diagnostics.revision_capture import AttemptLock
from epsbench.utils.canonical import canonical_json_bytes
from tests.unit.test_canonical_paired import fake_renderer, fake_state


def initialise(root: Path) -> None:
    q.initialise_ledger(root, {"synthetic_cpu_binding": True}, {"synthetic_cpu_runtime": True})


def test_plan_is_finite_with_no_extra_ordinary_segmentation() -> None:
    attempts = q.fixed_attempts()
    assert len(attempts) == 8 and len({attempt.name for attempt in attempts}) == 8
    assert sum(len(attempt.modalities) for attempt in attempts) == 160
    assert sum(attempt.modalities.count("canonical_pair") for attempt in attempts) == 64
    assert (
        sum(attempt.modalities.count("counterfactual_segmentation") for attempt in attempts) == 32
    )
    assert all("segmentation" not in attempt.modalities for attempt in attempts)
    assert q.plan()["limits"]["contexts"] == 32


def test_existing_output_cannot_be_reinitialized(tmp_path: Path) -> None:
    initialise(tmp_path / "study")
    with pytest.raises(q.CanonicalQualificationFailure, match="already exists"):
        initialise(tmp_path / "study")


@pytest.mark.parametrize("terminal", [None, "failed"])
def test_failure_or_orphan_cannot_continue(tmp_path: Path, terminal: str | None) -> None:
    root = tmp_path / "study"
    initialise(root)
    first = q.fixed_attempts()[0]
    q.append_revision(root, "reserved", first)
    if terminal:
        q.append_revision(root, terminal, first, {"synthetic_cpu_failure": True})
    with pytest.raises(q.CanonicalQualificationFailure, match="permanently stops"):
        q.validate_ledger(root)
    before = [path.read_bytes() for path in sorted((root / "canonical-paired-ledger").iterdir())]
    for attempt in (first, q.fixed_attempts()[1]):
        with pytest.raises(q.CanonicalQualificationFailure):
            q.append_revision(root, "reserved", attempt)
    after = [path.read_bytes() for path in sorted((root / "canonical-paired-ledger").iterdir())]
    assert before == after


def test_ledger_is_bound_and_monotonic_to_exact_completion(tmp_path: Path) -> None:
    root = tmp_path / "study"
    initialise(root)
    with pytest.raises(q.CanonicalQualificationFailure):
        q.append_revision(root, "reserved", q.fixed_attempts()[1])
    for attempt in q.fixed_attempts():
        q.append_revision(root, "reserved", attempt)
        q.append_revision(root, "complete", attempt, {"synthetic_cpu_completion": True})
    records = q.validate_ledger(root)
    assert len(records) == 17
    assert [r["attempt_ordinal"] for r in records if r["event"] == "complete"] == list(range(8))
    with pytest.raises(q.CanonicalQualificationFailure):
        q.append_revision(root, "reserved", q.fixed_attempts()[0])


@pytest.mark.parametrize("mutation", ["plan", "hash", "name", "fields", "foreign", "gap"])
def test_ledger_rejects_tampering(tmp_path: Path, mutation: str) -> None:
    root = tmp_path / "study"
    initialise(root)
    q.append_revision(root, "reserved", q.fixed_attempts()[0])
    q.append_revision(root, "complete", q.fixed_attempts()[0])
    path = root / "canonical-paired-ledger" / "revision-0001.json"
    if mutation == "foreign":
        (path.parent / "foreign.json").write_text("{}")
    elif mutation == "gap":
        path.unlink()
    elif mutation == "plan":
        path = path.parent / "revision-0000.json"
        value = json.loads(path.read_bytes())
        value["plan"]["limits"]["native_render_calls"] = 161
        path.write_bytes(canonical_json_bytes(value))
    else:
        value = json.loads(path.read_bytes())
        if mutation == "hash":
            value["predecessor_sha256"] = "0" * 64
        elif mutation == "name":
            value["attempt_name"] = "replacement"
        elif mutation == "fields":
            value["authority"] = "invented"
        path.write_bytes(canonical_json_bytes(value))
    with pytest.raises(q.CanonicalQualificationFailure):
        q.validate_ledger(root)


def test_reservation_precedes_runtime_output_and_graphics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "study"
    initialise(root)
    monkeypatch.setattr(q, "validate_source_linkage", lambda source: source)
    reached: list[str] = []

    def blocked(_backend: str) -> None:
        assert q.validate_ledger(root, allow_stopped=True)[-1]["event"] == "reserved"
        assert not (root / "datasets").exists()
        reached.append("runtime")
        raise q.CanonicalQualificationFailure("synthetic CPU preflight failure")

    monkeypatch.setattr(q, "require_supported_runtime", blocked)
    monkeypatch.setattr(
        q, "generate_attempt", lambda *_args: pytest.fail("generator must not be reached")
    )
    with pytest.raises(q.CanonicalQualificationFailure, match="preflight failure"):
        q.run_attempt(root, tmp_path, q.fixed_attempts()[0])
    assert reached == ["runtime"]
    assert (root / "canonical-paired-attempt.lock").exists()
    records = q.validate_ledger(root, allow_stopped=True)
    assert records[-1]["event"] == "failed"
    with pytest.raises(q.CanonicalQualificationFailure, match="permanently stops"):
        q.run_attempt(root, tmp_path, q.fixed_attempts()[0])


def test_orphan_lock_prevents_new_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "study"
    initialise(root)
    monkeypatch.setattr(q, "validate_source_linkage", lambda source: source)
    lock = AttemptLock(root)
    lock.path = root / "canonical-paired-attempt.lock"
    lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="already exists"):
            q.run_attempt(root, tmp_path, q.fixed_attempts()[0])
        assert len(q.validate_ledger(root)) == 1
    finally:
        import os

        assert lock.fd is not None
        os.close(lock.fd)


def test_tree_hashes_cover_volatile_file_and_reject_links(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    (root / "run.json").write_text("{}")
    snapshot = q.tree_snapshot(root)
    assert snapshot[0]["path"] == "run.json"
    (root / "run.json").write_text('{"changed":true}')
    assert q.tree_snapshot(root) != snapshot
    (root / "alias").symlink_to(root / "run.json")
    with pytest.raises(q.CanonicalQualificationFailure, match="linked"):
        q.tree_snapshot(root)


def _monitor_fixture(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, list[str]]:
    calls: list[str] = []

    def constructor(model: Any, *_args: Any, **_kwargs: Any) -> Any:
        calls.append("construct")
        renderer = fake_renderer()
        renderer._model = model
        renderer.width, renderer.height = 160, 120
        renderer._rect.width, renderer._rect.height = 160, 120
        renderer.update_scene = lambda *_args, **_kwargs: None
        renderer.close = lambda: calls.append("close")
        return renderer

    def draw(*_args: Any) -> None:
        calls.append("draw")

    def read(color: Any, depth: Any, *_args: Any) -> None:
        calls.append("read")
        color[:] = 1
        if depth is not None:
            depth[:] = 0.5

    module = SimpleNamespace(
        Renderer=constructor, mjr_render=draw, mjr_readPixels=read, mjtRndFlag=mujoco.mjtRndFlag
    )
    monkeypatch.setattr(q, "bind_pristine_sdk", lambda _module: {"synthetic_cpu_runtime": True})

    def context(renderer: Any, *_args: Any) -> dict[str, Any]:
        state = fake_state(renderer)
        return {
            **state["context_runtime"],
            "offscreen_attachments": state["offscreen_attachments"],
            "osmesa_context_identity": 77,
        }

    monkeypatch.setattr(q, "observe_shared_context", context)
    monkeypatch.setattr(q, "observe_canonical_paired_state", fake_state)
    return module, calls


def test_monitor_counts_owned_calls_and_restores_exact_entrypoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, calls = _monitor_fixture(monkeypatch)
    originals = module.Renderer, module.mjr_render, module.mjr_readPixels
    model = SimpleNamespace(vis=SimpleNamespace(quality=SimpleNamespace(offsamples=0)))
    with q.NativeMonitor(module, q.fixed_attempts()[4]) as monitor:
        for _episode in range(4):
            renderer = module.Renderer(model)
            for _pose in range(2):
                for paired in (False, True):
                    renderer.update_scene(object(), camera=0)
                    renderer._scene.flags[int(mujoco.mjtRndFlag.mjRND_SEGMENT)] = paired
                    renderer._scene.flags[int(mujoco.mjtRndFlag.mjRND_IDCOLOR)] = paired
                    module.mjr_render(renderer._rect, renderer._scene, renderer._mjr_context)
                    module.mjr_readPixels(
                        np.empty((120, 160, 3), dtype=np.uint8),
                        np.empty((120, 160), dtype=np.float32) if paired else None,
                        renderer._rect,
                        renderer._mjr_context,
                    )
            renderer.close()
    monitor.validate_complete()
    assert calls.count("construct") == calls.count("close") == 4
    assert calls.count("draw") == calls.count("read") == 16
    assert (module.Renderer, module.mjr_render, module.mjr_readPixels) == originals


def test_constructor_budget_rejects_fifth_before_native_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, calls = _monitor_fixture(monkeypatch)
    original = module.Renderer
    model = SimpleNamespace(vis=SimpleNamespace(quality=SimpleNamespace(offsamples=0)))
    with q.NativeMonitor(module, q.fixed_attempts()[4]) as monitor:
        for _ in range(4):
            module.Renderer(model)
        with pytest.raises(q.CanonicalQualificationFailure, match="budget"):
            module.Renderer(model)
        with pytest.raises(q.CanonicalQualificationFailure, match="permanently stopped"):
            module.Renderer(model)
    assert calls.count("construct") == calls.count("close") == 4
    assert module.Renderer is original and monitor.failed


def test_failed_constructor_is_accounted_and_hooks_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, calls = _monitor_fixture(monkeypatch)
    original = module.Renderer

    def fail(*_args: Any) -> Any:
        raise RuntimeError("synthetic observation failure")

    monkeypatch.setattr(q, "observe_shared_context", fail)
    with pytest.raises(RuntimeError, match="observation failure"):
        with q.NativeMonitor(module, q.fixed_attempts()[4]) as monitor:
            module.Renderer(
                SimpleNamespace(vis=SimpleNamespace(quality=SimpleNamespace(offsamples=0)))
            )
    assert calls == ["construct", "close"]
    assert module.Renderer is original
    assert monitor.constructor_attempts[0]["status"] == "failed"
    assert monitor.constructor_attempts[0]["close_complete"] is True


def test_completed_prefix_rejects_unrecorded_output(tmp_path: Path) -> None:
    root = tmp_path / "study"
    initialise(root)
    (root / "datasets" / "foreign").mkdir(parents=True)
    with pytest.raises(q.CanonicalQualificationFailure, match="membership"):
        q.validate_completed_prefix(root, q.validate_ledger(root))
