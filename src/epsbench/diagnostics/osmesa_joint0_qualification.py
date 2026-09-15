"""Bounded candidate-only OSMesa joint0 qualification; execution is fail-closed."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

STATUS = "PARTIAL_EVIDENCE_ALIGNMENT_UNSPECIFIED"
SCHEMA = "osmesa_joint0_candidate/v1"
ROOT_SEED = 1729
ORDINARY_MODALITIES = ("rgb", "depth", "segmentation")
COUNTERFACTUAL_MODALITY = "counterfactual_segmentation"
_LOCK = threading.Lock()


class QualificationFailure(RuntimeError):
    pass


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest_file(path: Path) -> str:
    return digest_bytes(path.read_bytes())


@dataclass(frozen=True)
class Attempt:
    ordinal: int
    family: str
    profile: str
    repeat: int

    @property
    def name(self) -> str:
        return f"{self.ordinal:02d}-{self.family}-{self.profile}-repeat-{self.repeat}"

    @property
    def expected_calls(self) -> int:
        return 8 if self.family == "single_occluder" else 6


def fixed_attempts() -> tuple[Attempt, ...]:
    order: list[Attempt] = []
    for family in ("single_occluder", "corridor"):
        for profile in ("legacy_solid_base_v1", "legacy_solid_alternate_v1"):
            for repeat in (0, 1):
                order.append(Attempt(len(order), family, profile, repeat))
    return tuple(order)


def plan() -> dict[str, object]:
    attempts = fixed_attempts()
    return {
        "schema": SCHEMA,
        "status_on_completion": STATUS,
        "backend": "osmesa",
        "quality": {"offsamples": 0},
        "seed": ROOT_SEED,
        "episodes_per_attempt": 4,
        "component_topology": False,
        "ordinary_poses": ["before", "after"],
        "ordinary_modalities": list(ORDINARY_MODALITIES),
        "single_occluder_extra": COUNTERFACTUAL_MODALITY,
        "attempts": [
            a.__dict__ | {"name": a.name, "expected_native_render_readbacks": a.expected_calls * 4}
            for a in attempts
        ],
        "limits": {
            "contexts": 32,
            "ordinary_poses": 64,
            "ordinary_sdk_modality_renders": 192,
            "native_render_readbacks": 224,
        },
    }


def validate_plan(value: dict[str, object]) -> None:
    if value != plan():
        raise QualificationFailure("immutable qualification plan differs")


def require_osmesa_linux() -> None:
    if platform.system() == "Windows" or os.environ.get("MUJOCO_GL") != "osmesa":
        raise QualificationFailure("requires Linux/WSL with MUJOCO_GL=osmesa")


def git_binding(
    root: Path, lock: Path, configs: Iterable[Path], registry: Path
) -> dict[str, object]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise QualificationFailure("source worktree not clean")
    return {
        "head": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "lock_sha256": digest_file(lock),
        "config_sha256": {p.name: digest_file(p) for p in configs},
        "registry_sha256": digest_file(registry),
    }


class RendererAdapter:
    """Replaces only the constructor during one batch and restores it in finally."""

    def __init__(self, mujoco: Any, observe: Callable[[Any], dict[str, object]]) -> None:
        self.mujoco, self.observe = mujoco, observe
        self.original = mujoco.Renderer
        self.thread = threading.get_ident()
        self.calls = 0
        self.observations: list[dict[str, object]] = []

    def __enter__(self) -> RendererAdapter:
        if not _LOCK.acquire(blocking=False):
            raise QualificationFailure("qualification renderer hook already active")
        if self.mujoco.Renderer is not self.original:
            _LOCK.release()
            raise QualificationFailure("preexisting Renderer replacement")

        def constructor(model: Any, *args: Any, **kwargs: Any) -> Any:
            if threading.get_ident() != self.thread:
                raise QualificationFailure("Renderer constructor wrong thread")
            if self.calls:
                raise QualificationFailure("Renderer constructor called more than once")
            self.calls += 1
            model.vis.quality.offsamples = 0
            renderer = self.original(model, *args, **kwargs)
            facts = self.observe(renderer)
            if (
                facts.get("offsamples") != 0
                or facts.get("sample_count") != 0
                or not facts.get("osmesa_context")
                or facts.get("gl_error") not in (0, None)
            ):
                raise QualificationFailure("zero-sample OSMesa renderer facts invalid")
            self.observations.append(facts)
            return renderer

        self.mujoco.Renderer = constructor
        return self

    def __exit__(self, typ: object, value: object, tb: object) -> None:
        try:
            if self.mujoco.Renderer is not self.original:
                self.mujoco.Renderer = self.original
            if self.mujoco.Renderer is not self.original:
                raise QualificationFailure("Renderer restoration failed")
        finally:
            _LOCK.release()


def arrays_equal(left: Path, right: Path) -> None:
    a, b = np.load(left, allow_pickle=False), np.load(right, allow_pickle=False)
    if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b):
        raise QualificationFailure(f"exact array mismatch: {left.name}")


def compare_repeat_artifacts(left: Path, right: Path) -> list[str]:
    """Exact deterministic domain; run.json is the sole volatile exclusion."""
    names = sorted(
        p.relative_to(left).as_posix()
        for p in left.rglob("*")
        if p.is_file() and p.name != "run.json"
    )
    other = sorted(
        p.relative_to(right).as_posix()
        for p in right.rglob("*")
        if p.is_file() and p.name != "run.json"
    )
    if names != other:
        raise QualificationFailure("repeat artifact membership differs")
    for name in names:
        a, b = left / name, right / name
        if a.suffix == ".npy":
            arrays_equal(a, b)
        elif digest_file(a) != digest_file(b):
            raise QualificationFailure(f"repeat artifact bytes differ: {name}")
    return names


def write_ledger(root: Path, event: dict[str, object]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "qualification-ledger.jsonl"
    existing = path.read_text().splitlines() if path.exists() else []
    if existing and json.loads(existing[-1]).get("event") in {"reserved", "failed"}:
        raise QualificationFailure("terminal/reserved attempt prevents continuation")
    with path.open("a", encoding="utf-8") as f:
        f.write(canonical(event).decode() + "\n")
    return path


def reserve(root: Path, attempt: Attempt, bindings: dict[str, object]) -> Path:
    return write_ledger(
        root,
        {
            "event": "reserved",
            "attempt": attempt.name,
            "plan_sha256": digest_bytes(canonical(plan())),
            "bindings": bindings,
        },
    )


def failure(root: Path, attempt: Attempt, error: BaseException) -> Path:
    return write_ledger(
        root,
        {"event": "failed", "attempt": attempt.name, "error": f"{type(error).__name__}: {error}"},
    )
