"""Operator CLI for the bounded capture-contract revision study.

No command constructs a graphics context except next after durable reservation.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from epsbench.diagnostics.revision_capture import (
    ARCHIVE_SHA256,
    ArtifactWriter,
    RevisionCaptureFailure,
    canonical_json_bytes,
    fixed_cells,
    initialise_ledger,
    load_cell_result,
    publish_bytes,
    sha256_file,
    validate_ledger,
    verify_input_archive,
)
from epsbench.diagnostics.revision_runner import PreparedEpisode, StackFactory, run_attempt


def _git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], text=True, encoding="utf-8").strip()


def _init(args: argparse.Namespace) -> None:
    archive = verify_input_archive(args.input_archive)
    head = _git("rev-parse", "HEAD")
    tree = _git("rev-parse", "HEAD^{tree}")
    if head != args.source_head:
        raise RevisionCaptureFailure("checked-out source head differs from explicit binding")
    lock_path = Path(args.dependency_lock)
    if not lock_path.is_file():
        raise RevisionCaptureFailure("dependency lock does not exist")
    plan = {
        "review_profile": "DUAL_REVIEW",
        "evidence_class": "PUBLIC_REPOSITORY_ONLY",
        "phase_gate_effect": "NONE",
        "archive_sha256": archive["archive_sha256"],
        "manifest_sha256": archive["manifest_sha256"],
        "source_head": head,
        "source_tree": tree,
        "dependency_lock_path": lock_path.as_posix(),
        "dependency_lock_sha256": sha256_file(lock_path),
        "root_seed": 1729,
        "episode_indices": [0, 1, 2, 3],
        "render": {"width": 160, "height": 120, "fovy_degrees": 55.0},
        "appearance_profile": "legacy_solid_base_v1",
        "config_sha256": {
            "corridor": sha256_file(Path("configs/corridor_v0.yaml")),
            "single_occluder": sha256_file(Path("configs/benchmark_v0.yaml")),
        },
        "appearance_registry_sha256": sha256_file(Path("configs/appearance_candidates_v0.yaml")),
        "seed_registry_sha256": sha256_file(Path("configs/evaluation_seed_candidates_v0.yaml")),
        "fixed_provider": ("epsbench.diagnostics.revision_mujoco:provide_fixed_episode"),
        "cells": [cell.name for cell in fixed_cells()],
        "maximum_frames": 96,
        "maximum_renderer_constructions": 64,
        "maximum_modality_render_calls": 288,
    }
    terminal = initialise_ledger(args.output_root, plan)
    print(
        json.dumps(
            {"status": "initialised", "ledger": str(terminal), "archive_sha256": ARCHIVE_SHA256},
            sort_keys=True,
        )
    )


def _handoff_create(args: argparse.Namespace) -> None:
    record = {
        "schema": "revision_capture_handoff/v1",
        "runtime": args.runtime,
        "token": args.token,
        "observed_root": str(args.output_root.resolve()),
    }
    target = args.output_root / "handoff" / f"{args.runtime}.json"
    publish_bytes(target, canonical_json_bytes(record))
    print(json.dumps(record, sort_keys=True))


def _handoff_verify(args: argparse.Namespace) -> None:
    records: dict[str, dict[str, Any]] = {}
    for runtime in ("windows", "wsl"):
        path = args.output_root / "handoff" / f"{runtime}.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema") != "revision_capture_handoff/v1":
            raise RevisionCaptureFailure("invalid handoff record")
        records[runtime] = value
    if records["windows"]["token"] == records["wsl"]["token"]:
        raise RevisionCaptureFailure("two-way handoff tokens must be distinct")
    own = records[args.runtime]
    peer = records["wsl" if args.runtime == "windows" else "windows"]
    if own["token"] != args.own_token or peer["token"] != args.peer_token:
        raise RevisionCaptureFailure("handoff token verification failed")
    record = {
        "schema": "revision_capture_handoff_verification/v1",
        "runtime": args.runtime,
        "own_token": args.own_token,
        "peer_token": args.peer_token,
        "observed_root": str(args.output_root.resolve()),
    }
    target = args.output_root / "handoff" / f"verified-{args.runtime}.json"
    publish_bytes(target, canonical_json_bytes(record))
    print(json.dumps(record, sort_keys=True))


def _configure_backend(backend: str) -> None:
    if backend == "wgl":
        if platform.system() != "Windows" or os.environ.get("MUJOCO_GL"):
            raise RevisionCaptureFailure("WGL requires native Windows and no MUJOCO_GL override")
        return
    if backend == "osmesa":
        if platform.system() == "Windows":
            raise RevisionCaptureFailure("OSMesa requires WSL/Linux")
        if os.environ.get("MUJOCO_GL") not in (None, "osmesa"):
            raise RevisionCaptureFailure("wrong MUJOCO_GL for OSMesa")
        if os.environ.get("PYOPENGL_PLATFORM") not in (None, "osmesa"):
            raise RevisionCaptureFailure("wrong PYOPENGL_PLATFORM for OSMesa")
        os.environ["MUJOCO_GL"] = "osmesa"
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        return
    raise RevisionCaptureFailure("unsupported backend")


def _next(args: argparse.Namespace) -> None:
    records = validate_ledger(args.output_root)
    completed = [item for item in records if item["event"] == "complete"]
    if len(completed) >= len(fixed_cells()):
        raise RevisionCaptureFailure("fixed matrix is complete")
    cell = fixed_cells()[len(completed)]

    def prepare() -> tuple[PreparedEpisode, StackFactory]:
        # Runs only after lock acquisition and durable reservation.
        if cell.backend != args.backend:
            raise RevisionCaptureFailure(
                f"wrong next backend: expected {cell.backend}, received {args.backend}"
            )
        _configure_backend(args.backend)
        from epsbench.diagnostics.revision_mujoco import provide_fixed_episode

        result = provide_fixed_episode(cell)
        if not isinstance(result[0], PreparedEpisode) or not callable(result[1]):
            raise RevisionCaptureFailure("fixed provider returned invalid preparation")
        return result

    terminal = run_attempt(
        args.output_root,
        args.input_archive,
        cell,
        prepare,
        geom_objtype=args.geom_objtype,
    )
    print(
        json.dumps(
            {"status": "complete", "cell": cell.name, "terminal_revision": str(terminal)},
            sort_keys=True,
        )
    )


def _analyse(args: argparse.Namespace) -> None:
    from epsbench.diagnostics.revision_analysis import analyze_capture_results

    records = validate_ledger(args.output_root)
    receipts = []
    for item in records:
        if item["event"] == "complete":
            directory = args.output_root / "cells" / item["cell_name"]
            receipts.append(load_cell_result(directory))
    result = analyze_capture_results(receipts)
    analysis_root = args.output_root / "analysis"
    writer = ArtifactWriter(analysis_root)
    counter = 0

    def materialize(value: Any) -> Any:
        nonlocal counter
        if isinstance(value, np.ndarray):
            name = f"array-{counter:06d}.npy"
            counter += 1
            return writer.array(name, value, validated=True)
        if isinstance(value, dict):
            return {str(key): materialize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [materialize(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    report = materialize(result)
    reference = writer.json("capture-revision-analysis.json", report, validated=True)
    print(json.dumps({"status": "complete", "analysis": reference}, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--output-root", type=Path, required=True)
    init.add_argument("--input-archive", type=Path, required=True)
    init.add_argument("--source-head", required=True)
    init.add_argument("--dependency-lock", type=Path, default=Path("uv.lock"))
    init.set_defaults(func=_init)
    for name in ("handoff-create", "handoff-verify"):
        command = sub.add_parser(name)
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--runtime", choices=("windows", "wsl"), required=True)
        if name == "handoff-create":
            command.add_argument("--token", required=True)
            command.set_defaults(func=_handoff_create)
        else:
            command.add_argument("--own-token", required=True)
            command.add_argument("--peer-token", required=True)
            command.set_defaults(func=_handoff_verify)
    nxt = sub.add_parser("next")
    nxt.add_argument("--output-root", type=Path, required=True)
    nxt.add_argument("--input-archive", type=Path, required=True)
    nxt.add_argument("--backend", choices=("wgl", "osmesa"), required=True)
    nxt.add_argument("--geom-objtype", type=int, default=5, choices=(5,))
    nxt.set_defaults(func=_next)
    analyse = sub.add_parser("analyze")
    analyse.add_argument("--output-root", type=Path, required=True)
    analyse.set_defaults(func=_analyse)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
