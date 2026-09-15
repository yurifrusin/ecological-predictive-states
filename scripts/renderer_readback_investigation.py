"""Operator CLI for the finite renderer-readback investigation.

Only ``next`` can construct a graphics context, and only after lock, reservation,
archive, handoff, source, SDK, and runtime checks.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from pathlib import Path

from epsbench.diagnostics.renderer_readback import (
    ARCHIVE_SHA256,
    MANIFEST_SHA256,
    SDK_RENDERER_SHA256,
    SDK_VERSION,
    STUDY_HOST,
    fixed_cells,
    initialise_ledger,
    run_attempt,
    validate_ledger,
    validate_sdk,
    verify_reference_archive,
)
from epsbench.diagnostics.revision_capture import (
    RevisionCaptureFailure,
    canonical_json_bytes,
    detect_study_runtime,
    publish_bytes,
    sha256_file,
    translate_study_root,
)
from epsbench.diagnostics.revision_runner import PreparedEpisode, StackFactory


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True, encoding="utf-8").strip()


def _init(args: argparse.Namespace) -> None:
    if platform.node().upper() != STUDY_HOST:
        raise RevisionCaptureFailure("renderer-readback initialization host differs")
    verify_reference_archive(args.reference_archive)
    head = _git("rev-parse", "HEAD")
    tree = _git("rev-parse", "HEAD^{tree}")
    if head != args.source_head:
        raise RevisionCaptureFailure("checked-out source head differs from binding")
    if _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise RevisionCaptureFailure("initial source worktree is not clean")
    plan = {
        "review_profile": "DUAL_REVIEW",
        "evidence_class": "PUBLIC_REPOSITORY_ONLY",
        "phase_gate_effect": "NONE",
        "archive_sha256": ARCHIVE_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "source_head": head,
        "source_tree": tree,
        "dependency_lock_sha256": sha256_file(Path("uv.lock")),
        "sdk_version": SDK_VERSION,
        "sdk_renderer_sha256": SDK_RENDERER_SHA256,
        "root_seed": 1729,
        "render": {"width": 160, "height": 120},
        "host": STUDY_HOST,
        "runtimes": ["windows", "wsl"],
        "config_sha256": {
            "corridor": sha256_file(Path("configs/corridor_v0.yaml")),
            "single_occluder": sha256_file(Path("configs/benchmark_v0.yaml")),
        },
        "cells": [cell.name for cell in fixed_cells()],
        "maximum_renderer_contexts": 8,
        "maximum_pose_endpoints": 16,
        "maximum_modality_render_calls": 48,
        "maximum_sdk_readbacks": 48,
    }
    terminal = initialise_ledger(args.output_root, plan)
    print(json.dumps({"status": "initialised", "ledger": str(terminal)}, sort_keys=True))


def _handoff_create(args: argparse.Namespace) -> None:
    runtime = detect_study_runtime()
    host = platform.node()
    if args.runtime != runtime or host.upper() != STUDY_HOST or not args.token:
        raise RevisionCaptureFailure("handoff runtime, host, or token invalid")
    record = {
        "schema": "renderer_readback_handoff/v1",
        "runtime": runtime,
        "host": host,
        "token": args.token,
        "observed_root": str(args.output_root.resolve()),
    }
    publish_bytes(
        args.output_root / "renderer-readback-handoff" / f"{runtime}.json",
        canonical_json_bytes(record),
    )
    print(json.dumps(record, sort_keys=True))


def _handoff_verify(args: argparse.Namespace) -> None:
    runtime = detect_study_runtime()
    host = platform.node()
    if args.runtime != runtime or host.upper() != STUDY_HOST:
        raise RevisionCaptureFailure("handoff verification runtime or host invalid")
    directory = args.output_root / "renderer-readback-handoff"
    sources = {
        name: json.loads((directory / f"{name}.json").read_text("utf-8"))
        for name in ("windows", "wsl")
    }
    peer = "wsl" if runtime == "windows" else "windows"
    if (
        sources[runtime].get("token") != args.own_token
        or sources[peer].get("token") != args.peer_token
        or args.own_token == args.peer_token
    ):
        raise RevisionCaptureFailure("handoff verification tokens invalid")
    translated = translate_study_root(str(sources[runtime]["observed_root"]), runtime)
    if translated != sources[peer].get("observed_root"):
        raise RevisionCaptureFailure("handoff root translation differs")
    record = {
        "schema": "renderer_readback_handoff_verification/v1",
        "runtime": runtime,
        "host": host,
        "own_token": args.own_token,
        "peer_token": args.peer_token,
        "expected_peer_root": translated,
        "observed_root": str(args.output_root.resolve()),
    }
    publish_bytes(directory / f"verified-{runtime}.json", canonical_json_bytes(record))
    print(json.dumps(record, sort_keys=True))


def _configure_backend(backend: str) -> None:
    if backend == "wgl":
        if platform.system() != "Windows" or os.environ.get("MUJOCO_GL"):
            raise RevisionCaptureFailure("WGL requires native Windows without override")
    elif backend == "osmesa":
        if (
            platform.system() == "Windows"
            or os.environ.get("MUJOCO_GL") not in (None, "osmesa")
            or os.environ.get("PYOPENGL_PLATFORM") not in (None, "osmesa")
        ):
            raise RevisionCaptureFailure("OSMesa requires WSL/Linux OSMesa environment")
        os.environ["MUJOCO_GL"] = "osmesa"
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    else:
        raise RevisionCaptureFailure("unsupported backend")


def _next(args: argparse.Namespace) -> None:
    records = validate_ledger(args.output_root)
    completed = [record for record in records if record["event"] == "complete"]
    if len(completed) >= len(fixed_cells()):
        raise RevisionCaptureFailure("fixed renderer-readback matrix is complete")
    cell = fixed_cells()[len(completed)]

    def prepare() -> tuple[PreparedEpisode, StackFactory, object, int]:
        if args.backend != cell.backend:
            raise RevisionCaptureFailure(
                f"wrong next backend: expected {cell.backend}, received {args.backend}"
            )
        _configure_backend(args.backend)
        import mujoco
        from OpenGL import GL

        from epsbench.diagnostics.revision_mujoco import provide_fixed_episode

        _path, depth_map = validate_sdk(mujoco)
        prepared, factory = provide_fixed_episode(cell)
        if not isinstance(prepared, PreparedEpisode) or not callable(factory):
            raise RevisionCaptureFailure("fixed provider returned invalid preparation")
        return prepared, factory, GL, depth_map

    terminal = run_attempt(
        args.output_root,
        args.reference_archive,
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--output-root", type=Path, required=True)
    init.add_argument("--reference-archive", type=Path, required=True)
    init.add_argument("--source-head", required=True)
    init.set_defaults(func=_init)
    for name in ("handoff-create", "handoff-verify"):
        command = commands.add_parser(name)
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--runtime", choices=("windows", "wsl"), required=True)
        if name == "handoff-create":
            command.add_argument("--token", required=True)
            command.set_defaults(func=_handoff_create)
        else:
            command.add_argument("--own-token", required=True)
            command.add_argument("--peer-token", required=True)
            command.set_defaults(func=_handoff_verify)
    nxt = commands.add_parser("next")
    nxt.add_argument("--output-root", type=Path, required=True)
    nxt.add_argument("--reference-archive", type=Path, required=True)
    nxt.add_argument("--backend", choices=("wgl", "osmesa"), required=True)
    nxt.add_argument("--geom-objtype", type=int, choices=(5,), default=5)
    nxt.set_defaults(func=_next)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
