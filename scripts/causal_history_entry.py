"""Held causal container entrypoint; independent worker deadline and retained failures."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / "src"))


def worker(args: argparse.Namespace) -> int:
    from epsbench.diagnostics.causal_history_execution import (
        BUDGET,
        Binding,
        CaptureRetention,
        FiniteArchive,
        source_binding,
    )
    from epsbench.diagnostics.causal_history_lifecycle import (
        EXPOSURE_PATH,
        SEAL_PATH,
        BoundedReader,
        BoundSource,
        ExposureReceipt,
        SealReceipt,
        evaluate,
        expose_targets,
        seal_forecasts,
    )
    from epsbench.diagnostics.causal_history_sequence import (
        MEMBERS,
        canonical,
        digest,
        parse,
    )

    binding = Binding(**parse(args.binding.encode(), 8192))
    if source_binding(SOURCE, binding.image, binding.purpose, binding.dummy_task) != binding:
        raise ValueError("exact clean source/image binding required")
    tasks = binding.tasks
    if args.task not in tasks:
        raise ValueError("fixed task/purpose domain")
    for name, value in {
        "EPS_CAUSAL_PURPOSE": binding.purpose,
        "EPS_CAUSAL_SOURCE_HEAD": binding.source_head,
        "EPS_CAUSAL_SOURCE_TREE": binding.source_tree,
        "EPS_CAUSAL_IMAGE": binding.image,
    }.items():
        os.environ[name] = value

    class FailedSink(FiniteArchive):
        def _flush(self) -> None:
            raise OSError("synthetic failed sink")

    sink_type = FailedSink if args.task == "dummy-sink-failure" else FiniteArchive
    sink = sink_type(
        Path("/retained/history"), binding.root, BUDGET, expected_history=args.expected_history
    )
    try:
        # Exact host reservation is needed even when the worker is called directly.
        reader = BoundedReader(sink)
        reservation = parse(reader.control(f"operations/{args.task}-reserved.json", 8192))
        if reservation["task"] != args.task or reservation["binding"] != vars(binding):
            raise ValueError("durable matching attempt reservation required")
        # Completed task prefix is required; namespace changes cannot skip/retry work.
        index = tasks.index(args.task)
        actual = {t for t in tasks if f"operations/{t}-complete.json" in sink.committed}
        if actual != set(tasks[:index]):
            raise ValueError("exact fixed prior completion prefix")
        if args.task.startswith("dummy-"):
            if args.task == "dummy-complete":
                sink.put("dummy/complete", b"tiny mechanical qualification only")
            elif args.task == "dummy-interrupted":
                sink.start("dummy/interrupted")
                sink.reserve_staging(8)
                sink.chunk(b"prefix")
                raise RuntimeError("synthetic interrupted output")
            elif args.task == "dummy-sink-failure":
                sink.start("dummy/sink-failure")
            elif args.task == "dummy-timeout":
                import time

                time.sleep(600)
            elif args.task == "dummy-overflow":
                sink.start("dummy/overflow")
                sink.reserve_staging(4 * 1024**2)
        else:
            from epsbench.diagnostics.causal_history_runtime import causal_candidate

            if not causal_candidate():
                raise ValueError("distinct causal resource/runtime admission required")
            expected_versions = {
                "mujoco": "3.12.0",
                "numpy": "2.4.6",
                "PyOpenGL": "3.1.10",
                "glfw": "2.10.2",
            }
            if platform.python_version() != "3.11.15" or any(
                importlib.metadata.version(k) != v for k, v in expected_versions.items()
            ):
                raise ValueError("exact pinned native versions required")
            renderer: dict[str, Any] = {
                "profile": "causal-osmesa-docker-v1",
                "python": "3.11.15",
                "installed": expected_versions,
                "backend": "osmesa",
                "image": binding.image,
            }
            sink.put(f"operations/{args.task}-runtime.json", canonical(renderer))
            payload = (
                SOURCE / "configs/development/causal_history_fixture_design_v1.json"
            ).read_bytes()
            if args.task == "compile-only":
                from epsbench.diagnostics.causal_history_native import (
                    NativeBackend,
                    build_sequence_xml,
                )
                from epsbench.diagnostics.causal_history_sequence import CompiledEvidence

                for member in MEMBERS:
                    xml = build_sequence_xml(payload, member)
                    backend = NativeBackend(xml, compile_only=True)
                    try:
                        compiled = CompiledEvidence.model_validate_json(canonical(backend.compiled))
                        visual, stats = compiled.visual, compiled.statistics
                        sink.put(
                            f"operations/compiled-{member}.json",
                            canonical(
                                {
                                    "member": member,
                                    "xml": digest(xml.encode()),
                                    "compiled": compiled.model_dump(mode="json"),
                                    "actual_near": visual.znear * stats.extent,
                                    "actual_far": visual.zfar * stats.extent,
                                    "renderers": 0,
                                    "draws": 0,
                                    "reads": 0,
                                }
                            ),
                        )
                    finally:
                        backend.close()
            elif args.task.startswith("capture-"):
                from epsbench.diagnostics.causal_history_native import (
                    NativeBackend,
                    produce_sequence,
                )

                member = args.task.removeprefix("capture-")
                retention = CaptureRetention(sink, payload, member, binding, renderer)
                produce_sequence(
                    payload,
                    member,
                    factory=lambda xml: NativeBackend(xml, progress=retention),
                    progress=retention,
                )
                retention.encode(partial=False)
            elif args.task == "seal":
                seal_forecasts(
                    BoundSource(
                        head=binding.source_head,
                        tree=binding.source_tree,
                        image=binding.image,
                        renderer=renderer,
                    ),
                    reader,
                )
            elif args.task == "evaluate-inspect":
                seal_data = reader.control(SEAL_PATH)
                seal = SealReceipt(binding.root, digest(seal_data))
                if EXPOSURE_PATH in sink.used:
                    raise ValueError("evaluation attempt already exposed; no retry")
                exposure: ExposureReceipt = expose_targets(seal, reader)
                evaluate(seal, exposure, reader)
                from epsbench.diagnostics.causal_history_lifecycle import inspect

                inspect(seal, exposure, reader)
        sink.put(
            f"operations/{args.task}-complete.json",
            canonical({"task": args.task, "binding": vars(binding), "complete": True}),
        )
        return 0
    except BaseException as error:
        if not sink.poisoned:
            sink.failure((args.task[:80] + ":" + type(error).__name__ + ":" + str(error))[:128])
        raise
    finally:
        sink.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--binding", required=True)
    parser.add_argument("--expected-history", required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    from epsbench.diagnostics.causal_history_execution import WORK_SECONDS

    if sys.platform != "linux":
        raise ValueError("in-container deadline requires qualified Linux process groups")
    # Supervisor owns its one worker and cannot refund or retry. No output log side channel.
    proc = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--task",
            args.task,
            "--binding",
            args.binding,
            "--expected-history",
            args.expected_history,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    timeout = 2 if args.task.startswith("dummy-") else WORK_SECONDS
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=2)
        # Interrupted suffix remains untouched and host witnesses terminal timeout.
        return 124


if __name__ == "__main__":
    raise SystemExit(main())
