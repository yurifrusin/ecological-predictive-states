"""Constrained in-container capture/evaluator. No host outcome decoding."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--expected-history", required=True)
    parser.add_argument("--binding", required=True)
    args = parser.parse_args()
    from epsbench.diagnostics.a1_execution import BUDGET, Binding, source_binding
    from epsbench.diagnostics.a1_retention import RetainedArchive

    binding = Binding(**json.loads(args.binding))
    if source_binding(SOURCE, binding.image, binding.purpose) != binding:
        raise ValueError("image checkout differs from prospective source")
    if args.task.startswith("dummy-") != (binding.purpose == "a1_dummy_qualification_v1"):
        raise ValueError("dummy/native purpose mismatch")

    # This entrypoint never creates, truncates or substitutes a retained history.
    class FailedSink(RetainedArchive):
        def _flush(self) -> None:
            raise OSError("synthetic sink failure")

    archive_type = FailedSink if args.task == "dummy-sink-failure" else RetainedArchive
    archive = archive_type(
        Path("/retained/history"), binding.root, BUDGET, expected_history=args.expected_history
    )
    try:
        native_packages = subprocess.check_output(
            [
                "dpkg-query",
                "-W",
                "-f=${Package}=${Version}\\n",
                "libosmesa6",
                "libgl1-mesa-dri",
                "libglfw3",
            ],
            timeout=10,
        )
        if len(native_packages) > 8192:
            raise ValueError("native package observation bound exceeded")
        archive.put(
            f"operations/{args.task}-runtime.json",
            json.dumps(
                {
                    "python": platform.python_version(),
                    "installed": {
                        name: importlib.metadata.version(name)
                        for name in ("mujoco", "numpy", "PyOpenGL", "glfw")
                    },
                    "native_library_packages": native_packages.decode(),
                    "binding": vars(binding),
                },
                sort_keys=True,
            ).encode(),
        )
        if args.task == "dummy-complete":
            archive.put("dummy/complete", b"dummy only; no producer imports")
        elif args.task == "dummy-interrupted":
            archive.start("dummy/incomplete")
            archive.reserve_staging(8)
            archive.chunk(b"prefix")
            raise RuntimeError("synthetic interrupted output")
        elif args.task == "dummy-sink-failure":
            archive.start("dummy/failed-sink")
        elif args.task == "dummy-timeout":
            import time

            time.sleep(600)
        elif args.task.startswith("cell-"):
            ordinal = int(args.task.removeprefix("cell-"))
            if args.task != f"cell-{ordinal:02d}" or not 0 <= ordinal < 8:
                raise ValueError("unknown fixed cell")
            from epsbench.data.generate import generate_dataset
            from epsbench.diagnostics.a1_action_contrast import cases

            archive.put(
                f"operations/cell-{ordinal:02d}-budget.json",
                json.dumps(
                    {
                        "ordinal": ordinal,
                        "prefix": f"cells/{ordinal:02d}",
                        "contexts": 1,
                        "render_read_pairs": 6,
                        "cell_seconds": 300,
                        "binding": vars(binding),
                    },
                    sort_keys=True,
                ).encode(),
            )
            generate_dataset(
                cases(SOURCE)[ordinal].config,
                1,
                Path(f"/output/cells/{ordinal:02d}"),
                capture_mode="canonical_paired",
                publisher=archive,
                publication_prefix=f"cells/{ordinal:02d}",
            )
            archive.put(f"operations/cell-{ordinal:02d}-complete.json", b'{"complete":true}')
        elif args.task in {"evaluate", "inspect-development"}:
            from epsbench.diagnostics.a1_archive_journal import ArchiveJournal
            from epsbench.diagnostics.a1_files import A1CanonicalLoader, A1Files
            from epsbench.diagnostics.a1_lifecycle import EvaluationJournal, assemble, evaluate
            from epsbench.schema import ModalityPermissionSet

            entries = tuple(archive.entries())  # fixed prefix; copies cannot extend this list
            for name, length, _ in entries:
                if not name.startswith("cells/"):
                    continue
                if args.task == "inspect-development" and not name.startswith(
                    ("cells/00/", "cells/01/")
                ):
                    continue
                path = Path("/output") / name
                path.parent.mkdir(parents=True, exist_ok=True)
                archive.reserve_copy(f"copies/evaluate/{archive.sequence:08d}", length)
                with path.open("xb") as stream:
                    for chunk in archive.read_chunks(name):
                        if stream.write(chunk) != len(chunk):
                            raise OSError("short restored artifact write")
            if args.task == "inspect-development":
                from epsbench.data.inspect import create_inspection_image

                for ordinal in (0, 1):
                    create_inspection_image(
                        Path(f"/output/cells/{ordinal:02d}"),
                        0,
                        Path(f"/output/inspection/cell-{ordinal:02d}.png"),
                        publisher=archive,
                        publication_prefix="inspection/development",
                    )
                archive.put("operations/inspect-development-complete.json", b'{"complete":true}')
                return 0
            loaders = tuple(
                A1CanonicalLoader(
                    Path(f"/output/cells/{i:02d}"), ModalityPermissionSet.all_modalities()
                )
                for i in range(8)
            )
            import hashlib

            from epsbench.diagnostics.a1_lifecycle import digest

            addendum = digest(
                {
                    name: hashlib.sha256(
                        (SOURCE / "docs/protocols" / name).read_bytes()
                    ).hexdigest()
                    for name in ("a1-prospective-controls.md", "a1-docker-execution.md")
                }
            )
            files = A1Files.membership(
                SOURCE, binding.source_head, binding.source_tree, addendum, loaders
            )
            journal = EvaluationJournal(ArchiveJournal(archive))
            bundle = assemble(
                files.study, SOURCE, files.before_reader(), files.development_reader(), journal
            )
            report = evaluate(bundle, journal, files.evaluator(bundle, journal))
            archive.put(
                "journal/final-report.json",
                json.dumps(report, sort_keys=True, allow_nan=False).encode(),
            )
        else:
            raise ValueError("unknown task")
        if not args.task.startswith("cell-"):
            archive.put(f"operations/{args.task}-complete.json", b'{"complete":true}')
    except Exception as error:
        if not archive.poisoned:
            archive.failure(type(error).__name__)
        raise
    finally:
        archive.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
