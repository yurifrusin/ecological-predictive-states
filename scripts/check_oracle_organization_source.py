"""Guarded public source checks: deny historical apparatus and all optimizer steps."""

from __future__ import annotations

import importlib.abc
import json
import os
import sys
from pathlib import Path
from typing import Any


class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: object = None, target: object = None) -> None:
        if fullname.startswith(
            ("mujoco", "OpenGL", "glfw", "epsbench.sim", "epsbench.data", "epsbench.annotations")
        ) or (
            fullname.startswith("epsbench.diagnostics.")
            and not fullname.startswith("epsbench.diagnostics.oracle_organization_")
        ):
            raise RuntimeError("outside public source scope: " + fullname)
        return None


def main() -> int:
    if sys.argv[1:]:
        raise ValueError("no extra source-check arguments admitted")
    sys.dont_write_bytecode = True
    sys.meta_path.insert(0, Guard())
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ.pop("PYTEST_ADDOPTS", None)
    os.environ.pop("PYTEST_PLUGINS", None)
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))
    import torch

    def denied(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("optimizer execution denied by public source-check guard")

    # Patch every exposed optimizer class, including overrides of base step.
    for value in vars(torch.optim).values():
        if isinstance(value, type) and issubclass(value, torch.optim.Optimizer):
            value.__init__ = denied  # type: ignore[method-assign]
            value.step = denied  # type: ignore[method-assign, assignment]
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    import pytest

    from epsbench.diagnostics.oracle_organization_readiness import source_report

    result = pytest.main(
        [
            "--noconftest",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            "-q",
            "tests/test_oracle_organization.py",
            "tests/test_oracle_region_extension.py",
        ]
    )
    print(json.dumps(source_report(), sort_keys=True))
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
