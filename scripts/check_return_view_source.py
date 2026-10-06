"""Only distinct synthetic return-view checks; no candidate or native evaluation."""

from __future__ import annotations

import os
import sys

from check_a1_source import Guard, check_loaded


def main() -> int:
    if sys.argv[1:]:
        raise ValueError("no source-check arguments accepted")
    check_loaded()
    sys.meta_path.insert(0, Guard())
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ.pop("PYTEST_ADDOPTS", None)
    os.environ.pop("PYTEST_PLUGINS", None)
    from epsbench.diagnostics import causal_history_fixture, return_view_analytic

    def candidate_denied(*args: object, **kwargs: object) -> None:
        raise RuntimeError("fixed or historical candidate evaluation denied during source checks")

    causal_history_fixture.check_candidate = candidate_denied
    return_view_analytic.Producer.frame = candidate_denied
    import pytest

    result = pytest.main(["--noconftest", "-o", "addopts=", "tests/test_return_view.py", "-q"])
    check_loaded()
    print("Synthetic smoke/serialization/inspection only; fixed candidate UNEVALUATED; launch HELD")
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
