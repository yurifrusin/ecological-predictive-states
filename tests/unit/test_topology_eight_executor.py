"""Synthetic lifecycle checks; injected captures never construct a renderer."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from epsbench.diagnostics import topology_eight_executor as e
from epsbench.diagnostics.revision_capture import AttemptLock, publish_bytes

SOURCE = Path(__file__).resolve().parents[2]


def test_membership_is_byte_identical_and_finite() -> None:
    cells = e.fixed_cells(SOURCE)
    assert len(cells) == 8
    assert [cell["ordinal"] for cell in cells] == list(range(8))
    assert {cell["episode_seed_decimal"] for cell in cells} == {"1703363364368450807"}
    assert len({cell["configuration_sha256"] for cell in cells}) == 4
    assert cells[0]["name"] == "00-forward-base-r0"
    assert cells[-1]["name"] == "07-reverse-alternate-r1"


def run_fake(root: Path, **changes: Any) -> tuple[dict[str, Any], list[int]]:
    calls: list[int] = []

    def capture(cell: dict[str, Any], output: Path, baseline: object) -> dict[str, Any]:
        calls.append(cell["ordinal"])
        output.mkdir(parents=True)
        (output / "synthetic.json").write_text("{}", encoding="utf-8")
        return {"contexts": 1, "render_pairs": 6, "runtime": {"synthetic": True}}

    def assess(*paths: Path) -> dict[str, Any]:
        paths[1].mkdir(parents=True)
        (paths[1] / "synthetic.json").write_text("{}", encoding="utf-8")
        for path in paths[2:]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic-only")
        return {"target": "valid_negative", "synthetic": True}

    def compare(output: Path, cells: tuple[str, ...]) -> dict[str, Any]:
        assert len(cells) == 8
        return {"hypothesis_positive": False, "valid_negative": True}

    args: dict[str, Any] = {
        "capture": capture,
        "assess": assess,
        "compare": compare,
        "recheck": lambda: None,
        "phase_start": lambda *_: None,
        "phase_complete": lambda *_: None,
    }
    args.update(changes)
    result = e._execute_once(
        root, {"schema": e.VERSION, "cells": list(e.fixed_cells(SOURCE))}, **args
    )
    return result, calls


def test_all_valid_negative_controls_complete_without_extra_capture(tmp_path: Path) -> None:
    root = tmp_path / "synthetic-study"
    result, calls = run_fake(root)
    assert result["valid_negative"] is True
    assert calls == list(range(8))
    assert not (root / "capture-attempt.lock").exists()
    records = [json.loads(p.read_bytes()) for p in sorted((root / "ledger").glob("*.json"))]
    assert [r["event"] for r in records] == ["initialised"] + [
        "reserved",
        "complete",
        "released",
    ] * 8 + ["finished"]
    previous = None
    for path, record in zip(sorted((root / "ledger").glob("*.json")), records, strict=True):
        assert record["predecessor_sha256"] == previous
        previous = e.digest(path.read_bytes())
    index = json.loads((root / "evidence-index.json").read_bytes())
    assert all(item["path"] != "evidence-index.json" for item in index["files"])
    with pytest.raises(e.TopologyExecutionFailure, match="existing namespace"):
        run_fake(root)


@pytest.mark.parametrize("failure", ["capture", "assessment", "recheck", "budget", "runtime"])
def test_failure_stops_permanently_and_retains_lock(tmp_path: Path, failure: str) -> None:
    root = tmp_path / "synthetic-study"
    calls: list[int] = []
    checks = 0

    def capture(cell: dict[str, Any], output: Path, baseline: object) -> dict[str, Any]:
        calls.append(cell["ordinal"])
        if failure == "capture":
            raise RuntimeError("failed synthetic constructor")
        output.mkdir(parents=True)
        (output / "synthetic.json").write_text("{}", encoding="utf-8")
        return {
            "contexts": 1,
            "render_pairs": 7 if failure == "budget" else 6,
            "runtime": {"synthetic": cell["ordinal"] if failure == "runtime" else True},
        }

    def assess(*paths: Path) -> dict[str, Any]:
        if failure == "assessment":
            raise RuntimeError("synthetic integrity failure")
        paths[1].mkdir(parents=True)
        (paths[1] / "synthetic.json").write_text("{}", encoding="utf-8")
        for path in paths[2:]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic-only")
        return {"target": "negative"}

    def recheck() -> None:
        nonlocal checks
        checks += 1
        if failure == "recheck" and checks == 2:
            raise RuntimeError("synthetic source drift")

    with pytest.raises(RuntimeError):
        run_fake(root, capture=capture, assess=assess, recheck=recheck)
    assert calls == ([0, 1] if failure == "runtime" else [0])
    assert (root / "capture-attempt.lock").is_file()
    before = e.inventory(root)
    with pytest.raises(e.TopologyExecutionFailure, match="existing namespace"):
        run_fake(root)
    assert e.inventory(root) == before


@pytest.mark.parametrize(
    "failure", ["terminal_visible", "release_after_unlink", "confirmation_visible"]
)
def test_irreversible_publication_failure_never_advances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    root = tmp_path / "synthetic-study"
    publications: list[str] = []

    def publisher(path: Path, data: bytes) -> object:
        result = publish_bytes(path, data)
        publications.append(path.name)
        if path.parent.name == "ledger":
            record = json.loads(data)
            if (failure == "terminal_visible" and record["event"] == "complete") or (
                failure == "confirmation_visible" and record["event"] == "released"
            ):
                raise OSError("visible file but publication failed")
        return result

    if failure == "release_after_unlink":
        original = AttemptLock.release_after_success

        def release(self: AttemptLock, terminal: Path) -> None:
            original(self, terminal)
            raise OSError("post-unlink fsync failure")

        monkeypatch.setattr(AttemptLock, "release_after_success", release)
    with pytest.raises(OSError):
        run_fake(root, publisher=publisher)
    revisions = [json.loads(p.read_bytes()) for p in sorted((root / "ledger").glob("*.json"))]
    assert not any(r["event"] == "reserved" and r["ordinal"] == 1 for r in revisions)
    assert (root / "capture-attempt.lock").exists() == (failure == "terminal_visible")
    with pytest.raises(e.TopologyExecutionFailure):
        run_fake(root)


def test_ledger_detects_changed_prefix_and_foreign_files(tmp_path: Path) -> None:
    ledger = e.OperationalLedger(tmp_path)
    ledger.append("initialised", None, {})
    path = tmp_path / "ledger/revision-0000.json"
    original = path.read_bytes()
    path.write_bytes(original + b" ")
    with pytest.raises(e.TopologyExecutionFailure, match="changed"):
        ledger.append("reserved", 0, {})
    path.write_bytes(original)
    (path.parent / "foreign.json").write_text("{}", encoding="utf-8")
    with pytest.raises(e.TopologyExecutionFailure, match="prefix differs"):
        ledger.append("reserved", 0, {})


def test_preflight_is_typed_and_missing_authority_fails_before_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "platform.system", lambda: pytest.fail("platform queried before schema denial")
    )
    missing = tmp_path / "preflight.json"
    missing.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        e.validate_preflight(SOURCE, tmp_path / e.BASENAME, missing)


def test_membership_mutation_is_rejected_before_plan(tmp_path: Path) -> None:
    path = tmp_path / "docs/protocols/topology-eight-transition-v1-membership.json"
    path.parent.mkdir(parents=True)
    records = copy.deepcopy(list(e.fixed_cells(SOURCE)))
    records[0]["root_seed"] = 1
    path.write_text(json.dumps({"records": records}), encoding="utf-8")
    with pytest.raises(e.TopologyExecutionFailure, match="byte identity"):
        e.fixed_cells(tmp_path)


def test_confirmed_cell_corruption_stops_before_next_capture(tmp_path: Path) -> None:
    root = tmp_path / "synthetic-study"
    checks = 0

    def recheck() -> None:
        nonlocal checks
        checks += 1
        if checks == 3:
            (root / "datasets/00-forward-base-r0/synthetic.json").write_bytes(b"changed")

    with pytest.raises(e.TopologyExecutionFailure, match="confirmed cell evidence changed"):
        run_fake(root, recheck=recheck)
    records = [json.loads(p.read_bytes()) for p in sorted((root / "ledger").glob("*.json"))]
    assert not any(r["event"] == "reserved" and r["ordinal"] == 1 for r in records)
    assert not (root / "capture-attempt.lock").exists()


def test_failed_constructor_native_receipt_is_retained_outside_ledger(tmp_path: Path) -> None:
    root = tmp_path / "synthetic-study"
    partial = {"constructor_attempts": [{"ordinal": 0, "completed": False}], "native_events": []}

    def capture(*args: Any) -> dict[str, Any]:
        error = RuntimeError("synthetic failed constructor")
        setattr(error, "native_receipt", partial)
        raise error

    with pytest.raises(RuntimeError, match="failed constructor"):
        run_fake(root, capture=capture)
    failure = json.loads((root / "operator/failure.json").read_bytes())
    assert failure["native_partial_receipt"] == partial
    assert failure["lock_present"] is True


def test_preflight_pins_complete_software_apparatus() -> None:
    assert e.ENVIRONMENT["MUJOCO_GL"] == e.ENVIRONMENT["PYOPENGL_PLATFORM"] == "osmesa"
    assert e.ENVIRONMENT["LIBGL_ALWAYS_SOFTWARE"] == "1"
    assert e.ENVIRONMENT["GALLIUM_DRIVER"] == "llvmpipe"
