"""Synthetic archive binding/chronology tests; no subprocess or native imports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from epsbench.diagnostics import a1_lifecycle as life
from epsbench.diagnostics.a1_archive_journal import ArchiveJournal
from epsbench.diagnostics.a1_execution import (
    BUDGET,
    Binding,
    DockerController,
    HostReceipts,
    create_arguments,
    history,
    initialize,
    validate_archive_mount,
)
from epsbench.diagnostics.a1_retention import Budget, RetainedArchive
from tests.test_a1_controls import SOURCE, SyntheticReaders

BINDING = Binding("a" * 40, "b" * 40, "c" * 64, "sha256:" + "d" * 64)


def test_single_regular_file_mount_and_purpose_domains(tmp_path: Path) -> None:
    archive = tmp_path / "retained"
    archive.write_bytes(b"owned regular file")
    args = create_arguments("name", "token", BINDING, archive, "cell-00", "a" * 64)
    assert args.count("--mount") == 1
    assert (
        args[args.index("--mount") + 1]
        == f"type=bind,src={archive.resolve()},dst=/retained/history"
    )
    assert "--memory=8g" in args and "--memory-swap=8g" in args and "--log-driver=none" in args
    dummy = Binding(
        BINDING.source_head,
        BINDING.source_tree,
        BINDING.configuration_root,
        BINDING.image,
        "a1_dummy_qualification_v1",
    )
    assert dummy.root != BINDING.root
    with pytest.raises(ValueError, match="regular"):
        create_arguments("name", "token", BINDING, tmp_path, "cell-00", "a" * 64)


def test_mount_source_mismatch_denied(tmp_path: Path) -> None:
    archive = tmp_path / "retained"
    source = str(archive.resolve())
    info: dict[str, Any] = {
        "HostConfig": {
            "Mounts": [{"Type": "bind", "Source": source, "Target": "/retained/history"}]
        },
        "Mounts": [
            {"Type": "bind", "Source": source, "Destination": "/retained/history", "RW": True}
        ],
    }
    validate_archive_mount(info, archive)
    info["Mounts"][0]["Source"] = source + "-other"
    with pytest.raises(ValueError, match="source"):
        validate_archive_mount(info, archive)


def test_witnessed_checkpoint_and_torn_anchor_fail_closed(tmp_path: Path) -> None:
    receipts = HostReceipts(tmp_path / "receipts", BINDING, initial=True)
    expected = initialize(tmp_path / "retained", BINDING, receipts)
    assert expected == history(tmp_path / "retained")
    receipts.close()
    restored = HostReceipts(tmp_path / "receipts", BINDING, initial=False)
    assert restored.records[-1]["history"] == expected
    controller = DockerController("DO_NOT_EXECUTE", tmp_path / "retained", BINDING, restored)
    with pytest.raises(ValueError, match="mismatch"):
        controller.phase("development", "synthetic decision", "e" * 64)
    with pytest.raises(ValueError, match="first two"):
        controller.phase("continuation", "synthetic decision", expected)
    restored.close()
    with (tmp_path / "receipts").open("ab") as stream:
        stream.write(b"torn")
    with pytest.raises(ValueError, match="torn"):
        HostReceipts(tmp_path / "receipts", BINDING, initial=False)


def test_failed_phase_retained_and_retry_denied(tmp_path: Path, monkeypatch: Any) -> None:
    receipts = HostReceipts(tmp_path / "receipts", BINDING, initial=True)
    expected = initialize(tmp_path / "retained", BINDING, receipts)
    controller = DockerController("DO_NOT_EXECUTE", tmp_path / "retained", BINDING, receipts)

    def fail(*args: object) -> str:
        raise RuntimeError("synthetic launch failure")

    monkeypatch.setattr(controller, "launch", fail)
    with pytest.raises(RuntimeError, match="synthetic"):
        controller.phase("development", "synthetic decision", expected)
    assert receipts.records[-1]["kind"] == "FAILURE"
    assert any(r["kind"] == "PHASE" for r in receipts.records)
    with pytest.raises(ValueError, match="failed"):
        controller.phase("development", "another decision cannot reset", expected)
    receipts.close()


def test_archive_index_copy_budget_and_changed_content(tmp_path: Path) -> None:
    budget = Budget(staging=8192, shared=0, archive=32768, terminal_reserve=1024)
    archive = RetainedArchive(tmp_path / "retained", "a" * 64, budget)
    archive.put("cells/00/file", b"exact bytes")
    archive.reserve_copy("copies/00", 11)
    expected = archive.history()
    archive.close()
    resumed = RetainedArchive(tmp_path / "retained", "a" * 64, budget, expected_history=expected)
    assert resumed.staging_reserved == 22
    assert b"".join(resumed.read_chunks("cells/00/file")) == b"exact bytes"
    resumed.reserve_copy("copies/01", 11)
    assert b"".join(resumed.read_chunks("cells/00/file")) == b"exact bytes"
    with pytest.raises(ValueError, match="duplicate"):
        resumed.start("cells/00/file")
    resumed.close()


def test_same_archive_complete_seal_exposure_and_restart(tmp_path: Path) -> None:
    reader = SyntheticReaders(tmp_path)
    archive = RetainedArchive(tmp_path / "retained", "a" * 64, BUDGET)
    reader.journal = life.EvaluationJournal(ArchiveJournal(archive))
    bundle = life.assemble(reader.study, SOURCE, reader, reader, reader.journal)
    assert reader.reads == []
    report = life.evaluate(bundle, reader.journal, reader)
    assert report["coverage"] == {"members": 8, "held_out": 4}
    assert reader.reads == [4, 5, 6, 7]
    assert archive.staging_reserved < archive.budget.staging
    expected = archive.history()
    archive.close()
    resumed = RetainedArchive(tmp_path / "retained", "a" * 64, BUDGET, expected_history=expected)
    journal = life.EvaluationJournal(ArchiveJournal(resumed))
    assert journal.exposed_seal() == bundle.seal
    with pytest.raises(ValueError, match="irreversible"):
        journal.deny_changed("f" * 64)
    resumed.close()


def test_backing_failure_denies_target_release(tmp_path: Path, monkeypatch: Any) -> None:
    reader = SyntheticReaders(tmp_path)
    archive = RetainedArchive(tmp_path / "retained", "a" * 64, BUDGET)
    reader.journal = life.EvaluationJournal(ArchiveJournal(archive))
    bundle = life.assemble(reader.study, SOURCE, reader, reader, reader.journal)

    def fail() -> None:
        raise OSError("synthetic sink failure")

    monkeypatch.setattr(archive, "_flush", fail)
    with pytest.raises(OSError):
        life.evaluate(bundle, reader.journal, reader)
    assert reader.reads == []
    assert archive.poisoned
    archive.close()
