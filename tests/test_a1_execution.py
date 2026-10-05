"""Synthetic archive binding/chronology tests; no subprocess or native imports."""

from __future__ import annotations

import json
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
        controller.phase("development", "synthetic decision", "e" * 64, restored.root)
    with pytest.raises(ValueError, match="first two"):
        controller.phase("continuation", "synthetic decision", expected, restored.root)
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
        controller.phase("development", "synthetic decision", expected, receipts.root)
    assert receipts.records[-1]["kind"] == "FAILURE"
    assert any(r["kind"] == "PHASE" for r in receipts.records)
    with pytest.raises(ValueError, match="failed"):
        controller.phase("development", "another decision cannot reset", expected, receipts.root)
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


def simulated_launches(controller: DockerController, monkeypatch: Any) -> None:
    """Independent synthetic completion bytes; never creates a Docker process."""

    def launch(task: str, expected: str, deadline: float) -> str:
        receipts = controller.receipts
        token = f"{len(receipts.records):032x}"
        receipts.append(
            {
                "kind": "ATTEMPT",
                "task": task,
                "name": "eps-a1-" + token,
                "token": token,
                "expected_history": expected,
                "seconds": 300,
            }
        )
        for verb in ("create", "inspect", "start", "inspect", "inspect", "rm"):
            receipts.append(
                {
                    "kind": "COMMAND",
                    "args": [verb],
                    "exit": 0,
                    "timeout": False,
                    "overflow": False,
                    "stdout": "",
                    "stderr": "",
                }
            )
        archive = RetainedArchive(
            controller.archive, controller.binding.root, BUDGET, expected_history=expected
        )
        archive.put(f"operations/{task}-complete.json", b'{"synthetic":true}')
        observed = archive.history()
        archive.close()
        receipts.append({"kind": "CHECKPOINT", "task": task, "history": observed})
        return observed

    monkeypatch.setattr(controller, "launch", launch)


def development_prefix(tmp_path: Path, monkeypatch: Any) -> tuple[str, str]:
    receipts = HostReceipts(tmp_path / "receipts", BINDING, initial=True)
    expected = initialize(tmp_path / "retained", BINDING, receipts)
    controller = DockerController("DO_NOT_EXECUTE", tmp_path / "retained", BINDING, receipts)
    simulated_launches(controller, monkeypatch)
    observed = controller.phase("development", "synthetic development", expected, receipts.root)
    root = receipts.root
    receipts.close()
    return observed, root


def rechain(path: Path, records: list[dict[str, Any]]) -> None:
    previous = "0" * 64
    payload = bytearray()
    for sequence, value in enumerate(records):
        body = {
            k: v for k, v in value.items() if k not in {"sequence", "previous_sha256", "sha256"}
        }
        body.update(sequence=sequence, previous_sha256=previous)
        previous = life.digest(body)
        payload.extend(
            json.dumps(
                {**body, "sha256": previous}, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
            + b"\n"
        )
    path.write_bytes(payload)


def test_mutated_elapsed_checksum_and_recomputed_stale_authorization(
    tmp_path: Path, monkeypatch: Any
) -> None:
    expected, root = development_prefix(tmp_path, monkeypatch)
    path = tmp_path / "receipts"
    records = [json.loads(line) for line in path.read_bytes().splitlines()]
    records[-1]["elapsed"] = 0
    path.write_bytes(
        b"".join(
            json.dumps(r, sort_keys=True, separators=(",", ":")).encode() + b"\n" for r in records
        )
    )
    with pytest.raises(ValueError, match="checksum"):
        HostReceipts(path, BINDING, initial=False)
    rechain(path, records)
    receipts = HostReceipts(path, BINDING, initial=False)
    controller = DockerController("DO_NOT_EXECUTE", tmp_path / "retained", BINDING, receipts)
    with pytest.raises(ValueError, match="stale"):
        controller.phase("continuation", "old authorization", expected, root)
    receipts.close()


@pytest.mark.parametrize("elapsed", [-1, True, "0", float("inf"), float("nan")])
def test_invalid_elapsed_denied_even_with_valid_json(
    tmp_path: Path, monkeypatch: Any, elapsed: Any
) -> None:
    development_prefix(tmp_path, monkeypatch)
    path = tmp_path / "receipts"
    records = [json.loads(line) for line in path.read_bytes().splitlines()]
    records[-1]["elapsed"] = elapsed
    if type(elapsed) is float:
        path.write_bytes(
            b"".join(
                json.dumps(r, sort_keys=True, separators=(",", ":")).encode() + b"\n"
                for r in records
            )
        )
    else:
        rechain(path, records)
    with pytest.raises(ValueError):
        HostReceipts(path, BINDING, initial=False)


def test_invalid_rechained_chronology_denied(tmp_path: Path, monkeypatch: Any) -> None:
    development_prefix(tmp_path, monkeypatch)
    path = tmp_path / "receipts"
    records = [json.loads(line) for line in path.read_bytes().splitlines()]
    # Remove the successful first task attempt but retain its checkpoint and rechecksum.
    records = [r for r in records if not (r["kind"] == "ATTEMPT" and r["task"] == "cell-00")]
    rechain(path, records)
    with pytest.raises(ValueError, match="command"):
        HostReceipts(path, BINDING, initial=False)


def test_valid_prefix_rollback_cannot_erase_consumed_continuation(
    tmp_path: Path, monkeypatch: Any
) -> None:
    expected, root = development_prefix(tmp_path, monkeypatch)
    path = tmp_path / "receipts"
    prefix = path.read_bytes()
    receipts = HostReceipts(path, BINDING, initial=False)
    controller = DockerController("DO_NOT_EXECUTE", tmp_path / "retained", BINDING, receipts)

    def fail(*args: object) -> str:
        raise RuntimeError("crash before first continuation native launch")

    monkeypatch.setattr(controller, "launch", fail)
    with pytest.raises(RuntimeError, match="crash"):
        controller.phase("continuation", "synthetic continuation", expected, root)
    receipts.close()
    assert history(tmp_path / "retained") != expected  # consumption is independently durable
    path.write_bytes(prefix)
    restored = HostReceipts(path, BINDING, initial=False)
    controller = DockerController("DO_NOT_EXECUTE", tmp_path / "retained", BINDING, restored)
    with pytest.raises(ValueError, match="mismatch"):
        controller.phase("continuation", "replayed old authorization", expected, root)
    restored.close()


def test_unchanged_valid_continuation_and_dummy_replay(tmp_path: Path, monkeypatch: Any) -> None:
    expected, root = development_prefix(tmp_path, monkeypatch)
    receipts = HostReceipts(tmp_path / "receipts", BINDING, initial=False)
    controller = DockerController("DO_NOT_EXECUTE", tmp_path / "retained", BINDING, receipts)
    simulated_launches(controller, monkeypatch)
    result = controller.phase("continuation", "synthetic continuation", expected, root)
    assert result == history(tmp_path / "retained")
    receipts.close()
    restored = HostReceipts(tmp_path / "receipts", BINDING, initial=False)
    assert [r["phase"] for r in restored.records if r["kind"] == "COMPLETE"] == [
        "development",
        "continuation",
    ]
    restored.close()
    dummy = Binding(
        BINDING.source_head,
        BINDING.source_tree,
        BINDING.configuration_root,
        BINDING.image,
        "a1_dummy_qualification_v1",
    )
    dummy_receipts = HostReceipts(tmp_path / "dummy-receipts", dummy, initial=True)
    dummy_expected = initialize(tmp_path / "dummy-archive", dummy, dummy_receipts)
    dummy_controller = DockerController(
        "DO_NOT_EXECUTE", tmp_path / "dummy-archive", dummy, dummy_receipts
    )
    simulated_launches(dummy_controller, monkeypatch)
    dummy_controller.launch("dummy-complete", dummy_expected, 0)
    dummy_receipts.close()
    HostReceipts(tmp_path / "dummy-receipts", dummy, initial=False).close()


def test_crash_between_archive_consumption_and_host_phase(tmp_path: Path, monkeypatch: Any) -> None:
    expected, root = development_prefix(tmp_path, monkeypatch)
    path = tmp_path / "receipts"
    prefix = path.read_bytes()
    receipts = HostReceipts(path, BINDING, initial=False)
    original = receipts.append

    def crash(value: dict[str, Any]) -> None:
        if value["kind"] == "PHASE":
            raise OSError("synthetic crash after archive fsync before host PHASE")
        original(value)

    monkeypatch.setattr(receipts, "append", crash)
    controller = DockerController("DO_NOT_EXECUTE", tmp_path / "retained", BINDING, receipts)
    with pytest.raises(OSError, match="synthetic crash"):
        controller.phase("continuation", "authorized before crash", expected, root)
    assert not any(r["kind"] == "PHASE" and r["phase"] == "continuation" for r in receipts.records)
    assert receipts.records[-1]["kind"] == "FAILURE"
    receipts.close()
    path.write_bytes(prefix)
    restored = HostReceipts(path, BINDING, initial=False)
    controller = DockerController("DO_NOT_EXECUTE", tmp_path / "retained", BINDING, restored)
    with pytest.raises(ValueError, match="mismatch"):
        controller.phase("continuation", "stale after crash", expected, root)
    restored.close()


def test_legal_uncertain_phase_prefix_denies_restart(tmp_path: Path, monkeypatch: Any) -> None:
    expected, root = development_prefix(tmp_path, monkeypatch)
    path = tmp_path / "receipts"
    receipts = HostReceipts(path, BINDING, initial=False)
    controller = DockerController("DO_NOT_EXECUTE", tmp_path / "retained", BINDING, receipts)

    def fail(*args: object) -> str:
        raise RuntimeError("synthetic termination before attempt")

    monkeypatch.setattr(controller, "launch", fail)
    with pytest.raises(RuntimeError):
        controller.phase("continuation", "synthetic authorization", expected, root)
    records = list(receipts.records)
    receipts.close()
    phase_index = next(
        i for i, r in enumerate(records) if r["kind"] == "PHASE" and r["phase"] == "continuation"
    )
    path.write_bytes(
        b"".join(
            json.dumps(r, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            for r in records[: phase_index + 1]
        )
    )
    restored = HostReceipts(path, BINDING, initial=False)
    controller = DockerController("DO_NOT_EXECUTE", tmp_path / "retained", BINDING, restored)
    with pytest.raises(ValueError, match="already attempted"):
        controller.phase(
            "continuation",
            "cannot finish uncertain prefix",
            history(tmp_path / "retained"),
            restored.root,
        )
    restored.close()
