"""Synthetic byte, admission, failure and recovery checks; no native imports."""

from __future__ import annotations

import ast
import hashlib
import json
import struct
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pytest
from PIL import Image

from epsbench.data.output import OutputWriter
from epsbench.diagnostics.a1_retention import Budget, Recovery, RetainedArchive, recover
from epsbench.utils.canonical import write_canonical_json

STUDY = "a" * 64
BUDGET = Budget(staging=8192, shared=0, archive=16384, terminal_reserve=1024)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state(path: Path, budget: Budget = BUDGET) -> Recovery:
    return recover(path, STUDY, budget, digest(path))


def test_serializer_bytes_and_ack(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    archive = RetainedArchive(tmp_path / "archive", STUDY, BUDGET)
    writer = OutputWriter(stage, archive)
    array = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    writer.array(stage / "array.npy", array)
    np.save(tmp_path / "original.npy", array, allow_pickle=False)
    writer.png(stage / "image.png", array)
    Image.fromarray(array, mode="RGB").save(
        tmp_path / "original.png", compress_level=9, optimize=False
    )
    writer.canonical(stage / "data.json", {"z": [1], "a": "é"})
    write_canonical_json(tmp_path / "original.json", {"z": [1], "a": "é"})
    writer.text(stage / "run.json", '{\n  "volatile": true\n}\n')
    (tmp_path / "original-run.json").write_text('{\n  "volatile": true\n}\n', encoding="utf-8")
    for actual, original in (
        ("array.npy", "original.npy"),
        ("image.png", "original.png"),
        ("data.json", "original.json"),
        ("run.json", "original-run.json"),
    ):
        assert (stage / actual).read_bytes() == (tmp_path / original).read_bytes()
    recovered = state(archive.path)
    assert len(recovered.committed) == 4
    assert recovered.incomplete == ()
    assert recovered.suffix_bytes == 0
    archive.close()


def test_cap_exact_and_one_byte_over(tmp_path: Path) -> None:
    # Compute the frame size before admission, including its four-byte length.
    budget = Budget(staging=10, shared=0, archive=2048, terminal_reserve=512)
    archive = RetainedArchive(tmp_path / "archive", STUDY, budget)
    size = archive.stream.tell()
    value = {"sequence": archive.sequence, "kind": "FAILURE", "reason": "x"}
    frame_size = 4 + len(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
    exact = Budget(staging=10, shared=0, archive=size + frame_size + 512, terminal_reserve=512)
    archive.budget = exact
    archive._append({"kind": "FAILURE", "reason": "x"})
    assert archive.stream.tell() == exact.archive - exact.terminal_reserve
    with pytest.raises(ValueError, match="admission"):
        archive._append({"kind": "FAILURE", "reason": "x"})
    archive.failure("terminal capacity remains")
    assert archive.stream.tell() <= exact.archive
    archive.close()
    archive = RetainedArchive(tmp_path / "over", STUDY, budget)
    archive.budget = Budget(staging=10, shared=0, archive=exact.archive - 1, terminal_reserve=512)
    before = archive.stream.tell()
    with pytest.raises(ValueError, match="admission"):
        archive._append({"kind": "FAILURE", "reason": "x"})
    assert archive.stream.tell() == before
    archive.close()


@pytest.mark.parametrize("path", ["../escape", "/root", "x/../y", "x//y", "x\\y", "C:x", "./x"])
def test_paths_denied(tmp_path: Path, path: str) -> None:
    archive = RetainedArchive(tmp_path / "archive", STUDY, BUDGET)
    with pytest.raises(ValueError, match="path"):
        archive.start(path)
    archive.close()


def test_duplicate_and_wrong_binding(tmp_path: Path) -> None:
    archive = RetainedArchive(tmp_path / "archive", STUDY, BUDGET)
    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"one")
    archive.start("artifact")
    archive.reserve_staging(3)
    with pytest.raises(ValueError, match="reservation"):
        archive.publish(artifact, "different")
    archive.publish(artifact, "artifact")
    with pytest.raises(ValueError, match="duplicate"):
        archive.start("artifact")
    receipt = digest(archive.path)
    archive.close()
    resumed = RetainedArchive(archive.path, STUDY, BUDGET, expected_history=receipt)
    with pytest.raises(ValueError, match="duplicate"):
        resumed.start("artifact")
    resumed.close()


def test_serialization_interruption_keeps_receipts(tmp_path: Path) -> None:
    archive = RetainedArchive(tmp_path / "archive", STUDY, BUDGET)
    writer = OutputWriter(tmp_path, archive)

    def interrupted(stream: BinaryIO) -> None:
        stream.write(b"partial")
        raise RuntimeError("interruption")

    with pytest.raises(RuntimeError):
        writer._write(tmp_path / "partial", interrupted)
    assert (tmp_path / "partial").read_bytes() == b"partial"
    assert state(archive.path).incomplete == ("partial",)
    assert state(archive.path).committed == ()
    assert b'"kind":"FAILURE"' in archive.path.read_bytes()
    archive.close()


def test_staging_admission_before_write(tmp_path: Path) -> None:
    archive = RetainedArchive(tmp_path / "archive", STUDY, BUDGET)
    writer = OutputWriter(tmp_path, archive)
    writer.text(tmp_path / "one", "x" * BUDGET.staging)
    with pytest.raises(ValueError, match="staging"):
        writer.text(tmp_path / "two", "x")
    assert (tmp_path / "two").read_bytes() == b""
    assert writer.staged_bytes == BUDGET.staging
    archive.close()


def test_failed_flush_no_ack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = RetainedArchive(tmp_path / "archive", STUDY, BUDGET)
    archive.start("file")
    archive.reserve_staging(4)
    (tmp_path / "file").write_bytes(b"data")

    def fail() -> None:
        raise OSError("flush failed")

    monkeypatch.setattr(archive, "_flush", fail)
    with pytest.raises(OSError):
        archive.publish(tmp_path / "file", "file")
    assert archive.pending == "file"
    assert archive.poisoned
    with pytest.raises(ValueError, match="failed sink"):
        archive.failure("flush failure")
    archive.close()


def test_committed_prefix_and_torn_suffix_retained(tmp_path: Path) -> None:
    archive = RetainedArchive(tmp_path / "archive", STUDY, BUDGET)
    writer = OutputWriter(tmp_path, archive)
    writer.text(tmp_path / "done", "retained")
    archive.start("unfinished")
    archive.close()
    with archive.path.open("ab") as stream:
        stream.write(struct.pack("!I", 100) + b'{"kind":')
    before = archive.path.read_bytes()
    recovered = state(archive.path)
    assert recovered.committed[0][0] == "done"
    assert recovered.incomplete == ("unfinished",)
    assert recovered.suffix_bytes == 12
    with pytest.raises(ValueError, match="INCOMPLETE_UNCOMMITTED"):
        RetainedArchive(archive.path, STUDY, BUDGET, expected_history=digest(archive.path))
    assert archive.path.read_bytes() == before


@pytest.mark.parametrize(
    "record",
    [
        {"sequence": 9, "kind": "START", "path": "file"},
        {"sequence": 1, "kind": "UNKNOWN"},
        {"sequence": 1, "kind": "START", "path": "../escape"},
        {"sequence": 1, "kind": "COMMIT", "length": 1, "sha256": "a" * 64},
    ],
)
def test_bad_record_denied(tmp_path: Path, record: dict[str, Any]) -> None:
    archive = RetainedArchive(tmp_path / "archive", STUDY, BUDGET)
    archive.close()
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    with archive.path.open("ab") as stream:
        stream.write(struct.pack("!I", len(payload)) + payload)
    with pytest.raises(ValueError):
        state(archive.path)


def test_hash_and_oversized_record_denied(tmp_path: Path) -> None:
    archive = RetainedArchive(tmp_path / "archive", STUDY, BUDGET)
    archive.start("file")
    archive._append({"kind": "COMMIT", "length": 0, "sha256": "a" * 64})
    archive.close()
    with pytest.raises(ValueError, match="hash"):
        state(archive.path)
    other = RetainedArchive(tmp_path / "other", STUDY, BUDGET)
    other.close()
    with other.path.open("ab") as stream:
        stream.write(struct.pack("!I", 100 * 1024 + 1))
    with pytest.raises(ValueError, match="length"):
        state(other.path)


def test_changed_missing_and_new_namespace_denied(tmp_path: Path) -> None:
    archive = RetainedArchive(tmp_path / "archive", STUDY, BUDGET)
    receipt = digest(archive.path)
    archive.failure("retained failure")
    archive.close()
    with pytest.raises(ValueError, match="history"):
        recover(archive.path, STUDY, BUDGET, receipt)
    with pytest.raises(ValueError, match="study"):
        recover(archive.path, "b" * 64, BUDGET, digest(archive.path))
    with pytest.raises(FileNotFoundError):
        RetainedArchive(tmp_path / "fresh", STUDY, BUDGET, expected_history=receipt)
    with pytest.raises(ValueError, match="resume"):
        RetainedArchive(archive.path, STUDY, BUDGET)


def test_generator_has_no_direct_writes() -> None:
    # Inspect source under the native-import guard, without importing generation.
    source = Path("src/epsbench/data/generate.py").read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            assert not (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in {"save", "write_text", "write_bytes"}
            )
            assert not (isinstance(node.func, ast.Name) and node.func.id == "write_canonical_json")
    assert "publisher: ArtifactPublisher | None = None" in source


def test_hardlinks_denied(tmp_path: Path) -> None:
    archive = RetainedArchive(tmp_path / "archive", STUDY, BUDGET)
    original = tmp_path / "original"
    original.write_bytes(b"data")
    (tmp_path / "alias").hardlink_to(original)
    archive.start("alias")
    with pytest.raises(ValueError, match="regular"):
        archive.publish(tmp_path / "alias", "alias")
    archive.failure("hardlink denied")
    archive.close()


def test_new_writer_and_resume_cannot_reset_staging(tmp_path: Path) -> None:
    archive = RetainedArchive(tmp_path / "archive", STUDY, BUDGET)
    OutputWriter(tmp_path, archive).text(tmp_path / "one", "x" * BUDGET.staging)
    with pytest.raises(ValueError, match="staging"):
        OutputWriter(tmp_path, archive).text(tmp_path / "two", "x")
    assert state(archive.path).staging_reserved == BUDGET.staging
    archive.close()
    clean = RetainedArchive(tmp_path / "clean", STUDY, BUDGET)
    OutputWriter(tmp_path, clean).text(tmp_path / "three", "x" * BUDGET.staging)
    receipt = digest(clean.path)
    clean.close()
    resumed = RetainedArchive(clean.path, STUDY, BUDGET, expected_history=receipt)
    resumed.start("four")
    with pytest.raises(ValueError, match="staging"):
        resumed.reserve_staging(1)
    resumed.failure("quota retained across restart")
    resumed.close()


def test_interrupted_transfer_preserves_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = RetainedArchive(tmp_path / "archive", STUDY, BUDGET)
    writer = OutputWriter(tmp_path, archive)
    writer.text(tmp_path / "done", "done")
    append = archive._append

    def interrupted(values: dict[str, Any], *, terminal: bool = False) -> None:
        if values["kind"] == "COMMIT":
            raise OSError("transfer interrupted before acknowledgment")
        append(values, terminal=terminal)

    monkeypatch.setattr(archive, "_append", interrupted)
    with pytest.raises(OSError):
        writer.text(tmp_path / "partial", "transferred but uncommitted")
    recovered = state(archive.path)
    assert recovered.committed[0][0] == "done"
    assert recovered.incomplete == ("partial",)
    assert b"transfer" in (tmp_path / "partial").read_bytes()
    archive.close()


def test_canonical_serialization_failure_has_start_receipt(tmp_path: Path) -> None:
    archive = RetainedArchive(tmp_path / "archive", STUDY, BUDGET)
    with pytest.raises(ValueError):
        OutputWriter(tmp_path, archive).canonical(tmp_path / "invalid.json", {"bad": float("nan")})
    assert state(archive.path).incomplete == ("invalid.json",)
    assert b'"kind":"FAILURE"' in archive.path.read_bytes()
    archive.close()


def test_changed_transfer_denies_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = RetainedArchive(tmp_path / "archive", STUDY, BUDGET)
    artifact = tmp_path / "file"
    artifact.write_bytes(b"data")
    archive.start("file")
    archive.reserve_staging(4)
    append = archive._append

    def changing(values: dict[str, Any], *, terminal: bool = False) -> None:
        append(values, terminal=terminal)
        if values["kind"] == "CHUNK":
            artifact.write_bytes(b"changed")

    monkeypatch.setattr(archive, "_append", changing)
    with pytest.raises(ValueError, match="changed"):
        archive.publish(artifact, "file")
    archive.failure("read stability failed")
    assert state(archive.path).committed == ()
    archive.close()


def test_symlink_denied(tmp_path: Path) -> None:
    archive = RetainedArchive(tmp_path / "archive", STUDY, BUDGET)
    actual = tmp_path / "actual"
    actual.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(actual, target_is_directory=True)
    except OSError:
        archive.close()
        pytest.skip("host does not grant symlink creation")
    with pytest.raises(ValueError, match="linked"):
        OutputWriter(tmp_path, archive).text(link / "file", "data")
    archive.close()


@pytest.mark.parametrize(
    "data", ["not base64!", "", "eA==" * (64 * 1024 + 1)], ids=["invalid", "empty", "oversized"]
)
def test_invalid_chunk_denied(tmp_path: Path, data: str) -> None:
    archive = RetainedArchive(tmp_path / "archive", STUDY, BUDGET)
    archive.start("file")
    archive.reserve_staging(1)
    try:
        archive._append({"kind": "CHUNK", "data": data})
    except ValueError:
        # A record larger than the frame bound is rejected before any write.
        pass
    else:
        archive.close()
        with pytest.raises(ValueError):
            state(archive.path)
        return
    archive.close()


def test_cell_prefixes_share_budget_and_archive(tmp_path: Path) -> None:
    archive = RetainedArchive(tmp_path / "archive", STUDY, BUDGET)
    for cell in ("cell-0", "cell-1"):
        root = tmp_path / cell
        root.mkdir()
        OutputWriter(root, archive, publication_prefix=cell).text(root / "run.json", "x")
    recovered = state(archive.path)
    assert [item[0] for item in recovered.committed] == ["cell-0/run.json", "cell-1/run.json"]
    assert recovered.staging_reserved == 2
    archive.close()


@pytest.mark.parametrize("prefix", ["../cell", "/cell", "cell//a", "cell\\a", "C:cell"])
def test_invalid_prefix_denied(tmp_path: Path, prefix: str) -> None:
    with pytest.raises(ValueError, match="prefix"):
        OutputWriter(tmp_path, publication_prefix=prefix)
