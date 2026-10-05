"""Guarded file-backed acceptance on explicitly synthetic canonical records."""

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from epsbench.data.loader import PermissionDeniedError
from epsbench.diagnostics import a1_lifecycle as life
from epsbench.diagnostics.a1_action_contrast import cases
from epsbench.diagnostics.a1_files import A1CanonicalLoader, A1Files
from epsbench.schema import ModalityPermissionSet
from tests.a1_file_fixtures import make_dataset

SOURCE = Path(__file__).resolve().parents[1]


def files(tmp_path: Path, *, future_summary_delta: int = 0) -> A1Files:
    loaders = tuple(
        A1CanonicalLoader(
            make_dataset(
                tmp_path / str(c.ordinal),
                c.config,
                future_summary_delta=future_summary_delta if c.partition == "held_out" else 0,
            ),
            ModalityPermissionSet.all_modalities(),
        )
        for c in cases(SOURCE)
    )
    return A1Files.membership(SOURCE, "b" * 40, "c" * 40, "d" * 64, loaders)


def test_file_membership_and_complete_retention_before_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads: list[tuple[Path, str]] = []
    original = A1CanonicalLoader._load_npy

    def tracked(self: A1CanonicalLoader, record: Any) -> Any:
        reads.append((self.root, record.path))
        return original(self, record)

    monkeypatch.setattr(A1CanonicalLoader, "_load_npy", tracked)
    data = files(tmp_path)
    assert len(data.study.members) == 8
    assert all("segmentation_0" in path for _, path in reads)
    journal = life.EvaluationJournal(tmp_path)
    with pytest.raises(ValueError, match="four held-out forecasts"):
        data.evaluator(life.Bundle(data.study, "a" * 64, (), ""), journal)
    bundle = life.assemble(
        data.study, SOURCE, data.before_reader(), data.development_reader(), journal
    )
    held = {loader.root for loader in data.loaders[4:]}
    assert all("segmentation_0" in path for root, path in reads if root in held)
    evaluator = data.evaluator(bundle, journal)
    with pytest.raises(ValueError, match="exposure required"):
        evaluator.read_held_out_target(data.study.members[4])
    journal.verify_commit(bundle)
    count = len(reads)
    journal.expose(bundle)
    result = life.evaluate(bundle, journal, evaluator)
    assert result["coverage"] == {"members": 8, "held_out": 4}
    fate_reads = [(root, path) for root, path in reads[count:] if "before_fate_codes" in path]
    assert {root for root, _ in fate_reads} == held
    assert len(fate_reads) == 4


def test_partition_permissions_and_changed_files_deny_release(tmp_path: Path) -> None:
    data = files(tmp_path)
    with pytest.raises(ValueError, match="denies held-out"):
        data.development_reader().read_development_target(data.study.members[4])
    loader = data.loaders[4]
    loader.permissions = ModalityPermissionSet(allowed=frozenset())
    with pytest.raises(PermissionDeniedError):
        data.before_reader().read_before(data.study.members[4])
    loader.permissions = ModalityPermissionSet.all_modalities()
    path = loader.root / "episode-000000/segmentation_0.npy"
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="hash mismatch"):
        data.before_reader().read_before(data.study.members[4])


def test_changed_manifest_and_membership_rejected(tmp_path: Path) -> None:
    data = files(tmp_path)
    with pytest.raises(ValueError, match="eight"):
        A1Files.membership(SOURCE, "b" * 40, "c" * 40, "d" * 64, data.loaders[:-1])
    with pytest.raises(ValueError):
        A1Files.membership(SOURCE, "b" * 40, "c" * 40, "d" * 64, (data.loaders[0],) * 8)
    member = replace(data.study.members[4], dataset=life.DatasetIdentity("f" * 64))
    with pytest.raises(ValueError, match="unknown"):
        data.before_reader().read_before(member)
    path = data.loaders[4].root / "manifest.json"
    manifest = data.loaders[4].read_dataset_manifest()
    manifest = manifest.model_copy(update={"source_provenance_sha256": "f" * 64})
    path.write_text(manifest.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance hash"):
        data.before_reader().read_before(data.study.members[4])


def test_target_file_hash_and_exposure_membership_binding(tmp_path: Path) -> None:
    data = files(tmp_path)
    journal = life.EvaluationJournal(tmp_path)
    bundle = life.assemble(
        data.study, SOURCE, data.before_reader(), data.development_reader(), journal
    )
    evaluator = data.evaluator(bundle, journal)
    journal.expose(bundle)
    path = data.loaders[4].root / "episode-000000/before_fate_codes.npy"
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="hash mismatch"):
        evaluator.read_held_out_target(data.study.members[4])
    journal.path.write_text(
        '{"seal":"' + bundle.seal + '","study":"a1_contrasting_action_v1",'
        '"whole_membership_exposed":true}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exposure binding"):
        evaluator.read_held_out_target(data.study.members[5])


def test_failed_persistence_never_opens_held_out_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = files(tmp_path)
    journal = life.EvaluationJournal(tmp_path)
    bundle = life.assemble(
        data.study, SOURCE, data.before_reader(), data.development_reader(), journal
    )
    evaluator = data.evaluator(bundle, journal)
    reads: list[int] = []
    original = A1CanonicalLoader.read_a1_fate

    def tracked(self: A1CanonicalLoader, artifact: Any) -> Any:
        reads.append(1)
        return original(self, artifact)

    monkeypatch.setattr(A1CanonicalLoader, "read_a1_fate", tracked)

    def fail(_descriptor: int) -> None:
        raise OSError("synthetic durability failure")

    monkeypatch.setattr(os, "fsync", fail)
    with pytest.raises(OSError, match="durability failure"):
        life.evaluate(bundle, journal, evaluator)
    assert reads == []


@pytest.mark.parametrize(
    "field",
    [
        "dataset_logical_sha256",
        "source_provenance_sha256",
        "renderer_execution_provenance_sha256",
        "content_provenance_binding_sha256",
    ],
)
def test_declared_provenance_hash_mutation_rejected(tmp_path: Path, field: str) -> None:
    data = files(tmp_path)
    loader = data.loaders[4]
    manifest = loader.read_dataset_manifest().model_copy(update={field: "f" * 64})
    (loader.root / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance hash"):
        data.before_reader().read_before(data.study.members[4])


def test_actual_transition_and_unsafe_path_rejected(tmp_path: Path) -> None:
    data = files(tmp_path)
    loader = data.loaders[4]
    manifest = loader.read_dataset_manifest()
    transition = loader.root / manifest.episodes[0].transition.path
    transition.write_bytes(transition.read_bytes() + b" ")
    with pytest.raises(ValueError, match="byte count mismatch"):
        data.before_reader().read_before(data.study.members[4])
    value = manifest.model_dump(mode="json")
    value["episodes"][0]["transition"]["path"] = "../transition.json"
    (loader.root / "manifest.json").write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        data.before_reader().read_before(data.study.members[4])


def test_future_summary_mutation_cannot_change_forecasts(tmp_path: Path) -> None:
    original = files(tmp_path / "original")
    changed = files(tmp_path / "changed", future_summary_delta=1)
    journal1 = tmp_path / "journal1"
    journal2 = tmp_path / "journal2"
    journal1.mkdir()
    journal2.mkdir()
    first = life.assemble(
        original.study,
        SOURCE,
        original.before_reader(),
        original.development_reader(),
        life.EvaluationJournal(journal1),
    )
    second = life.assemble(
        changed.study,
        SOURCE,
        changed.before_reader(),
        changed.development_reader(),
        life.EvaluationJournal(journal2),
    )
    assert first.study.root != second.study.root
    assert first.seal != second.seal
    assert any(np.count_nonzero(f.forecast.original.boundary_scores) for f in first.forecasts)
    for left, right in zip(first.forecasts, second.forecasts, strict=True):
        assert left.own_view_digest == right.own_view_digest
        assert left.forecast.alignment == right.forecast.alignment
        assert (
            np.count_nonzero(left.forecast.original.boundary_scores)
            + np.count_nonzero(left.forecast.wrong_action)
        ) > 0
        for a, b in (
            (left.forecast.original.boundary_scores, right.forecast.original.boundary_scores),
            (left.forecast.original.template_scores, right.forecast.original.template_scores),
            (left.forecast.aligned, right.forecast.aligned),
            (left.forecast.wrong_action, right.forecast.wrong_action),
        ):
            assert np.array_equal(a, b)
