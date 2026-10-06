from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from epsbench.diagnostics.return_view_analytic import CONFIG_SHA256, Frame, adapt, validate_config
from epsbench.diagnostics.return_view_core import (
    REQUIRED,
    HistoryView,
    collision_bijection,
    persistence,
    return_cache,
    score,
)
from epsbench.diagnostics.return_view_execution import (
    LIMIT,
    PATHS,
    attempt,
    launch,
    validate_and_inspect,
)
from epsbench.schema import ModalityPermissionSet


def example(member: int, index: int) -> Frame:
    # Distinct two-by-three synthetic input; no fixed analytic scene is instantiated.
    raw = np.array([[1, 2, 3], [0, 2, 3 if member == 0 else 2]], dtype=np.int64)
    if index in (1, 2):
        raw[1, 2] = 2
    mapping = {i: (i + member * 10, f"surface-{i + member * 10:016x}") for i in (1, 2, 3)}
    return Frame(raw, np.zeros((*raw.shape, 3)), adapt(raw, index, mapping))


def view(member: int, recent: bool) -> HistoryView:
    return HistoryView(
        tuple(example(member, i).observation for i in ((1, 2) if recent else (0, 1, 2))),
        (Fraction(1), Fraction(2)),
        Fraction(-3),
        ModalityPermissionSet(allowed=REQUIRED),
        recent,
    )


def test_collision_and_causal_rules() -> None:
    a, b = view(0, True), view(1, True)
    rename = collision_bijection(a, b)
    assert rename is not None
    assert view(0, False).canonical_bytes() != view(1, False).canonical_bytes(rename)
    assert score(return_cache(view(0, False)), example(0, 3).observation)["exact"]
    assert not score(persistence(a), example(0, 3).observation)["exact"]
    assert (
        json.loads(a.canonical_bytes())["observations"][0]["boundary"][0]["ownership"] == "unknown"
    )
    with pytest.raises(PermissionError):
        return_cache(a)
    with pytest.raises(PermissionError):
        persistence(view(0, False))
    with pytest.raises(ValueError):
        HistoryView(a.observations[:1], a.executed, a.announced, a.permissions, True)
    with pytest.raises(PermissionError):
        HistoryView(
            a.observations, a.executed, a.announced, ModalityPermissionSet.all_modalities(), True
        )
    with pytest.raises(ValueError):
        validate_config({"schema": "unapproved-example"})


def test_action_closure_unknown_and_snapshot() -> None:
    a = view(0, False)
    different = HistoryView(a.observations, a.executed, Fraction(1), a.permissions, False)
    assert score(return_cache(different), example(0, 3).observation)["unknown"] == 3
    with pytest.raises(ValueError):
        a.observations[0].raster.segmentation[0, 0] = 4
    assert "world_points" not in a.canonical_bytes().decode()
    assert "raw_labels" not in a.canonical_bytes().decode()


def test_one_attempt_seal_then_whole_exposure(tmp_path: Path) -> None:
    root = tmp_path / "synthetic"
    calls = []

    def producer(member: int, index: int) -> Frame:
        if index == 3:
            seal = json.loads((root / "seal.json").read_bytes())
            exposure = json.loads((root / "exposure.json").read_bytes())
            assert seal["members"] == exposure["members"] == [0, 1]
            assert all(
                f"forecast-{m}-{r}.npz" in seal["hashes"] for m in range(2) for r in range(2)
            )
        calls.append((member, index))
        return example(member, index)

    result = attempt(root, producer, b"{}", b"{}", (Fraction(1), Fraction(2)), Fraction(-3))
    assert result["status"] == "ADMITTED_WITNESS"
    assert calls == [(m, i) for m in range(2) for i in range(3)] + [(0, 3), (1, 3)]
    assert sum(p.stat().st_size for p in root.iterdir()) <= LIMIT
    assert sum(PATHS.values()) <= LIMIT
    with np.load(root / "frame-0-0.npz", allow_pickle=False) as data:
        assert data["segmentation"].shape == (2, 3)
        assert data["edges"].shape[1] == 5
    before = calls.copy()
    with pytest.raises(FileExistsError):
        attempt(root, producer, b"{}", b"{}", (Fraction(1), Fraction(2)), Fraction(-3))
    assert calls == before
    assert validate_and_inspect(root)["file_count"] == 19
    (root / "result.json").write_bytes(b"{}")
    with pytest.raises(ValueError):
        validate_and_inspect(root)


def test_failure_retains_namespace_and_launch_held(tmp_path: Path) -> None:
    root = tmp_path / "failed"

    def fail(member: int, index: int) -> Frame:
        raise RuntimeError("synthetic failure")

    with pytest.raises(RuntimeError):
        attempt(root, fail, b"{}", b"{}", (Fraction(1), Fraction(2)), Fraction(-3))
    assert (root / "attempt.json").exists()
    assert json.loads((root / "failure.json").read_bytes())["no_retry"]
    auth = tmp_path / "decision.json"
    auth.write_text("{}")
    with pytest.raises(PermissionError):
        launch(tmp_path / "held", auth)
    assert not (tmp_path / "held").exists()


def test_future_only_token_is_explicitly_unscored() -> None:
    full = view(0, False)
    a = full.observations
    raw = np.array([[1, 2, 0], [0, 2, 2]], dtype=np.int64)
    mapping = {i: (i, f"surface-{i:016x}") for i in (1, 2, 3)}
    recent = HistoryView(
        tuple(adapt(raw, i, mapping) for i in (1, 2)),
        full.executed,
        full.announced,
        full.permissions,
        True,
    )
    result = score(persistence(recent), a[0])
    assert result["not_observed_unscored"] == 1
    assert result["exact"] is False


def test_target_failure_keeps_whole_exposure(tmp_path: Path) -> None:
    root = tmp_path / "target-failed"

    def producer(member: int, index: int) -> Frame:
        if index == 3:
            assert json.loads((root / "exposure.json").read_bytes())["members"] == [0, 1]
            raise RuntimeError("distinct synthetic target failure")
        return example(member, index)

    with pytest.raises(RuntimeError):
        attempt(root, producer, b"{}", b"{}", (Fraction(1), Fraction(2)), Fraction(-3))
    assert (root / "seal.json").exists()
    assert (root / "exposure.json").exists()
    assert (root / "failure.json").exists()
    assert not (root / "frame-1-3.npz").exists()


WRONG_ATTESTATIONS: tuple[object, ...] = (1, 1.0, "true", None, [], {}, False)


@pytest.mark.parametrize(
    ("field", "wrong"),
    [
        (field, value)
        for field in ("authorize_one_analytic_attempt", "independent_exact_head_reviews_complete")
        for value in WRONG_ATTESTATIONS
    ]
    + [
        (field, 1)
        for field in ("schema", "source_head", "config_sha256", "authorization_path", "output_path")
    ],
)
def test_launch_decision_strict_types_before_producer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, wrong: object
) -> None:
    from epsbench.diagnostics import return_view_execution as execution

    root = tmp_path / "synthetic-never-launched"
    decision_path = tmp_path / "synthetic-malformed-decision.json"

    # Synthetic Git binding: never issue a real exact-source launch decision.
    def metadata(command: list[str], **kwargs: object) -> str:
        return "synthetic-source-head" if command[1] == "rev-parse" else ""

    def producer_denied(*args: object, **kwargs: object) -> None:
        pytest.fail("malformed decision reached Producer construction")

    monkeypatch.setattr(subprocess, "check_output", metadata)
    monkeypatch.setattr(execution, "Producer", producer_denied)
    decision: dict[str, object] = {
        "schema": "return-view-launch-decision-v1",
        "source_head": "synthetic-source-head",
        "config_sha256": CONFIG_SHA256,
        "authorization_path": str(decision_path.resolve()),
        "output_path": str(root.resolve()),
        "authorize_one_analytic_attempt": True,
        "independent_exact_head_reviews_complete": True,
    }
    decision[field] = wrong
    decision_path.write_text(json.dumps(decision))
    with pytest.raises(PermissionError):
        execution.launch(root, decision_path)
    assert not root.exists()


@pytest.mark.parametrize("shape", [None, [], True, "object"])
def test_launch_decision_requires_json_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shape: object
) -> None:
    from epsbench.diagnostics import return_view_execution as execution

    def metadata(command: list[str], **kwargs: object) -> str:
        return "synthetic-source-head" if command[1] == "rev-parse" else ""

    def producer_denied(*args: object, **kwargs: object) -> None:
        pytest.fail("non-object decision reached Producer construction")

    monkeypatch.setattr(subprocess, "check_output", metadata)
    monkeypatch.setattr(execution, "Producer", producer_denied)
    decision_path = tmp_path / "synthetic-non-object.json"
    decision_path.write_text(json.dumps(shape))
    root = tmp_path / "synthetic-never-launched"
    with pytest.raises(PermissionError):
        execution.launch(root, decision_path)
    assert not root.exists()
