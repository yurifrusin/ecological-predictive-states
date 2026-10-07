"""Handwritten fixed-cell returns; fake operation signals, never real ray computation."""

from __future__ import annotations

import json
from fractions import Fraction as Q
from pathlib import Path
from typing import Any

import pytest

from epsbench.diagnostics import event_cell_mapping as m
from epsbench.diagnostics.mask_development_qualification import Study
from epsbench.schema import Modality, ModalityPermissionSet
from epsbench.utils.canonical import canonical_json_bytes, sha256_bytes

ACCESS = ModalityPermissionSet(allowed=frozenset({Modality.PRIVILEGED_GENERATION_RECORDS}))
IDENTITY = {"episode": "a" * 32, "tokens": [f"surface-{i:016x}" for i in (1, 2, 3)]}
CONTEXT = {
    "version": m.VERSION,
    "head": "b" * 40,
    "tree": "c" * 40,
    "versions": {"fixture": "handwritten-v1"},
    "sources": {n: "d" * 64 for n in m.SOURCES},
    "cell_sha256": sha256_bytes(m.CELL_BYTES),
}


def handwritten(index: int) -> list[list[int]]:
    rows = [[0] * 32 for _ in range(32)]
    for r in range(14, 20):
        for c in ((31,), (29, 30), (), (27, 28))[index]:
            rows[r][c] = 2
    for r in range(15, 18):
        for c in ((28,), (27, 28), (29,), (26,))[index]:
            rows[r][c] = 3
    for c in ((17,), (15, 16), (18,), (13,))[index]:
        rows[21][c] = 1
    return rows


class Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


class Fake:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.calls: list[tuple[int, str]] = []
        self.mode = "normal"
        self.clock = Clock()
        self.external_bytes = 0

    def produce(self, study: Study, witness: dict[str, Any], check: Any) -> dict[str, Any]:
        index = witness["index"]
        self.calls.append((index, "raw"))
        assert study.payload == m.CELL_BYTES
        assert witness["producer_lateral"] == str(m.POSITIONS[index])
        assert study.boxes[1].lower == (Q(22, 5), Q(3), Q(1, 5))
        assert (self.path / f"{index}-raw-attempt.json").exists()
        if index >= 2:
            assert (self.path / "seal.json").exists()
            for branch in (0, 1):
                assert set(m.read(self.path, f"before-{branch}.json")["candidates"]) == set(
                    m.METHODS
                )
        labels = handwritten(index)
        if self.mode == "contradiction":
            labels[0][0] = 1
        if self.mode == "timeout" and index == 2:
            self.clock.value = 61
        if self.mode == "budget" and index == 2:
            self.external_bytes = 5 * 1024 * 1024
        return {"labels": labels, "ties": []}

    def audit(self, study: Study, witness: dict[str, Any], check: Any) -> dict[str, Any]:
        index = witness["index"]
        self.calls.append((index, "audit"))
        assert (self.path / f"{index}-raw-return.json").exists()
        labels = handwritten(index)
        if self.mode in ("contradiction", "disagreement"):
            labels[0][0] = 1
        return {
            "labels": labels,
            "ties": [],
            "boundaries": [[0, 0]] if self.mode == "boundary" else [],
            "supports": [[1, 1, False], [1, 1, False]],
            "footprints": [
                {
                    "polygon": [["0", "0"], ["1", "0"], ["0", "1"]],
                    "depths": ["3", "4"],
                    "domain": True,
                }
            ]
            * 2,
        }


def run(path: Path, fake: Fake, **kwargs: Any) -> dict[str, Any]:
    return m.run_mapping(
        path,
        ACCESS,
        CONTEXT,
        fake,
        lambda: None,
        kwargs.pop("external", lambda: 0),
        lambda p: None,
        clock=fake.clock,
        allocate=lambda: IDENTITY,
        **kwargs,
    )


def test_complete_fake_smoke_retained_inspection_and_fixed_scores(tmp_path: Path) -> None:
    path = tmp_path / "cell"
    fake = Fake(path)
    result = run(path, fake)
    assert result["completion"] == "COMPLETE"
    assert result["cell_outcome"] == "PASS"
    assert result["qualification"] == "EXACT_CELL_MAPPING_AGREEMENT"
    assert fake.calls == [(i, k) for i in range(4) for k in ("raw", "audit")]
    assert m.inspect(path, ACCESS, lambda: None) == result
    assert len(fake.calls) == 8  # Inspector did not make physical calls.
    expected = {
        "stable": ("1/2", "1/2"),
        "replay": ("1/4", "1/12"),
        "half": ("1/4", "1/4"),
        "visibility_saturation": ("1/6", "0"),
    }
    for method, pair in expected.items():
        assert tuple(b[method]["report"]["brier"] for b in result["branches"]) == pair
    for branch in (0, 1):
        before = m.read(path, f"before-{branch}.json")
        assert before["features"]["endpoints"][0]["pairs"] == []
        assert before["features"]["endpoints"][1]["pairs"] == [[1, 2]]
        encoded = json.dumps(before["structured"])
        assert "surface-" not in encoded and "27/10" not in encoded
    print("Handwritten fake smoke: retention, validation and inspection; no physical qualification")


@pytest.mark.parametrize(
    "mode,outcome",
    [("disagreement", "INCONCLUSIVE"), ("boundary", "INCONCLUSIVE"), ("contradiction", "FAIL")],
)
def test_qualified_contradiction_distinct_from_unresolved_truth(
    tmp_path: Path, mode: str, outcome: str
) -> None:
    path = tmp_path / "cell"
    fake = Fake(path)
    fake.mode = mode
    result = run(path, fake)
    assert result["cell_outcome"] == outcome
    assert result["completion"] == "STOPPED_INCOMPLETE"
    assert len(fake.calls) == 2
    assert (path / "0-raw-return.json").exists() and (path / "0-audit-return.json").exists()
    assert m.inspect(path, ACCESS, lambda: None) == result


def test_timeout_keeps_return_before_stopping_and_no_future_retry(tmp_path: Path) -> None:
    path = tmp_path / "cell"
    fake = Fake(path)
    fake.mode = "timeout"
    result = run(path, fake)
    assert result["cell_outcome"] == "INCONCLUSIVE"
    assert len(fake.calls) == 5
    assert (path / "2-raw-return.json").exists()
    assert not (path / "2-audit-attempt.json").exists()
    assert (path / "seal.json").exists()
    assert m.inspect(path, ACCESS, lambda: None) == result
    with pytest.raises(FileExistsError):
        run(path, fake)
    assert len(fake.calls) == 5


def test_inspector_tamper_source_and_call_arguments_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "cell"
    fake = Fake(path)
    run(path, fake)

    def denied() -> None:
        raise ValueError("changed source")

    with pytest.raises(ValueError, match="changed source"):
        m.inspect(path, ACCESS, denied)
    target = path / "2-raw-attempt.json"
    data = m.read(path, target.name)
    data["arguments"]["producer_lateral"] = "0"
    target.write_bytes(canonical_json_bytes(data))
    with pytest.raises(ValueError, match="hash"):
        m.inspect(path, ACCESS, lambda: None)
    terminal = m.read(path, "terminal.json")
    terminal["files"][target.name] = sha256_bytes(target.read_bytes())
    (path / "terminal.json").write_bytes(canonical_json_bytes(terminal))
    with pytest.raises(ValueError, match="arguments"):
        m.inspect(path, ACCESS, lambda: None)


def test_single_retained_fetch_and_no_actual_adapter_or_collector() -> None:
    from epsbench.diagnostics import mask_development_qualification as old
    from epsbench.diagnostics import occupancy_reference, restricted_exact_raster

    for operation in (
        old.collect,
        old.ExactAdapter().produce,
        restricted_exact_raster.raster,
        occupancy_reference.audit,
        restricted_exact_raster.box_hit,
    ):
        with pytest.raises(RuntimeError, match="forbidden"):
            operation()  # type: ignore[call-arg]
    once = m.Once(None)  # type: ignore[arg-type]
    assert once.observation(2) is None
    with pytest.raises(PermissionError):
        once.observation(2)


def test_identity_failure_preserved_without_physical_calls(tmp_path: Path) -> None:
    path = tmp_path / "cell"
    fake = Fake(path)
    result = m.run_mapping(
        path,
        ACCESS,
        CONTEXT,
        fake,
        lambda: None,
        lambda: 0,
        lambda p: None,
        clock=fake.clock,
        allocate=lambda: {"episode": "bad", "tokens": []},
    )
    assert result["cell_outcome"] == "INCONCLUSIVE" and fake.calls == []
    assert m.inspect(path, ACCESS, lambda: None) == result


def test_budget_account_includes_archive_external_review_and_reserve(tmp_path: Path) -> None:
    store = m.Store(tmp_path, lambda: 0, lambda: 100)
    assert store.account(m.RESERVE) == 2 * (100 + m.LOG + m.RESERVE) + m.REVIEW
    store.bytes = m.LIMIT
    with pytest.raises(OSError):
        store.check()


def test_external_growth_consumes_reserve_stops_but_retains_return(tmp_path: Path) -> None:
    path = tmp_path / "cell"
    fake = Fake(path)
    fake.mode = "budget"
    result = run(path, fake, external=lambda: fake.external_bytes)
    assert len(fake.calls) == 5
    assert (path / "2-raw-return.json").exists()
    assert not (path / "2-audit-attempt.json").exists()
    assert result["cell_outcome"] == "INCONCLUSIVE"
    assert m.inspect(path, ACCESS, lambda: None) == result


def test_initial_budget_denies_claim_before_identity(tmp_path: Path) -> None:
    path = tmp_path / "unclaimed"
    fake = Fake(path)
    with pytest.raises(OSError):
        run(path, fake, external=lambda: 5 * 1024 * 1024)
    assert not path.exists() and fake.calls == []


def test_closure_timeout_demotes_complete_fake_packet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "cell"
    fake = Fake(path)
    write = m.Store.write

    def slow_closure(self: m.Store, name: str, value: Any, *, closure: bool = False) -> None:
        write(self, name, value, closure=closure)
        if name == "terminal.json":
            fake.clock.value = 61

    monkeypatch.setattr(m.Store, "write", slow_closure)
    result = run(path, fake)
    assert len(fake.calls) == 8 and result["cell_outcome"] == "INCONCLUSIVE"
    assert m.inspect(path, ACCESS, lambda: None) == result


def test_source_snapshot_matches_exact_files(tmp_path: Path) -> None:
    # Verify against this checkout only: no model/runtime import or physical operation.
    root = Path(__file__).resolve().parents[1]
    context = m.source_context(root, {"fixture": "handwritten-v1"})
    m.verify_source(root, context)
    context["sources"][m.SOURCES[-1]] = "0" * 64
    with pytest.raises(ValueError, match="source"):
        m.verify_source(root, context)


@pytest.mark.parametrize(
    "access",
    [
        None,
        ModalityPermissionSet(allowed=frozenset()),
        ModalityPermissionSet(allowed=frozenset({Modality.SURFACE_REGIONS})),
        ModalityPermissionSet(
            allowed=frozenset({Modality.PRIVILEGED_GENERATION_RECORDS, Modality.SURFACE_REGIONS})
        ),
    ],
)
def test_permission_denial_precedes_all_side_effects(tmp_path: Path, access: Any) -> None:
    path = tmp_path / "unclaimed"
    signals = []

    def side_effect() -> Any:
        signals.append("access")
        raise AssertionError("permission check must precede access")

    with pytest.raises(PermissionError):
        m.run_mapping(
            path,
            access,
            CONTEXT,
            Fake(path),
            side_effect,
            side_effect,
            lambda p: side_effect(),
            clock=side_effect,
            allocate=side_effect,
        )
    with pytest.raises(PermissionError):
        m.inspect(path, access, side_effect)
    assert signals == [] and not path.exists()
