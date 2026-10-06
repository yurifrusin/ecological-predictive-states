"""Symbolic operator checks: no actual seeds, membership, producer or frames."""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from epsbench.diagnostics.restricted_learning_collection import (
    PACKAGING_RESERVE,
    Decision,
    _prepare,
)
from epsbench.diagnostics.restricted_learning_collection import (
    _qualify as _operator_qualify,
)
from epsbench.diagnostics.restricted_learning_contract import VERSION, digest
from epsbench.diagnostics.restricted_learning_retention import Archive
from epsbench.diagnostics.visible_forecast_contract import _json
from epsbench.utils.canonical import canonical_json_bytes

LOCK = canonical_json_bytes({"version": "SYNTHETIC_SOURCE_ONLY", "symbol": "no-membership"})


class FakeBackend:
    def __init__(self) -> None:
        self.calls = 0
        self.fail_prepare = False
        self.fail_qualification = False

    def source(self, root: Path, decision: Decision) -> tuple[tuple[str, bytes], ...]:
        return (("symbolic-source", b"SYNTHETIC_SOURCE_ONLY"),)

    def prepare(
        self, seeds: tuple[bytes, ...], decision: Decision, files: tuple[tuple[str, bytes], ...]
    ) -> bytes:
        assert len(seeds) == 7
        if self.fail_prepare:
            raise ValueError("symbolic retained failure")
        return LOCK

    def validate(self, lock: bytes, seeds: tuple[bytes, ...], root: Path) -> Any:
        assert lock == LOCK
        return "symbolic-lock"

    def qualify(
        self, lock: Any, seeds: tuple[bytes, ...], root: Path, archive: Archive, receipt: bytes
    ) -> None:
        self.calls += 1
        archive.event("MEMBERSHIP_LOCK", "membership-lock.json", LOCK)
        assert archive.max_bytes < PACKAGING_RESERVE
        if self.fail_qualification:
            archive.write("orphan.bin", b"retained-symbolic-partial")
            raise ValueError("symbolic qualification failure")
        archive.event("SEAL_A", "seal-a.json", b"{}")

    def inspect(self, root: Path) -> dict[str, Any]:
        return {
            "failed": False,
            "seal_a": digest(b"{}"),
            "seal_b": None,
            "events": 2,
            "history_root": digest(b"symbolic-history"),
        }


def decision(membership: str | None = None) -> Decision:
    return Decision("a" * 64, "b" * 40, "c" * 40, "d" * 64, membership)


def symbolic_bytes(size: int) -> bytes:
    # Deliberately repeated placeholders; fake backend never derives actual membership.
    return b"S" * size


def _qualify(
    attempt: Path, root: Path, authorized: Decision, backend: FakeBackend
) -> dict[str, Any]:
    # Explicit externally supplied fixture receipt, never an actual launch identity.
    receipt = canonical_json_bytes(
        {
            "version": VERSION,
            "source_head": authorized.source_head,
            "source_tree": authorized.source_tree,
            "membership": authorized.membership,
            "owner_decision": "EXACT_SOURCE_COLLECTION_AUTHORIZED",
        }
    )
    return _operator_qualify(attempt, root, authorized, backend, receipt)


def test_preparation_exclusive_and_private(tmp_path: Path) -> None:
    attempt = tmp_path / "private"
    backend = FakeBackend()
    receipt = _prepare(attempt, tmp_path, decision(), backend, symbolic_bytes)
    assert receipt["disposition"] == "PRECOMMITTED"
    assert receipt["membership"] == digest(LOCK)
    assert not (attempt / "dataset").exists()
    assert len(list(attempt.glob("seed-*.bin"))) == 7
    with pytest.raises(ValueError, match="fresh"):
        _prepare(attempt, tmp_path, decision(), backend, lambda _: pytest.fail("must not redraw"))


def test_preimages_retained_before_validation_failure(tmp_path: Path) -> None:
    backend = FakeBackend()
    backend.fail_prepare = True
    attempt = tmp_path / "private"
    with pytest.raises(ValueError, match="symbolic"):
        _prepare(attempt, tmp_path, decision(), backend, symbolic_bytes)
    assert len(list(attempt.glob("seed-*.bin"))) == 7
    assert (attempt / "operator-failure.json").exists()
    assert not (attempt / "precommit.json").exists()
    with pytest.raises(PermissionError, match="terminal"):
        _qualify(attempt, tmp_path, decision(digest(LOCK)), backend)
    assert (attempt / "qualification-claim.json").exists()


def test_separate_exact_membership_and_single_qualification(tmp_path: Path) -> None:
    attempt = tmp_path / "private"
    backend = FakeBackend()
    _prepare(attempt, tmp_path, decision(), backend, symbolic_bytes)
    receipt = _qualify(attempt, tmp_path, decision(digest(LOCK)), backend)
    assert receipt["disposition"] == "QUALIFIED"
    assert backend.calls == 1
    assert set(receipt) == {
        "disposition",
        "membership",
        "precommit",
        "private_attempt_root",
        "seal_a",
        "history_root",
        "events",
        "retained_bytes",
        "packaging_reserve",
        "phase_gate_effect",
    }
    with pytest.raises(FileExistsError):
        _qualify(attempt, tmp_path, decision(digest(LOCK)), backend)
    assert backend.calls == 1


@pytest.mark.parametrize("artifact", ["seed-0.bin", "source-0.bin", "membership-lock.json"])
def test_changed_private_preimages_terminal(tmp_path: Path, artifact: str) -> None:
    attempt = tmp_path / "private"
    backend = FakeBackend()
    _prepare(attempt, tmp_path, decision(), backend, symbolic_bytes)
    (attempt / artifact).write_bytes(b"tampered")
    with pytest.raises(ValueError):
        _qualify(attempt, tmp_path, decision(digest(LOCK)), backend)
    assert backend.calls == 0
    assert (attempt / "qualification-failure.json").exists()
    with pytest.raises(FileExistsError):
        _qualify(attempt, tmp_path, decision(digest(LOCK)), backend)


def test_wrong_source_and_membership_never_collect(tmp_path: Path) -> None:
    attempt = tmp_path / "private"
    backend = FakeBackend()
    _prepare(attempt, tmp_path, decision(), backend, symbolic_bytes)
    with pytest.raises(ValueError, match="precommit"):
        _qualify(attempt, tmp_path, decision("f" * 64), backend)
    assert backend.calls == 0


def test_failed_qualification_retains_orphans_no_resume(tmp_path: Path) -> None:
    attempt = tmp_path / "private"
    backend = FakeBackend()
    _prepare(attempt, tmp_path, decision(), backend, symbolic_bytes)
    backend.fail_qualification = True
    with pytest.raises(ValueError, match="qualification"):
        _qualify(attempt, tmp_path, decision(digest(LOCK)), backend)
    assert (attempt / "dataset" / "orphan.bin").read_bytes() == b"retained-symbolic-partial"
    assert (attempt / "dataset" / "operator-failure.json").exists()
    with pytest.raises(FileExistsError):
        _qualify(attempt, tmp_path, decision(digest(LOCK)), backend)
    assert backend.calls == 1


def test_explicit_decision_shape() -> None:
    with pytest.raises(ValueError):
        Decision("human-readable-claim", "b" * 40, "c" * 40, "d" * 64)


@pytest.mark.parametrize("after_seal", [False, True])
def test_terminal_pending_matches_retained_phase(tmp_path: Path, after_seal: bool) -> None:
    class PhaseBackend(FakeBackend):
        def validate(self, lock: bytes, seeds: tuple[bytes, ...], root: Path) -> Any:
            assert lock == LOCK
            return SimpleNamespace(
                decision_roster=("symbolic/decision",), forecast_roster=("symbolic/forecast",)
            )

        def inspect(self, root: Path) -> dict[str, Any]:
            raise ValueError("symbolic post-SealA inspection failure")

    attempt = tmp_path / "private"
    backend = PhaseBackend()
    _prepare(attempt, tmp_path, decision(), backend, symbolic_bytes)
    backend.fail_qualification = not after_seal
    with pytest.raises(ValueError, match="symbolic"):
        _qualify(attempt, tmp_path, decision(digest(LOCK)), backend)
    failure = _json((attempt / "dataset" / "failure.json").read_bytes())
    assert failure["phase"] == ("FORECAST" if after_seal else "QUALIFICATION")
    assert failure["pending"] == (["symbolic/forecast"] if after_seal else ["symbolic/decision"])
    events = sorted((attempt / "dataset").glob("event-*.json"))
    assert _json(events[-1].read_bytes())["kind"] == "FAILURE"
    with pytest.raises(FileExistsError):
        _qualify(attempt, tmp_path, decision(digest(LOCK)), backend)
    assert backend.calls == 1


def test_external_receipt_contract_static_without_producer_import(tmp_path: Path) -> None:
    producer = Path(__file__).resolve().parents[1] / (
        "src/epsbench/diagnostics/restricted_learning_producer.py"
    )
    parsed = ast.parse(producer.read_text(encoding="utf-8"))
    assert any(
        isinstance(node, ast.Constant) and node.value == "EXACT_SOURCE_COLLECTION_AUTHORIZED"
        for node in ast.walk(parsed)
    )
    attempt = tmp_path / "private"
    backend = FakeBackend()
    _prepare(attempt, tmp_path, decision(), backend, symbolic_bytes)
    external = decision(digest(LOCK))
    # An identity commitment must never be rewritten into the adapter discriminator.
    with pytest.raises(PermissionError, match="external"):
        _operator_qualify(attempt, tmp_path, external, backend, external.receipt())
    assert backend.calls == 0
    assert _json((attempt / "qualification-claim.json").read_bytes())["owner_decision"] == "a" * 64
