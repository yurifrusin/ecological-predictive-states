"""Symbolic fake-provider checks; never enumerate candidate geometry membership."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction as Q
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from epsbench.diagnostics.boundary_observation import VisibleRaster
from epsbench.diagnostics.restricted_learning_contract import (
    ARCHITECTURE,
    BIJECTIONS,
    LIMITS,
    REQUIRED,
    ROSTER,
    VERSION,
    ZERO,
    Fit,
    Forecast,
    InputEvidence,
    StreamingView,
    digest,
)
from epsbench.diagnostics.restricted_learning_membership import Geometry, MembershipLock
from epsbench.diagnostics.restricted_learning_retention import (
    Archive,
    Lifecycle,
    QualifiedTarget,
    SyntheticLock,
)
from epsbench.diagnostics.restricted_learning_sampling import Entropy, shuffled
from epsbench.diagnostics.restricted_learning_scoring import (
    GeometryScore,
    Score,
    balanced_loss,
    bootstrap,
    combine,
    decide,
    score,
)
from epsbench.diagnostics.visible_forecast_contract import CausalInput, TokenFrame
from epsbench.schema import ModalityPermissionSet
from epsbench.utils.canonical import canonical_json_bytes

A = "surface-ffffffffffffffff"
B = "surface-0000000000000000"
H = "0" * 64
ACCESS = ModalityPermissionSet(allowed=REQUIRED)


def evidence() -> InputEvidence:
    a = np.zeros((32, 32), dtype=np.bool_)
    b = a.copy()
    a[:, :16] = True
    b[:, 16:] = True
    return InputEvidence(
        CausalInput(
            (
                TokenFrame(0, (32, 32), ((A, a), (B, b))),
                TokenFrame(1, (32, 32), ((A, a),)),
                TokenFrame(2, (32, 32), ((A, a),)),
            ),
            (ZERO, ZERO),
            (Q(0), Q(3, 4), Q(0)),
            ACCESS,
            LIMITS,
        )
    )


def prediction(source: InputEvidence, unknown: bool = False) -> Forecast:
    return Forecast(
        source.digest,
        H,
        H,
        Fit(16, "init-0", "relational"),
        ((A, 0.8), (B, None if unknown else 0.25)),
    )


def test_streaming_arrivals_boundaries_and_no_replay() -> None:
    view = evidence().streaming()
    with pytest.raises(PermissionError):
        _ = view.announced
    first = view.next_observation()
    assert first.handles == (A, B)  # Raster order, opposite opaque lexical order.
    assert first.first_seen == (0, 0) and first.present == (1, 1, 0)
    assert first.boundary[0] == (Q(32, 992), Q(0))
    assert first.boundary[0] == first.boundary[2]
    assert len(BIJECTIONS) == 6 and len(set(BIJECTIONS)) == 6
    assert view.next_observation().visible == (1, 0, 0)
    last = view.next_observation()
    assert view.announced == (Q(0), Q(3, 4), Q(0))
    assert last.handles == (A, B) and all(pair == (Q(0), Q(0)) for pair in last.boundary)
    with pytest.raises(PermissionError):
        view.next_observation()
    with pytest.raises(ValueError):
        replace(last, present=(True, True, False))
    with pytest.raises(ValueError):
        replace(last, handles=list(last.handles))  # type: ignore[arg-type]


def test_stream_terminal_after_failure_and_permission_before_fetch() -> None:
    class Bad:
        calls = 0

        def raster(self, index: int) -> VisibleRaster:
            self.calls += 1
            raise ValueError("failed observation")

    p = Bad()
    with pytest.raises(PermissionError):
        StreamingView(p, ModalityPermissionSet(allowed=frozenset()), (ZERO, ZERO), ZERO)
    assert p.calls == 0
    v = StreamingView(p, ACCESS, (ZERO, ZERO), ZERO)
    with pytest.raises(ValueError):
        v.next_observation()
    with pytest.raises(PermissionError):
        v.next_observation()
    assert p.calls == 1


def test_input_serialization_strict_and_immutable() -> None:
    source = evidence()
    assert InputEvidence.from_bytes(source.canonical_bytes(), ACCESS).digest == source.digest
    with pytest.raises(PermissionError):
        InputEvidence.from_bytes(b"malformed", ModalityPermissionSet(allowed=frozenset()))
    data = source.canonical_bytes().replace(b'"architecture":', b'"depth":0,"architecture":')
    with pytest.raises(ValueError):
        InputEvidence.from_bytes(data, ACCESS)
    for _, mask in source.prefix.frames[0].masks:
        with pytest.raises(ValueError):
            mask[0, 0] = False


@pytest.mark.parametrize("bad", [True, 0, -0.1, 1.1, float("nan"), float("inf")])
def test_probability_rejects_malformed(bad: Any) -> None:
    with pytest.raises(ValueError):
        replace(prediction(evidence()), channels=((A, bad), (B, 0.5)))


def test_forecast_exact_bindings_unknown_and_closed_keys() -> None:
    source = evidence()
    f = prediction(source)
    assert Forecast.from_bytes(f.canonical_bytes(), source) == f
    with pytest.raises(ValueError):
        Forecast.from_bytes(replace(f, channels=((A, 0.8),)).canonical_bytes(), source)
    with pytest.raises(ValueError):
        Forecast.from_bytes(f.canonical_bytes().replace(b'"run":', b'"run":"x","run":'), source)
    with pytest.raises(ValueError):
        replace(f, channels=((A, 0.5), (A, 0.5)))
    with pytest.raises(ValueError):
        replace(f, channels=([A, 0.5],))  # type: ignore[arg-type]
    result = score(prediction(source, True), source, ((A, True), (B, False)))
    assert result["all"].denominator == 2 and result["all"].unknown == 1
    assert result["all"].coverage == 0.5 and result["current-absent"].interval == (0.0, 1.0)
    assert result["current-absent"].risk is None
    assert result["future-visible"].denominator == 1
    with pytest.raises(ValueError):
        balanced_loss(result)


@pytest.mark.parametrize(
    "args", [(True, 0, 0.0), (1, 2, 0.0), (1, 0, float("nan")), (1, 0, -0.1), (1, 0, 1.1)]
)
def test_score_strict_bounds(args: tuple[Any, ...]) -> None:
    with pytest.raises(ValueError):
        Score(*args)


def test_geometry_denominators_and_empty_strata() -> None:
    source = evidence()
    rows = score(prediction(source), source, ((A, True), (B, False)))
    expected = tuple(str(i) for i in range(8))
    g = GeometryScore("symbolic", "fit", tuple((k, rows) for k in expected))
    assert g.aggregate(expected)["all"].denominator == 16
    assert combine(()).interval is None
    with pytest.raises(ValueError):
        replace(g, decisions=g.decisions[:-1]).aggregate(expected)


def test_sampling_symbolic_and_common_bootstrap() -> None:
    # Fixed non-study bytes only; no seed allocation or candidate domain.
    entropy = Entropy("SYNTHETIC_SOURCE_ONLY", b"fixture")
    assert all(0 <= entropy.below(7) < 7 for _ in range(100))
    assert sorted(shuffled(("a", "b", "c"), "SYNTHETIC_SOURCE_ONLY", b"fixture")) == ["a", "b", "c"]
    values = tuple(float(i) / 32 for i in range(32))
    samples = bootstrap((values, values), bytes(32), "M0-occupancy-development-v1/bootstrap-v1")
    assert len(samples[0]) == 10000 and samples[0] == samples[1]
    result = decide(
        (0.04,) * 32,
        (0.0,) * 32,
        (0.0,) * 32,
        (0.1,) * 32,
        ((0.03,) * 32,) * 5,
        ((0.03,) * 32,) * 2,
        bytes(32),
    )
    assert result.disposition == "SUPPORT"  # Numerical helper only, not actual pilot evidence.
    result = decide(
        (0.01,) * 32,
        (0.0,) * 32,
        (0.0,) * 32,
        (0.1,) * 32,
        ((0.03,) * 32,) * 5,
        ((0.03,) * 32,) * 2,
        bytes(32),
    )
    assert result.disposition == "NOT_SUPPORTED"


def test_actual_membership_rejects_partial_without_enumeration() -> None:
    assert (
        len(ROSTER) == 54
        and sum(
            f.condition in ("relational", "dense", "action-zero", "memory-reset") for f in ROSTER
        )
        == 24
    )
    with pytest.raises(ValueError):
        MembershipLock((), (), (), (), H, H, (), (), H, "0" * 40, "0" * 40, ())
    with pytest.raises(ValueError):
        Geometry(Q(1, 2), Q(1, 2), Q(0), Q(3), Q(0))


class Fake:
    def __init__(self, lifecycle: Lifecycle) -> None:
        self.lifecycle = lifecycle
        self.source = evidence()
        self.calls: list[str] = []

    def prefix(self, decision: str) -> tuple[InputEvidence, bytes]:
        self.calls.append("prefix")
        assert self.lifecycle.archive.events[0]["kind"] == "MEMBERSHIP_LOCK"
        audit = {
            "raw": H,
            "reference": H,
            "ties": [],
            "boundaries": [],
            "statuses": {"support": "UNKNOWN_DOMAIN"},
        }
        return self.source, canonical_json_bytes(
            {
                "version": VERSION,
                "audits": [audit] * 3,
                "decision_occlusion": "COMPLETE_IN_FRUSTUM_OCCLUSION",
            }
        )

    def target(self, decision: str) -> QualifiedTarget:
        self.calls.append("target")
        assert decision in self.lifecycle.inputs
        annotation = {
            "version": VERSION,
            "descriptor": H,
            "raw": H,
            "reference": H,
            "ties": [],
            "boundaries": [],
            "statuses": {"support": "UNKNOWN_DOMAIN"},
        }
        return QualifiedTarget(((A, True), (B, False)), canonical_json_bytes(annotation))


def test_synthetic_smoke_validation_inspection_and_seal_order(tmp_path: Path) -> None:
    archive = Archive(tmp_path / "fake")
    lifecycle = Lifecycle(SyntheticLock(), archive)
    fake = Fake(lifecycle)
    a = lifecycle.qualify(fake)
    assert fake.calls == ["prefix", "target"] and digest(archive.read("seal-a.json")) == a
    assert b"SOURCE_SMOKE_ONLY" in archive.read("seal-a.json")
    f = prediction(fake.source)
    b = lifecycle.seal_forecasts(((SyntheticLock().forecast_roster[0], f),))
    assert a != b and b"SYNTHETIC_SOURCE_ONLY" in archive.read("seal-b.json")
    assert archive.inspect()["events"] == 7
    with pytest.raises(PermissionError):
        lifecycle.evaluate(bytes(32), b"{}")
    with pytest.raises(PermissionError):
        lifecycle.training_material("symbolic")
    with pytest.raises(PermissionError):
        lifecycle.qualify(fake)
    print("Synthetic smoke generated; immutable seals, validation and inspection verified")


def test_missing_forecast_failure_retained_and_no_retry(tmp_path: Path) -> None:
    archive = Archive(tmp_path / "failed")
    lifecycle = Lifecycle(SyntheticLock(), archive)
    lifecycle.qualify(Fake(lifecycle))
    with pytest.raises(ValueError):
        lifecycle.seal_forecasts(())
    assert archive.failed and b"FAILED_CLOSED" in archive.read("failure.json")
    with pytest.raises(PermissionError):
        lifecycle.seal_forecasts(())


def test_tamper_and_write_once_and_timeout(tmp_path: Path) -> None:
    now = [0.0]
    archive = Archive(tmp_path / "tamper", clock=lambda: now[0], seconds=2.0)
    archive.event("FAKE", "fake.json", b"{}")
    with pytest.raises(FileExistsError):
        archive.write("fake.json", b"{}")
    (archive.root / "fake.json").write_bytes(b"changed")
    with pytest.raises(ValueError):
        archive.inspect()
    now[0] = 2.0
    with pytest.raises(RuntimeError):
        archive.write("late.json", b"{}")
    with pytest.raises(ValueError):
        archive.read("../outside.json")


def test_public_spec_identity() -> None:
    text = Path("docs/RESTRICTED_OCCUPANCY_ARCHITECTURE_V1.md").read_text(encoding="utf-8")
    assert ARCHITECTURE in text and "99913" in text and "99984" in text


def test_label_only_export_and_non_candidate_reference() -> None:
    from dataclasses import fields

    from epsbench.diagnostics.occupancy_reference import Solid, face_hit, footprint, support_hit
    from epsbench.diagnostics.restricted_learning_retention import VisibilityLabels

    labels = VisibilityLabels(((A, True), (B, False)))
    assert [f.name for f in fields(labels)] == ["channels"]
    assert not hasattr(labels, "annotations") and not hasattr(labels, "descriptor")
    with pytest.raises(ValueError):
        VisibilityLabels(((A, True), (A, False)))
    box = Solid((Q(2), Q(2), Q(2)), (Q(3), Q(3), Q(3)))
    assert face_hit((Q(0), Q(0), Q(0)), (Q(1), Q(1), Q(1)), box) == (Q(2), True)
    assert footprint((Q(0), Q(0), Q(0)), box).domain
    assert support_hit((Q(0), Q(0), Q(1)), (Q(0), Q(1), Q(-1)), (Q(4), Q(7))) == (Q(1), False)


def test_durable_read_only_inspection_after_synthetic_seals(tmp_path: Path) -> None:
    from epsbench.diagnostics.restricted_learning_retention import inspect_archive

    root = tmp_path / "durable"
    life = Lifecycle(SyntheticLock(), Archive(root))
    fake = Fake(life)
    life.qualify(fake)
    life.seal_forecasts(((SyntheticLock().forecast_roster[0], prediction(fake.source)),))
    assert inspect_archive(root, allow_synthetic=True)["events"] == 7
    with pytest.raises(PermissionError):
        inspect_archive(root)
    (root / "event-4.json").unlink()
    with pytest.raises(ValueError):
        inspect_archive(root, allow_synthetic=True)


def _sealed_symbolic(root: Path) -> Lifecycle:
    life = Lifecycle(SyntheticLock(), Archive(root))
    fake = Fake(life)
    life.qualify(fake)
    life.seal_forecasts(((SyntheticLock().forecast_roster[0], prediction(fake.source)),))
    return life


def _coherent_rewrite(root: Path, path: str, update: Any) -> None:
    """Test adversary rewrites hashes too: semantic closure must still reject it."""
    import json

    data = json.loads((root / path).read_bytes())
    update(data)
    payload = canonical_json_bytes(data)
    (root / path).write_bytes(payload)
    previous = None
    files = sorted(root.glob("event-*.json"), key=lambda p: int(p.stem.split("-")[1]))
    for event_path in files:
        event = json.loads(event_path.read_bytes())
        event["previous"] = previous
        if event["path"] == path:
            event["sha256"] = digest(payload)
        encoded = canonical_json_bytes(event)
        event_path.write_bytes(encoded)
        previous = digest(encoded)


def test_er54_0001_valid_failure_durable_identity_and_orphan_preservation(tmp_path: Path) -> None:
    from epsbench.diagnostics.restricted_learning_retention import inspect_archive

    root = tmp_path / "failure"
    life = Lifecycle(SyntheticLock(), Archive(root))
    life.qualify(Fake(life))
    with pytest.raises(ValueError):
        life.seal_forecasts(())
    summary = inspect_archive(root, allow_synthetic=True)
    assert summary["failed"] and summary["events"] == 6
    assert summary["failure_sha256"] == digest((root / "failure.json").read_bytes())
    assert life.archive.inspect()["archive_root"] == summary["archive_root"]
    (root / "partial-orphan.bin").write_bytes(b"partial evidence")
    assert inspect_archive(root, allow_synthetic=True)["archive_root"] == summary["archive_root"]
    assert (root / "partial-orphan.bin").read_bytes() == b"partial evidence"
    _coherent_rewrite(root, "failure.json", lambda p: p.update(failure_kind="IO"))
    changed = inspect_archive(root, allow_synthetic=True)
    assert changed["failure_sha256"] != summary["failure_sha256"]
    assert changed["archive_root"] != summary["archive_root"]
    (root / "failure.json").write_bytes(b"{}")
    with pytest.raises(ValueError):
        inspect_archive(root, allow_synthetic=True)
    with pytest.raises(ValueError):
        life.archive.inspect()


@pytest.mark.parametrize(
    "mutation",
    [
        {"last_event": H},
        {"pending": []},
        {"phase": "QUALIFICATION"},
        {"status": "RECOVERED"},
        {"used_bytes": True},
        {"membership": H},
        {"extra": 0},
    ],
)
def test_er54_0001_closed_failure_chronology(tmp_path: Path, mutation: dict[str, Any]) -> None:
    from epsbench.diagnostics.restricted_learning_retention import inspect_archive

    root = tmp_path / "closed-failure"
    life = Lifecycle(SyntheticLock(), Archive(root))
    life.qualify(Fake(life))
    with pytest.raises(ValueError):
        life.seal_forecasts(())
    _coherent_rewrite(root, "failure.json", lambda p: p.update(mutation))
    with pytest.raises(ValueError):
        inspect_archive(root, allow_synthetic=True)


@pytest.mark.parametrize(
    "mutation",
    [
        {"forecasts": {}, "bindings": {}, "roster": []},
        {"forecasts": {}},
        {"bindings": {}},
        {"roster": []},
        {"roster": ["extra"]},
        {"extra": 0},
        {"bindings": {"16/init-0/relational": ["1" * 64, H]}},
        {"forecasts": {"16/init-0/relational/synthetic/source-only": H}},
    ],
)
def test_er54_0002_closed_complete_sealb(tmp_path: Path, mutation: dict[str, Any]) -> None:
    from epsbench.diagnostics.restricted_learning_retention import inspect_archive

    root = tmp_path / "seal"
    _sealed_symbolic(root)
    _coherent_rewrite(root, "seal-b.json", lambda p: p.update(mutation))
    with pytest.raises(ValueError):
        inspect_archive(root, allow_synthetic=True)


@pytest.mark.parametrize(
    "path,mutation",
    [
        ("forecast-0.json", {"input": H}),
        ("forecast-0.json", {"condition": "dense"}),
        ("forecast-0.json", {"channels": {A: {"probability": 0.8}}}),
        ("target-0.json", {"truth": {A: True}}),
        ("target-0.json", {"truth": {A: 1, B: False}}),
        ("seal-a.json", {"inputs": {}}),
        ("seal-a.json", {"targets": {}}),
        ("seal-a.json", {"extra": 0}),
    ],
)
def test_er54_0002_offline_member_bindings(
    tmp_path: Path, path: str, mutation: dict[str, Any]
) -> None:
    from epsbench.diagnostics.restricted_learning_retention import inspect_archive

    root = tmp_path / "member"
    _sealed_symbolic(root)
    _coherent_rewrite(root, path, lambda p: p.update(mutation))
    with pytest.raises(ValueError):
        inspect_archive(root, allow_synthetic=True)


def test_er54_0002_phase_paths_and_terminal_failure(tmp_path: Path) -> None:
    import json

    from epsbench.diagnostics.restricted_learning_retention import inspect_archive

    root = tmp_path / "phase"
    life = _sealed_symbolic(root)
    event = json.loads((root / "event-6.json").read_bytes())
    event["kind"] = "SEAL_A"
    (root / "event-6.json").write_bytes(canonical_json_bytes(event))
    with pytest.raises(ValueError):
        inspect_archive(root, allow_synthetic=True)
    root2 = tmp_path / "terminal"
    failed = Lifecycle(SyntheticLock(), Archive(root2))
    failed.qualify(Fake(failed))
    with pytest.raises(ValueError):
        failed.seal_forecasts(())
    with pytest.raises(RuntimeError):
        failed.archive.event("REPORT", "late.json", b"{}")
    assert life.seal_b is not None


def test_large_symbolic_pending_failure_reserve_at_payload_limit(tmp_path: Path) -> None:
    # Symbolic strings test finite serialization bounds, never candidate membership.
    pending = tuple("SYMBOLIC_SOURCE_ONLY/" + str(i) + "x" * 120 for i in range(13824))
    reserve = len(canonical_json_bytes(list(pending))) + 4096
    archive = Archive(tmp_path / "reserve", max_bytes=4 * 1024**2, failure_reserve=reserve)
    archive.event("MEMBERSHIP_LOCK", "membership-lock.json", SyntheticLock().canonical_bytes())
    available = archive.max_bytes - archive.failure_reserve - archive.used
    archive.write("bounded-symbolic.bin", b"x" * available)
    with pytest.raises(RuntimeError, match="bound"):
        archive.write("over-limit.bin", b"x")
    archive.fail(pending, RuntimeError("bounded symbolic failure"))
    assert archive.used <= archive.max_bytes
    assert archive.events[-1]["kind"] == "FAILURE"
    tiny = Lifecycle(SyntheticLock(), Archive(tmp_path / "tiny", max_bytes=65536))
    assert tiny.archive.failure_reserve == 32768
