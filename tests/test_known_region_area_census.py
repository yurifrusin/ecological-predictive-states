"""Handwritten fake geometry returns and nonscientific ZIP fixture bytes only."""

from __future__ import annotations

import json
from fractions import Fraction as Q
from pathlib import Path
from typing import Any

import pytest

from epsbench.diagnostics import known_region_area_census as m
from epsbench.schema import ModalityPermissionSet
from epsbench.utils.canonical import canonical_json_bytes, sha256_bytes

ACCESS = ModalityPermissionSet(allowed=m.PRIVILEGED)
CONTEXT = {
    "head": "a" * 40,
    "tree": "b" * 40,
    "sources": dict.fromkeys(m.SOURCES, "c" * 64),
    "versions": {"purpose": "fake-only"},
}


class Fake:
    def __init__(self, packet: Path) -> None:
        self.packet = packet
        self.calls: list[tuple[int, int, str]] = []
        self.mode = "normal"
        self.time = 0.0

    def labels(self, e: int, v: int) -> list[list[int]]:
        if v < 2 or self.mode == "no_contrast":
            grid = [[2 if c < 16 else 3 for c in range(32)] for _ in range(32)]
        else:
            grid = [[2 if v == 2 else 3] * 32 for _ in range(32)]
        return [list(reversed(r)) for r in grid] if e % 2 else grid

    def produce(self, study: Any, witness: dict[str, Any], check: Any) -> Any:
        e, v = witness["episode_slot"], witness["view"]
        check()
        if v >= 2:
            assert (self.packet / "seal.json").is_file()
            for i in range(4):
                b = json.loads((self.packet / f"before-{i}.json").read_bytes())
                assert len(b["candidates"]) == 6 and all(
                    len(c) == 2 for c in b["candidates"].values()
                )
        self.calls.append((e, v, "raw"))
        if self.mode == "late_error" and e == 3 and v == 3:
            raise OSError("fake later producer failure")
        if self.mode == "timeout" and e == 0 and v == 0:
            self.time = 61
            check()
        return {"labels": self.labels(e, v), "ties": []}

    def audit(self, study: Any, witness: dict[str, Any], check: Any) -> Any:
        e, v = witness["episode_slot"], witness["view"]
        self.calls.append((e, v, "audit"))
        check()
        grid = self.labels(e, v)
        if self.mode == "normal" and e % 2 and v >= 2:
            grid = self.labels(e, 3 if v == 2 else 2)
        if self.mode == "disagreement" and e == 0 and v == 0:
            grid[0][0] = 0
        f = {
            "polygon": [["-1/2", "-1/2"], ["1/2", "-1/2"], ["1/2", "1/2"], ["-1/2", "1/2"]],
            "depths": ["3", "4"],
            "domain": True,
        }
        value = {
            "labels": grid,
            "ties": [],
            "boundaries": [],
            "supports": [[512, 512, False]] * 2,
            "footprints": [f, f],
        }
        if self.mode == "boundary":
            value["boundaries"] = [[0, 0]]
        return value


class MirroredFake(Fake):
    def labels(self, e: int, v: int) -> list[list[int]]:
        if v >= 2 and e % 2 and self.mode != "no_contrast":
            v = 3 if v == 2 else 2
        return super().labels(e, v)

    def audit(self, study: Any, witness: dict[str, Any], check: Any) -> Any:
        mode = self.mode
        # Parent class's extra swap is unused: normal physical-sign mirror supplied here.
        if mode == "normal":
            self.mode = "mirrored"
        try:
            return super().audit(study, witness, check)
        finally:
            self.mode = mode


def operate(
    tmp_path: Path, mode: str = "normal", protect: Any = None, external: Any = None
) -> tuple[dict[str, Any], Fake, Path]:
    packet = tmp_path / "packet"
    fake = MirroredFake(packet)
    fake.mode = mode
    output = m.run(
        packet,
        ACCESS,
        fake,
        CONTEXT,
        lambda: None,
        protect or (lambda p: None),
        external or (lambda: 0),
        lambda: fake.time,
        0.0,
    )
    return output, fake, packet


def test_fixed_manifest_mirrors_and_dyadic_commands() -> None:
    assert len(m.ORDER) == 32 and len(m.MANIFEST) < 65536
    for e in (0, 2):
        a, b = m.study(e), m.study(e + 1)
        for x, y in zip(a.boxes, b.boxes, strict=True):
            assert y.lower == (-x.upper[0], *x.lower[1:])
            assert y.upper == (-x.lower[0], *x.upper[1:])
        assert Q(a.executed.delta_lateral) == Q(5, 8)
        assert Q(b.executed.delta_lateral) == Q(-5, 8)
        assert tuple(Q(x.delta_lateral) for x in a.announced) == (Q(-3, 8), Q(3, 8))


def test_smoke_global_seal_single_calls_group_scores_replay_and_zip(tmp_path: Path) -> None:
    output, fake, packet = operate(tmp_path)
    assert output["outer_outcome"] == "PASS", output
    assert tuple(fake.calls) == m.ORDER
    report = output["inner"]["analysis"]
    assert report["residual"] == "PASS"
    assert report["aggregate"]["parents"][0]["controls"]["persistence"]["mean_regret"] == "1/2"
    assert report["aggregate"]["all_controls"]["zero"]["mse"] == "1/2"
    assert report["aggregate"]["all_controls"]["replay"]["mse"] == "1/4"
    assert m.inspect(packet, ACCESS, lambda: None) == report
    assert tuple(fake.calls) == m.ORDER
    assert output["archive_files"]["terminal.json"] == sha256_bytes(
        (packet / "terminal.json").read_bytes()
    )
    assert "closure.json" not in output["archive_files"]
    print(
        "Fake census smoke:32 ordered calls,48 seals,two ancestry means; "
        "retained inspection and ZIP verified"
    )


def test_no_contrast_zero_regret_stops_not_prediction_error_rescue(tmp_path: Path) -> None:
    output, _, _ = operate(tmp_path, "no_contrast")
    assert output["outer_outcome"] == "PASS"
    report = output["inner"]["analysis"]
    assert report["residual"] == "FAIL"
    assert report["aggregate"]["next"] == "STOP_DECISION_RESIDUAL_ROUTE"
    assert report["aggregate"]["all_controls"]["zero"]["mse"] == "1/4"
    assert report["aggregate"]["all_controls"]["zero"]["mean_regret"] == "0"


@pytest.mark.parametrize(
    "mode,apparatus",
    [
        ("disagreement", "FAIL"),
        ("boundary", "INCONCLUSIVE"),
        ("late_error", "INCONCLUSIVE"),
        ("timeout", "INCONCLUSIVE"),
    ],
)
def test_failure_classes_preserve_evidence_without_partial_aggregate(
    tmp_path: Path, mode: str, apparatus: str
) -> None:
    output, fake, packet = operate(tmp_path, mode)
    assert output["inner"]["analysis"]["apparatus"] == apparatus
    assert output["inner"]["analysis"]["aggregate"] is None
    assert output["inner"]["failure"] is not None
    assert m.inspect(packet, ACCESS, lambda: None) == output["inner"]["analysis"]
    assert len(fake.calls) <= 32
    if mode == "disagreement":
        assert len(fake.calls) == 2
    if mode == "late_error":
        assert len(fake.calls) == 31


def test_permission_first_no_callback_provider_filesystem_access(tmp_path: Path) -> None:
    def denied(*a: Any) -> Any:
        raise AssertionError("side effect before typed denial")

    class NoAccess:
        def __getattribute__(self, name: str) -> Any:
            return denied()

    p = tmp_path / "packet"
    access = ModalityPermissionSet(allowed=frozenset())
    with pytest.raises(PermissionError):
        m.run(p, access, NoAccess(), {}, denied, denied, denied, denied, 0.0)
    with pytest.raises(PermissionError):
        m.inspect(p, access, denied)
    with pytest.raises(PermissionError):
        m.archive(p, tmp_path / "originals.zip", access, denied)
    assert not p.exists()


def test_seal_error_before_any_future(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = m.before

    def broken(ps: Any) -> Any:
        data = original(ps)
        data["candidates"]["zero"][0] = "00"
        return data

    monkeypatch.setattr(m, "before", broken)
    output, fake, _ = operate(tmp_path)
    assert output["outer_outcome"] == "INCONCLUSIVE"
    assert all(v < 2 for _, v, _ in fake.calls)


def test_tamper_and_false_failure_terminal_rejected(tmp_path: Path) -> None:
    _, _, packet = operate(tmp_path)
    terminal = json.loads((packet / "terminal.json").read_bytes())
    terminal["failure"] = {"type": "OSError", "message": "fake failure"}
    (packet / "terminal.json").write_bytes(canonical_json_bytes(terminal))
    with pytest.raises(ValueError):
        m.inspect(packet, ACCESS, lambda: None)


def test_archive_exclusive_failure_outer_dominates_inner_immutable(tmp_path: Path) -> None:
    (tmp_path / "originals.zip").write_bytes(b"existing fixture preserved")
    output, _, packet = operate(tmp_path)
    assert output["inner"]["analysis"]["apparatus"] == "PASS"
    assert output["outer_outcome"] == "INCONCLUSIVE"
    assert output["failure"]["type"] == "FileExistsError"
    assert (tmp_path / "originals.zip").read_bytes() == b"existing fixture preserved"
    assert m.inspect(packet, ACCESS, lambda: None)["apparatus"] == "PASS"


def test_nonscientific_zip_bytes_and_mutation_failure(tmp_path: Path) -> None:
    packet = tmp_path / "packet"
    packet.mkdir()
    (packet / "fixture.txt").write_bytes(b"fixture only")
    catalog = m.archive(packet, tmp_path / "originals.zip", ACCESS, lambda: None)
    assert catalog == {"fixture.txt": sha256_bytes(b"fixture only")}
    with pytest.raises(FileExistsError):
        m.archive(packet, tmp_path / "originals.zip", ACCESS, lambda: None)
    with pytest.raises(ValueError):
        m.archive(packet, packet / "nested.zip", ACCESS, lambda: None)


def test_protection_and_byte_cap_errors_preserved(tmp_path: Path) -> None:
    def failed(p: Path) -> None:
        raise OSError("fake protection failure")

    output, _, packet = operate(tmp_path, protect=failed)
    assert output["outer_outcome"] == "INCONCLUSIVE"
    assert output["inner"]["failure"]["message"] == "fake protection failure"
    assert m.inspect(packet, ACCESS, lambda: None)["apparatus"] == "INCONCLUSIVE"


def test_return_crossing_deadline_is_retained_before_demotion(tmp_path: Path) -> None:
    packet = tmp_path / "packet"

    class Late(MirroredFake):
        def produce(self, study: Any, witness: dict[str, Any], check: Any) -> Any:
            result = super().produce(study, witness, check)
            self.time = 61
            return result

    fake = Late(packet)
    output = m.run(
        packet,
        ACCESS,
        fake,
        CONTEXT,
        lambda: None,
        lambda p: None,
        lambda: 0,
        lambda: fake.time,
        0.0,
    )
    assert output["outer_outcome"] == "INCONCLUSIVE"
    assert (packet / "0-0-raw-return.json").exists()
    assert json.loads((packet / "0-0-raw-return.json").read_bytes())["labels"] == fake.labels(0, 0)
    assert fake.calls == [(0, 0, "raw")]
    assert output["inner"]["failure"]["type"] == "TimeoutError"
    assert m.inspect(packet, ACCESS, lambda: None)["apparatus"] == "INCONCLUSIVE"


def test_source_identity_callbacks_bounded_not_per_pixel(tmp_path: Path) -> None:
    packet = tmp_path / "packet"
    verifications = []

    class ManyChecks(MirroredFake):
        def produce(self, study: Any, witness: dict[str, Any], check: Any) -> Any:
            for _ in range(1024):
                check()
            return super().produce(study, witness, check)

    fake = ManyChecks(packet)
    output = m.run(
        packet,
        ACCESS,
        fake,
        CONTEXT,
        lambda: verifications.append(1),
        lambda p: None,
        lambda: 0,
        lambda: fake.time,
        0.0,
    )
    assert output["outer_outcome"] == "PASS"
    assert 18 <= len(verifications) <= 22


def test_replay_private_access_denied_before_any_read(tmp_path: Path) -> None:
    with pytest.raises(PermissionError):
        m._replay(tmp_path / "absent", {}, [], ModalityPermissionSet(allowed=frozenset()))


def test_inclusive_cap_stops_without_calls_or_unbounded_closure(tmp_path: Path) -> None:
    output, fake, _ = operate(tmp_path, external=lambda: m.LIMIT)
    assert output["outer_outcome"] == "INCONCLUSIVE"
    assert fake.calls == []
    assert output["closure_not_written"] is True
    assert not (tmp_path / "closure.json").exists()


def test_post_archive_timeout_dominates_immutable_inner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = tmp_path / "packet"
    fake = MirroredFake(packet)
    original = m.archive

    def late(*args: Any) -> Any:
        result = original(*args)
        fake.time = 61
        return result

    monkeypatch.setattr(m, "archive", late)
    output = m.run(
        packet,
        ACCESS,
        fake,
        CONTEXT,
        lambda: None,
        lambda p: None,
        lambda: 0,
        lambda: fake.time,
        0.0,
    )
    assert output["inner"]["analysis"]["apparatus"] == "PASS"
    assert output["outer_outcome"] == "INCONCLUSIVE"
    assert output["failure"]["type"] == "TimeoutError"
    assert (tmp_path / "originals.zip").exists()
    assert m.inspect(packet, ACCESS, lambda: None)["apparatus"] == "PASS"


def test_archive_detects_original_mutation(tmp_path: Path) -> None:
    packet = tmp_path / "packet"
    packet.mkdir()
    fixture = packet / "fixture"
    fixture.write_bytes(b"first")
    calls = []

    def check() -> None:
        calls.append(1)
        if len(calls) == 2:
            fixture.write_bytes(b"changed")

    with pytest.raises(ValueError):
        m.archive(packet, tmp_path / "originals.zip", ACCESS, check)
    assert fixture.read_bytes() == b"changed" and (tmp_path / "originals.zip").exists()


@pytest.mark.parametrize("name", ["terminal.json", "closure.json"])
def test_terminal_or_closure_write_crosses_whole_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    packet = tmp_path / "packet"
    fake = MirroredFake(packet)
    original = Path.open

    class Wrapped:
        def __init__(self, handle: Any) -> None:
            self.handle = handle

        def __enter__(self) -> Any:
            self.handle.__enter__()
            return self

        def __exit__(self, *args: Any) -> Any:
            return self.handle.__exit__(*args)

        def write(self, data: bytes) -> Any:
            result = self.handle.write(data)
            fake.time = 61
            return result

        def flush(self) -> None:
            self.handle.flush()

    def opened(path: Path, *args: Any, **kwargs: Any) -> Any:
        handle = original(path, *args, **kwargs)
        return Wrapped(handle) if path.name == name and args and args[0] == "xb" else handle

    monkeypatch.setattr(Path, "open", opened)
    output = m.run(
        packet,
        ACCESS,
        fake,
        CONTEXT,
        lambda: None,
        lambda p: None,
        lambda: 0,
        lambda: fake.time,
        0.0,
    )
    assert output["outer_outcome"] == "INCONCLUSIVE"
    assert output["inner"]["analysis"]["apparatus"] == "PASS"
    assert output["failure"]["type"] == "TimeoutError"
    if name == "closure.json":
        assert output["closure_write_overrun"] is True
    assert m.inspect(packet, ACCESS, lambda: None)["apparatus"] == "PASS"


def test_timeout_precedence_preserves_initiating_contradiction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = tmp_path / "packet"
    fake = Fake(packet)
    fake.mode = "unmirrored"
    original = m._replay

    def delayed(*args: Any, **kwargs: Any) -> Any:
        try:
            return original(*args, **kwargs)
        except m.Contradiction:
            fake.time = 61
            raise

    monkeypatch.setattr(m, "_replay", delayed)
    output = m.run(
        packet,
        ACCESS,
        fake,
        CONTEXT,
        lambda: None,
        lambda p: None,
        lambda: 0,
        lambda: fake.time,
        0.0,
    )
    assert output["inner"]["failure"]["type"] == "TimeoutError"
    assert "Contradiction" in output["inner"]["failure"]["message"]
    assert output["inner"]["analysis"]["apparatus"] == "INCONCLUSIVE"
    assert m.inspect(packet, ACCESS, lambda: None)["apparatus"] == "INCONCLUSIVE"


def test_exclusive_closure_failure_retains_returned_inner_and_archive(tmp_path: Path) -> None:
    (tmp_path / "closure.json").write_bytes(b"existing fixture")
    output, _, packet = operate(tmp_path)
    assert output["outer_outcome"] == "INCONCLUSIVE"
    assert output["failure"]["type"] == "FileExistsError"
    assert output["inner"]["analysis"]["apparatus"] == "PASS"
    assert (tmp_path / "closure.json").read_bytes() == b"existing fixture"
    assert (tmp_path / "originals.zip").exists()
    assert m.inspect(packet, ACCESS, lambda: None)["apparatus"] == "PASS"
