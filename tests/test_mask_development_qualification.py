"""Public synthetic qualification fixtures; real development members remain unexecuted."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from epsbench.diagnostics import mask_development_qualification as c
from epsbench.utils.canonical import canonical_json_bytes, sha256_bytes

DEFINITION = b"PUBLIC FAKE DEFINITION"
ADOPTION = b"PUBLIC FAKE ADOPTION"


def manifest() -> bytes:
    def a(q: float) -> dict[str, Any]:
        return {
            "name": "lateral_left" if q < 0 else "lateral_right",
            "delta_forward": 0.0,
            "delta_lateral": q,
            "delta_yaw": 0.0,
        }

    return canonical_json_bytes(
        {
            "version": c.VERSION,
            "definition_sha256": sha256_bytes(DEFINITION),
            "adoption_sha256": sha256_bytes(ADOPTION),
            "calibration": {
                "size": 32,
                "forward": "-3",
                "elevation": "1",
                "up_y": "0",
                "fovy": "90",
                "clipping": "NONE",
            },
            "support": ["8", "8"],
            "boxes": [
                {"lower": ["0", "1", "1"], "upper": ["1", "2", "2"]},
                {"lower": ["0", "3", "1"], "upper": ["1", "4", "2"]},
            ],
            "prefix": ["-1", "0"],
            "executed_action": a(1.0),
            "announced_actions": [a(-0.5), a(0.5)],
            "steps": ["-1/2", "1/2"],
            "signals": list(c.SIGNALS),
            "limits": {
                "producer_calls": 4,
                "audits": 4,
                "seconds": 60,
                "inclusive_bytes": c.LIMIT,
                "review_reserve_bytes": c.REVIEW_RESERVE,
            },
        }
    )


def context(raw: bytes) -> dict[str, Any]:
    return {
        "head": "a" * 40,
        "tree": "b" * 40,
        "manifest_sha256": sha256_bytes(raw),
        "sources": dict.fromkeys(c.SOURCE_FILES, "c" * 64),
        "versions": {"fixture": "PUBLIC_FAKE"},
        "purpose": c.VERSION,
    }


class Fake:
    def __init__(
        self, output: Path, mode: str = "PASS", failure: tuple[str, int] | None = None
    ) -> None:
        self.output, self.mode, self.failure = output, mode, failure
        self.produced = self.audited = 0
        self.last: dict[str, Any] = {}

    def produce(self, study: c.Study, witness: dict[str, Any], check: Any) -> Any:
        index = self.produced
        self.produced += 1
        check()
        if self.failure == ("raw", index):
            raise RuntimeError("fake production failure")
        if index >= 2:
            assert (self.output / "seal.json").is_file()
            assert all((self.output / f"action{i}-candidate.json").is_file() for i in range(2))
            assert witness["producer_lateral"] == witness["audit_lateral"]
            assert witness["branch"] == index - 2 and witness["target_index"] == 2
        flat = [0] * 1024
        if index == 0:
            flat[:100] = [1] * 100
            flat[120:150] = [2] * 30
            if self.mode == "NO_REMEMBERED":
                flat[:100] = [0] * 100
        elif index == 1:
            flat[10:20] = [2] * 10
        elif index == 2:
            flat[:20] = [1] * 20
            flat[200:220] = [2] * 20
        else:
            flat[400:420] = [2] * 20
            flat[500:510] = [3] * 10
        if index >= 2 and self.mode == "METADATA_ONLY":
            flat = [0] * 1024
            flat[10:20] = [2] * 10
        if index == 2 and self.mode == "ZERO_ERROR":
            flat = [0] * 1024
            flat[10:20] = [2] * 10
        if index == 3 and self.mode == "NO_NEW":
            flat[500:510] = [0] * 10
        self.last = {"labels": [flat[i * 32 : (i + 1) * 32] for i in range(32)], "ties": []}
        return self.last

    def audit(self, study: c.Study, witness: dict[str, Any], check: Any) -> Any:
        index = self.audited
        self.audited += 1
        check()
        if self.failure == ("audit", index):
            raise RuntimeError("fake reference failure")
        data = {
            **self.last,
            "boundaries": [],
            "supports": [[10, 10, False]] * 2,
            "footprints": [
                {
                    "polygon": [["0", "0"], ["1", "0"], ["0", "1"]],
                    "depths": ["1", "2"],
                    "domain": True,
                }
            ]
            * 2,
        }
        if self.failure == ("boundary", index):
            data["boundaries"] = [[0, 0]]
        if self.failure == ("domain", index):
            data["footprints"] = [{**data["footprints"][0], "domain": False}] * 2
        if self.failure == ("disagreement", index):
            data["labels"] = [[0] * 32 for _ in range(32)]
        return data


def run(output: Path, fake: Fake, source_check: Any = None) -> tuple[Any, Any]:
    raw = manifest()
    ctx = context(raw)
    result = c.collect(
        raw,
        DEFINITION,
        ADOPTION,
        ctx,
        output,
        lambda p: None,
        source_check or (lambda: None),
        fake,
        lambda: 0,
        allocate=lambda: {
            "episode": "d" * 32,
            "tokens": [f"surface-{i:016x}" for i in (30, 10, 20)],
        },
    )
    return result, ctx


@pytest.mark.parametrize("mode", ["PASS", "NO_NEW", "NO_REMEMBERED", "ZERO_ERROR", "METADATA_ONLY"])
def test_qualification_and_semantic_signal(tmp_path: Path, mode: str) -> None:
    output = tmp_path / "fresh"
    fake = Fake(output, mode)
    terminal, ctx = run(output, fake)
    assert terminal["completion"] == "COMPLETE" and terminal["first_failure"] is None
    assert fake.produced == fake.audited == 4
    inspected = c.inspect(output, ctx, lambda: None)
    assert inspected["development_outcome"] == ("PASS" if mode == "PASS" else "FAIL")
    assert all(r["C"] == r["N"] and r["U"] == 0 for r in inspected["branches"])
    if mode == "METADATA_ONLY":
        assert (
            not inspected["signals"]["action_contrast"]
            and not inspected["signals"]["both_targets_novel"]
        )
    if mode == "PASS":
        assert inspected["signals"]["remembered_channel"] and inspected["signals"]["new_positive"]


@pytest.mark.parametrize(
    "kind,index",
    [
        ("raw", 0),
        ("audit", 0),
        ("raw", 2),
        ("audit", 2),
        ("boundary", 2),
        ("domain", 2),
        ("disagreement", 2),
    ],
)
def test_first_failure_raw_retention_and_stop(tmp_path: Path, kind: str, index: int) -> None:
    output = tmp_path / "fresh"
    fake = Fake(output, failure=(kind, index))
    terminal, ctx = run(output, fake)
    assert terminal["completion"] == "STOPPED_INCOMPLETE"
    assert (
        terminal["first_failure"] is not None
        and terminal["analysis"]["development_outcome"] == "INCONCLUSIVE"
    )
    assert fake.produced <= index + 1 and fake.audited <= index + 1
    if kind != "raw":
        assert (output / f"{'prefix' if index < 2 else 'target'}{index % 2}-raw.json").exists()
    assert c.inspect(output, ctx, lambda: None)["development_outcome"] == "INCONCLUSIVE"
    before = (fake.produced, fake.audited)
    with pytest.raises(FileExistsError):
        run(output, fake)
    assert before == (fake.produced, fake.audited)


@pytest.mark.parametrize("where", ["seal", "candidate"])
def test_complete_global_seal_before_any_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str
) -> None:
    original = c.Sink.json

    def tamper(self: c.Sink, name: str, value: Any, terminal: bool = False) -> None:
        original(self, name, value, terminal)
        if name == "seal.json":
            (
                self.path / ("seal.json" if where == "seal" else "action1-candidate.json")
            ).write_bytes(b"{}")

    monkeypatch.setattr(c.Sink, "json", tamper)
    output = tmp_path / "fresh"
    fake = Fake(output)
    terminal, _ = run(output, fake)
    assert fake.produced == fake.audited == 2
    assert terminal["analysis"]["development_outcome"] == "INCONCLUSIVE"
    assert terminal["first_failure"]["inspection_error"]


def test_persistent_source_failure_keeps_local_scores_unverified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = c.Sink.json
    failed = False

    def mark(self: c.Sink, name: str, value: Any, terminal: bool = False) -> None:
        nonlocal failed
        original(self, name, value, terminal)
        if name == "branch0-analysis.json":
            failed = True

    def source_check() -> None:
        if failed:
            raise ValueError("persistent source mismatch")

    monkeypatch.setattr(c.Sink, "json", mark)
    output = tmp_path / "fresh"
    fake = Fake(output)
    terminal, _ = run(output, fake, source_check)
    assert fake.produced == fake.audited == 3
    assert terminal["analysis"]["development_outcome"] == "INCONCLUSIVE"
    assert terminal["first_failure"]["prior_analysis_unverified"]["branches"]


@pytest.mark.parametrize(
    "field", ["episode", "branch", "announced_action", "source_head", "producer_lateral"]
)
def test_branch_witness_rejection(field: str) -> None:
    study = c.parse(manifest())
    mapping = {"episode": "d" * 32, "tokens": [f"surface-{i:016x}" for i in (1, 2, 3)]}
    labels = tuple(tuple([1, 2] + [0] * 30) for _ in range(32))
    frames = (
        c.visible(labels, 0, tuple(mapping["tokens"])),
        c.visible(labels, 1, tuple(mapping["tokens"])),
    )
    p = c.projections(study, frames, mapping, "a" * 40)[0]
    expected = c.witness(study, mapping, context(manifest()), "target", 0, p)
    observed = dict(expected)
    observed[field] = "WRONG"
    with pytest.raises(ValueError):
        c.Target(c.visible(labels, 2, tuple(mapping["tokens"])), observed, expected)


def test_sink_external_and_log_inclusive_cap(tmp_path: Path) -> None:
    sink = c.Sink(tmp_path, lambda: 0.0, lambda: 5 * 1024 * 1024)
    with pytest.raises(OSError, match="inclusive"):
        sink.write("excess", b"x" * 1024 * 1024)
    assert not (tmp_path / "excess").exists()


def test_public_smoke_offline_without_any_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "fresh"
    terminal, ctx = run(output, Fake(output))

    def forbidden(*a: Any, **k: Any) -> Any:
        raise AssertionError("offline regeneration")

    monkeypatch.setattr(c.ExactAdapter, "produce", forbidden)
    monkeypatch.setattr(c.ExactAdapter, "audit", forbidden)
    result = c.inspect(output, ctx, lambda: None)
    assert result == terminal["analysis"]
    print(
        "Mask development PUBLIC_FAKE_ONLY:",
        json.dumps(
            {
                "outcome": result["development_outcome"],
                "productions": 4,
                "audits": 4,
                "model": "NONE",
                "actual_members": "UNEXECUTED",
            },
            sort_keys=True,
        ),
    )


@pytest.mark.parametrize("case", ["definition", "action", "membership", "duplicate", "limits"])
def test_manifest_rejection_before_allocation(tmp_path: Path, case: str) -> None:
    p = json.loads(manifest())
    if case == "definition":
        p["definition_sha256"] = "f" * 64
    if case == "action":
        p["announced_actions"][0]["delta_lateral"] = -0.25
    if case == "membership":
        p["prefix"][1] = "1/2"
    if case == "limits":
        p["limits"]["producer_calls"] = 5
    raw = canonical_json_bytes(p)
    if case == "duplicate":
        raw = raw.replace(b"{", b'{"version":"duplicate",', 1)
    output = tmp_path / "fresh"
    fake = Fake(output)
    with pytest.raises(ValueError):
        c.collect(
            raw,
            DEFINITION,
            ADOPTION,
            context(raw),
            output,
            lambda p: None,
            lambda: None,
            fake,
            lambda: 0,
        )
    assert not output.exists() and fake.produced == 0


@pytest.mark.parametrize(
    "field", ["external_bytes", "bytes_before_terminal", "completion", "calls", "extra_file"]
)
def test_offline_terminal_accounting_rejects(tmp_path: Path, field: str) -> None:
    output = tmp_path / "fresh"
    terminal, ctx = run(output, Fake(output))
    if field == "extra_file":
        (output / "unexpected.json").write_bytes(b"{}")
        terminal["files"]["unexpected.json"] = sha256_bytes(b"{}")
        terminal["bytes_before_terminal"] += 2
    elif field == "calls":
        terminal["calls"][0]["witness"]["episode"] = "f" * 32
    elif field == "completion":
        terminal["completion"] = "UNKNOWN"
    else:
        terminal[field] = -1
    (output / "terminal.json").write_bytes(canonical_json_bytes(terminal))
    with pytest.raises(ValueError):
        c.inspect(output, ctx, lambda: None)


def test_independent_endpoint_lattice_blocks_false_availability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(c, "lattice", lambda raster: [])
    output = tmp_path / "fresh"
    fake = Fake(output)
    terminal, ctx = run(output, fake)
    assert fake.produced == fake.audited == 2
    assert not (output / "seal.json").exists()
    assert terminal["analysis"]["development_outcome"] == "INCONCLUSIVE"
    assert c.inspect(output, ctx, lambda: None)["development_outcome"] == "INCONCLUSIVE"


def test_cooperative_deadline_preserves_first_attempt(tmp_path: Path) -> None:
    value = 0.0

    class Late(Fake):
        def produce(self, study: c.Study, witness: dict[str, Any], check: Any) -> Any:
            nonlocal value
            value = 61.0
            return super().produce(study, witness, check)

    raw = manifest()
    ctx = context(raw)
    output = tmp_path / "fresh"
    fake = Late(output)
    terminal = c.collect(
        raw,
        DEFINITION,
        ADOPTION,
        ctx,
        output,
        lambda p: None,
        lambda: None,
        fake,
        lambda: 0,
        clock=lambda: value,
        allocate=lambda: {"episode": "d" * 32, "tokens": [f"surface-{i:016x}" for i in (1, 2, 3)]},
    )
    assert (
        terminal["first_failure"]["type"] == "TimeoutError"
        and fake.produced == 1
        and fake.audited == 0
    )
    assert (output / "call-01-attempt.json").exists()
    assert c.inspect(output, ctx, lambda: None)["development_outcome"] == "INCONCLUSIVE"


def test_return_retention_failure_keeps_return_in_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = c.Sink.json

    def fail(self: c.Sink, name: str, value: Any, terminal: bool = False) -> None:
        if name == "prefix0-raw.json":
            raise OSError("fake write failure")
        original(self, name, value, terminal)

    monkeypatch.setattr(c.Sink, "json", fail)
    output = tmp_path / "fresh"
    fake = Fake(output)
    terminal, ctx = run(output, fake)
    assert fake.produced == 1 and fake.audited == 0
    assert terminal["first_failure"]["unretained_return"]["returned"] == fake.last
    assert c.inspect(output, ctx, lambda: None)["development_outcome"] == "INCONCLUSIVE"


@pytest.mark.parametrize("kind", ["write", "deadline"])
@pytest.mark.parametrize("mode", ["PASS", "NO_NEW"])
def test_late_closure_failure_never_promotes_local_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, mode: str
) -> None:
    original = c.Sink.json
    value = 0.0

    def fail(self: c.Sink, name: str, data: Any, terminal: bool = False) -> None:
        nonlocal value
        if name == "branch1-analysis.json":
            if kind == "write":
                raise OSError("late fake report write failure")
            value = 61.0
        original(self, name, data, terminal)

    monkeypatch.setattr(c.Sink, "json", fail)
    raw = manifest()
    ctx = context(raw)
    output = tmp_path / "fresh"
    fake = Fake(output, mode)
    result = c.collect(
        raw,
        DEFINITION,
        ADOPTION,
        ctx,
        output,
        lambda p: None,
        lambda: None,
        fake,
        lambda: 0,
        clock=lambda: value,
        allocate=lambda: {
            "episode": "d" * 32,
            "tokens": [f"surface-{i:016x}" for i in (30, 10, 20)],
        },
    )
    assert fake.produced == fake.audited == 4
    assert all(call["completed"] and call["retained"] for call in result["calls"])
    assert result["completion"] == "STOPPED_INCOMPLETE"
    assert result["first_failure"]["type"] == ("OSError" if kind == "write" else "TimeoutError")
    assert result["analysis"]["development_outcome"] == "INCONCLUSIVE"
    assert result["analysis"]["qualification"] == "UNRESOLVED"
    local = result["analysis"]["local_analysis"]
    assert local["development_outcome"] == ("PASS" if mode == "PASS" else "FAIL")
    assert len(local["branches"]) == 2 and "signals" in local
    assert (output / "branch0-analysis.json").is_file()
    assert not (output / "branch1-analysis.json").exists()
    assert c.inspect(output, ctx, lambda: None) == result["analysis"]
    with pytest.raises(FileExistsError):
        run(output, fake)
    assert fake.produced == fake.audited == 4
    result["analysis"] = local
    (output / "terminal.json").write_bytes(canonical_json_bytes(result))
    with pytest.raises(ValueError, match="stopped closure"):
        c.inspect(output, ctx, lambda: None)
