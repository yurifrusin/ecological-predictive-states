"""Handwritten masks/fake cohorts only; proposed grid outcomes remain unproduced."""

from __future__ import annotations

import json
import subprocess
from fractions import Fraction as Q
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from epsbench.diagnostics import finite_mask_motion as c
from epsbench.diagnostics import mask_development_qualification as shared
from epsbench.diagnostics.restricted_mask_projection import Projection
from epsbench.schema import ModalityPermissionSet
from epsbench.utils.canonical import sha256_bytes

DEFINITION = b"PUBLIC HANDWRITTEN FIXTURE, NOT THE PROPOSED GRID OUTCOME"
ADOPTION = b"FAKE SOURCE CHECK ONLY"


def mask(*points: tuple[int, int]) -> Any:
    a = np.zeros((32, 32), dtype=np.bool_)
    for r, col in points:
        a[r, col] = True
    return a


def projection(previous: Any, current: Any, step: Q = Q(1, 2)) -> Projection:
    study = c.grid(DEFINITION, ADOPTION)[0]
    frames = tuple(
        shared.visible(
            shared.rows((a.astype(int)).tolist()),
            i,
            ("surface-0000000000000001", "surface-0000000000000002", "surface-0000000000000003"),
        )
        for i, a in enumerate((previous, current))
    )
    mapping = {"episode": "d" * 32, "tokens": [f"surface-{i:016x}" for i in (1, 2, 3)]}
    return shared.projections(study, frames, mapping, "a" * 40)[0 if step < 0 else 1]  # type: ignore[arg-type]


def test_shifts_and_signed_rounding() -> None:
    original = mask((0, 0), (5, 10), (31, 31))
    assert np.array_equal(c.shift(original, 2), mask((0, 2), (5, 12)))
    assert np.array_equal(c.shift(original, -2), mask((5, 8), (31, 29)))
    assert not c.shift(original, 32).any()
    assert np.array_equal(original, mask((0, 0), (5, 10), (31, 31)))
    for value, expected in [
        (Q(1, 2), 1),
        (Q(-1, 2), -1),
        (Q(3, 2), 2),
        (Q(-3, 2), -2),
        (Q(1, 3), 0),
        (Q(-1, 3), 0),
        (Q(0), 0),
    ]:
        assert c.round_away(value) == expected
    with pytest.raises(ValueError):
        c.shift(original.astype(int), 1)
    with pytest.raises(ValueError):
        c.round_away(0.5)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kind", ["unique", "empty_current", "empty_previous", "tied", "boundary", "unchanged"]
)
def test_estimator_fallbacks(kind: str) -> None:
    p = mask((10, 10), (12, 12))
    current = c.shift(p, 3)
    expected = (3, "UNIQUE_INTERIOR_IMPROVEMENT")
    if kind == "empty_current":
        current, expected = mask(), (0, "EMPTY_CURRENT")
    elif kind == "empty_previous":
        p, expected = mask(), (0, "EMPTY_PREVIOUS")
    elif kind == "tied":
        p, current, expected = mask((5, 10)), mask((5, 9), (5, 11)), (0, "TIED")
    elif kind == "boundary":
        current, expected = c.shift(p, 8), (0, "BOUNDARY")
    elif kind == "unchanged":
        current, expected = p, (0, "NO_IMPROVEMENT")
    value = c.estimate(p, current)
    assert (value["k"], value["reason"]) == expected
    # Independent literal pixel enumeration, including zero-filled clipping.
    assert value["costs"] == [
        sum(
            bool(p[r, col - k]) != bool(current[r, col])
            if 0 <= col - k < 32
            else bool(current[r, col])
            for r in range(32)
            for col in range(32)
        )
        for k in range(-8, 9)
    ]


def test_true_reversed_binding_and_remembered() -> None:
    p = projection(mask((10, 10)), mask((10, 13)))
    correct, details = c.forecast(p, "motion")
    reverse, _ = c.forecast(p, "reversed")
    assert details["nodes"][0]["forecast_shift"] == 2
    a, b = json.loads(correct), json.loads(reverse)
    assert a["binding_sha256"] == b["binding_sha256"] == sha256_bytes(p.binding_bytes())
    name = p.alignment[0]
    assert np.array_equal(np.array(a["candidate"]["known"][name]), mask((10, 15)))
    assert np.array_equal(np.array(b["candidate"]["known"][name]), mask((10, 11)))
    remembered = projection(mask((10, 10)), mask())
    for method in c.METHODS:
        candidate, _ = c.forecast(remembered, method)
        assert remembered.alignment == p.alignment
        assert not np.array(json.loads(candidate)["candidate"]["known"][name]).any()


def test_symmetric_overlap_and_token_equivariance() -> None:
    previous, current = np.zeros((32, 32), dtype=np.int32), np.zeros((32, 32), dtype=np.int32)
    previous[10, 8], previous[10, 18] = 1, 2
    current[10, 11], current[10, 15] = 1, 2
    p = projection(previous, current)
    saved, details = c.forecast(p, "motion")
    assert details["conflict_pixels"] == 1
    assert all(not np.array(a).any() for a in json.loads(saved)["candidate"]["known"].values())
    # Swapping opaque association permutes the feature/output channels.
    p2 = projection(
        np.where(previous == 1, 2, np.where(previous == 2, 1, 0)),
        np.where(current == 1, 2, np.where(current == 2, 1, 0)),
    )
    _, d2 = c.forecast(p2, "motion")
    assert d2["conflict_pixels"] == 1
    assert details["nodes"] == list(reversed(d2["nodes"]))


def test_permission_context_and_actions_fail_closed() -> None:
    p = projection(mask((10, 10)), mask((10, 13)))
    object.__setattr__(p, "access", ModalityPermissionSet(allowed=frozenset()))
    with pytest.raises(PermissionError):
        c.forecast(p, "motion")
    p = projection(mask((10, 10)), mask((10, 13)))
    object.__setattr__(p, "availability", (False, True))
    with pytest.raises(ValueError):
        c.forecast(p, "motion")
    p = projection(mask((10, 10)), mask((10, 13)))
    object.__setattr__(p, "_binding", b"{}")
    with pytest.raises(ValueError):
        c.forecast(p, "motion")


def test_common_geometry_schema_and_frozen_grid() -> None:
    studies = c.grid(DEFINITION, ADOPTION)
    assert len(studies) == 12
    assert len({c.physical_fingerprint(s) for s in studies}) == 12
    assert all(s.extent == (Q(2), Q(2)) and s.prefix == (Q(-3, 4), Q(0)) for s in studies)
    assert c.physical_fingerprint(studies[0]) == c.physical_fingerprint(
        c.grid(b"different source", b"different adoption")[0]
    )
    with pytest.raises(ValueError, match="known physical"):
        c.membership(DEFINITION, ADOPTION, (c.physical_fingerprint(studies[0]),))


class Fake:
    def __init__(self, output: Path, failure: tuple[str, int] | None = None) -> None:
        self.output, self.failure = output, failure
        self.raw = self.audit_count = 0
        self.last: dict[str, Any] = {}
        self.phases: list[str] = []

    def produce(self, study: shared.Study, witness: dict[str, Any], check: Any) -> Any:
        index = self.raw
        self.raw += 1
        self.phases.append(witness["phase"])
        check()
        if witness["phase"] == "target":
            assert self.raw > 24 and self.audit_count >= 24
            seal = json.loads((self.output / "global-seal.json").read_bytes())
            assert len(seal["forecasts"]) == 96
            assert all((self.output / n).is_file() for n in seal["forecasts"])
        if self.failure == ("raw", index):
            raise RuntimeError("handwritten first raw failure")
        a = np.zeros((32, 32), dtype=np.int32)
        # Deliberately handwriting arbitrary optical observations, never rasterizing study.
        col = 10 if witness["phase"] == "prefix" and witness["branch"] == 0 else 13
        if witness["phase"] == "target":
            col += -2 if witness["branch"] == 0 else 2
        a[10, col] = 2
        a[25, 10] = 1
        if witness["phase"] == "target":
            a[5, 20] = 3  # NEW stays in the full denominator, all methods miss it.
        self.last = {"labels": a.tolist(), "ties": []}
        return self.last

    def audit(self, study: shared.Study, witness: dict[str, Any], check: Any) -> Any:
        index = self.audit_count
        self.audit_count += 1
        check()
        if self.failure == ("audit", index):
            raise RuntimeError("handwritten first audit failure")
        return {
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


def run(
    output: Path,
    fake: Fake,
    *,
    clock: Any = None,
    external: Any = None,
    process_peak: Any = None,
    close: Any = None,
) -> tuple[Any, Any]:
    manifest = c.membership(DEFINITION, ADOPTION, ())
    ctx = {
        "head": "a" * 40,
        "tree": "b" * 40,
        "manifest_sha256": sha256_bytes(manifest),
        "sources": dict.fromkeys(c.SOURCE_FILES, "c" * 64),
        "versions": {"fixture": "HANDWRITTEN"},
        "purpose": c.VERSION,
    }
    count = 0

    def allocate() -> Any:
        nonlocal count
        count += 1
        return {
            "episode": f"{count:032x}",
            "tokens": [f"surface-{count * 10 + i:016x}" for i in range(3)],
        }

    kw = {} if clock is None else {"clock": clock}
    result = c.collect(
        manifest,
        DEFINITION,
        ADOPTION,
        ctx,
        output,
        lambda p: None,
        lambda: None,
        fake,
        external or (lambda: 0),
        process_peak or (lambda: 1024),
        1024,
        close or (lambda value: None),
        allocate=allocate,
        **kw,
    )
    return result, ctx


def test_fake_cohort_global_seal_and_neutral_counts(tmp_path: Path) -> None:
    output = tmp_path / "cohort"
    fake = Fake(output)
    terminal, ctx = run(output, fake)
    assert terminal["completion"] == "COMPLETE", terminal["first_failure"]
    assert fake.raw == fake.audit_count == 48
    assert fake.phases == ["prefix"] * 24 + ["target"] * 24
    result = c.inspect(output, ctx, lambda: None, targets=True)
    assert result == terminal["analysis"]
    assert result["finite_outcome"] == "POSITIVE"
    assert result["losses"]["motion"] == "1/3072"
    assert all(len(g["branches"]) == 2 for g in result["groups"])
    for group in result["groups"]:
        for branch in group["branches"]:
            for method in c.METHODS:
                r = branch[method]
                assert (r["N"], r["C"], r["U"]) == (3072, 3072, 0)
                assert r["channels"][-1]["FN"] == 1
                assert r["undefined_iou_count"] == 0
    (output / "g00-b0-motion-candidate.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="sealed"):
        c.inspect(output, ctx, lambda: None)


@pytest.mark.parametrize("failure", [("raw", 0), ("audit", 4), ("raw", 24), ("audit", 25)])
def test_first_failure_no_retry_and_partial_retention(
    tmp_path: Path, failure: tuple[str, int]
) -> None:
    output = tmp_path / "failed"
    fake = Fake(output, failure)
    closed: list[dict[str, Any]] = []
    terminal, _ = run(output, fake, close=closed.append)
    assert terminal["completion"] == "STOPPED_INCOMPLETE"
    assert terminal["analysis"]["finite_outcome"] == "INCONCLUSIVE"
    assert terminal["first_failure"]["type"] == "RuntimeError"
    assert len(closed) == 1
    assert len(terminal["calls"]) == 2 * failure[1] + (1 if failure[0] == "raw" else 2)
    assert (output / "terminal.json").is_file()
    assert (output / "global-seal.json").exists() == (failure[1] >= 24)


def test_wrong_branch_and_single_fetch() -> None:
    p = projection(mask((10, 10)), mask((10, 13)))
    study = c.grid(DEFINITION, ADOPTION)[0]
    mapping = {"episode": "d" * 32, "tokens": [f"surface-{i:016x}" for i in (1, 2, 3)]}
    ctx = {
        "manifest_sha256": sha256_bytes(study.payload),
        "head": "a" * 40,
        "tree": "b" * 40,
        "sources": {},
    }
    w = shared.witness(study, mapping, ctx, "target", 1, p)
    value = shared.visible(
        shared.rows(np.zeros((32, 32), dtype=int).tolist()), 2, tuple(mapping["tokens"])
    )
    with pytest.raises(ValueError):
        shared.Target(value, {**w, "branch": 0}, w)
    target = shared.Target(value, w, w)
    target.raster(2)
    with pytest.raises(PermissionError):
        target.raster(2)
    with pytest.raises(ValueError):
        shared.witness(study, mapping, ctx, "target", 0, p)


def test_post_terminal_expiration_external_closure(tmp_path: Path, monkeypatch: Any) -> None:
    current = [0.0]
    old = c.Sink.write

    def write(self: Any, name: str, raw: bytes, terminal: bool = False) -> None:
        old(self, name, raw, terminal)
        if name == "terminal.json":
            current[0] = 121.0

    monkeypatch.setattr(c.Sink, "write", write)
    closed: list[dict[str, Any]] = []
    with pytest.raises(RuntimeError, match="closure"):
        run(
            tmp_path / "expired",
            Fake(tmp_path / "expired", ("raw", 0)),
            clock=lambda: current[0],
            close=closed.append,
        )
    assert closed[0]["analysis"]["finite_outcome"] == "INCONCLUSIVE"
    assert closed[0]["local_terminal_qualified"] is False


def test_resource_measurements_fail_closed(tmp_path: Path) -> None:
    closed: list[dict[str, Any]] = []
    output = tmp_path / "cap"
    fake = Fake(output)
    with pytest.raises(RuntimeError, match="closure"):
        run(output, fake, process_peak=lambda: c.PROCESS_LIMIT + 1, close=closed.append)
    assert fake.raw == 0 and closed[0]["analysis"]["finite_outcome"] == "INCONCLUSIVE"
    assert len(closed) == 1


def test_group_weighting_ties_and_nondiscrimination() -> None:
    groups: list[dict[str, Any]] = [
        {
            "losses": {"motion": "1/4", "persistence": "1/4", "empty": "1/2", "reversed": "1/2"},
            "prefix_change": True,
            "targets_persistence": False,
            "target_action_contrast": True,
            "correct_reversed_identical": False,
        }
        for _ in range(12)
    ]
    result = c.summarize(groups)
    assert result["finite_outcome"] == "NEGATIVE"
    assert all(x["ties"]["persistence"] for x in result["group_differences"])
    groups[0]["losses"]["motion"] = "0"
    result = c.summarize(groups)
    assert result["finite_outcome"] == "POSITIVE" and result["losses"]["motion"] == "11/48"
    for group in groups:
        group["prefix_change"] = False
    assert c.summarize(groups)["finite_outcome"] == "NEGATIVE"


def test_empty_inventory_complete_new_and_undefined_iou() -> None:
    p = projection(mask(), mask())
    saved, details = c.forecast(p, "motion")
    c.verify_forecast(p, "motion", saved, details)
    raster = shared.visible(
        shared.rows(np.zeros((32, 32), dtype=int).tolist()),
        2,
        ("surface-0000000000000001", "surface-0000000000000002", "surface-0000000000000003"),
    )

    class Provider:
        def raster(self, index: int) -> Any:
            assert index == 2
            return raster

    value = p.evaluate(saved, Provider())
    assert (value.report["N"], value.report["C"], value.report["U"], value.report["E"]) == (
        1024,
        1024,
        0,
        0,
    )
    rows = c.count_channels(p, saved, value)
    assert rows == [
        {
            "channel": "NEW",
            "prediction_pixels": 0,
            "target_pixels": 0,
            "TP": 0,
            "FP": 0,
            "FN": 0,
            "iou": None,
        }
    ]


def test_zero_measurement_and_post_write_accounting(tmp_path: Path, monkeypatch: Any) -> None:
    output = tmp_path / "zero"
    closed: list[dict[str, Any]] = []
    with pytest.raises(RuntimeError, match="closure"):
        run(output, Fake(output), process_peak=lambda: 0, close=closed.append)
    assert closed[0]["analysis"]["finite_outcome"] == "INCONCLUSIVE"
    overhead = [0]
    old = c.Sink.write

    def write(self: Any, name: str, raw: bytes, terminal: bool = False) -> None:
        old(self, name, raw, terminal)
        if name == "terminal.json":
            overhead[0] = c.LIMIT

    monkeypatch.setattr(c.Sink, "write", write)
    output = tmp_path / "accounting"
    closed = []
    with pytest.raises(RuntimeError, match="closure"):
        run(output, Fake(output, ("raw", 0)), external=lambda: overhead[0], close=closed.append)
    assert closed[0]["local_terminal_qualified"] is False


def test_independent_arithmetic_rejects_rule_candidate_change() -> None:
    p = projection(mask((10, 10)), mask((10, 13)))
    saved, details = c.forecast(p, "motion")
    c.verify_forecast(p, "motion", saved, details)
    details["nodes"][0]["forecast_shift"] = -2
    with pytest.raises(ValueError, match="independent"):
        c.verify_forecast(p, "motion", saved, details)


@pytest.mark.parametrize("name", ["g11-b1-reversed-candidate.json", "g00-prefix0-raw.json"])
def test_missing_forecast_or_unretained_return_preserved(
    tmp_path: Path, monkeypatch: Any, name: str
) -> None:
    old = c.Sink.write

    def write(self: Any, filename: str, raw: bytes, terminal: bool = False) -> None:
        if filename == name:
            raise OSError("handwritten retention failure")
        old(self, filename, raw, terminal)

    monkeypatch.setattr(c.Sink, "write", write)
    output = tmp_path / "partial"
    fake = Fake(output)
    closed: list[dict[str, Any]] = []
    terminal, _ = run(output, fake, close=closed.append)
    assert terminal["completion"] == "STOPPED_INCOMPLETE"
    assert terminal["analysis"]["finite_outcome"] == "INCONCLUSIVE"
    assert "target" not in fake.phases and not (output / "global-seal.json").exists()
    assert terminal["first_failure"]["message"] == "handwritten retention failure"
    if name.endswith("raw.json"):
        assert fake.raw == 1 and fake.audit_count == 0
        assert terminal["first_failure"]["unretained_return"]["returned"] == fake.last
        assert terminal["calls"][0]["completed"] and not terminal["calls"][0]["retained"]
    else:
        assert fake.raw == fake.audit_count == 24
    assert closed == [terminal]


def test_exact_source_bytes_head_and_runtime_binding(tmp_path: Path, monkeypatch: Any) -> None:
    manifest = c.membership(DEFINITION, ADOPTION, ())
    sources = {}
    for name in c.SOURCE_FILES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"HANDWRITTEN SOURCE IDENTITY FIXTURE")
        sources[name] = sha256_bytes(target.read_bytes())
    context = {
        "head": "a" * 40,
        "tree": "b" * 40,
        "manifest_sha256": sha256_bytes(manifest),
        "sources": sources,
        "versions": {"fixture": "FAKE"},
        "purpose": c.VERSION,
    }

    class Result:
        stdout = "a" * 40 + "\n" + "b" * 40 + "\n"

    commands = []

    def run(command: Any, **kwargs: Any) -> Any:
        commands.append(command)
        return Result()

    monkeypatch.setattr(subprocess, "run", run)
    c.verify_source(tmp_path, context, manifest)
    assert commands[0][1:3] == ["-c", "safe.directory=" + tmp_path.resolve().as_posix()]
    with pytest.raises(ValueError, match="head/tree"):
        c.verify_source(tmp_path, {**context, "head": "d" * 40}, manifest)
    (tmp_path / c.SOURCE_FILES[0]).write_bytes(b"CHANGED")
    with pytest.raises(ValueError, match="source bytes"):
        c.verify_source(tmp_path, context, manifest)
    with pytest.raises(ValueError, match="cohort/runtime"):
        c.check_context({**context, "versions": {}}, manifest)
