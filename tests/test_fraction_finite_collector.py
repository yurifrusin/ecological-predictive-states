"""Public fictional provider outcomes; never generate the private study members."""

from __future__ import annotations

import json
from fractions import Fraction as Q
from pathlib import Path
from typing import Any

import pytest

from epsbench.diagnostics import fraction_finite_collector as c
from epsbench.utils.canonical import canonical_json_bytes, sha256_bytes

PROPOSAL = b"PUBLIC SYNTHETIC DEFINITION ONLY"
ADOPTION = b"PUBLIC SYNTHETIC ADOPTION ONLY"


def manifest() -> bytes:
    # Deliberately different public dummy parameters, not study membership.
    units = []
    for start, depth in (("-1", "4"), ("1", "5")):
        units.append(
            {
                "boxes": [
                    {"lower": ["-1", "1", "1"], "upper": ["1", "2", "2"]},
                    {"lower": ["-1", depth, "1"], "upper": ["1", "6", "3"]},
                ],
                "prefix": [start, "0"],
                "actions": ["1/2", "-1/2"],
            }
        )
    return canonical_json_bytes(
        {
            "version": c.VERSION,
            "proposal_sha256": sha256_bytes(PROPOSAL),
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
            "units": units,
            "criterion": {
                "margin": "1/1024",
                "informative": list(c.INFORMATIVE),
                "precedence": "FAIL_PASS_INCONCLUSIVE",
            },
            "limits": {
                "producer_calls": 8,
                "audits": 8,
                "seconds": 60,
                "inclusive_bytes": c.LIMIT,
                "review_reserve_bytes": c.REVIEW_RESERVE,
            },
        }
    )


def context(payload: bytes) -> dict[str, Any]:
    return {
        "head": "a" * 40,
        "tree": "b" * 40,
        "manifest_sha256": sha256_bytes(payload),
        "sources": dict.fromkeys(c.SOURCE_FILES, "c" * 64),
        "versions": {"fixture": "PUBLIC_FAKE"},
        "purpose": c.VERSION,
    }


class Clock:
    value = 0.0

    def __call__(self) -> float:
        return self.value


class Fake:
    def __init__(
        self,
        output: Path,
        modes: tuple[str, str] = ("PASS", "PASS"),
        failure: tuple[str, int] | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.output, self.modes, self.failure, self.clock = output, modes, failure, clock
        self.produced = self.audited = 0
        self.last: dict[str, Any] = {}

    def produce(self, unit: c.Unit, lateral: Q, extent: tuple[Q, Q], check: Any) -> dict[str, Any]:
        index = self.produced
        self.produced += 1
        if index >= 4:
            assert (self.output / "seal.json").exists()
            assert len(list(self.output.glob("*-forecast.json"))) == 4
        if self.failure == ("raw", index):
            raise RuntimeError("synthetic producer failure")
        if self.failure == ("timeout", index):
            assert self.clock is not None
            self.clock.value = 61
        check()
        if index < 4:
            count = 2 if index % 2 == 0 else 6
        else:
            i, q = (index - 4) // 2, (index - 4) % 2
            mode = self.modes[i]
            count = (8 if q == 0 else 4) if i == 0 else (4 if q == 0 else 8)
            if mode == "FAIL":
                count = 5 if q == 0 else 7
            elif mode == "UNINFORMATIVE":
                count = 6
        flat = [1] * count + [2] * 3 + [0] * (1024 - count - 3)
        self.last = {"labels": [flat[r * 32 : (r + 1) * 32] for r in range(32)], "ties": []}
        return self.last

    def audit(self, unit: c.Unit, lateral: Q, extent: tuple[Q, Q], check: Any) -> dict[str, Any]:
        index = self.audited
        self.audited += 1
        if self.failure == ("audit", index):
            raise RuntimeError("synthetic audit failure")
        check()
        data = {
            **self.last,
            "boundaries": [],
            "supports": [[10, 10, False], [10, 10, False]],
            "footprints": [
                {
                    "polygon": [["0", "0"], ["1", "0"], ["0", "1"]],
                    "depths": ["1", "2"],
                    "domain": True,
                }
            ]
            * 2,
        }
        if self.failure == ("tie", index):
            data["ties"] = [[0, 0]]
        elif self.failure == ("boundary", index):
            data["boundaries"] = [[0, 0]]
        elif self.failure == ("domain", index):
            data["footprints"] = [{**data["footprints"][0], "domain": False}] * 2
        elif self.failure == ("disagreement", index):
            data["labels"] = [[0] * 32 for _ in range(32)]
        return data


def run(
    output: Path, fake: Fake, clock: Clock | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = manifest()
    ctx = context(payload)
    counter = 0

    def allocate() -> dict[str, Any]:
        nonlocal counter
        counter += 1
        return {
            "episode_key": f"{counter:032x}",
            "tokens": ["surface-" + f"{counter * 10 + i:016x}" for i in range(3)],
        }

    result = c.collect(
        payload,
        PROPOSAL,
        ADOPTION,
        ctx,
        output,
        lambda p: None,
        lambda: None,
        fake,
        clock or Clock(),
        allocate,
    )
    return result, ctx


@pytest.mark.parametrize(
    "modes,expected",
    [
        (("PASS", "PASS"), "PASS"),
        (("FAIL", "PASS"), "FAIL"),
        (("PASS", "UNINFORMATIVE"), "INCONCLUSIVE"),
        (("FAIL", "UNINFORMATIVE"), "FAIL"),
    ],
)
def test_complete_fixed_outcomes_and_offline_inspection(
    tmp_path: Path, modes: tuple[str, str], expected: str
) -> None:
    output = tmp_path / "fresh"
    fake = Fake(output, modes)
    terminal, ctx = run(output, fake)
    assert terminal["analysis"]["scientific_outcome"] == expected
    assert terminal["completion"] == "COMPLETE" and terminal["first_failure"] is None
    assert fake.produced == fake.audited == 8
    inspected = c.inspect(output, ctx)
    assert inspected["scientific_outcome"] == expected
    assert inspected["inclusive_accounted_bytes"] <= c.LIMIT
    assert inspected["units"][0]["complete_coverage"] is True
    if modes[0] == "PASS":
        assert inspected["units"][0]["delta"] == "-1/1024"


@pytest.mark.parametrize(
    "kind", ["raw", "audit", "tie", "boundary", "domain", "disagreement", "timeout"]
)
def test_valid_negative_survives_incomplete_other_unit(tmp_path: Path, kind: str) -> None:
    output = tmp_path / "fresh"
    clock = Clock()
    fake = Fake(output, ("FAIL", "PASS"), (kind, 6), clock)
    terminal, ctx = run(output, fake, clock)
    assert terminal["completion"] == "STOPPED_INCOMPLETE"
    assert terminal["first_failure"] is not None
    assert terminal["analysis"]["scientific_outcome"] == "FAIL"
    assert c.inspect(output, ctx)["scientific_outcome"] == "FAIL"
    assert fake.produced <= 7 and fake.audited <= 7
    assert terminal["unattempted_producer_calls"] == 1


def test_prefix_failure_never_seals_or_accesses_targets(tmp_path: Path) -> None:
    output = tmp_path / "fresh"
    fake = Fake(output, failure=("audit", 1))
    terminal, ctx = run(output, fake)
    assert not (output / "seal.json").exists() and fake.produced == fake.audited == 2
    assert (output / "u0-p1-raw.json").exists()
    assert terminal["analysis"]["scientific_outcome"] == "INCONCLUSIVE"
    assert c.inspect(output, ctx)["completion"] == "STOPPED_INCOMPLETE"


def test_global_disk_tamper_before_any_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "fresh"
    original = c.Sink.json

    def corrupt(self: c.Sink, name: str, value: Any, terminal: bool = False) -> None:
        original(self, name, value, terminal)
        if name == "seal.json":
            (self.path / "u1-q1-forecast.json").write_bytes(b"{}")

    monkeypatch.setattr(c.Sink, "json", corrupt)
    fake = Fake(output)
    terminal, ctx = run(output, fake)
    assert fake.produced == fake.audited == 4
    assert terminal["completion"] == "STOPPED_INCOMPLETE"
    with pytest.raises(ValueError, match="seal"):
        c.inspect(output, ctx)


def test_exclusive_attempt_and_no_silent_retry(tmp_path: Path) -> None:
    output = tmp_path / "fresh"
    fake = Fake(output, failure=("raw", 0))
    run(output, fake)
    saved = (output / "terminal.json").read_bytes()
    with pytest.raises(FileExistsError):
        run(output, fake)
    assert fake.produced == 1 and fake.audited == 0
    assert (output / "terminal.json").read_bytes() == saved


@pytest.mark.parametrize(
    "case",
    ["rational", "extra", "duplicate_key", "calibration", "return", "margin", "cap", "definition"],
)
def test_manifest_domain_binding_before_access(tmp_path: Path, case: str) -> None:
    payload = manifest()
    value = json.loads(payload)
    if case == "rational":
        value["support"][0] = "16/2"
    elif case == "extra":
        value["hidden"] = 1
    elif case == "calibration":
        value["calibration"]["clipping"] = "FINITE"
    elif case == "return":
        value["units"][0]["actions"][1] = "-1"
    elif case == "margin":
        value["criterion"]["margin"] = "0"
    elif case == "cap":
        value["limits"]["producer_calls"] = 9
    elif case == "definition":
        value["proposal_sha256"] = "wrong"
    payload = canonical_json_bytes(value)
    if case == "duplicate_key":
        payload = payload.replace(b"{", b'{"version":"fraction-finite-study-v1",', 1)
    fake = Fake(tmp_path / "fresh")
    with pytest.raises(ValueError):
        c.collect(
            payload,
            PROPOSAL,
            ADOPTION,
            context(payload),
            fake.output,
            lambda p: None,
            lambda: None,
            fake,
        )
    assert not fake.output.exists() and not fake.produced


def test_observed_label_boundary_hides_privileged_role_order() -> None:
    tokens = tuple("surface-" + f"{x:016x}" for x in (30, 20, 10))
    labels = tuple(tuple([1, 3] + [0] * 30) for _ in range(32))
    observed = c.visible(labels, 0, tokens)
    assert set(observed.identities) == {(1, tokens[2]), (2, tokens[0])}
    assert tokens[1] not in dict(observed.identities).values()
    assert observed.segmentation[0, 0] == 2 and observed.segmentation[0, 1] == 1


def test_sink_failure_retains_raw_before_failed_reference_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = c.Sink.write

    def fail(self: c.Sink, name: str, data: bytes, terminal: bool = False) -> None:
        if name == "u0-p0-audit.json":
            raise OSError("synthetic sink failure")
        original(self, name, data, terminal)

    monkeypatch.setattr(c.Sink, "write", fail)
    output = tmp_path / "fresh"
    terminal, ctx = run(output, Fake(output))
    assert (output / "u0-p0-raw.json").exists()
    assert terminal["calls"][1]["completed"] is True and terminal["calls"][1]["retained"] is False
    assert terminal["first_failure"]["type"] == "OSError"
    assert c.inspect(output, ctx)["scientific_outcome"] == "INCONCLUSIVE"


def test_resource_accounting_reserves_duplicate_archive_and_review(tmp_path: Path) -> None:
    sink = c.Sink(tmp_path, Clock())
    with pytest.raises(OSError, match="inclusive"):
        sink.write("too-large", b"x" * ((c.LIMIT - c.REVIEW_RESERVE) // 2))
    assert not (tmp_path / "too-large").exists()
    sink.json("terminal.json", {"first_failure": "budget refusal"}, terminal=True)
    assert 2 * sink.bytes + c.REVIEW_RESERVE <= c.LIMIT


def test_retained_only_verifier_does_not_call_producer_or_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "fresh"
    terminal, ctx = run(output, Fake(output))

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("offline inspector called an algorithm")

    monkeypatch.setattr(c.ExactAdapter, "produce", forbidden)
    monkeypatch.setattr(c.ExactAdapter, "audit", forbidden)
    inspected = c.inspect(output, ctx)
    assert inspected["scientific_outcome"] == terminal["analysis"]["scientific_outcome"]
    print(
        "Public fake finite smoke:",
        json.dumps(
            {
                "scientific_outcome": inspected["scientific_outcome"],
                "completion": inspected["completion"],
                "producer_calls": 8,
                "audits": 8,
                "original_bytes": inspected["original_bytes"],
                "inclusive_accounted_bytes": inspected["inclusive_accounted_bytes"],
                "actual_study": "UNEXECUTED",
                "physical_truth": "PUBLIC_FAKE_ONLY",
            },
            sort_keys=True,
        ),
    )


@pytest.mark.parametrize("case", ["terminal", "forecast", "unbound"])
def test_strict_retained_inspection_tamper(tmp_path: Path, case: str) -> None:
    output = tmp_path / "fresh"
    _, ctx = run(output, Fake(output))
    if case == "unbound":
        (output / "unexpected").write_bytes(b"x")
    elif case == "forecast":
        (output / "u0-q0-forecast.json").write_bytes(b"{}")
    else:
        t = json.loads((output / "terminal.json").read_bytes())
        t["analysis"]["scientific_outcome"] = "FAIL"
        (output / "terminal.json").write_bytes(canonical_json_bytes(t))
    with pytest.raises(ValueError):
        c.inspect(output, ctx)


@pytest.mark.parametrize("earlier", ["PASS", "FAIL"])
def test_global_tamper_invalidates_prior_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, earlier: str
) -> None:
    original = c.Sink.json

    def tamper(self: c.Sink, name: str, value: Any, terminal: bool = False) -> None:
        original(self, name, value, terminal)
        if name == "u0-analysis.json":
            (self.path / "seal.json").write_bytes(b"{}")

    monkeypatch.setattr(c.Sink, "json", tamper)
    output = tmp_path / "fresh"
    fake = Fake(output, (earlier, "PASS"))
    terminal, ctx = run(output, fake)
    assert terminal["analysis"]["scientific_outcome"] == "INCONCLUSIVE"
    assert terminal["first_failure"]["prior_analysis_unverified"]["scientific_outcome"] == (
        "FAIL" if earlier == "FAIL" else "INCONCLUSIVE"
    )
    assert terminal["first_failure"]["inspection_error"]
    assert fake.produced == 6
    assert (output / "u0-analysis.json").exists()
    with pytest.raises(ValueError):
        c.inspect(output, ctx)
