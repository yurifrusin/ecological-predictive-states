"""Geometry-unit Brier accounting and paired bootstrap; no predictor/target producer."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from epsbench.diagnostics.restricted_learning_contract import Forecast, InputEvidence


@dataclass(frozen=True)
class Score:
    denominator: int
    unknown: int
    known_sum: float

    def __post_init__(self) -> None:
        if (
            type(self.denominator) is not int
            or type(self.unknown) is not int
            or not (0 <= self.unknown <= self.denominator)
            or type(self.known_sum) is not float
            or not math.isfinite(self.known_sum)
            or not (0 <= self.known_sum <= self.denominator - self.unknown)
        ):
            raise ValueError("strict nonnegative score/count bounds required")

    @property
    def coverage(self) -> float | None:
        return (
            None if not self.denominator else (self.denominator - self.unknown) / self.denominator
        )

    @property
    def conditional(self) -> float | None:
        n = self.denominator - self.unknown
        return self.known_sum / n if n else None

    @property
    def interval(self) -> tuple[float, float] | None:
        if not self.denominator:
            return None
        return self.known_sum / self.denominator, (self.known_sum + self.unknown) / self.denominator

    @property
    def risk(self) -> float | None:
        return self.conditional if self.denominator and not self.unknown else None


def combine(scores: Iterable[Score]) -> Score:
    values = tuple(scores)
    return Score(
        sum(s.denominator for s in values),
        sum(s.unknown for s in values),
        math.fsum(s.known_sum for s in values),
    )


def score(
    forecast: Forecast, evidence: InputEvidence, truth: tuple[tuple[str, bool], ...]
) -> dict[str, Score]:
    forecast.validate_input(evidence)  # Target access is the lifecycle's responsibility.
    if (
        type(truth) is not tuple
        or any(type(y) is not bool for _, y in truth)
        or (
            len({h for h, _ in truth}) != len(truth)
            or {h for h, _ in truth} != set(evidence.prefix.inventory)
        )
    ):
        raise ValueError("exact independently qualified observed-token truth required")
    visible = {h for h, _ in evidence.prefix.frames[-1].masks}
    targets = dict(truth)
    rows: dict[str, list[Score]] = {
        s: []
        for s in ("all", "current-absent", "current-visible", "future-visible", "future-absent")
    }
    for h, p in forecast.channels:
        y = targets[h]
        item = Score(1, int(p is None), 0.0 if p is None else (p - float(y)) ** 2)
        for stratum in (
            "all",
            "current-visible" if h in visible else "current-absent",
            "future-visible" if y else "future-absent",
        ):
            rows[stratum].append(item)
    return {k: combine(v) for k, v in rows.items()}


def balanced_loss(rows: dict[str, Score]) -> float:
    values = [rows[s].risk for s in ("current-absent", "current-visible") if rows[s].denominator]
    if not values or any(v is None for v in values):
        raise ValueError("nonempty complete assertion strata required for loss")
    return math.fsum(v for v in values if v is not None) / len(values)


@dataclass(frozen=True)
class GeometryScore:
    geometry: str
    fit: str
    decisions: tuple[tuple[str, dict[str, Score]], ...]

    def aggregate(self, expected: tuple[str, ...]) -> dict[str, Score]:
        if (
            len(self.decisions) != 8
            or len({k for k, _ in self.decisions}) != 8
            or ({k for k, _ in self.decisions} != set(expected))
        ):
            raise ValueError("all eight dependent decisions retained; no dropping")
        names = {"all", "current-absent", "current-visible", "future-visible", "future-absent"}
        if any(set(v) != names for _, v in self.decisions):
            raise ValueError("all score strata required")
        return {s: combine(v[s] for _, v in self.decisions) for s in sorted(names)}


def bootstrap(
    values: tuple[tuple[float, ...], ...], seed: bytes, domain: str
) -> tuple[tuple[float, ...], ...]:
    """One common geometry resample for every endpoint; seed allocated only later.

    SHA256 counter stream with unbiased rejection into 32 indices; deterministic
    integer domain independent of endpoint values. No views/tokens/init resampling.
    """
    if (
        type(seed) is not bytes
        or len(seed) != 32
        or domain != "M0-occupancy-development-v1/bootstrap-v1"
    ):
        raise ValueError("precommitted private bootstrap seed/domain required")
    if not values or any(len(v) != 32 or any(not math.isfinite(x) for x in v) for v in values):
        raise ValueError("complete finite 32-geometry paired endpoint vectors required")
    output: list[list[float]] = [[] for _ in values]
    counter = 0
    for _ in range(10000):
        indices: list[int] = []
        while len(indices) < 32:
            # 32 divides 2^64 exactly: no modulo bias/rejection is necessary here.
            data = hashlib.sha256(
                domain.encode() + b"\0" + seed + counter.to_bytes(8, "little")
            ).digest()
            counter += 1
            indices.extend(int.from_bytes(data[j : j + 8], "little") % 32 for j in range(0, 32, 8))
        for target, v in zip(output, values, strict=True):
            target.append(math.fsum(v[i] for i in indices) / 32)
    return tuple(tuple(v) for v in output)


@dataclass(frozen=True)
class EndpointResult:
    disposition: str
    low_improvement: float
    low_interval: tuple[float, float]
    high_absent_upper: float
    high_visible_upper: float


def decide(
    low_improvement: tuple[float, ...],
    high_absent_penalty: tuple[float, ...],
    high_visible_penalty: tuple[float, ...],
    low_candidate: tuple[float, ...],
    control_improvements: tuple[tuple[float, ...], ...],
    ablation_penalties: tuple[tuple[float, ...], ...],
    seed: bytes,
) -> EndpointResult:
    """Inputs are geometry means after averaging three paired initializations.

    Lifecycle must supply complete qualified/fair chronology and 100% coverage;
    this numerical rule cannot certify those prerequisites or advance a gate.
    """
    vectors = (
        low_improvement,
        high_absent_penalty,
        high_visible_penalty,
        low_candidate,
        *control_improvements,
        *ablation_penalties,
    )
    if len(control_improvements) != 5 or len(ablation_penalties) != 2:
        raise ValueError("all five controls and both diagnostics required")
    samples = bootstrap(vectors, seed, "M0-occupancy-development-v1/bootstrap-v1")
    interval = tuple(float(v) for v in np.percentile(samples[0], [2.5, 97.5], method="linear"))
    absent, visible = (float(np.percentile(samples[j], 95, method="linear")) for j in (1, 2))
    improvement = math.fsum(low_improvement) / 32
    passed = improvement >= 0.02 and interval[0] > 0 and absent <= 0.02 and visible <= 0.02
    passed &= math.fsum(low_candidate) / 32 <= 0.15
    passed &= all(math.fsum(v) / 32 >= 0.02 for v in (*control_improvements, *ablation_penalties))
    return EndpointResult(
        "SUPPORT" if passed else "NOT_SUPPORTED",
        improvement,
        (interval[0], interval[1]),
        absent,
        visible,
    )


def paired_means(rows: tuple[tuple[float, float, float], ...]) -> tuple[float, ...]:
    if len(rows) != 32 or any(len(r) != 3 or any(not math.isfinite(v) for v in r) for r in rows):
        raise ValueError("all 32 geometries and three paired initialization risks required")
    return tuple(math.fsum(r) / 3 for r in rows)
