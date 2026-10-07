"""Evaluator-only read-only export boundary; never given to model code."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from epsbench.diagnostics.restricted_learning_contract import (
    INITIALIZATIONS,
    REQUIRED,
    InputEvidence,
    Observation,
    digest,
    sha,
)
from epsbench.diagnostics.restricted_learning_membership import (
    INIT_DOMAIN,
    MembershipLock,
    decisions,
)
from epsbench.diagnostics.restricted_learning_retention import (
    QualifiedTarget,
    _control_rule,
    inspect_archive,
    reject_reparse,
)
from epsbench.diagnostics.restricted_learning_sampling import commitment
from epsbench.diagnostics.visible_forecast_contract import _json
from epsbench.schema import ModalityPermissionSet
from epsbench.utils.canonical import canonical_json_bytes


@dataclass(frozen=True)
class Example:
    observations: tuple[Observation, Observation, Observation]
    announced: tuple[float, float, float]
    truth: tuple[bool, ...]

    def __post_init__(self) -> None:
        if (
            type(self.observations) is not tuple
            or len(self.observations) != 3
            or any(
                type(o) is not Observation or o.index != i for i, o in enumerate(self.observations)
            )
        ):
            raise ValueError("immutable lawful three-observation example required")
        if (
            type(self.truth) is not tuple
            or len(self.truth) != len(self.observations[-1].handles)
            or any(type(y) is not bool for y in self.truth)
        ):
            raise ValueError("exact observed-inventory training labels required")
        previous: tuple[str, ...] = ()
        first_seen: tuple[int, ...] = ()
        for o in self.observations:
            if (
                o.handles[: len(previous)] != previous
                or o.first_seen[: len(first_seen)] != first_seen
                or any(
                    o.first_seen[i] != o.index or not o.visible[i]
                    for i in range(len(previous), len(o.handles))
                )
            ):
                raise ValueError("causal visible arrivals and persistent first-seen required")
            previous, first_seen = o.handles, o.first_seen
        import math

        if any(type(v) is not float or not math.isfinite(v) for v in self.announced):
            raise ValueError("finite float command required")
        if type(self.announced) is not tuple or len(self.announced) != 3:
            raise ValueError("immutable command required")


@dataclass(frozen=True)
class TrainingMaterial:
    budget: int
    examples: tuple[Example, ...]
    development: tuple[Example, ...]
    order: tuple[int, ...]
    initialization_values: tuple[int, int, int]
    data_root: str
    schedule_root: str
    seal_a: str
    labels_root: str

    def __post_init__(self) -> None:
        for root in (self.data_root, self.schedule_root, self.seal_a, self.labels_root):
            sha(root)
        if (
            self.budget not in (16, 64)
            or type(self.examples) is not tuple
            or type(self.development) is not tuple
            or any(type(e) is not Example for e in (*self.examples, *self.development))
        ):
            raise ValueError("typed train/development material only")
        if (
            len(self.examples) != self.budget * 8
            or len(self.development) != 128
            or type(self.order) is not tuple
            or len(self.order) != 16000
            or any(type(i) is not int or not 0 <= i < len(self.examples) for i in self.order)
        ):
            raise ValueError("full fixed budget/schedule required")
        if (
            type(self.initialization_values) is not tuple
            or len(set(self.initialization_values)) != 3
            or any(type(i) is not int or not 0 <= i < 2**64 for i in self.initialization_values)
        ):
            raise ValueError("three paired uint64 initializations required")


def lawful_example(source: InputEvidence, truth: tuple[tuple[str, bool], ...]) -> Example:
    view = source.streaming()
    observations = tuple(view.next_observation() for _ in range(3))
    labels = dict(truth)
    if set(labels) != set(observations[-1].handles):
        raise ValueError("exact training label inventory required")
    command = tuple(float(v) for v in view.announced)
    return Example(
        (observations[0], observations[1], observations[2]),
        (command[0], command[1], command[2]),
        tuple(labels[h] for h in observations[-1].handles),
    )


class ReadOnlyExport:
    """Privileged evaluator capability: holds identifiers, never passed to learners.

    Authenticated receipt roots are explicit launch inputs supplied by the evaluator.
    Inspection occurs once; all released bytes are checked against retained SealA.
    """

    def __init__(self, root: Path, accepted_archive: str, accepted_seal_a: str) -> None:
        sha(accepted_archive)
        sha(accepted_seal_a)
        report = inspect_archive(root)
        if (
            report["failed"]
            or report["archive_root"] != accepted_archive
            or report["seal_a"] != accepted_seal_a
            or report["seal_b"] is not None
        ):
            raise PermissionError("authenticated closed qualified archive required")
        self._root = root
        self._accepted_archive = accepted_archive
        self._seal_a = accepted_seal_a
        self._lock = MembershipLock.from_bytes(self._read("membership-lock.json"))
        self._seal = _json(self._read("seal-a.json"))

    def _read(self, name: str) -> bytes:
        path = self._root / name
        reject_reparse(path)
        if not path.is_file() or path.stat().st_size > 4 * 1024**2:
            raise ValueError("bounded regular retained artifact required")
        return path.read_bytes()

    def _material(self, decision: str) -> tuple[InputEvidence, tuple[tuple[str, bool], ...]]:
        if self._lock.split(decision.split("/")[0]) == "evaluation":
            raise PermissionError("evaluation targets cannot be exported")
        i = self._lock.decision_roster.index(decision)
        raw, target = self._read(f"input-{i}.json"), self._read(f"target-{i}.json")
        if (
            digest(raw) != self._seal["inputs"][decision]
            or digest(target) != self._seal["targets"][decision]
        ):
            raise ValueError("qualified payload changed after boundary inspection")
        return InputEvidence.from_bytes(
            raw, ModalityPermissionSet(allowed=REQUIRED)
        ), QualifiedTarget.from_bytes(target).truth

    def export(
        self,
        schedules: tuple[tuple[str, ...], tuple[str, ...]],
        initializations: tuple[int, int, int],
    ) -> tuple[TrainingMaterial, TrainingMaterial]:
        if (
            type(schedules) is not tuple
            or len(schedules) != 2
            or type(initializations) is not tuple
            or len(initializations) != 3
        ):
            raise ValueError("complete evaluator supplied committed schedule/init values required")
        for label, value in zip(INITIALIZATIONS, initializations, strict=True):
            if (
                type(value) is not int
                or not 0 <= value < 2**64
                or commitment(INIT_DOMAIN, value.to_bytes(8, "little"))
                != dict(self._lock.initialization_commitments)[label]
            ):
                raise ValueError("initialization commitment differs")
        development = tuple(
            lawful_example(*self._material(d))
            for g in self._lock.development
            for d in decisions(g.identity)
        )
        result = []
        for budget, schedule in zip((16, 64), schedules, strict=True):
            geometries = (
                self._lock.nested_train
                if budget == 16
                else tuple(g.identity for g in self._lock.train)
            )
            roster = tuple(d for g in geometries for d in decisions(g))
            schedule_root = digest(canonical_json_bytes(list(schedule)))
            if (
                type(schedule) is not tuple
                or len(schedule) != 16000
                or schedule_root != dict(self._lock.schedule_commitments)[budget]
                or set(schedule) != set(roster)
            ):
                raise ValueError("exact complete committed schedule required")
            counts = Counter(d.split("/")[0] for d in schedule)
            if set(counts.values()) != {16000 // budget}:
                raise ValueError("geometry balanced schedule required")
            for start in range(0, 16000, budget):
                if {d.split("/")[0] for d in schedule[start : start + budget]} != set(geometries):
                    raise ValueError("complete geometry cycle required")
            for g in geometries:
                sequence = tuple(d for d in schedule if d.startswith(g + "/"))
                first = sequence[:8]
                if set(first) != set(decisions(g)) or any(
                    d != first[i % 8] for i, d in enumerate(sequence)
                ):
                    raise ValueError("committed complete per-geometry decision cycles required")
            pairs = tuple(self._material(d) for d in roster)
            examples = tuple(lawful_example(*p) for p in pairs)
            data_root = digest(canonical_json_bytes([p[0].digest for p in pairs]))
            indexes = {d: i for i, d in enumerate(roster)}
            result.append(
                TrainingMaterial(
                    budget,
                    examples,
                    development,
                    tuple(indexes[d] for d in schedule),
                    initializations,
                    data_root,
                    schedule_root,
                    self._seal_a,
                    digest(canonical_json_bytes([list(p[1]) for p in pairs])),
                )
            )
        self._materials = (result[0], result[1])
        return self._materials

    def forecast_inputs(self) -> tuple[tuple[str, InputEvidence], ...]:
        result = []
        for g in self._lock.evaluation:
            for decision in decisions(g.identity):
                i = self._lock.decision_roster.index(decision)
                raw = self._read(f"input-{i}.json")
                if digest(raw) != self._seal["inputs"][decision]:
                    raise ValueError("qualified evaluation input changed")
                result.append(
                    (
                        decision,
                        InputEvidence.from_bytes(raw, ModalityPermissionSet(allowed=REQUIRED)),
                    )
                )
        return tuple(result)

    def control_rules(
        self,
    ) -> dict[tuple[str, int], tuple[str, dict[tuple[tuple[str, ...], bool], float]]]:
        inputs, targets = {}, {}
        for g in self._lock.train:
            for decision in decisions(g.identity):
                i = self._lock.decision_roster.index(decision)
                source, _ = self._material(decision)
                inputs[decision] = source
                targets[decision] = QualifiedTarget.from_bytes(self._read(f"target-{i}.json"))
        return {
            (c, b): _control_rule(c, b, self._lock, inputs, targets)
            for b in (16, 64)
            for c in ("persistence", "absent", "visible", "half", "train-frequency")
        }
