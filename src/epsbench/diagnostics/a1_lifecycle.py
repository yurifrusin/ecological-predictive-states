"""Fixed-study cooperative chronology. No renderer or predictor target capability.

The journal directory is a protected evaluator dependency, never a run namespace.
It must be retained across attempts. This is not confinement of malicious Python.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

from epsbench.data.loader import BeforeActionEcologicalView
from epsbench.diagnostics.a1_action_contrast import (
    STUDY,
    DevelopmentTemplate,
    _action_key,
    cases,
)
from epsbench.diagnostics.a1_controls import ControlledForecast, controlled_forecast, score_controls
from epsbench.utils.canonical import logical_array_hash

ALGORITHM = "a1_rank_pair_lower_median_v1"


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def array_identity(values: npt.NDArray[np.generic]) -> dict[str, object]:
    return {
        "dtype": values.dtype.str,
        "shape": values.shape,
        "sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
    }


def view_identity(view: BeforeActionEcologicalView) -> str:
    return digest(
        {
            "action": view.action.model_dump(mode="json"),
            "segmentation": array_identity(view.segmentation),
            "surfaces": view.surfaces,
            "boundaries": [b.model_dump(mode="json") for b in view.boundaries],
        }
    )


@dataclass(frozen=True)
class Member:
    ordinal: int
    partition: str
    configuration_digest: str
    episode: str
    transition: str
    before_digest: str
    action_digest: str
    fate_digest: str
    provenance_digest: str

    def __post_init__(self) -> None:
        if not self.episode or not self.transition:
            raise ValueError("canonical episode and transition required")
        for name in (
            "configuration_digest",
            "before_digest",
            "action_digest",
            "fate_digest",
            "provenance_digest",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("canonical digest required")


@dataclass(frozen=True)
class Study:
    source_head: str
    source_tree: str
    members: tuple[Member, ...]
    addendum_digest: str
    algorithm: str = ALGORITHM

    def __post_init__(self) -> None:
        object.__setattr__(self, "members", tuple(self.members))
        if len(self.source_head) != 40 or len(self.source_tree) != 40:
            raise ValueError("immutable source head/tree required")
        if self.algorithm != ALGORITHM or len(self.addendum_digest) != 64:
            raise ValueError("fixed algorithm/addendum required")
        if tuple(m.ordinal for m in self.members) != tuple(range(8)):
            raise ValueError("complete ordered eight-member membership required")
        if tuple(m.partition for m in self.members) != ("development",) * 4 + ("held_out",) * 4:
            raise ValueError("fixed development/held-out partitions required")
        if (
            len({m.episode for m in self.members}) != 8
            or len({m.transition for m in self.members}) != 8
        ):
            raise ValueError("duplicate canonical member identity")

    @property
    def root(self) -> str:
        return digest(asdict(self))

    def validate_source_cases(self, source: Path) -> None:
        expected = cases(source)
        if any(
            m.configuration_digest != digest(c.config.model_dump(mode="json"))
            or m.action_digest != digest(c.config.action.model_dump(mode="json"))
            for m, c in zip(self.members, expected, strict=True)
        ):
            raise ValueError("fixed configuration/action membership differs")


@dataclass(frozen=True)
class Target:
    member: Member
    source_head: str
    source_tree: str
    membership_root: str
    codes: npt.NDArray[np.uint8]

    def __post_init__(self) -> None:
        if self.codes.dtype != np.uint8 or self.codes.shape != (120, 160):
            raise ValueError("canonical target shape/dtype differs")
        if np.any(self.codes > 5) or logical_array_hash(self.codes) != self.member.fate_digest:
            raise ValueError("canonical before-fate binding differs")
        object.__setattr__(
            self,
            "codes",
            np.frombuffer(self.codes.tobytes(), dtype=np.uint8).reshape(self.codes.shape),
        )


class BeforeReader(Protocol):
    def read_before(self, member: Member) -> BeforeActionEcologicalView: ...


class DevelopmentReader(Protocol):
    def read_development_target(self, member: Member) -> Target: ...


class HeldOutReader(Protocol):
    def read_held_out_target(self, member: Member) -> Target: ...


@dataclass(frozen=True)
class SealedMember:
    member: Member
    own_view_digest: str
    correct_action_digest: str
    switched_action_digest: str
    forecast: ControlledForecast


def forecast_identity(item: SealedMember) -> dict[str, object]:
    prediction = item.forecast
    return {
        "member": asdict(item.member),
        "own": item.own_view_digest,
        "correct_action": item.correct_action_digest,
        "switched_action": item.switched_action_digest,
        "original_rule": array_identity(prediction.original.boundary_scores),
        "unaligned": array_identity(prediction.original.template_scores),
        "aligned": array_identity(prediction.aligned),
        "wrong_action": array_identity(prediction.wrong_action),
        "diagnostics": asdict(prediction.alignment),
    }


@dataclass(frozen=True)
class Bundle:
    study: Study
    development_root: str
    forecasts: tuple[SealedMember, ...]
    seal: str

    def logical_root(self) -> str:
        return digest(
            {
                "scope": "own_before_action_only_v1",
                "study": asdict(self.study),
                "development_root": self.development_root,
                "forecasts": [forecast_identity(f) for f in self.forecasts],
            }
        )

    def validate(self) -> None:
        if tuple(f.member for f in self.forecasts) != self.study.members[4:]:
            raise ValueError("complete ordered four held-out forecasts required")
        if any(
            f.own_view_digest != f.member.before_digest
            or f.correct_action_digest != f.member.action_digest
            for f in self.forecasts
        ):
            raise ValueError("own view/action binding differs")
        if any(f.forecast.original.boundary_scores.shape != (120, 160) for f in self.forecasts):
            raise ValueError("fixed raster dimensions differ")
        if self.logical_root() != self.seal:
            raise ValueError("forecast seal changed")


def check_view(member: Member, view: BeforeActionEcologicalView) -> None:
    _action_key(view)
    if view.segmentation.shape != (120, 160) or view_identity(view) != member.before_digest:
        raise ValueError("own before identity/dimensions differ")
    if digest(view.action.model_dump(mode="json")) != member.action_digest:
        raise ValueError("actual action differs")


def check_target(study: Study, member: Member, target: Target) -> None:
    if (
        target.member != member
        or target.source_head != study.source_head
        or target.source_tree != study.source_tree
        or target.membership_root != study.root
    ):
        raise ValueError("target member/source/membership binding differs")
    if logical_array_hash(target.codes) != member.fate_digest:
        raise ValueError("target content changed")


class EvaluationJournal:
    """Durable whole-study exposure anchor; retain this directory across attempts.

    Construction is evaluator-only. A new output namespace never changes the fixed
    filename. Owner confirmation of no prior exposure remains a launch condition.
    """

    def __init__(self, protected_directory: Path) -> None:
        self.path = protected_directory / f"{STUDY}-target-exposure.json"

    def exposed_seal(self) -> str | None:
        if not self.path.exists():
            return None
        record = json.loads(self.path.read_text(encoding="utf-8"))
        if record["study"] != STUDY or record["whole_membership_exposed"] is not True:
            raise ValueError("invalid exposure history; release denied")
        return str(record["seal"])

    def deny_changed(self, seal: str | None = None) -> None:
        previous = self.exposed_seal()
        if previous is not None and previous != seal:
            raise ValueError("irreversible whole-membership target exposure")

    def retained_content(self, bundle: Bundle) -> str:
        maps = []
        for item in bundle.forecasts:
            f = item.forecast
            maps.append(
                [
                    base64.b64encode(a.tobytes(order="C")).decode("ascii")
                    for a in (
                        f.original.boundary_scores,
                        f.original.template_scores,
                        f.aligned,
                        f.wrong_action,
                    )
                ]
            )
        return json.dumps(
            {
                "seal": bundle.seal,
                "study": asdict(bundle.study),
                "development_root": bundle.development_root,
                "forecasts": [forecast_identity(f) for f in bundle.forecasts],
                "maps_float64_c_order": maps,
            },
            sort_keys=True,
            allow_nan=False,
        )

    def commit(self, bundle: Bundle) -> None:
        bundle.validate()
        self.deny_changed()
        destination = self.path.with_name(f"{STUDY}-bundle-{bundle.seal}.json")
        content = self.retained_content(bundle)
        descriptor, name = tempfile.mkstemp(prefix="a1-commit-", dir=self.path.parent)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        self.deny_changed()

    def verify_commit(self, bundle: Bundle) -> None:
        destination = self.path.with_name(f"{STUDY}-bundle-{bundle.seal}.json")
        if destination.read_text(encoding="utf-8") != self.retained_content(bundle):
            raise ValueError("retained complete bundle differs")
        with destination.open("r+b") as stream:
            os.fsync(stream.fileno())

    def retain_evaluation(self, bundle: Bundle, value: dict[str, object]) -> None:
        history = self.path.with_name(f"{STUDY}-evaluation-{bundle.seal}.jsonl")
        with history.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def expose(self, bundle: Bundle) -> None:
        bundle.validate()
        self.verify_commit(bundle)
        self.deny_changed(bundle.seal)
        if self.exposed_seal() is not None:
            # A prior failed fsync cannot be converted into release by a retry.
            with self.path.open("r+b") as stream:
                os.fsync(stream.fileno())
            return
        record = {
            "study": STUDY,
            "whole_membership_exposed": True,
            "membership_root": bundle.study.root,
            "seal": bundle.seal,
            "members": [asdict(m) for m in bundle.study.members[4:]],
        }
        # Exclusive creation prevents concurrent attempts from replacing history.
        # Any write/fsync failure leaves a conservative record and denies release.
        with self.path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False))
            stream.flush()
            os.fsync(stream.fileno())


def assemble(
    study: Study,
    source: Path,
    before_reader: BeforeReader,
    development_reader: DevelopmentReader,
    journal: EvaluationJournal,
) -> Bundle:
    """No held-out outcome capability is accepted by this assembly stage."""
    journal.deny_changed()
    study.validate_source_cases(source)
    templates: list[DevelopmentTemplate] = []
    template_bindings: list[object] = []
    for member in study.members[:4]:
        view = before_reader.read_before(member)
        check_view(member, view)
        target = development_reader.read_development_target(member)
        check_target(study, member, target)
        template = DevelopmentTemplate(view, (target.codes == 1).astype(np.float64))
        templates.append(template)
        template_bindings.append(
            {
                "member": asdict(member),
                "view": view_identity(view),
                "mask": array_identity(template.deletion_scores),
            }
        )
    fixed = tuple(templates)
    forecasts: list[SealedMember] = []
    for member in study.members[4:]:
        own = before_reader.read_before(member)
        check_view(member, own)
        # Calls receive no paths, member identities, reader or held-out collection.
        prediction = controlled_forecast(own, fixed)
        switched = own.action.model_dump(mode="json")
        switched.update(
            name="lateral_left" if own.action.delta_lateral > 0 else "lateral_right",
            delta_lateral=-own.action.delta_lateral,
        )
        forecasts.append(
            SealedMember(
                member, view_identity(own), member.action_digest, digest(switched), prediction
            )
        )
    unsealed = Bundle(study, digest(template_bindings), tuple(forecasts), "")
    committed = Bundle(
        study, unsealed.development_root, unsealed.forecasts, unsealed.logical_root()
    )
    committed.validate()
    journal.commit(committed)
    return committed


def evaluate(
    bundle: Bundle,
    journal: EvaluationJournal,
    reader: HeldOutReader,
) -> dict[str, object]:
    bundle.validate()
    journal.expose(bundle)  # durable whole-membership exposure precedes FIRST read
    results: list[dict[str, object]] = []
    for item in bundle.forecasts:
        bundle.validate()
        try:
            target = reader.read_held_out_target(item.member)
            check_target(bundle.study, item.member, target)
            result = score_controls(item.forecast, target.codes)
            journal.retain_evaluation(bundle, {"ordinal": item.member.ordinal, "result": result})
            results.append(result)
        except Exception as error:
            journal.retain_evaluation(
                bundle,
                {
                    "ordinal": item.member.ordinal,
                    "failure_type": type(error).__name__,
                    "status": "EVALUATION_FAILED",
                },
            )
            raise
    interpretable = all(r["status"] == "FINITE_CASE_ONLY" for r in results)

    def strict_comparison(field: str) -> bool:
        if not interpretable:
            return False
        for result in results:
            correct, comparator = result["boundary_average_precision"], result[field]
            if not isinstance(correct, float) or not isinstance(comparator, float):
                return False
            if correct <= comparator:
                return False
        return True

    return {
        "seal": bundle.seal,
        "cases": results,
        "coverage": {"members": 8, "held_out": 4},
        "original_primary_all_four": strict_comparison("template_average_precision"),
        "additional_alignment_all_four": strict_comparison("aligned_average_precision"),
        "action_specificity_all_four": strict_comparison("wrong_action_average_precision"),
        "phase_gate_effect": "NONE",
    }
