"""Prospective, untrained A1 caller/evaluator; native launch remains held.

Predictors receive only BeforeActionEcologicalView, never paths or target records.
No target-dependent fitting, renderer import, runtime selection or supervision here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

from epsbench.config import SingleOccluderConfig, load_config
from epsbench.data.loader import BeforeActionEcologicalView
from epsbench.schema import BoundaryAxis, BoundaryKind, BoundaryOwnerSide, DatasetManifest

STUDY = "a1_contrasting_action_v1"


class CaptureHeldError(RuntimeError):
    """Specific launch authority, qualification and hard caps remain unresolved."""


@dataclass(frozen=True)
class Case:
    ordinal: int
    partition: str
    config: SingleOccluderConfig


def cases(source: Path) -> tuple[Case, ...]:
    base = load_config(source / "configs/benchmark_v0.yaml")
    if not isinstance(base, SingleOccluderConfig):
        raise ValueError("A1 requires the existing single-occluder configuration")
    if (
        base.seed != 1729
        or (base.render.width, base.render.height) != (160, 120)
        or base.appearance.profile_id != "legacy_solid_base_v1"
        or base.camera.height != 1.25
        or base.camera.field_of_view_degrees != 55.0
    ):
        raise ValueError("A1 base inputs differ from the prospective concept")
    result: list[Case] = []
    for forward, partition in ((-3.0, "development"), (-2.5, "held_out")):
        for lateral in (-0.35, 0.35):
            for delta in (-0.7, 0.7):
                payload = base.model_dump(mode="python")
                payload["camera"].update(
                    before_lateral=lateral, after_lateral=lateral + delta, forward=forward
                )
                payload["action"].update(
                    name="lateral_left" if delta < 0 else "lateral_right", delta_lateral=delta
                )
                result.append(
                    Case(len(result), partition, SingleOccluderConfig.model_validate(payload))
                )
    return tuple(result)


def plan(source: Path) -> dict[str, object]:
    return {
        "study": STUDY,
        "review_profile": "DUAL_REVIEW",
        "evidence_class": "PUBLIC_REPOSITORY_ONLY",
        "phase_gate_effect": "NONE",
        "launch": "HELD",
        "cases": [
            {"ordinal": c.ordinal, "partition": c.partition, "config": c.config.model_dump()}
            for c in cases(source)
        ],
        "limits": {
            "contexts": 8,
            "native_render_read_pairs": 48,
            "cpu_threads": 2,
            "llvmpipe_threads": 2,
            "memory_bytes": 8 * 1024**3,
            "output_bytes": 1024**3,
            "cell_seconds": 300,
            "total_seconds": 2700,
        },
        "hard_aggregate_caps": "UNRESOLVED; RSS/output polling is not a hard cap",
        "runtime_split": "existing 3.14.4 controller / pinned 3.11.15 producer; unqualified",
    }


class Producer(Protocol):
    def __call__(
        self,
        config: SingleOccluderConfig,
        episodes: int,
        output: Path,
        *,
        component_topology: bool,
        capture_mode: str,
    ) -> DatasetManifest: ...


def call_cell_candidate(case: Case, output: Path, producer: Producer) -> DatasetManifest:
    """Dependency-injected native caller specification, with launch disabled.

    No caller can turn a boolean or path into launch authority. This candidate
    deliberately stops before invoking even an injected producer. A later exact-
    head reviewed launch change must resolve hard caps and explicit authorization.
    """
    raise CaptureHeldError(
        "A1 capture held: exact-head reviews, specific launch authority, target qualification "
        "and hard aggregate memory/output caps remain unresolved"
    )


def _snapshot(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.frombuffer(values.tobytes(order="C"), dtype=np.float64).reshape(values.shape)


def _action_key(view: BeforeActionEcologicalView) -> tuple[str, float]:
    action = view.action
    if action.name not in {"lateral_left", "lateral_right"} or abs(action.delta_lateral) != 0.7:
        raise ValueError("A1 forecasts require the fixed left/right 0.7 action")
    return action.name, action.delta_lateral


def boundary_risk(view: BeforeActionEcologicalView) -> npt.NDArray[np.float64]:
    """Rank deletion risk, without claiming calibrated probabilities or band width.

    For the fixed camera axes, rightward motion moves nearer owners left relative
    to nonowners. The mirrored rule applies to leftward motion. Only before-frame
    horizontal occluding contours contribute; inverse pixel distance breaks ranks.
    """
    _, delta = _action_key(view)
    scores: npt.NDArray[np.float64] = np.zeros(view.segmentation.shape, dtype=np.float64)
    labels = dict(view.surfaces)
    columns = np.arange(scores.shape[1], dtype=np.float64)
    for edge in view.boundaries:
        if edge.axis != BoundaryAxis.HORIZONTAL or edge.kind != BoundaryKind.OCCLUDING_CONTOUR:
            continue
        expected = (
            BoundaryOwnerSide.POSITIVE_AXIS_SIDE
            if delta > 0
            else BoundaryOwnerSide.NEGATIVE_AXIS_SIDE
        )
        if edge.owner_side != expected:
            continue
        affected = edge.negative_surface_id if delta > 0 else edge.positive_surface_id
        if affected is None:
            raise ValueError("occluding contour lacks its affected surface")
        side = columns <= edge.column if delta > 0 else columns > edge.column
        eligible = (view.segmentation[edge.row] == labels[affected]) & side
        distance = np.abs(columns + 0.5 - (edge.column + 1.0))
        contribution = np.where(eligible, 1.0 / (1.0 + distance), 0.0)
        scores[edge.row] = np.maximum(scores[edge.row], contribution)
    return _snapshot(scores)


def boundary_layout(view: BeforeActionEcologicalView) -> npt.NDArray[np.uint8]:
    """ID-invariant current owned-boundary descriptor for template retrieval."""
    layout = np.zeros(view.segmentation.shape, dtype=np.uint8)
    for edge in view.boundaries:
        if edge.axis == BoundaryAxis.HORIZONTAL and edge.kind == BoundaryKind.OCCLUDING_CONTOUR:
            layout[edge.row, edge.column] = (
                1 if edge.owner_side == BoundaryOwnerSide.NEGATIVE_AXIS_SIDE else 2
            )
    return layout


@dataclass(frozen=True)
class DevelopmentTemplate:
    before: BeforeActionEcologicalView
    deletion_scores: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        values = self.deletion_scores
        if values.shape != self.before.segmentation.shape or not np.all(
            (values == 0) | (values == 1)
        ):
            raise ValueError("development template must be a same-shape binary deletion map")
        object.__setattr__(self, "deletion_scores", _snapshot(values.astype(np.float64)))


def template_copy(
    view: BeforeActionEcologicalView, development: tuple[DevelopmentTemplate, ...]
) -> npt.NDArray[np.float64]:
    key = _action_key(view)
    eligible = [d for d in development if _action_key(d.before) == key]
    if not eligible:
        raise ValueError("no development template for the commanded action")
    layout = boundary_layout(view)
    if any(d.before.segmentation.shape != view.segmentation.shape for d in eligible):
        raise ValueError("template raster dimensions differ")
    # Equal distances retain prospective development order, never target-derived ties.
    nearest = min(
        eligible, key=lambda d: int(np.count_nonzero(boundary_layout(d.before) != layout))
    )
    return _snapshot(nearest.deletion_scores)


@dataclass(frozen=True)
class Forecast:
    boundary_scores: npt.NDArray[np.float64]
    template_scores: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.boundary_scores.shape != self.template_scores.shape:
            raise ValueError("forecast rasters differ")
        for name in ("boundary_scores", "template_scores"):
            values = getattr(self, name)
            if values.ndim != 2 or not np.all(np.isfinite(values) & (values >= 0) & (values <= 1)):
                raise ValueError("forecast scores must be finite two-dimensional ranks in [0,1]")
            object.__setattr__(self, name, _snapshot(values.astype(np.float64)))


def forecast(
    view: BeforeActionEcologicalView, development: tuple[DevelopmentTemplate, ...]
) -> Forecast:
    return Forecast(boundary_risk(view), template_copy(view, development))


def average_precision(
    scores: npt.NDArray[np.float64], truth: npt.NDArray[np.bool_]
) -> float | None:
    """Grouped-threshold AP: tied pixels enter together; no favorable tie order."""
    positives = int(np.count_nonzero(truth))
    if not positives:
        return None
    order = np.argsort(-scores, kind="stable")
    ranked, labels = scores[order], truth[order]
    ends = np.r_[np.flatnonzero(ranked[:-1] != ranked[1:]), len(ranked) - 1]
    true_positive = np.cumsum(labels, dtype=np.int64)[ends]
    recall = true_positive / positives
    precision = true_positive / (ends + 1)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def score_forecast(
    prediction: Forecast, before_fate_codes: npt.NDArray[np.uint8]
) -> dict[str, object]:
    """Separate evaluation stage; preserves every canonical category count."""
    codes = before_fate_codes
    if codes.dtype != np.uint8 or codes.shape != prediction.boundary_scores.shape:
        raise ValueError("target must be a same-shape canonical uint8 before-fate map")
    if np.any(codes > 5):
        raise ValueError("unknown canonical before-fate event code")
    counts = np.bincount(codes.ravel(), minlength=6)
    eligible = (codes == 0) | (codes == 1)
    truth = codes[eligible] == 1
    return {
        "before_fate_code_counts": [int(count) for count in counts],
        "scored_domain": "stable_transport_or_occluding_deletion",
        "excluded_categories_reported": [
            "frame_exit",
            "boundary_ambiguous",
            "no_controlled_surface",
            "unresolved_occlusion",
        ],
        "boundary_average_precision": average_precision(
            prediction.boundary_scores[eligible], truth
        ),
        "template_average_precision": average_precision(
            prediction.template_scores[eligible], truth
        ),
        "status": "INCONCLUSIVE" if counts[1] == 0 or counts[5] else "FINITE_CASE_ONLY",
    }
