"""One fixed A1 alignment control; no target access or registration search."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from epsbench.data.loader import BeforeActionEcologicalView
from epsbench.diagnostics.a1_action_contrast import (
    DevelopmentTemplate,
    Forecast,
    _action_key,
    _snapshot,
    boundary_layout,
    boundary_risk,
    score_forecast,
)
from epsbench.schema import Action, BoundaryAxis, BoundaryKind


@dataclass(frozen=True)
class Alignment:
    selected_ordinal: int
    hamming_distance: int
    template_edges: int
    own_edges: int
    pairs: int
    unmatched_template: int
    unmatched_own: int
    support: str
    median_index: int | None
    shift: int
    nonzero_before: int
    retained: int
    clipped: int


def contours(view: BeforeActionEcologicalView) -> dict[tuple[int, str], list[int]]:
    groups: dict[tuple[int, str], list[int]] = {}
    for edge in view.boundaries:
        if (
            edge.frame_index == 0
            and edge.axis == BoundaryAxis.HORIZONTAL
            and (edge.kind == BoundaryKind.OCCLUDING_CONTOUR)
        ):
            group = groups.setdefault((edge.row, edge.owner_side.value), [])
            if edge.column in group:
                raise ValueError("duplicate contour")
            group.append(edge.column)
    for group in groups.values():
        group.sort()
    return groups


@dataclass(frozen=True)
class ControlledForecast:
    original: Forecast
    aligned: npt.NDArray[np.float64]
    wrong_action: npt.NDArray[np.float64]
    alignment: Alignment

    def __post_init__(self) -> None:
        for name in ("aligned", "wrong_action"):
            values = getattr(self, name)
            Forecast(values, self.original.boundary_scores)
            object.__setattr__(self, name, _snapshot(values.astype(np.float64)))


def controlled_forecast(
    own: BeforeActionEcologicalView, development: tuple[DevelopmentTemplate, ...]
) -> ControlledForecast:
    key = _action_key(own)
    layout = boundary_layout(own)
    eligible = [(i, d) for i, d in enumerate(development) if _action_key(d.before) == key]
    if not eligible:
        raise ValueError("no same-action development template")
    if any(d.before.segmentation.shape != own.segmentation.shape for _, d in eligible):
        raise ValueError("template raster dimensions differ")
    ordinal, selected = min(
        eligible, key=lambda item: int(np.count_nonzero(boundary_layout(item[1].before) != layout))
    )
    distance = int(np.count_nonzero(boundary_layout(selected.before) != layout))
    template, current = contours(selected.before), contours(own)
    shifts = sorted(
        c - t
        for group in sorted(template.keys() & current.keys())
        for t, c in zip(template[group], current[group], strict=False)
    )
    index = (len(shifts) - 1) // 2 if shifts else None
    shift = shifts[index] if index is not None else 0
    copied = selected.deletion_scores
    aligned = np.zeros_like(copied)
    width = copied.shape[1]
    if abs(shift) < width:
        if shift >= 0:
            aligned[:, shift:] = copied[:, : width - shift]
        else:
            aligned[:, : width + shift] = copied[:, -shift:]
    total = int(np.count_nonzero(copied))
    retained = int(np.count_nonzero(aligned))
    nt, nc = sum(map(len, template.values())), sum(map(len, current.values()))
    opposite = Action(
        name="lateral_left" if key[1] > 0 else "lateral_right",
        delta_lateral=-key[1],
        delta_forward=0.0,
        delta_yaw=0.0,
    )
    switched = BeforeActionEcologicalView(opposite, own.segmentation, own.surfaces, own.boundaries)
    return ControlledForecast(
        Forecast(boundary_risk(own), copied),
        aligned,
        boundary_risk(switched),
        Alignment(
            ordinal,
            distance,
            nt,
            nc,
            len(shifts),
            nt - len(shifts),
            nc - len(shifts),
            "SUPPORTED" if shifts else "NO_ALIGNMENT_SUPPORT",
            index,
            shift,
            total,
            retained,
            total - retained,
        ),
    )


def score_controls(
    prediction: ControlledForecast, codes: npt.NDArray[np.uint8]
) -> dict[str, object]:
    original = score_forecast(prediction.original, codes)
    aligned = score_forecast(Forecast(prediction.aligned, prediction.aligned), codes)
    wrong = score_forecast(Forecast(prediction.wrong_action, prediction.wrong_action), codes)
    correct_ap, wrong_ap = (
        original["boundary_average_precision"],
        wrong["boundary_average_precision"],
    )
    difference = prediction.original.boundary_scores - prediction.wrong_action
    return {
        **original,
        "aligned_average_precision": aligned["boundary_average_precision"],
        "wrong_action_average_precision": wrong_ap,
        "signed_action_ap_difference": correct_ap - wrong_ap
        if isinstance(correct_ap, float) and isinstance(wrong_ap, float)
        else None,
        "action_maps_equal": bool(np.array_equal(difference, np.zeros_like(difference))),
        "action_unequal_pixels": int(np.count_nonzero(difference)),
        "action_max_absolute_difference": float(np.max(np.abs(difference))),
    }
