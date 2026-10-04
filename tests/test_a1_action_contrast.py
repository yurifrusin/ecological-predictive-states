from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from epsbench.config import SingleOccluderConfig
from epsbench.data.loader import BeforeActionEcologicalView, DatasetLoader, PermissionDeniedError
from epsbench.diagnostics.a1_action_contrast import (
    CaptureHeldError,
    DevelopmentTemplate,
    Forecast,
    average_precision,
    boundary_risk,
    call_cell_candidate,
    cases,
    forecast,
    plan,
    score_forecast,
)
from epsbench.schema import (
    Action,
    BoundaryAxis,
    BoundaryKind,
    BoundaryOwnerSide,
    DatasetManifest,
    Modality,
    ModalityPermissionSet,
    OrientedBoundaryElement,
    SurfaceReference,
    TransitionRecord,
)

FIRST = "surface-0000000000000001"
SECOND = "surface-0000000000000002"


def before(
    delta: float = 0.7, offset: int = 3, labels: tuple[int, int] = (17, 42)
) -> BeforeActionEcologicalView:
    image = np.full((5, 10), labels[0], dtype=np.int32)
    image[:, offset : offset + 3] = labels[1]
    edges = tuple(
        OrientedBoundaryElement(
            frame_index=0,
            axis=BoundaryAxis.HORIZONTAL,
            row=row,
            column=column,
            negative_surface_id=negative,
            positive_surface_id=positive,
            kind=BoundaryKind.OCCLUDING_CONTOUR,
            owner_side=owner,
            owner_surface_id=SECOND,
        )
        for row in range(5)
        for column, negative, positive, owner in (
            (offset - 1, FIRST, SECOND, BoundaryOwnerSide.POSITIVE_AXIS_SIDE),
            (offset + 2, SECOND, FIRST, BoundaryOwnerSide.NEGATIVE_AXIS_SIDE),
        )
    )
    return BeforeActionEcologicalView(
        Action(
            name="lateral_right" if delta > 0 else "lateral_left",
            delta_lateral=delta,
            delta_forward=0.0,
            delta_yaw=0.0,
        ),
        image,
        ((FIRST, labels[0]), (SECOND, labels[1])),
        edges,
    )


def test_plan_is_separate_eight_case_membership() -> None:
    source = Path(__file__).resolve().parents[1]
    values = cases(source)
    assert [v.partition for v in values] == ["development"] * 4 + ["held_out"] * 4
    assert [v.config.action.delta_lateral for v in values] == [-0.7, 0.7] * 4
    assert len(values) == 8
    for first, second in zip(values[::2], values[1::2], strict=True):
        assert first.config.camera.before_lateral == second.config.camera.before_lateral
        assert first.config.camera.forward == second.config.camera.forward
        assert first.config.appearance == second.config.appearance
    assert plan(source)["launch"] == "HELD"


def test_caller_cannot_invoke_a_producer_without_future_launch_change() -> None:
    def forbidden(
        config: SingleOccluderConfig,
        episodes: int,
        output: Path,
        *,
        component_topology: bool,
        capture_mode: str,
    ) -> DatasetManifest:
        pytest.fail("source candidate invoked a native producer")

    with pytest.raises(CaptureHeldError, match="hard aggregate"):
        call_cell_candidate(
            cases(Path(__file__).resolve().parents[1])[0], Path("unused"), forbidden
        )


def test_before_view_contains_only_current_optical_fields_and_immutable_pixels() -> None:
    view = before()
    assert {f.name for f in fields(view)} == {"action", "segmentation", "surfaces", "boundaries"}
    with pytest.raises(ValueError):
        view.segmentation.setflags(write=True)
    future = view.boundaries[0].model_copy(update={"frame_index": 1})
    with pytest.raises(ValueError, match="future-frame"):
        BeforeActionEcologicalView(view.action, view.segmentation, view.surfaces, (future,))
    with pytest.raises(ValueError, match="exactly visible"):
        BeforeActionEcologicalView(view.action, view.segmentation, view.surfaces[:-1], ())


@pytest.mark.parametrize(
    "missing",
    [Modality.EXECUTED_ACTION, Modality.SURFACE_REGIONS, Modality.ORIENTED_BOUNDARY_OWNERSHIP],
)
def test_before_permission_denial_precedes_transition_access(
    missing: Modality, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader = object.__new__(DatasetLoader)
    required = {
        Modality.EXECUTED_ACTION,
        Modality.SURFACE_REGIONS,
        Modality.ORIENTED_BOUNDARY_OWNERSHIP,
    }
    loader.permissions = ModalityPermissionSet(allowed=frozenset(required - {missing}))

    def forbidden(*args: object) -> None:
        pytest.fail("permission denial accessed a transition")

    monkeypatch.setattr(loader, "_transition", forbidden)
    with pytest.raises(PermissionDeniedError):
        loader.read_before_action(999)


def test_action_flip_changes_risk_side_and_opaque_relabeling_changes_nothing() -> None:
    right, left = boundary_risk(before()), boundary_risk(before(-0.7))
    assert np.argmax(right[0]) == 2
    assert np.argmax(left[0]) == 6
    assert not np.array_equal(right, left)
    assert np.array_equal(right, boundary_risk(before(labels=(999, 3))))


def test_loader_projects_only_before_fields_and_visible_surface_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = before()
    third = "surface-0000000000000003"
    future = view.boundaries[0].model_copy(update={"frame_index": 1})
    transition = cast(
        TransitionRecord,
        SimpleNamespace(
            action=view.action,
            before=SimpleNamespace(segmentation=object()),
            surfaces=tuple(
                SurfaceReference(surface_id=identifier, segmentation_label=label)
                for identifier, label in (*view.surfaces, (third, 777))
            ),
            oriented_boundary_ownership=SimpleNamespace(elements=(*view.boundaries, future)),
        ),
    )
    loader = object.__new__(DatasetLoader)
    loader.permissions = ModalityPermissionSet.ecological_only()
    monkeypatch.setattr(loader, "_transition", lambda _: transition)
    monkeypatch.setattr(loader, "_load_npy", lambda _: view.segmentation.copy())
    projected = loader.read_before_action(0)
    assert projected.surfaces == view.surfaces
    assert projected.boundaries == view.boundaries
    assert np.array_equal(projected.segmentation, view.segmentation)
    with pytest.raises(ValueError):
        projected.segmentation.setflags(write=True)


def test_before_view_rejects_out_of_image_boundaries() -> None:
    view = before()
    edge = view.boundaries[0].model_copy(update={"column": 10})
    with pytest.raises(ValueError, match="edge lattice"):
        BeforeActionEcologicalView(view.action, view.segmentation, view.surfaces, (edge,))


def test_unseen_boundary_shift_can_beat_template_copy_in_closed_form() -> None:
    development = before(offset=3)
    target = np.zeros((5, 10), dtype=np.float64)
    target[:, 2] = 1
    templates = (DevelopmentTemplate(development, target),)
    held_out = before(offset=4)
    prediction = forecast(held_out, templates)
    codes = np.zeros((5, 10), dtype=np.uint8)
    codes[:, 3] = 1
    result = score_forecast(prediction, codes)
    assert result["boundary_average_precision"] == 1.0
    assert result["template_average_precision"] == 0.1
    with pytest.raises(ValueError):
        prediction.boundary_scores.setflags(write=True)


def test_scoring_reports_ambiguity_exit_and_inconclusive_negative_results() -> None:
    values = np.zeros((2, 3), dtype=np.float64)
    prediction = Forecast(values, values)
    codes = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
    result = score_forecast(prediction, codes)
    assert result["before_fate_code_counts"] == [1] * 6
    assert result["status"] == "INCONCLUSIVE"
    empty = score_forecast(prediction, np.zeros((2, 3), dtype=np.uint8))
    assert empty["boundary_average_precision"] is None
    assert empty["status"] == "INCONCLUSIVE"


def test_average_precision_groups_ties_without_label_dependent_order() -> None:
    scores = np.array([1.0, 1.0, 0.0], dtype=np.float64)
    for labels in ([True, False, True], [False, True, True]):
        assert average_precision(scores, np.array(labels, dtype=np.bool_)) == pytest.approx(7 / 12)


def test_forecast_rejects_nonfinite_scores_and_unknown_target_codes() -> None:
    with pytest.raises(ValueError, match="finite"):
        Forecast(np.array([[np.nan]], dtype=np.float64), np.zeros((1, 1), dtype=np.float64))
    prediction = Forecast(np.zeros((2, 3), dtype=np.float64), np.zeros((2, 3), dtype=np.float64))
    with pytest.raises(ValueError, match="unknown"):
        score_forecast(prediction, np.full((2, 3), 6, dtype=np.uint8))
