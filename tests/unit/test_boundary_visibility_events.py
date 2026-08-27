from __future__ import annotations

from collections.abc import Sequence

import mujoco
import numpy as np
import pytest

from epsbench.annotations import (
    AfterOriginCode,
    AnalyticCamera,
    BeforeFateCode,
    TransportReasonCode,
    classify_oriented_boundary_lattice,
    compute_analytic_transport,
    compute_raw_boundary_visibility_analysis,
    verify_attachment_contract,
)
from epsbench.annotations.boundary_events import (
    _derive_directional_events,
    _whole_surface_events,
)

IDENTITY_ROTATION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def _camera(x: float = 0.0) -> AnalyticCamera:
    return AnalyticCamera(
        world_position=(x, 0.0, 0.0),
        world_rotation_row_major=IDENTITY_ROTATION,
        vertical_field_of_view_degrees=60.0,
    )


def _foreground_background_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody>"
        '<geom name="background" type="plane" pos="0 0 -5" size="20 20 0.1"/>'
        '<geom name="foreground" type="box" pos="0 0 -2" size="0.35 0.8 0.8"/>'
        "</worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _counterfactual(
    first: Sequence[Sequence[int]],
    second: Sequence[Sequence[int]],
) -> dict[int, np.ndarray]:
    return {
        0: np.asarray(first, dtype=np.int32),
        1: np.asarray(second, dtype=np.int32),
    }


def test_finite_foreground_contour_is_owned_by_surface_revealing_background() -> None:
    model, data = _foreground_background_model()
    transport = compute_analytic_transport(
        model,
        data,
        (0, 1),
        80,
        60,
        _camera(),
        _camera(),
    )
    analysis = compute_raw_boundary_visibility_analysis(
        model,
        data,
        {"background": 0, "foreground": 1},
        (),
        80,
        60,
        _camera(),
        _camera(),
        transport,
    )
    occluding = [item for item in analysis.boundary_elements if item.kind == "occluding_contour"]
    assert occluding
    assert {item.owner_raw_geom_id for item in occluding} == {1}
    assert all(
        {item.negative_raw_geom_id, item.positive_raw_geom_id} == {0, 1} for item in occluding
    )


def test_compiled_panel_support_contact_is_attached_and_has_no_unilateral_owner() -> None:
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody>"
        '<geom name="support" type="plane" size="4 4 0.1"/>'
        '<geom name="panel" type="box" pos="0 0 0.5" size="0.5 0.1 0.5"/>'
        "</worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    evidence = verify_attachment_contract(
        model,
        data,
        {"support": 0, "panel": 1},
        (("support", "panel"),),
    )
    assert evidence.pair_evidence[0].observed_contact
    records = classify_oriented_boundary_lattice(
        0,
        np.asarray([[0, 1]], dtype=np.int32),
        _counterfactual([[-1, -1]], [[-1, -1]]),
        frozenset({("horizontal", 0, 0, frozenset((0, 1)))}),
    )
    assert records[0].kind == "attached_junction"
    assert records[0].owner_side == "none"
    assert records[0].owner_raw_geom_id is None


def test_local_attachment_edge_does_not_override_lateral_ownership_for_same_pair() -> None:
    pair = frozenset((0, 1))
    records = classify_oriented_boundary_lattice(
        0,
        np.asarray([[0, 1, 0]], dtype=np.int32),
        {
            0: np.full((1, 3), -1, dtype=np.int32),
            1: np.asarray([[-1, 0, -1]], dtype=np.int32),
        },
        frozenset({("horizontal", 0, 0, pair)}),
    )
    assert records[0].kind == "attached_junction"
    assert records[0].on_projected_attachment_locus
    assert records[1].kind == "occluding_contour"
    assert records[1].owner_raw_geom_id == 1
    assert not records[1].on_projected_attachment_locus


def test_two_perpendicular_axis_aligned_boxes_are_verified_as_attached() -> None:
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody>"
        '<geom name="floor" type="box" pos="0 1 -0.05" size="1 1 0.05"/>'
        '<geom name="wall" type="box" pos="-1 1 1" size="0.05 1 1"/>'
        "</worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    evidence = verify_attachment_contract(
        model,
        data,
        {"floor": 0, "wall": 1},
        (("floor", "wall"),),
    )
    assert evidence.attached_raw_pairs == frozenset((frozenset((0, 1)),))


@pytest.mark.parametrize("declared", [(), (("first", "second"),)])
def test_attachment_contract_rejects_undeclared_or_missing_contact(
    declared: tuple[tuple[str, str], ...],
) -> None:
    separation = 0.0 if not declared else 4.0
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody>"
        '<geom name="first" type="box" pos="0 0 0" size="1 1 1"/>'
        f'<geom name="second" type="box" pos="{2.0 + separation} 0 0" size="1 1 1"/>'
        "</worldbody></mujoco>"
    )
    with pytest.raises(ValueError, match="contact"):
        verify_attachment_contract(
            model,
            mujoco.MjData(model),
            {"first": 0, "second": 1},
            declared,
        )


def test_attachment_contract_binds_semantic_name_to_compiled_geom_name() -> None:
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom name="compiled" type="box" size="1 1 1"/></worldbody></mujoco>'
    )
    with pytest.raises(ValueError, match="name/raw-ID binding"):
        verify_attachment_contract(
            model,
            mujoco.MjData(model),
            {"fabricated": 0},
            (),
        )


@pytest.mark.parametrize(
    "geom_xml",
    [
        '<geom name="surface" type="sphere" size="1"/>',
        '<geom name="surface" type="box" euler="0 0 45" size="1 1 1"/>',
    ],
)
def test_attachment_contract_fails_closed_for_unsupported_geometry(geom_xml: str) -> None:
    model = mujoco.MjModel.from_xml_string(f"<mujoco><worldbody>{geom_xml}</worldbody></mujoco>")
    with pytest.raises(ValueError, match=r"supports only|axis-aligned"):
        verify_attachment_contract(
            model,
            mujoco.MjData(model),
            {"surface": 0},
            (),
        )


def test_controlled_surface_against_uncontrolled_space_owns_silhouette() -> None:
    records = classify_oriented_boundary_lattice(
        0,
        np.asarray([[0, -1]], dtype=np.int32),
        {0: np.asarray([[-1, -1]], dtype=np.int32)},
    )
    assert len(records) == 1
    assert records[0].kind == "controlled_silhouette"
    assert records[0].owner_side == "negative_axis_side"
    assert records[0].owner_raw_geom_id == 0


def test_local_three_surface_junction_is_ambiguous_and_unowned() -> None:
    assignment = np.asarray([[0, 1], [2, 1]], dtype=np.int32)
    counterfactual = {raw_id: np.full((2, 2), -1, dtype=np.int32) for raw_id in range(3)}
    records = classify_oriented_boundary_lattice(0, assignment, counterfactual)
    incident = next(item for item in records if item.axis == "horizontal" and item.row == 0)
    assert incident.kind == "multi_surface_junction_ambiguous"
    assert incident.owner_side == "none"


@pytest.mark.parametrize(
    ("first_next", "second_next"),
    [(-1, -1), (1, 0)],
)
def test_nonunique_counterfactual_continuation_is_unresolved_and_strictly_rejected(
    first_next: int,
    second_next: int,
) -> None:
    assignment = np.asarray([[0, 1]], dtype=np.int32)
    counterfactual = _counterfactual(
        [[first_next, -1]],
        [[-1, second_next]],
    )
    records = classify_oriented_boundary_lattice(0, assignment, counterfactual)
    assert records[0].kind == "unresolved_boundary"
    with pytest.raises(ValueError, match="unresolved"):
        classify_oriented_boundary_lattice(
            0,
            assignment,
            counterfactual,
            strict=True,
        )


def test_horizontal_and_vertical_owner_side_semantics_are_image_relative() -> None:
    horizontal = classify_oriented_boundary_lattice(
        0,
        np.asarray([[0, 1]], dtype=np.int32),
        _counterfactual([[1, -1]], [[-1, -1]]),
    )[0]
    vertical = classify_oriented_boundary_lattice(
        0,
        np.asarray([[1], [0]], dtype=np.int32),
        _counterfactual([[-1], [1]], [[-1], [-1]]),
    )[0]
    positive_horizontal = classify_oriented_boundary_lattice(
        0,
        np.asarray([[1, 0]], dtype=np.int32),
        _counterfactual([[-1, 1]], [[-1, -1]]),
    )[0]
    assert horizontal.axis == "horizontal"
    assert horizontal.owner_side == "negative_axis_side"
    assert vertical.axis == "vertical"
    assert vertical.owner_side == "positive_axis_side"
    assert positive_horizontal.owner_side == "positive_axis_side"


def test_static_camera_has_stable_interior_and_no_accretion_or_deletion() -> None:
    model, data = _foreground_background_model()
    transport = compute_analytic_transport(
        model,
        data,
        (0, 1),
        80,
        60,
        _camera(),
        _camera(),
    )
    analysis = compute_raw_boundary_visibility_analysis(
        model,
        data,
        {"background": 0, "foreground": 1},
        (),
        80,
        60,
        _camera(),
        _camera(),
        transport,
    )
    assert np.any(analysis.before_fate_codes == BeforeFateCode.STABLE_TRANSPORT)
    assert not np.any(analysis.before_fate_codes == BeforeFateCode.DELETION_AT_OCCLUDING_BOUNDARY)
    assert not np.any(
        analysis.after_origin_codes == AfterOriginCode.ACCRETION_AT_OCCLUDING_BOUNDARY
    )


def test_frame_exit_and_entry_are_not_deletion_or_accretion() -> None:
    reasons = np.asarray([[TransportReasonCode.TARGET_OUT_OF_FRAME]], dtype=np.uint8)
    source = np.asarray([[0]], dtype=np.int32)
    target = np.asarray([[-1]], dtype=np.int32)
    before, _, _ = _derive_directional_events(
        reasons,
        source,
        target,
        set(),
        True,
    )
    after, _, _ = _derive_directional_events(
        reasons,
        source,
        target,
        set(),
        False,
    )
    assert before[0, 0] == BeforeFateCode.FRAME_EXIT
    assert after[0, 0] == AfterOriginCode.FRAME_ENTRY


def test_only_actual_transport_boundary_band_is_event_ambiguous() -> None:
    reasons = np.asarray(
        [[TransportReasonCode.ANALYTIC_BOUNDARY_AMBIGUOUS, TransportReasonCode.OCCLUDED_AT_TARGET]],
        dtype=np.uint8,
    )
    codes, affected, owner = _derive_directional_events(
        reasons,
        np.asarray([[0, 0]], dtype=np.int32),
        np.asarray([[1, 1]], dtype=np.int32),
        {(1, 0)},
        True,
    )
    assert codes[0, 0] == BeforeFateCode.ANALYTIC_BOUNDARY_AMBIGUOUS
    assert codes[0, 1] == BeforeFateCode.DELETION_AT_OCCLUDING_BOUNDARY
    assert affected[0, 1] == 0
    assert owner[0, 1] == 1


def test_local_attached_seam_reason_cannot_become_causal_event() -> None:
    codes, affected, owner = _derive_directional_events(
        np.asarray([[TransportReasonCode.ANALYTIC_BOUNDARY_AMBIGUOUS]], dtype=np.uint8),
        np.asarray([[0]], dtype=np.int32),
        np.asarray([[1]], dtype=np.int32),
        {(1, 0)},
        True,
    )
    assert codes[0, 0] == BeforeFateCode.ANALYTIC_BOUNDARY_AMBIGUOUS
    assert affected[0, 0] == owner[0, 0] == -1


def test_whole_surface_appearance_and_disappearance_require_zero_crossing() -> None:
    before = np.asarray([[0, 1, 1]], dtype=np.int32)
    after = np.asarray([[0, 0, 2]], dtype=np.int32)
    events = _whole_surface_events((0, 1, 2, 3), before, after)
    by_id = {item.raw_geom_id: item for item in events}
    assert by_id[0].kind == "persistently_visible"
    assert by_id[1].kind == "disappearance"
    assert by_id[2].kind == "appearance"
    assert by_id[3].kind == "persistently_hidden_or_absent"


def test_unsupported_occluder_pair_remains_unresolved() -> None:
    reasons = np.asarray([[TransportReasonCode.OCCLUDED_AT_TARGET]], dtype=np.uint8)
    codes, affected, owner = _derive_directional_events(
        reasons,
        np.asarray([[0]], dtype=np.int32),
        np.asarray([[1]], dtype=np.int32),
        set(),
        True,
    )
    assert codes[0, 0] == BeforeFateCode.UNRESOLVED_OCCLUSION
    assert affected[0, 0] == owner[0, 0] == -1
    with pytest.raises(ValueError, match="unresolved"):
        _derive_directional_events(
            reasons,
            np.asarray([[0]], dtype=np.int32),
            np.asarray([[1]], dtype=np.int32),
            set(),
            True,
            strict=True,
        )
