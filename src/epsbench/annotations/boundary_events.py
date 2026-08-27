"""Analytic oriented boundaries and transport-causal ecological visibility events."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
from itertools import combinations
from typing import Final

import mujoco
import numpy as np
import numpy.typing as npt

from epsbench.annotations.optical_transport import (
    AnalyticCamera,
    AnalyticTransportArrays,
    TransportReasonCode,
    counterfactual_surface_assignments,
)

ORIENTED_BOUNDARY_METHOD: Final = "analytic_oriented_boundary_ownership_v1"
EDGE_LATTICE_CONVENTION: Final = "four_neighbour_sample_edge_lattice_v1"
BOUNDARY_KIND_DOMAIN: Final = "oriented_boundary_kind_domain_v1"
OWNER_SIDE_DOMAIN: Final = "oriented_boundary_owner_side_domain_v1"
ATTACHMENT_PUBLIC_CONTRACT_VERSION: Final = "scene_attachment_public_contract_v1"
ATTACHMENT_RULE: Final = "compiled_axis_aligned_plane_box_and_box_contact_v1"
ATTACHMENT_CONTACT_TOLERANCE: Final = 1e-12
ATTACHMENT_ROTATION_TOLERANCE: Final = 1e-12
COUNTERFACTUAL_CONTINUATION_RULE: Final = "counterfactual_nearest_surface_continuation_v1"
COUNTERFACTUAL_TIE_RULE: Final = "exactly_one_side_continues_v1"
JUNCTION_AMBIGUITY_RULE: Final = "edge_incident_3x2_or_2x3_multi_surface_v1"
SILHOUETTE_RULE: Final = "controlled_to_uncontrolled_side_owns_v1"
VISIBILITY_EVENT_METHOD: Final = "analytic_transport_boundary_causal_events_v1"
BEFORE_EVENT_CODE_DOMAIN: Final = "before_frame_fate_codes_v1"
AFTER_EVENT_CODE_DOMAIN: Final = "after_frame_origin_codes_v1"

Int32Array = npt.NDArray[np.int32]
UInt8Array = npt.NDArray[np.uint8]


class BeforeFateCode(IntEnum):
    STABLE_TRANSPORT = 0
    DELETION_AT_OCCLUDING_BOUNDARY = 1
    FRAME_EXIT = 2
    ANALYTIC_BOUNDARY_AMBIGUOUS = 3
    NO_CONTROLLED_SURFACE = 4
    UNRESOLVED_OCCLUSION = 5


class AfterOriginCode(IntEnum):
    STABLE_TRANSPORT = 0
    ACCRETION_AT_OCCLUDING_BOUNDARY = 1
    FRAME_ENTRY = 2
    ANALYTIC_BOUNDARY_AMBIGUOUS = 3
    NO_CONTROLLED_SURFACE = 4
    UNRESOLVED_OCCLUSION = 5


@dataclass(frozen=True)
class RawAttachmentPairEvidence:
    first_semantic_name: str
    second_semantic_name: str
    first_raw_geom_id: int
    second_raw_geom_id: int
    expected_attached: bool
    observed_contact: bool
    axis_interval_gaps: tuple[float, float, float]


@dataclass(frozen=True)
class RawAttachmentContractEvidence:
    method: str
    contact_tolerance: float
    rotation_tolerance: float
    geom_types: dict[str, str]
    geom_world_rotations_row_major: dict[str, tuple[float, ...]]
    pair_evidence: tuple[RawAttachmentPairEvidence, ...]
    attached_raw_pairs: frozenset[frozenset[int]]


@dataclass(frozen=True)
class RawBoundaryElement:
    frame_index: int
    axis: str
    row: int
    column: int
    negative_raw_geom_id: int | None
    positive_raw_geom_id: int | None
    kind: str
    owner_side: str
    owner_raw_geom_id: int | None
    negative_counterfactual_next_raw_geom_id: int | None
    positive_counterfactual_next_raw_geom_id: int | None


@dataclass(frozen=True)
class RawOccludingEventSummary:
    kind: str
    affected_raw_geom_id: int
    owner_raw_geom_id: int
    pixel_count: int


@dataclass(frozen=True)
class RawWholeSurfaceEvent:
    raw_geom_id: int
    kind: str
    before_visible_pixels: int
    after_visible_pixels: int


@dataclass(frozen=True)
class RawBoundaryVisibilityAnalysis:
    attachment_contract: RawAttachmentContractEvidence
    boundary_elements: tuple[RawBoundaryElement, ...]
    before_fate_codes: UInt8Array
    before_affected_raw_geom_ids: Int32Array
    before_owner_raw_geom_ids: Int32Array
    after_origin_codes: UInt8Array
    after_affected_raw_geom_ids: Int32Array
    after_owner_raw_geom_ids: Int32Array
    event_summaries: tuple[RawOccludingEventSummary, ...]
    whole_surface_events: tuple[RawWholeSurfaceEvent, ...]


def _geom_type_name(model: mujoco.MjModel, geom_id: int) -> str:
    geom_type = int(model.geom_type[geom_id])
    if geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
        return "plane"
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        return "box"
    raise ValueError(
        "attachment verification supports only controlled MuJoCo plane and box geoms; "
        f"geom {geom_id} has type {geom_type}"
    )


def _axis_aligned_half_extents(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], tuple[float, ...]]:
    rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    absolute = np.abs(rotation)
    nearest_axis = np.zeros((3, 3), dtype=np.float64)
    nearest_axis[np.arange(3), np.argmax(absolute, axis=1)] = np.sign(
        rotation[np.arange(3), np.argmax(absolute, axis=1)]
    )
    if len(set(np.argmax(absolute, axis=1).tolist())) != 3 or not np.allclose(
        rotation,
        nearest_axis,
        atol=ATTACHMENT_ROTATION_TOLERANCE,
        rtol=0.0,
    ):
        raise ValueError("attachment verification requires axis-aligned controlled geoms")
    local_half_extent = np.asarray(model.geom_size[geom_id], dtype=np.float64).copy()
    if _geom_type_name(model, geom_id) == "plane":
        local_half_extent[2] = 0.0
    world_half_extent = absolute @ local_half_extent
    position = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
    return position, world_half_extent, tuple(float(value) for value in rotation.reshape(-1))


def verify_attachment_contract(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    semantic_raw_geom_ids: dict[str, int],
    declared_attachment_pairs: tuple[tuple[str, str], ...],
) -> RawAttachmentContractEvidence:
    """Verify the complete declared/non-declared contact graph from compiled geometry."""

    if not semantic_raw_geom_ids:
        raise ValueError("attachment verification requires controlled apparatus surfaces")
    if len(set(semantic_raw_geom_ids.values())) != len(semantic_raw_geom_ids):
        raise ValueError("semantic apparatus names must map bijectively to raw geom IDs")
    mujoco.mj_forward(model, data)
    known_names = set(semantic_raw_geom_ids)
    declared = {frozenset((first, second)) for first, second in declared_attachment_pairs}
    if any(len(pair) != 2 or not pair.issubset(known_names) for pair in declared):
        raise ValueError("declared attachment pairs must name two known distinct surfaces")
    if len(declared) != len(declared_attachment_pairs):
        raise ValueError("declared attachment pairs must be unique")

    geom_types: dict[str, str] = {}
    rotations: dict[str, tuple[float, ...]] = {}
    bounds: dict[str, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = {}
    for name, raw_id in semantic_raw_geom_ids.items():
        if raw_id < 0 or raw_id >= model.ngeom:
            raise ValueError("semantic apparatus mapping contains an out-of-range raw geom ID")
        compiled_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, raw_id)
        if compiled_name != name:
            raise ValueError(
                "semantic apparatus name/raw-ID binding differs from compiled geometry"
            )
        geom_types[name] = _geom_type_name(model, raw_id)
        position, half_extent, rotation = _axis_aligned_half_extents(model, data, raw_id)
        rotations[name] = rotation
        bounds[name] = (position - half_extent, position + half_extent)

    evidence: list[RawAttachmentPairEvidence] = []
    attached_raw_pairs: set[frozenset[int]] = set()
    for first, second in combinations(sorted(semantic_raw_geom_ids), 2):
        first_min, first_max = bounds[first]
        second_min, second_max = bounds[second]
        gaps = np.maximum(first_min - second_max, second_min - first_max)
        observed_contact = bool(np.all(gaps <= ATTACHMENT_CONTACT_TOLERANCE))
        expected_attached = frozenset((first, second)) in declared
        if observed_contact != expected_attached:
            relation = "missing declared" if expected_attached else "undeclared"
            raise ValueError(f"attachment verification found {relation} contact: {first}/{second}")
        first_raw = semantic_raw_geom_ids[first]
        second_raw = semantic_raw_geom_ids[second]
        if observed_contact:
            attached_raw_pairs.add(frozenset((first_raw, second_raw)))
        evidence.append(
            RawAttachmentPairEvidence(
                first_semantic_name=first,
                second_semantic_name=second,
                first_raw_geom_id=first_raw,
                second_raw_geom_id=second_raw,
                expected_attached=expected_attached,
                observed_contact=observed_contact,
                axis_interval_gaps=tuple(float(value) for value in gaps),  # type: ignore[arg-type]
            )
        )
    return RawAttachmentContractEvidence(
        method=ATTACHMENT_RULE,
        contact_tolerance=ATTACHMENT_CONTACT_TOLERANCE,
        rotation_tolerance=ATTACHMENT_ROTATION_TOLERANCE,
        geom_types=geom_types,
        geom_world_rotations_row_major=rotations,
        pair_evidence=tuple(evidence),
        attached_raw_pairs=frozenset(attached_raw_pairs),
    )


def _local_controlled_assignments(
    assignment: Int32Array,
    axis: str,
    row: int,
    column: int,
) -> set[int]:
    height, width = assignment.shape
    if axis == "horizontal":
        local = assignment[max(0, row - 1) : min(height, row + 2), column : column + 2]
    else:
        local = assignment[row : row + 2, max(0, column - 1) : min(width, column + 2)]
    return {int(value) for value in np.unique(local) if int(value) >= 0}


def _classify_frame_boundaries(
    frame_index: int,
    assignment: Int32Array,
    counterfactual: dict[int, Int32Array],
    attached_pairs: frozenset[frozenset[int]],
) -> tuple[RawBoundaryElement, ...]:
    records: list[RawBoundaryElement] = []
    for axis, negative, positive in (
        ("horizontal", assignment[:, :-1], assignment[:, 1:]),
        ("vertical", assignment[:-1, :], assignment[1:, :]),
    ):
        for row_value, column_value in np.argwhere(negative != positive):
            row = int(row_value)
            column = int(column_value)
            negative_raw = int(negative[row, column])
            positive_raw = int(positive[row, column])
            negative_id = negative_raw if negative_raw >= 0 else None
            positive_id = positive_raw if positive_raw >= 0 else None
            owner_side = "none"
            owner_raw: int | None = None
            negative_next: int | None = None
            positive_next: int | None = None
            local_assignments = _local_controlled_assignments(
                assignment,
                axis,
                row,
                column,
            )
            if len(local_assignments) > 2:
                kind = "multi_surface_junction_ambiguous"
            elif negative_id is None or positive_id is None:
                kind = "controlled_silhouette"
                if negative_id is not None:
                    owner_side = "negative_axis_side"
                    owner_raw = negative_id
                else:
                    owner_side = "positive_axis_side"
                    owner_raw = positive_id
            elif frozenset((negative_id, positive_id)) in attached_pairs:
                kind = "attached_junction"
            else:
                negative_next_value = int(counterfactual[negative_id][row, column])
                if axis == "horizontal":
                    positive_row, positive_column = row, column + 1
                else:
                    positive_row, positive_column = row + 1, column
                positive_next_value = int(
                    counterfactual[positive_id][positive_row, positive_column]
                )
                negative_next = negative_next_value if negative_next_value >= 0 else None
                positive_next = positive_next_value if positive_next_value >= 0 else None
                negative_owns = negative_next_value == positive_id
                positive_owns = positive_next_value == negative_id
                if negative_owns ^ positive_owns:
                    kind = "occluding_contour"
                    if negative_owns:
                        owner_side = "negative_axis_side"
                        owner_raw = negative_id
                    else:
                        owner_side = "positive_axis_side"
                        owner_raw = positive_id
                else:
                    kind = "unresolved_boundary"
            records.append(
                RawBoundaryElement(
                    frame_index=frame_index,
                    axis=axis,
                    row=row,
                    column=column,
                    negative_raw_geom_id=negative_id,
                    positive_raw_geom_id=positive_id,
                    kind=kind,
                    owner_side=owner_side,
                    owner_raw_geom_id=owner_raw,
                    negative_counterfactual_next_raw_geom_id=negative_next,
                    positive_counterfactual_next_raw_geom_id=positive_next,
                )
            )
    return tuple(records)


def classify_oriented_boundary_lattice(
    frame_index: int,
    assignment: Int32Array,
    counterfactual: dict[int, Int32Array],
    attached_pairs: frozenset[frozenset[int]] = frozenset(),
    *,
    strict: bool = False,
) -> tuple[RawBoundaryElement, ...]:
    """Classify one analytic assignment lattice, optionally rejecting unresolved edges."""

    records = _classify_frame_boundaries(
        frame_index,
        assignment,
        counterfactual,
        attached_pairs,
    )
    unresolved = tuple(item for item in records if item.kind == "unresolved_boundary")
    if strict and unresolved:
        raise ValueError(f"boundary derivation left {len(unresolved)} unresolved edges")
    return records


def _supported_owner_pairs(
    boundary_elements: tuple[RawBoundaryElement, ...],
    frame_index: int,
) -> set[tuple[int, int]]:
    supported: set[tuple[int, int]] = set()
    for element in boundary_elements:
        if element.frame_index != frame_index or element.kind != "occluding_contour":
            continue
        owner = element.owner_raw_geom_id
        if owner is None:
            continue
        affected = (
            element.positive_raw_geom_id
            if owner == element.negative_raw_geom_id
            else element.negative_raw_geom_id
        )
        if affected is not None:
            supported.add((owner, affected))
    return supported


def _derive_directional_events(
    reasons: UInt8Array,
    source_assignment: Int32Array,
    target_hit_assignment: Int32Array,
    supported_owner_pairs: set[tuple[int, int]],
    attached_pairs: frozenset[frozenset[int]],
    forward: bool,
    *,
    strict: bool = False,
) -> tuple[UInt8Array, Int32Array, Int32Array]:
    stable_code = int(
        BeforeFateCode.STABLE_TRANSPORT if forward else AfterOriginCode.STABLE_TRANSPORT
    )
    occluding_code = int(
        BeforeFateCode.DELETION_AT_OCCLUDING_BOUNDARY
        if forward
        else AfterOriginCode.ACCRETION_AT_OCCLUDING_BOUNDARY
    )
    frame_code = int(BeforeFateCode.FRAME_EXIT if forward else AfterOriginCode.FRAME_ENTRY)
    ambiguous_code = int(
        BeforeFateCode.ANALYTIC_BOUNDARY_AMBIGUOUS
        if forward
        else AfterOriginCode.ANALYTIC_BOUNDARY_AMBIGUOUS
    )
    no_surface_code = int(
        BeforeFateCode.NO_CONTROLLED_SURFACE if forward else AfterOriginCode.NO_CONTROLLED_SURFACE
    )
    unresolved_code = int(
        BeforeFateCode.UNRESOLVED_OCCLUSION if forward else AfterOriginCode.UNRESOLVED_OCCLUSION
    )
    codes = np.full(reasons.shape, no_surface_code, dtype=np.uint8)
    affected = np.full(reasons.shape, -1, dtype=np.int32)
    owner = np.full(reasons.shape, -1, dtype=np.int32)
    codes[reasons == int(TransportReasonCode.VALID_TRANSPORT)] = stable_code
    codes[reasons == int(TransportReasonCode.TARGET_OUT_OF_FRAME)] = frame_code
    codes[reasons == int(TransportReasonCode.ANALYTIC_BOUNDARY_AMBIGUOUS)] = ambiguous_code
    occluded = reasons == int(TransportReasonCode.OCCLUDED_AT_TARGET)
    codes[occluded] = unresolved_code
    for pair in attached_pairs:
        first, second = tuple(pair)
        attached_ambiguity = occluded & (
            ((target_hit_assignment == first) & (source_assignment == second))
            | ((target_hit_assignment == second) & (source_assignment == first))
        )
        codes[attached_ambiguity] = ambiguous_code
    for owner_raw, affected_raw in sorted(supported_owner_pairs):
        supported = (
            occluded & (target_hit_assignment == owner_raw) & (source_assignment == affected_raw)
        )
        codes[supported] = occluding_code
        affected[supported] = np.int32(affected_raw)
        owner[supported] = np.int32(owner_raw)
    if strict and np.any(codes == unresolved_code):
        raise ValueError("visibility-event derivation left unresolved occlusion")
    return codes, affected, owner


def _event_summaries(
    before_codes: UInt8Array,
    before_affected: Int32Array,
    before_owner: Int32Array,
    after_codes: UInt8Array,
    after_affected: Int32Array,
    after_owner: Int32Array,
) -> tuple[RawOccludingEventSummary, ...]:
    counts: Counter[tuple[str, int, int]] = Counter()
    for kind, codes, event_code, affected, owner in (
        (
            "deletion",
            before_codes,
            int(BeforeFateCode.DELETION_AT_OCCLUDING_BOUNDARY),
            before_affected,
            before_owner,
        ),
        (
            "accretion",
            after_codes,
            int(AfterOriginCode.ACCRETION_AT_OCCLUDING_BOUNDARY),
            after_affected,
            after_owner,
        ),
    ):
        mask = codes == event_code
        pairs = np.stack((affected[mask], owner[mask]), axis=-1)
        for affected_raw, owner_raw in pairs.tolist():
            if affected_raw < 0 or owner_raw < 0:
                raise ValueError("causal visibility event lacks an owner/affected surface pair")
            counts[(kind, int(affected_raw), int(owner_raw))] += 1
    return tuple(
        RawOccludingEventSummary(
            kind=kind,
            affected_raw_geom_id=affected_raw,
            owner_raw_geom_id=owner_raw,
            pixel_count=count,
        )
        for (kind, affected_raw, owner_raw), count in sorted(counts.items())
    )


def _whole_surface_events(
    controlled_geom_ids: tuple[int, ...],
    before_assignment: Int32Array,
    after_assignment: Int32Array,
) -> tuple[RawWholeSurfaceEvent, ...]:
    records: list[RawWholeSurfaceEvent] = []
    for raw_id in sorted(controlled_geom_ids):
        before = int(np.count_nonzero(before_assignment == raw_id))
        after = int(np.count_nonzero(after_assignment == raw_id))
        if before == 0 and after > 0:
            kind = "appearance"
        elif before > 0 and after == 0:
            kind = "disappearance"
        elif before > 0 and after > 0:
            kind = "persistently_visible"
        else:
            kind = "persistently_hidden_or_absent"
        records.append(
            RawWholeSurfaceEvent(
                raw_geom_id=raw_id,
                kind=kind,
                before_visible_pixels=before,
                after_visible_pixels=after,
            )
        )
    return tuple(records)


def compute_raw_boundary_visibility_analysis(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    semantic_raw_geom_ids: dict[str, int],
    declared_attachment_pairs: tuple[tuple[str, str], ...],
    width: int,
    height: int,
    before_camera: AnalyticCamera,
    after_camera: AnalyticCamera,
    analytic_transport: AnalyticTransportArrays,
    *,
    strict: bool = True,
) -> RawBoundaryVisibilityAnalysis:
    """Derive the complete bounded Slice 4 oracle from compiled geometry and transport."""

    controlled_geom_ids = tuple(semantic_raw_geom_ids.values())
    attachment = verify_attachment_contract(
        model,
        data,
        semantic_raw_geom_ids,
        declared_attachment_pairs,
    )
    before_counterfactual = counterfactual_surface_assignments(
        model,
        data,
        controlled_geom_ids,
        width,
        height,
        before_camera,
    )
    after_counterfactual = counterfactual_surface_assignments(
        model,
        data,
        controlled_geom_ids,
        width,
        height,
        after_camera,
    )
    boundaries = classify_oriented_boundary_lattice(
        0,
        analytic_transport.before_surface_assignment,
        before_counterfactual,
        attachment.attached_raw_pairs,
    ) + classify_oriented_boundary_lattice(
        1,
        analytic_transport.after_surface_assignment,
        after_counterfactual,
        attachment.attached_raw_pairs,
    )
    unresolved_boundaries = tuple(
        element for element in boundaries if element.kind == "unresolved_boundary"
    )
    if strict and unresolved_boundaries:
        raise ValueError(
            f"canonical boundary derivation left {len(unresolved_boundaries)} unresolved edges"
        )
    forward_target = analytic_transport.forward.target_hit_assignment
    backward_target = analytic_transport.backward.target_hit_assignment
    if forward_target is None or backward_target is None:
        raise ValueError("visibility-event derivation requires privileged target-hit assignments")
    before_codes, before_affected, before_owner = _derive_directional_events(
        analytic_transport.forward.reasons,
        analytic_transport.before_surface_assignment,
        forward_target,
        _supported_owner_pairs(boundaries, 1),
        attachment.attached_raw_pairs,
        True,
    )
    after_codes, after_affected, after_owner = _derive_directional_events(
        analytic_transport.backward.reasons,
        analytic_transport.after_surface_assignment,
        backward_target,
        _supported_owner_pairs(boundaries, 0),
        attachment.attached_raw_pairs,
        False,
    )
    unresolved_events = int(
        np.count_nonzero(before_codes == int(BeforeFateCode.UNRESOLVED_OCCLUSION))
        + np.count_nonzero(after_codes == int(AfterOriginCode.UNRESOLVED_OCCLUSION))
    )
    if strict and unresolved_events:
        raise ValueError(
            f"canonical visibility-event derivation left {unresolved_events} unresolved pixels"
        )
    return RawBoundaryVisibilityAnalysis(
        attachment_contract=attachment,
        boundary_elements=boundaries,
        before_fate_codes=before_codes,
        before_affected_raw_geom_ids=before_affected,
        before_owner_raw_geom_ids=before_owner,
        after_origin_codes=after_codes,
        after_affected_raw_geom_ids=after_affected,
        after_owner_raw_geom_ids=after_owner,
        event_summaries=_event_summaries(
            before_codes,
            before_affected,
            before_owner,
            after_codes,
            after_affected,
            after_owner,
        ),
        whole_surface_events=_whole_surface_events(
            controlled_geom_ids,
            analytic_transport.before_surface_assignment,
            analytic_transport.after_surface_assignment,
        ),
    )
