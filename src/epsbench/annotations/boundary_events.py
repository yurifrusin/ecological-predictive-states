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
    RAY_DIRECTION_EPSILON,
    AnalyticCamera,
    AnalyticTransportArrays,
    TransportReasonCode,
    counterfactual_surface_assignments,
    focal_scales_from_vertical_fov,
    validate_analytic_camera,
)

ORIENTED_BOUNDARY_METHOD: Final = "analytic_oriented_boundary_ownership_v3"
EDGE_LATTICE_CONVENTION: Final = "four_neighbour_sample_edge_lattice_v1"
BOUNDARY_KIND_DOMAIN: Final = "oriented_boundary_kind_domain_v1"
OWNER_SIDE_DOMAIN: Final = "oriented_boundary_owner_side_domain_v1"
ATTACHMENT_PUBLIC_CONTRACT_VERSION: Final = "scene_attachment_public_contract_v3"
ATTACHMENT_RULE: Final = "projected_compiled_contact_locus_v2"
ATTACHMENT_CONTACT_MANIFOLD_RULE: Final = "compiled_axis_aligned_intersection_cell_v1"
ATTACHMENT_PROJECTION_CONVENTION: Final = "analytic_pinhole_pixel_centre_v1"
ATTACHMENT_PROJECTION_IN_FRONT_RULE: Final = "strict_forward_distance_greater_than_epsilon_v1"
ATTACHMENT_FEASIBILITY_RULE: Final = "per_constraint_inclusive_slack_except_strict_in_front_v1"
ATTACHMENT_EDGE_ASSOCIATION_RULE: Final = (
    "sample_connection_segment_intersects_projected_contact_cell_v1"
)
ATTACHMENT_ENDPOINT_TIE_RULE: Final = "inclusive_contact_endpoints_v1"
ATTACHMENT_MULTI_SURFACE_RULE: Final = "multi_surface_ambiguity_precedes_attachment_v1"
SUPPORTED_CONTACT_MANIFOLD_TYPES: Final = (
    "point",
    "axis_aligned_segment",
    "axis_aligned_rectangle",
    "axis_aligned_overlap_volume",
)
ATTACHMENT_CONTACT_TOLERANCE: Final = 1e-12
ATTACHMENT_FEASIBILITY_SLACK: Final = 1e-12
ATTACHMENT_ROTATION_TOLERANCE: Final = 1e-12
ATTACHMENT_IMAGE_TOLERANCE_PIXELS: Final = 0.5
COUNTERFACTUAL_CONTINUATION_RULE: Final = "counterfactual_nearest_surface_continuation_v1"
COUNTERFACTUAL_TIE_RULE: Final = "exactly_one_side_continues_v1"
JUNCTION_AMBIGUITY_RULE: Final = "edge_incident_3x2_or_2x3_multi_assignment_v2"
SILHOUETTE_RULE: Final = "controlled_to_uncontrolled_side_owns_v1"
VISIBILITY_EVENT_METHOD: Final = "analytic_transport_boundary_causal_events_v2"
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
    contact_manifold_type: str | None
    contact_world_min: tuple[float, float, float] | None
    contact_world_max: tuple[float, float, float] | None


@dataclass(frozen=True)
class RawAttachmentContractEvidence:
    method: str
    contact_manifold_rule: str
    supported_contact_manifold_types: tuple[str, ...]
    projection_convention: str
    projection_in_front_rule: str
    projection_in_front_epsilon: float
    feasibility_rule: str
    feasibility_slack: float
    edge_lattice_association_rule: str
    endpoint_tie_rule: str
    multi_surface_rule: str
    contact_tolerance: float
    rotation_tolerance: float
    image_tolerance_pixels: float
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
    on_projected_attachment_locus: bool


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
        contact_manifold_type: str | None = None
        contact_world_min: tuple[float, float, float] | None = None
        contact_world_max: tuple[float, float, float] | None = None
        if observed_contact:
            attached_raw_pairs.add(frozenset((first_raw, second_raw)))
            contact_min = np.maximum(first_min, second_min)
            contact_max = np.minimum(first_max, second_max)
            near_gap = contact_min > contact_max
            midpoint = (contact_min + contact_max) / 2.0
            contact_min = np.where(near_gap, midpoint, contact_min)
            contact_max = np.where(near_gap, midpoint, contact_max)
            manifold_dimensions = int(
                np.count_nonzero(contact_max - contact_min > ATTACHMENT_CONTACT_TOLERANCE)
            )
            contact_manifold_type = SUPPORTED_CONTACT_MANIFOLD_TYPES[manifold_dimensions]
            contact_world_min = tuple(float(value) for value in contact_min)  # type: ignore[assignment]
            contact_world_max = tuple(float(value) for value in contact_max)  # type: ignore[assignment]
        evidence.append(
            RawAttachmentPairEvidence(
                first_semantic_name=first,
                second_semantic_name=second,
                first_raw_geom_id=first_raw,
                second_raw_geom_id=second_raw,
                expected_attached=expected_attached,
                observed_contact=observed_contact,
                axis_interval_gaps=tuple(float(value) for value in gaps),  # type: ignore[arg-type]
                contact_manifold_type=contact_manifold_type,
                contact_world_min=contact_world_min,
                contact_world_max=contact_world_max,
            )
        )
    return RawAttachmentContractEvidence(
        method=ATTACHMENT_RULE,
        contact_manifold_rule=ATTACHMENT_CONTACT_MANIFOLD_RULE,
        supported_contact_manifold_types=SUPPORTED_CONTACT_MANIFOLD_TYPES,
        projection_convention=ATTACHMENT_PROJECTION_CONVENTION,
        projection_in_front_rule=ATTACHMENT_PROJECTION_IN_FRONT_RULE,
        projection_in_front_epsilon=RAY_DIRECTION_EPSILON,
        feasibility_rule=ATTACHMENT_FEASIBILITY_RULE,
        feasibility_slack=ATTACHMENT_FEASIBILITY_SLACK,
        edge_lattice_association_rule=ATTACHMENT_EDGE_ASSOCIATION_RULE,
        endpoint_tie_rule=ATTACHMENT_ENDPOINT_TIE_RULE,
        multi_surface_rule=ATTACHMENT_MULTI_SURFACE_RULE,
        contact_tolerance=ATTACHMENT_CONTACT_TOLERANCE,
        rotation_tolerance=ATTACHMENT_ROTATION_TOLERANCE,
        image_tolerance_pixels=ATTACHMENT_IMAGE_TOLERANCE_PIXELS,
        geom_types=geom_types,
        geom_world_rotations_row_major=rotations,
        pair_evidence=tuple(evidence),
        attached_raw_pairs=frozenset(attached_raw_pairs),
    )


LocalAttachmentEdge = tuple[str, int, int, frozenset[int]]


def _bounded_halfspace_feasible(
    coefficients: npt.NDArray[np.float64],
    limits: npt.NDArray[np.float64],
    feasibility_slacks: npt.NDArray[np.float64],
    dimensions: int,
) -> bool:
    """Decide a bounded binary64 linear-feasibility problem by its vertices."""

    if coefficients.ndim != 2 or coefficients.shape[1] != dimensions:
        raise ValueError("halfspace coefficients do not match the declared dimensions")
    if limits.shape != (coefficients.shape[0],) or feasibility_slacks.shape != limits.shape:
        raise ValueError("halfspace limits and feasibility slacks must match the constraints")
    if not (
        np.all(np.isfinite(coefficients))
        and np.all(np.isfinite(limits))
        and np.all(np.isfinite(feasibility_slacks))
        and np.all(feasibility_slacks >= 0.0)
    ):
        raise ValueError("halfspace constraints and feasibility slacks must be finite and valid")
    effective_limits = limits + feasibility_slacks
    if dimensions == 0:
        return bool(np.all(0.0 <= effective_limits))
    centre = np.full(dimensions, 0.5, dtype=np.float64)
    if np.all(coefficients @ centre <= effective_limits):
        return True
    for active in combinations(range(coefficients.shape[0]), dimensions):
        matrix = coefficients[np.asarray(active), :]
        if np.linalg.matrix_rank(matrix, tol=ATTACHMENT_CONTACT_TOLERANCE) != dimensions:
            continue
        try:
            candidate = np.linalg.solve(matrix, limits[np.asarray(active)])
        except np.linalg.LinAlgError:
            continue
        if np.all(coefficients @ candidate <= effective_limits):
            return True
    return False


def _contact_cell_projects_to_edge(
    pair: RawAttachmentPairEvidence,
    camera: AnalyticCamera,
    width: int,
    height: int,
    axis: str,
    row: int,
    column: int,
) -> bool:
    """Test exact projective intersection of a contact cell and one sample edge."""

    if pair.contact_world_min is None or pair.contact_world_max is None:
        raise ValueError("projected attachment requires a compiled contact manifold")
    lower = np.asarray(pair.contact_world_min, dtype=np.float64)
    upper = np.asarray(pair.contact_world_max, dtype=np.float64)
    spans = upper - lower
    varying_axes = np.flatnonzero(spans > ATTACHMENT_CONTACT_TOLERANCE)
    dimensions = int(varying_axes.size)
    if pair.contact_manifold_type != SUPPORTED_CONTACT_MANIFOLD_TYPES[dimensions]:
        raise ValueError("compiled contact manifold type does not match its metric extent")

    position = np.asarray(camera.world_position, dtype=np.float64)
    rotation = np.asarray(camera.world_rotation_row_major, dtype=np.float64).reshape(3, 3)
    camera_origin = np.asarray((lower - position) @ rotation, dtype=np.float64)
    camera_coefficients = np.zeros((3, dimensions), dtype=np.float64)
    for variable_index, world_axis in enumerate(varying_axes.tolist()):
        camera_coefficients[:, variable_index] = spans[world_axis] * rotation[world_axis, :]

    focal_x, focal_y = focal_scales_from_vertical_fov(
        width,
        height,
        camera.vertical_field_of_view_degrees,
    )
    image_tolerance = ATTACHMENT_IMAGE_TOLERANCE_PIXELS
    if axis == "horizontal":
        x_min, x_max = column + 0.5, column + 1.5
        y_min = row + 0.5 - image_tolerance
        y_max = row + 0.5 + image_tolerance
    elif axis == "vertical":
        x_min = column + 0.5 - image_tolerance
        x_max = column + 0.5 + image_tolerance
        y_min, y_max = row + 0.5, row + 1.5
    else:
        raise ValueError("attachment projection requires a known edge axis")
    u_min = (x_min - width / 2.0) / focal_x
    u_max = (x_max - width / 2.0) / focal_x
    v_min = (y_min - height / 2.0) / focal_y
    v_max = (y_max - height / 2.0) / focal_y

    linear_forms = (
        (
            np.asarray((0.0, 0.0, 1.0)),
            float(np.nextafter(-RAY_DIRECTION_EPSILON, -np.inf)),
            0.0,
        ),
        (np.asarray((-1.0, 0.0, -u_min)), 0.0, ATTACHMENT_FEASIBILITY_SLACK),
        (np.asarray((1.0, 0.0, u_max)), 0.0, ATTACHMENT_FEASIBILITY_SLACK),
        (np.asarray((0.0, 1.0, -v_min)), 0.0, ATTACHMENT_FEASIBILITY_SLACK),
        (np.asarray((0.0, -1.0, v_max)), 0.0, ATTACHMENT_FEASIBILITY_SLACK),
    )
    rows: list[npt.NDArray[np.float64]] = []
    limits: list[float] = []
    feasibility_slacks: list[float] = []
    for form, right_hand_side, feasibility_slack in linear_forms:
        rows.append(np.asarray(form @ camera_coefficients, dtype=np.float64))
        limits.append(float(right_hand_side - form @ camera_origin))
        feasibility_slacks.append(feasibility_slack)
    for index in range(dimensions):
        upper_bound = np.zeros(dimensions, dtype=np.float64)
        upper_bound[index] = 1.0
        rows.append(upper_bound)
        limits.append(1.0)
        feasibility_slacks.append(ATTACHMENT_FEASIBILITY_SLACK)
        rows.append(-upper_bound)
        limits.append(0.0)
        feasibility_slacks.append(ATTACHMENT_FEASIBILITY_SLACK)
    return _bounded_halfspace_feasible(
        np.asarray(rows, dtype=np.float64).reshape(len(rows), dimensions),
        np.asarray(limits, dtype=np.float64),
        np.asarray(feasibility_slacks, dtype=np.float64),
        dimensions,
    )


def projected_attachment_locus_edges(
    assignment: Int32Array,
    attachment: RawAttachmentContractEvidence,
    camera: AnalyticCamera,
    width: int,
    height: int,
) -> frozenset[LocalAttachmentEdge]:
    """Associate compiled contact manifolds with only their local image edges."""

    validate_analytic_camera(camera)
    focal_scales_from_vertical_fov(width, height, camera.vertical_field_of_view_degrees)
    if assignment.shape != (height, width):
        raise ValueError("attachment projection raster shape differs from assignment")
    pairs = {
        frozenset((item.first_raw_geom_id, item.second_raw_geom_id)): item
        for item in attachment.pair_evidence
        if item.observed_contact
    }
    edges: set[LocalAttachmentEdge] = set()
    for axis, negative, positive in (
        ("horizontal", assignment[:, :-1], assignment[:, 1:]),
        ("vertical", assignment[:-1, :], assignment[1:, :]),
    ):
        for row_value, column_value in np.argwhere(negative != positive):
            row = int(row_value)
            column = int(column_value)
            first = int(negative[row, column])
            second = int(positive[row, column])
            if first < 0 or second < 0:
                continue
            raw_pair = frozenset((first, second))
            evidence = pairs.get(raw_pair)
            if evidence is not None and _contact_cell_projects_to_edge(
                evidence,
                camera,
                width,
                height,
                axis,
                row,
                column,
            ):
                edges.add((axis, row, column, raw_pair))
    return frozenset(edges)


def _local_assignments(
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
    return {int(value) for value in np.unique(local)}


def _classify_frame_boundaries(
    frame_index: int,
    assignment: Int32Array,
    counterfactual: dict[int, Int32Array],
    projected_attachment_edges: frozenset[LocalAttachmentEdge],
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
            local_assignments = _local_assignments(
                assignment,
                axis,
                row,
                column,
            )
            raw_pair = frozenset(
                raw_id for raw_id in (negative_id, positive_id) if raw_id is not None
            )
            on_attachment_locus = (
                len(raw_pair) == 2 and (axis, row, column, raw_pair) in projected_attachment_edges
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
            elif on_attachment_locus:
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
                    on_projected_attachment_locus=on_attachment_locus,
                )
            )
    return tuple(records)


def classify_oriented_boundary_lattice(
    frame_index: int,
    assignment: Int32Array,
    counterfactual: dict[int, Int32Array],
    projected_attachment_edges: frozenset[LocalAttachmentEdge] = frozenset(),
    *,
    strict: bool = False,
) -> tuple[RawBoundaryElement, ...]:
    """Classify one analytic assignment lattice, optionally rejecting unresolved edges."""

    records = _classify_frame_boundaries(
        frame_index,
        assignment,
        counterfactual,
        projected_attachment_edges,
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
    before_attachment_edges = projected_attachment_locus_edges(
        analytic_transport.before_surface_assignment,
        attachment,
        before_camera,
        width,
        height,
    )
    after_attachment_edges = projected_attachment_locus_edges(
        analytic_transport.after_surface_assignment,
        attachment,
        after_camera,
        width,
        height,
    )
    boundaries = classify_oriented_boundary_lattice(
        0,
        analytic_transport.before_surface_assignment,
        before_counterfactual,
        before_attachment_edges,
    ) + classify_oriented_boundary_lattice(
        1,
        analytic_transport.after_surface_assignment,
        after_counterfactual,
        after_attachment_edges,
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
        True,
    )
    after_codes, after_affected, after_owner = _derive_directional_events(
        analytic_transport.backward.reasons,
        analytic_transport.after_surface_assignment,
        backward_target,
        _supported_owner_pairs(boundaries, 0),
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
