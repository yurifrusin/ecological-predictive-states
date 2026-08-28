"""Whole-dataset schema, alignment, identity, and artifact validation."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from epsbench.annotations import (
    ANALYTIC_TRANSPORT_METHOD,
    ATTACHMENT_CONTACT_MANIFOLD_RULE,
    ATTACHMENT_EDGE_ASSOCIATION_RULE,
    ATTACHMENT_ENDPOINT_TIE_RULE,
    ATTACHMENT_FEASIBILITY_RULE,
    ATTACHMENT_MULTI_SURFACE_RULE,
    ATTACHMENT_PROJECTION_CONVENTION,
    ATTACHMENT_PROJECTION_IN_FRONT_RULE,
    ATTACHMENT_PUBLIC_CONTRACT_VERSION,
    ATTACHMENT_RULE,
    BOUNDARY_KIND_DOMAIN,
    COUNTERFACTUAL_CONTINUATION_RULE,
    COUNTERFACTUAL_TIE_RULE,
    EDGE_LATTICE_CONVENTION,
    JUNCTION_AMBIGUITY_RULE,
    ORIENTED_BOUNDARY_METHOD,
    OWNER_SIDE_DOMAIN,
    RAY_DIRECTION_EPSILON,
    SILHOUETTE_RULE,
    SUPPORTED_CONTACT_MANIFOLD_TYPES,
    AfterOriginCode,
    AnalyticTransportArrays,
    BeforeFateCode,
    DirectionalTransportArrays,
    RawBoundaryVisibilityAnalysis,
    TransportReasonCode,
    derive_boundary_structure,
    derive_visibility,
)
from epsbench.appearance import (
    AppearanceRegistry,
    EvaluationSeedRegistry,
    appearance_profile_hash,
    appearance_registry_hash,
    profile_by_id,
    seed_registry_hash,
    validate_appearance_instance,
    validate_axis_isolation,
)
from epsbench.config import (
    BenchmarkConfig,
    CorridorConfig,
    SingleOccluderConfig,
    parse_config,
)
from epsbench.data.decoding import (
    ArtifactDecodeError,
    decode_json_artifact,
    decode_npy_artifact,
    decode_rgb_artifact,
)
from epsbench.data.identity import (
    compute_analytic_transport_hash,
    compute_boundary_numerical_contract_hash,
    compute_content_provenance_binding,
    compute_dataset_logical_hash,
    compute_ecological_label_hash,
    compute_oriented_boundary_hash,
    compute_renderer_execution_provenance_hash,
    compute_source_provenance_hash,
    compute_visibility_event_hash,
)
from epsbench.data.paths import (
    OwnedRegularFile,
    UnsafeDatasetManifestError,
    UnsafeOwnedFileError,
    open_dataset_manifest,
    open_owned_regular_file,
    sha256_open_file,
)
from epsbench.schema import (
    Action,
    AnalyticRendererFrameDiagnostic,
    AnalyticTransportDiagnostics,
    ArtifactRecord,
    AvailableDenseOpticalTransport,
    AvailableOcclusionAnnotation,
    AvailableOrientedBoundaryOwnership,
    BoundaryAxis,
    BoundaryKind,
    BoundaryOwnerSide,
    BoundaryVisibilityDiagnostics,
    CameraInstrumentation,
    CorridorInstrumentation,
    DatasetManifest,
    DirectionalOpticalTransport,
    DirectionalTransportDiagnostic,
    DirectionalVisibilityEventMap,
    EdgeLatticeCoordinateConvention,
    OccludingVisibilityEventSummary,
    OcclusionRelation,
    OrientedBoundaryElement,
    PrivilegedAttachmentContractEvidence,
    PrivilegedAttachmentPairEvidence,
    PrivilegedBoundaryElementEvidence,
    PrivilegedInstrumentation,
    SingleOccluderInstrumentation,
    TransitionRecord,
    WholeSurfaceEventKind,
    WholeSurfaceVisibilityEvent,
    parse_privileged_instrumentation_json,
)
from epsbench.sim import (
    CORRIDOR_SURFACE_NAMES,
    compile_corridor_scene_contract,
    compile_single_occluder_scene_contract,
    compute_corridor_analytic_transport,
    compute_corridor_boundary_visibility,
    compute_single_occluder_analytic_transport,
    compute_single_occluder_boundary_visibility,
    corridor_generation_seeds,
)
from epsbench.sim.compiled import CompiledSceneContract
from epsbench.utils.canonical import (
    canonical_json_bytes,
    logical_array_hash,
    sha256_bytes,
)
from epsbench.utils.seeding import derive_seed


class DatasetValidationError(ValueError):
    """Raised when a dataset fails the public data contract."""


class _ArtifactRegistry:
    """Reject duplicate logical paths and filesystem aliases across declared roles."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.paths: set[str] = set()
        self.resolved_paths: set[Path] = set()
        self.file_identities: set[tuple[int, int]] = set()

    @contextmanager
    def claim(self, record: ArtifactRecord) -> Iterator[OwnedRegularFile]:
        if record.path in self.paths:
            raise DatasetValidationError(f"duplicate artifact path: {record.path}")
        try:
            owned_context = open_owned_regular_file(self.root, record.path)
            with owned_context as owned:
                if owned.path in self.resolved_paths:
                    raise DatasetValidationError(
                        f"artifact path aliases another role: {record.path}"
                    )
                identity = (owned.device, owned.inode)
                if identity in self.file_identities:
                    raise DatasetValidationError(
                        f"artifact path aliases another role: {record.path}"
                    )
                if owned.byte_count != record.byte_count:
                    raise DatasetValidationError(f"artifact byte count mismatch: {record.path}")
                if sha256_open_file(owned) != record.file_sha256:
                    raise DatasetValidationError(f"artifact file hash mismatch: {record.path}")
                self.paths.add(record.path)
                self.resolved_paths.add(owned.path)
                self.file_identities.add(identity)
                yield owned
        except UnsafeOwnedFileError as error:
            raise DatasetValidationError(str(error)) from error


def _verify_json(
    root: Path,
    record: ArtifactRecord,
    registry: _ArtifactRegistry,
) -> dict[str, Any]:
    del root
    with registry.claim(record) as owned:
        try:
            return decode_json_artifact(owned.payload, record)
        except ArtifactDecodeError as error:
            raise DatasetValidationError(str(error)) from error


def _load_rgb(
    root: Path,
    record: ArtifactRecord,
    registry: _ArtifactRegistry,
) -> np.ndarray[Any, Any]:
    del root
    with registry.claim(record) as owned:
        try:
            return decode_rgb_artifact(owned.payload, record)
        except ArtifactDecodeError as error:
            raise DatasetValidationError(str(error)) from error


def _load_npy(
    root: Path,
    record: ArtifactRecord,
    registry: _ArtifactRegistry,
) -> np.ndarray[Any, Any]:
    del root
    with registry.claim(record) as owned:
        try:
            return decode_npy_artifact(owned.payload, record)
        except ArtifactDecodeError as error:
            raise DatasetValidationError(str(error)) from error


def _load_transport_direction(
    root: Path,
    direction: DirectionalOpticalTransport,
    expected_shape: tuple[int, int],
    registry: _ArtifactRegistry,
) -> DirectionalTransportArrays:
    vectors = _load_npy(root, direction.vectors_fixed, registry)
    validity = _load_npy(root, direction.validity, registry)
    reasons = _load_npy(root, direction.reasons, registry)
    if vectors.dtype != np.dtype("int32") or vectors.shape != (*expected_shape, 2):
        raise DatasetValidationError("analytic transport vector dtype or shape is invalid")
    if validity.dtype != np.dtype("uint8") or validity.shape != expected_shape:
        raise DatasetValidationError("analytic transport validity dtype or shape is invalid")
    if reasons.dtype != np.dtype("uint8") or reasons.shape != expected_shape:
        raise DatasetValidationError("analytic transport reason dtype or shape is invalid")
    if not np.all(np.isin(validity, (0, 1))):
        raise DatasetValidationError("analytic transport validity contains a value outside {0,1}")
    allowed_reason_codes = tuple(int(code) for code in TransportReasonCode)
    if not np.all(np.isin(reasons, allowed_reason_codes)):
        raise DatasetValidationError("analytic transport contains an unknown reason code")
    valid = validity == 1
    if not np.array_equal(valid, reasons == int(TransportReasonCode.VALID_TRANSPORT)):
        raise DatasetValidationError(
            "analytic transport validity and reason masks are inconsistent"
        )
    if np.any(vectors[~valid] != 0):
        raise DatasetValidationError("invalid analytic transport vectors must be canonical zero")
    return DirectionalTransportArrays(
        vectors_fixed=np.asarray(vectors, dtype=np.int32),
        validity=np.asarray(validity, dtype=np.uint8),
        reasons=np.asarray(reasons, dtype=np.uint8),
    )


def _load_analytic_transport(
    root: Path,
    transition: TransitionRecord,
    expected_shape: tuple[int, int],
    registry: _ArtifactRegistry,
) -> tuple[DirectionalTransportArrays, DirectionalTransportArrays]:
    transport = transition.analytic_optical_transport
    if not isinstance(transport, AvailableDenseOpticalTransport):
        raise DatasetValidationError(
            "generated scene families require available analytic transport"
        )
    if compute_analytic_transport_hash(transport) != transport.analytic_transport_sha256:
        raise DatasetValidationError("analytic transport identity mismatch")
    forward = _load_transport_direction(root, transport.forward, expected_shape, registry)
    backward = _load_transport_direction(root, transport.backward, expected_shape, registry)
    return forward, backward


def _require_exact_transport_recomputation(
    observed: tuple[DirectionalTransportArrays, DirectionalTransportArrays],
    expected: AnalyticTransportArrays,
) -> None:
    for direction_name, actual, recomputed in (
        ("forward", observed[0], expected.forward),
        ("backward", observed[1], expected.backward),
    ):
        if not np.array_equal(actual.vectors_fixed, recomputed.vectors_fixed):
            raise DatasetValidationError(
                f"{direction_name} analytic transport vectors differ from recomputation"
            )
        if not np.array_equal(actual.validity, recomputed.validity):
            raise DatasetValidationError(
                f"{direction_name} analytic transport validity differs from recomputation"
            )
        if not np.array_equal(actual.reasons, recomputed.reasons):
            raise DatasetValidationError(
                f"{direction_name} analytic transport reasons differ from recomputation"
            )


def _raw_surface_records(
    transition: TransitionRecord,
    instrumentation: PrivilegedInstrumentation,
) -> dict[int, Any]:
    by_id = {surface.surface_id: surface for surface in transition.surfaces}
    return {
        int(raw_id): by_id[opaque_id]
        for raw_id, opaque_id in instrumentation.raw_to_opaque_surface_ids.items()
    }


def _expected_oriented_boundary(
    analysis: RawBoundaryVisibilityAnalysis,
    raw_surfaces: dict[int, Any],
    width: int,
    height: int,
) -> AvailableOrientedBoundaryOwnership:
    elements = tuple(
        OrientedBoundaryElement(
            frame_index=item.frame_index,  # type: ignore[arg-type]
            axis=BoundaryAxis(item.axis),
            row=item.row,
            column=item.column,
            negative_surface_id=(
                raw_surfaces[item.negative_raw_geom_id].surface_id
                if item.negative_raw_geom_id is not None
                else None
            ),
            positive_surface_id=(
                raw_surfaces[item.positive_raw_geom_id].surface_id
                if item.positive_raw_geom_id is not None
                else None
            ),
            kind=BoundaryKind(item.kind),
            owner_side=BoundaryOwnerSide(item.owner_side),
            owner_surface_id=(
                raw_surfaces[item.owner_raw_geom_id].surface_id
                if item.owner_raw_geom_id is not None
                else None
            ),
        )
        for item in analysis.boundary_elements
    )
    boundary = AvailableOrientedBoundaryOwnership(
        status="available",
        method=ORIENTED_BOUNDARY_METHOD,
        raster_width=width,
        raster_height=height,
        coordinate_convention=EdgeLatticeCoordinateConvention(
            version=EDGE_LATTICE_CONVENTION,
            horizontal_negative_sample="pixel_centre_row_column_left",
            horizontal_positive_sample="pixel_centre_row_column_plus_1_right",
            horizontal_shape="height_by_width_minus_1",
            vertical_negative_sample="pixel_centre_row_column_top",
            vertical_positive_sample="pixel_centre_row_plus_1_column_bottom",
            vertical_shape="height_minus_1_by_width",
            no_boundary_representation="implicit_by_absent_sparse_record",
        ),
        boundary_kind_domain=BOUNDARY_KIND_DOMAIN,
        owner_side_domain=OWNER_SIDE_DOMAIN,
        attachment_rule=ATTACHMENT_RULE,
        attachment_public_contract_version=ATTACHMENT_PUBLIC_CONTRACT_VERSION,
        attachment_contact_manifold_rule=ATTACHMENT_CONTACT_MANIFOLD_RULE,
        attachment_supported_contact_manifold_types=SUPPORTED_CONTACT_MANIFOLD_TYPES,
        attachment_projection_convention=ATTACHMENT_PROJECTION_CONVENTION,
        attachment_projection_in_front_rule=ATTACHMENT_PROJECTION_IN_FRONT_RULE,
        attachment_feasibility_rule=ATTACHMENT_FEASIBILITY_RULE,
        attachment_edge_lattice_association_rule=ATTACHMENT_EDGE_ASSOCIATION_RULE,
        attachment_endpoint_tie_rule=ATTACHMENT_ENDPOINT_TIE_RULE,
        attachment_multi_surface_rule=ATTACHMENT_MULTI_SURFACE_RULE,
        numerical_contract_sha256=compute_boundary_numerical_contract_hash(),
        counterfactual_continuation_rule=COUNTERFACTUAL_CONTINUATION_RULE,
        counterfactual_tie_rule=COUNTERFACTUAL_TIE_RULE,
        junction_ambiguity_rule=JUNCTION_AMBIGUITY_RULE,
        silhouette_rule=SILHOUETTE_RULE,
        elements=elements,
        oriented_boundary_sha256="0" * 64,
    )
    return boundary.model_copy(
        update={"oriented_boundary_sha256": compute_oriented_boundary_hash(boundary)}
    )


def _expected_boundary_occlusion(
    boundary: AvailableOrientedBoundaryOwnership,
    oracle_rule: str,
) -> AvailableOcclusionAnnotation:
    frames_by_pair: dict[tuple[str, str], set[int]] = {}
    for element in boundary.elements:
        if element.kind != BoundaryKind.OCCLUDING_CONTOUR:
            continue
        owner = element.owner_surface_id
        if owner is None:
            raise DatasetValidationError("recomputed occluding contour lacks an owner")
        affected = (
            element.positive_surface_id
            if owner == element.negative_surface_id
            else element.negative_surface_id
        )
        if affected is None:
            raise DatasetValidationError("recomputed occluding contour lacks an affected surface")
        frames_by_pair.setdefault((owner, affected), set()).add(element.frame_index)
    return AvailableOcclusionAnnotation(
        status="available",
        oracle_rule=oracle_rule,  # type: ignore[arg-type]
        relations=tuple(
            OcclusionRelation(
                occluder_surface_id=owner,
                occluded_surface_id=affected,
                frame_indices=tuple(sorted(frames)),  # type: ignore[arg-type]
            )
            for (owner, affected), frames in sorted(frames_by_pair.items())
        ),
    )


def _raw_event_labels(
    raw_ids: np.ndarray[Any, Any],
    raw_surfaces: dict[int, Any],
) -> np.ndarray[Any, Any]:
    labels = np.zeros(raw_ids.shape, dtype=np.int32)
    for raw_id, surface in raw_surfaces.items():
        labels[raw_ids == raw_id] = np.int32(surface.segmentation_label)
    return labels


def _load_event_direction(
    root: Path,
    direction: DirectionalVisibilityEventMap,
    expected_shape: tuple[int, int],
    registry: _ArtifactRegistry,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    codes = _load_npy(root, direction.event_codes, registry)
    affected = _load_npy(root, direction.affected_surface_labels, registry)
    owner = _load_npy(root, direction.owner_surface_labels, registry)
    if codes.dtype != np.dtype("uint8") or codes.shape != expected_shape:
        raise DatasetValidationError("visibility-event code dtype or shape is invalid")
    if affected.dtype != np.dtype("int32") or affected.shape != expected_shape:
        raise DatasetValidationError("visibility-event affected-label dtype or shape is invalid")
    if owner.dtype != np.dtype("int32") or owner.shape != expected_shape:
        raise DatasetValidationError("visibility-event owner-label dtype or shape is invalid")
    if not np.all(np.isin(codes, tuple(range(6)))):
        raise DatasetValidationError("visibility-event map contains an unknown event code")
    causal = codes == 1
    if np.any(affected[causal] == 0) or np.any(owner[causal] == 0):
        raise DatasetValidationError("causal visibility events require owner/affected labels")
    if np.any(affected[~causal] != 0) or np.any(owner[~causal] != 0):
        raise DatasetValidationError("non-causal visibility events must use canonical zero labels")
    return codes, affected, owner


def _require_boundary_and_event_recomputation(
    root: Path,
    transition: TransitionRecord,
    instrumentation: PrivilegedInstrumentation,
    expected: RawBoundaryVisibilityAnalysis,
    expected_shape: tuple[int, int],
    registry: _ArtifactRegistry,
) -> None:
    raw_surfaces = _raw_surface_records(transition, instrumentation)
    expected_boundary = _expected_oriented_boundary(
        expected,
        raw_surfaces,
        expected_shape[1],
        expected_shape[0],
    )
    observed_boundary = transition.oriented_boundary_ownership
    if compute_oriented_boundary_hash(observed_boundary) != (
        observed_boundary.oriented_boundary_sha256
    ):
        raise DatasetValidationError("oriented-boundary identity mismatch")
    if observed_boundary != expected_boundary:
        raise DatasetValidationError("oriented-boundary records differ from recomputation")

    events = transition.ecological_visibility_events
    if compute_visibility_event_hash(events) != events.visibility_event_sha256:
        raise DatasetValidationError("visibility-event identity mismatch")
    before = _load_event_direction(root, events.before_fate, expected_shape, registry)
    after = _load_event_direction(root, events.after_origin, expected_shape, registry)
    expected_before = (
        expected.before_fate_codes,
        _raw_event_labels(expected.before_affected_raw_geom_ids, raw_surfaces),
        _raw_event_labels(expected.before_owner_raw_geom_ids, raw_surfaces),
    )
    expected_after = (
        expected.after_origin_codes,
        _raw_event_labels(expected.after_affected_raw_geom_ids, raw_surfaces),
        _raw_event_labels(expected.after_owner_raw_geom_ids, raw_surfaces),
    )
    if any(
        not np.array_equal(actual, recomputed)
        for actual, recomputed in zip(before, expected_before, strict=True)
    ):
        raise DatasetValidationError("before-frame visibility events differ from recomputation")
    if any(
        not np.array_equal(actual, recomputed)
        for actual, recomputed in zip(after, expected_after, strict=True)
    ):
        raise DatasetValidationError("after-frame visibility events differ from recomputation")

    summaries = tuple(
        sorted(
            (
                OccludingVisibilityEventSummary(
                    kind=item.kind,  # type: ignore[arg-type]
                    affected_surface_id=raw_surfaces[item.affected_raw_geom_id].surface_id,
                    owner_surface_id=raw_surfaces[item.owner_raw_geom_id].surface_id,
                    pixel_count=item.pixel_count,
                )
                for item in expected.event_summaries
            ),
            key=lambda item: (item.kind, item.affected_surface_id, item.owner_surface_id),
        )
    )
    whole = tuple(
        sorted(
            (
                WholeSurfaceVisibilityEvent(
                    surface_id=raw_surfaces[item.raw_geom_id].surface_id,
                    kind=WholeSurfaceEventKind(item.kind),
                    before_visible_pixels=item.before_visible_pixels,
                    after_visible_pixels=item.after_visible_pixels,
                )
                for item in expected.whole_surface_events
            ),
            key=lambda item: item.surface_id,
        )
    )
    if events.occluding_event_summaries != summaries:
        raise DatasetValidationError("visibility-event summaries differ from recomputation")
    if events.whole_surface_events != whole:
        raise DatasetValidationError("whole-surface events differ from recomputation")
    if np.any(before[0] == int(BeforeFateCode.UNRESOLVED_OCCLUSION)) or np.any(
        after[0] == int(AfterOriginCode.UNRESOLVED_OCCLUSION)
    ):
        raise DatasetValidationError("canonical visibility events contain unresolved occlusion")


def _expected_attachment_contract(
    analysis: RawBoundaryVisibilityAnalysis,
) -> PrivilegedAttachmentContractEvidence:
    contract = analysis.attachment_contract
    return PrivilegedAttachmentContractEvidence(
        method=contract.method,  # type: ignore[arg-type]
        contact_manifold_rule=contract.contact_manifold_rule,  # type: ignore[arg-type]
        supported_contact_manifold_types=contract.supported_contact_manifold_types,  # type: ignore[arg-type]
        projection_convention=contract.projection_convention,  # type: ignore[arg-type]
        projection_in_front_rule=contract.projection_in_front_rule,  # type: ignore[arg-type]
        projection_in_front_epsilon=contract.projection_in_front_epsilon,
        feasibility_rule=contract.feasibility_rule,  # type: ignore[arg-type]
        image_feasibility_slack=contract.image_feasibility_slack,
        edge_lattice_association_rule=contract.edge_lattice_association_rule,  # type: ignore[arg-type]
        endpoint_tie_rule=contract.endpoint_tie_rule,  # type: ignore[arg-type]
        multi_surface_rule=contract.multi_surface_rule,  # type: ignore[arg-type]
        contact_tolerance=contract.contact_tolerance,
        rotation_tolerance=contract.rotation_tolerance,
        image_tolerance_pixels=contract.image_tolerance_pixels,
        geom_types=contract.geom_types,  # type: ignore[arg-type]
        geom_world_rotations_row_major=contract.geom_world_rotations_row_major,
        pair_evidence=tuple(
            PrivilegedAttachmentPairEvidence(**item.__dict__) for item in contract.pair_evidence
        ),
    )


def _expected_boundary_diagnostics(
    analysis: RawBoundaryVisibilityAnalysis,
) -> BoundaryVisibilityDiagnostics:
    kind_counts = {kind: 0 for kind in BoundaryKind}
    owner_counts = {side: 0 for side in BoundaryOwnerSide}
    evidence: list[PrivilegedBoundaryElementEvidence] = []
    for item in analysis.boundary_elements:
        kind = BoundaryKind(item.kind)
        side = BoundaryOwnerSide(item.owner_side)
        kind_counts[kind] += 1
        owner_counts[side] += 1
        evidence.append(
            PrivilegedBoundaryElementEvidence(
                frame_index=item.frame_index,  # type: ignore[arg-type]
                axis=BoundaryAxis(item.axis),
                row=item.row,
                column=item.column,
                negative_raw_geom_id=item.negative_raw_geom_id,
                positive_raw_geom_id=item.positive_raw_geom_id,
                kind=kind,
                owner_side=side,
                owner_raw_geom_id=item.owner_raw_geom_id,
                negative_counterfactual_next_raw_geom_id=(
                    item.negative_counterfactual_next_raw_geom_id
                ),
                positive_counterfactual_next_raw_geom_id=(
                    item.positive_counterfactual_next_raw_geom_id
                ),
                on_projected_attachment_locus=item.on_projected_attachment_locus,
            )
        )
    return BoundaryVisibilityDiagnostics(
        boundary_method=ORIENTED_BOUNDARY_METHOD,
        counterfactual_continuation_rule="counterfactual_nearest_surface_continuation_v1",
        counterfactual_tie_rule="exactly_one_side_continues_v1",
        counterfactual_ray_direction_epsilon=RAY_DIRECTION_EPSILON,
        junction_ambiguity_rule=JUNCTION_AMBIGUITY_RULE,
        silhouette_rule="controlled_to_uncontrolled_side_owns_v1",
        visibility_event_method="analytic_transport_boundary_causal_events_v2",
        boundary_evidence=tuple(evidence),
        boundary_kind_counts=kind_counts,
        owner_side_counts=owner_counts,
        before_event_code_counts=tuple(
            int(value)
            for value in np.bincount(analysis.before_fate_codes.reshape(-1), minlength=6)[:6]
        ),  # type: ignore[arg-type]
        after_event_code_counts=tuple(
            int(value)
            for value in np.bincount(analysis.after_origin_codes.reshape(-1), minlength=6)[:6]
        ),  # type: ignore[arg-type]
    )


def _reconstruct_raw_segmentation(
    transition: TransitionRecord,
    instrumentation: PrivilegedInstrumentation,
    public_segmentation: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    labels = {surface.surface_id: surface.segmentation_label for surface in transition.surfaces}
    raw = np.full(public_segmentation.shape, -1, dtype=np.int32)
    for raw_id_text, opaque_id in instrumentation.raw_to_opaque_surface_ids.items():
        raw[public_segmentation == labels[opaque_id]] = int(raw_id_text)
    return raw


def _directional_transport_diagnostic(
    arrays: DirectionalTransportArrays,
) -> DirectionalTransportDiagnostic:
    total = int(arrays.validity.size)
    valid = int(np.count_nonzero(arrays.validity))
    counts = np.bincount(arrays.reasons.reshape(-1), minlength=5)
    return DirectionalTransportDiagnostic(
        total_pixels=total,
        valid_transport_pixels=valid,
        valid_transport_fraction=valid / total,
        reason_code_counts=tuple(int(count) for count in counts[:5]),  # type: ignore[arg-type]
    )


def _expected_analytic_transport_diagnostics(
    arrays: AnalyticTransportArrays,
    raw_segmentations: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]],
) -> AnalyticTransportDiagnostics:
    frame_diagnostics: list[AnalyticRendererFrameDiagnostic] = []
    for frame_index, (assignment, boundary, rendered) in enumerate(
        (
            (
                arrays.before_surface_assignment,
                arrays.before_boundary_ambiguous,
                raw_segmentations[0],
            ),
            (
                arrays.after_surface_assignment,
                arrays.after_boundary_ambiguous,
                raw_segmentations[1],
            ),
        )
    ):
        interior = ~boundary
        compared = int(np.count_nonzero(interior))
        agreeing = int(np.count_nonzero((assignment == rendered) & interior))
        frame_diagnostics.append(
            AnalyticRendererFrameDiagnostic(
                frame_index=frame_index,  # type: ignore[arg-type]
                compared_interior_pixels=compared,
                agreeing_interior_pixels=agreeing,
                interior_agreement_rate=agreeing / compared,
                unexplained_interior_disagreement_pixels=compared - agreeing,
                excluded_analytic_boundary_pixels=int(np.count_nonzero(boundary)),
            )
        )
    return AnalyticTransportDiagnostics(
        method=ANALYTIC_TRANSPORT_METHOD,
        renderer_cross_check="non_authoritative_exact_interior_agreement_v2",
        interior_agreement_requirement="zero_unexplained_disagreement_v1",
        frames=tuple(frame_diagnostics),  # type: ignore[arg-type]
        forward=_directional_transport_diagnostic(arrays.forward),
        backward=_directional_transport_diagnostic(arrays.backward),
    )


def _require_camera_action_alignment(
    config: BenchmarkConfig,
    transition: TransitionRecord,
    cameras: tuple[CameraInstrumentation, CameraInstrumentation],
    instrumentation: PrivilegedInstrumentation,
    compiled_scene: CompiledSceneContract,
) -> None:
    expected_action = Action.model_validate(config.action.model_dump(mode="python"))
    if transition.action != expected_action:
        raise DatasetValidationError("persisted action differs from resolved configuration")
    before, after = cameras
    if isinstance(config, SingleOccluderConfig):
        if not isinstance(instrumentation, SingleOccluderInstrumentation):
            raise DatasetValidationError("single-occluder config requires matching instrumentation")
        expected_before = np.asarray(
            (config.camera.before_lateral, config.camera.forward, config.camera.height),
            dtype=np.float64,
        )
        expected_after = np.asarray(
            (config.camera.after_lateral, config.camera.forward, config.camera.height),
            dtype=np.float64,
        )
    elif isinstance(config, CorridorConfig):
        if not isinstance(instrumentation, CorridorInstrumentation):
            raise DatasetValidationError("corridor config requires matching instrumentation")
        geometry = instrumentation.sampled_geometry
        expected_before = np.asarray(
            (
                geometry.camera_lateral_position,
                geometry.camera_before_forward_position,
                geometry.camera_height,
            ),
            dtype=np.float64,
        )
        expected_after = np.asarray(
            (
                geometry.camera_lateral_position,
                geometry.camera_after_forward_position,
                geometry.camera_height,
            ),
            dtype=np.float64,
        )
        if before != instrumentation.camera_before or after != instrumentation.camera_after:
            raise DatasetValidationError(
                "corridor camera artifacts differ from scene instrumentation"
            )
    else:
        raise DatasetValidationError("unsupported scene configuration")
    before_position = np.asarray(before.camera_world_position, dtype=np.float64)
    after_position = np.asarray(after.camera_world_position, dtype=np.float64)
    if not np.allclose(before_position, expected_before, atol=1e-12, rtol=0.0):
        raise DatasetValidationError("before camera pose differs from resolved configuration")
    if not np.allclose(after_position, expected_after, atol=1e-12, rtol=0.0):
        raise DatasetValidationError("after camera pose differs from resolved configuration")
    expected_delta = np.asarray(
        (
            transition.action.delta_lateral,
            transition.action.delta_forward,
            0.0,
        ),
        dtype=np.float64,
    )
    if not np.allclose(after_position - before_position, expected_delta, atol=1e-12, rtol=0.0):
        raise DatasetValidationError("persisted camera displacement differs from executed action")
    if not np.allclose(
        before.camera_world_rotation_row_major,
        after.camera_world_rotation_row_major,
        atol=1e-12,
        rtol=0.0,
    ):
        raise DatasetValidationError("camera rotation changed despite zero executed yaw")
    if not np.allclose(
        before.camera_world_rotation_row_major,
        compiled_scene.camera_world_rotation_row_major,
        atol=1e-12,
        rtol=0.0,
    ):
        raise DatasetValidationError("camera rotation differs from the compiled MuJoCo scene")
    if not np.allclose(
        before.camera_world_position,
        compiled_scene.camera_world_position,
        atol=1e-12,
        rtol=0.0,
    ):
        raise DatasetValidationError("before camera pose differs from the compiled MuJoCo scene")
    if not np.isclose(
        compiled_scene.camera_field_of_view_degrees,
        config.camera.field_of_view_degrees,
        atol=1e-12,
        rtol=0.0,
    ):
        raise DatasetValidationError("camera field of view differs from the compiled MuJoCo scene")


def _require_compiled_apparatus_contract(
    instrumentation: PrivilegedInstrumentation,
    compiled_scene: CompiledSceneContract,
) -> None:
    if instrumentation.raw_geom_ids != compiled_scene.raw_geom_ids:
        raise DatasetValidationError(
            "semantic apparatus names do not match compiled MuJoCo geom identifiers"
        )
    for name, expected_position in compiled_scene.raw_geom_world_positions.items():
        if not np.allclose(
            instrumentation.raw_geom_world_positions[name],
            expected_position,
            atol=1e-12,
            rtol=0.0,
        ):
            raise DatasetValidationError(
                "instrumented geom world position differs from compiled MuJoCo scene"
            )
    for name, expected_size in compiled_scene.raw_geom_compiled_sizes.items():
        if not np.allclose(
            instrumentation.raw_geom_compiled_sizes[name],
            expected_size,
            atol=1e-12,
            rtol=0.0,
        ):
            raise DatasetValidationError(
                "instrumented geom size differs from compiled MuJoCo scene"
            )
    if instrumentation.raw_geom_types != compiled_scene.raw_geom_types:
        raise DatasetValidationError("instrumented geom types differ from compiled MuJoCo scene")
    for name, expected_rotation in compiled_scene.raw_geom_world_rotations_row_major.items():
        if not np.allclose(
            instrumentation.raw_geom_world_rotations_row_major[name],
            expected_rotation,
            atol=1e-12,
            rtol=0.0,
        ):
            raise DatasetValidationError(
                "instrumented geom rotation differs from compiled MuJoCo scene"
            )


def _corridor_scene_content_hash(
    config: CorridorConfig,
    instrumentation: CorridorInstrumentation,
) -> str:
    geometry = instrumentation.sampled_geometry
    return sha256_bytes(
        canonical_json_bytes(
            {
                "scene_family": config.scene_family,
                "apparatus_version": "corridor_v1",
                "surfaces": {
                    "surface_names": list(CORRIDOR_SURFACE_NAMES),
                    "width": geometry.width,
                    "length": geometry.length,
                    "wall_height": geometry.wall_height,
                    "wall_thickness": 0.05,
                    "floor_thickness": 0.05,
                },
                "camera": {
                    "before_position": [
                        geometry.camera_lateral_position,
                        geometry.camera_before_forward_position,
                        geometry.camera_height,
                    ],
                    "after_position": [
                        geometry.camera_lateral_position,
                        geometry.camera_after_forward_position,
                        geometry.camera_height,
                    ],
                    "orientation_rule": "xyaxes_1_0_0_0_0_1",
                    "field_of_view_degrees": geometry.field_of_view_degrees,
                },
                "action": config.action.model_dump(mode="json"),
            }
        )
    )


def _single_occluder_scene_content_hash(config: SingleOccluderConfig) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "scene_family": config.scene_family,
                "apparatus_version": "single_occluder_v1",
                "surfaces": {
                    "support_surface": {
                        "type": "plane",
                        "position": [0.0, 0.0, 0.0],
                        "size": [4.0, 7.0, 0.1],
                    },
                    "background_surface": {
                        "type": "box",
                        "position": [0.0, 2.5, 1.05],
                        "half_size": [2.2, 0.05, 1.05],
                    },
                    "occluding_surface": {
                        "type": "box",
                        "position": [0.0, 0.8, 0.9],
                        "half_size": [0.55, 0.05, 0.9],
                    },
                },
                "camera": {
                    "before_position": [
                        config.camera.before_lateral,
                        config.camera.forward,
                        config.camera.height,
                    ],
                    "after_position": [
                        config.camera.after_lateral,
                        config.camera.forward,
                        config.camera.height,
                    ],
                    "orientation_rule": "xyaxes_1_0_0_0_0.16_1",
                    "field_of_view_degrees": config.camera.field_of_view_degrees,
                },
                "action": config.action.model_dump(mode="json"),
            }
        )
    )


def _require_corridor_instrumentation(
    config: CorridorConfig,
    episode_seed: int,
    instrumentation: CorridorInstrumentation,
) -> None:
    geometry_seed, remapping_seed, appearance_seed = corridor_generation_seeds(episode_seed)
    if instrumentation.generation_seeds.model_dump(mode="python") != {
        "episode_seed": episode_seed,
        "geometry_sampling_seed": geometry_seed,
        "surface_remapping_seed": remapping_seed,
        "appearance_seed": appearance_seed,
    }:
        raise DatasetValidationError("corridor generation seed namespaces are inconsistent")
    geometry = instrumentation.sampled_geometry
    geometry_rng = np.random.default_rng(geometry_seed)
    expected_width = float(
        geometry_rng.uniform(config.geometry.width.minimum, config.geometry.width.maximum)
    )
    expected_length = float(
        geometry_rng.uniform(config.geometry.length.minimum, config.geometry.length.maximum)
    )
    expected_sampled_values = (
        expected_width,
        expected_length,
        config.geometry.wall_height,
        config.camera.lateral_position,
        config.camera.starting_forward_position,
        config.camera.starting_forward_position + config.action.delta_forward,
        config.camera.height,
        config.camera.field_of_view_degrees,
    )
    observed_sampled_values = (
        geometry.width,
        geometry.length,
        geometry.wall_height,
        geometry.camera_lateral_position,
        geometry.camera_before_forward_position,
        geometry.camera_after_forward_position,
        geometry.camera_height,
        geometry.field_of_view_degrees,
    )
    if observed_sampled_values != expected_sampled_values:
        raise DatasetValidationError(
            "sampled corridor geometry differs from deterministic configuration"
        )
    expected_positions = {
        "corridor_floor": (0.0, geometry.length / 2.0, -0.05),
        "corridor_left_surface": (
            -geometry.width / 2.0,
            geometry.length / 2.0,
            geometry.wall_height / 2.0,
        ),
        "corridor_right_surface": (
            geometry.width / 2.0,
            geometry.length / 2.0,
            geometry.wall_height / 2.0,
        ),
        "corridor_end_surface": (0.0, geometry.length, geometry.wall_height / 2.0),
    }
    if set(instrumentation.apparatus_surface_names) != set(CORRIDOR_SURFACE_NAMES):
        raise DatasetValidationError("corridor apparatus surface membership is inconsistent")
    for name, expected_position in expected_positions.items():
        if not np.allclose(
            instrumentation.raw_geom_world_positions[name],
            expected_position,
            atol=1e-12,
            rtol=0.0,
        ):
            raise DatasetValidationError(
                "corridor raw geometry does not match sampled privileged geometry"
            )


def _require_corridor_raw_segmentation(
    root: Path,
    transition: TransitionRecord,
    instrumentation: CorridorInstrumentation,
    segmentations: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]],
    expected_shape: tuple[int, int],
    registry: _ArtifactRegistry,
) -> None:
    labels = {surface.surface_id: surface.segmentation_label for surface in transition.surfaces}
    raw_ids = set(instrumentation.raw_geom_ids.values())
    observed_raw_ids: set[int] = set()
    evidence_by_frame = sorted(
        instrumentation.raw_segmentation_frames,
        key=lambda item: item.frame_index,
    )
    for frame_index, public_segmentation in enumerate(segmentations):
        raw_segmentation = _load_npy(
            root,
            evidence_by_frame[frame_index].raw_segmentation,
            registry,
        )
        if (
            raw_segmentation.dtype != np.dtype("int32")
            or raw_segmentation.shape != expected_shape
            or raw_segmentation.shape != public_segmentation.shape
        ):
            raise DatasetValidationError("corridor raw segmentation dtype or shape is invalid")
        frame_raw_ids = {int(value) for value in np.unique(raw_segmentation)}
        if not frame_raw_ids.issubset(raw_ids | {-1}):
            raise DatasetValidationError("corridor raw segmentation contains an unknown raw ID")
        observed_raw_ids.update(frame_raw_ids - {-1})
        reconstructed = np.zeros(raw_segmentation.shape, dtype=np.int32)
        for raw_id_text, opaque_id in instrumentation.raw_to_opaque_surface_ids.items():
            reconstructed[raw_segmentation == int(raw_id_text)] = labels[opaque_id]
        if not np.array_equal(reconstructed, public_segmentation):
            raise DatasetValidationError(
                "corridor public segmentation does not match the raw-to-opaque mapping"
            )
    if observed_raw_ids != raw_ids:
        raise DatasetValidationError("corridor raw segmentation does not contain every surface")


def _require_occlusion_oracle(
    root: Path,
    transition: TransitionRecord,
    instrumentation: SingleOccluderInstrumentation,
    segmentations: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]],
    expected_shape: tuple[int, int],
    registry: _ArtifactRegistry,
) -> None:
    oracle = instrumentation.occlusion_oracle
    raw_ids = set(instrumentation.raw_geom_ids.values())
    if oracle.candidate_occluder_raw_geom_id not in raw_ids:
        raise DatasetValidationError("occlusion oracle names an unknown occluder raw ID")
    if oracle.candidate_occluded_raw_geom_id not in raw_ids:
        raise DatasetValidationError("occlusion oracle names an unknown occluded raw ID")
    if oracle.candidate_occluder_raw_geom_id != instrumentation.raw_geom_ids.get(
        "occluding_surface"
    ):
        raise DatasetValidationError("occlusion oracle candidate is not the apparatus occluder")
    if oracle.candidate_occluded_raw_geom_id != instrumentation.raw_geom_ids.get(
        "background_surface"
    ):
        raise DatasetValidationError("occlusion oracle candidate is not the apparatus background")

    mapping = instrumentation.raw_to_opaque_surface_ids
    occluder_id = mapping.get(str(oracle.candidate_occluder_raw_geom_id))
    occluded_id = mapping.get(str(oracle.candidate_occluded_raw_geom_id))
    if occluder_id is None or occluded_id is None:
        raise DatasetValidationError("occlusion candidates lack opaque surface mappings")
    labels = {surface.surface_id: surface.segmentation_label for surface in transition.surfaces}
    if occluder_id not in labels or occluded_id not in labels:
        raise DatasetValidationError("occlusion candidates map outside declared surfaces")
    opaque_to_raw = {opaque_id: int(raw_id) for raw_id, opaque_id in mapping.items()}
    if set(opaque_to_raw) != set(labels) or len(opaque_to_raw) != len(mapping):
        raise DatasetValidationError("apparatus raw-to-opaque mapping is not exact and bijective")

    evidence_by_frame = sorted(oracle.frames, key=lambda item: item.frame_index)
    supported_frames: list[int] = []
    for frame_index, ordinary_segmentation in enumerate(segmentations):
        evidence = evidence_by_frame[frame_index]
        counterfactual = _load_npy(
            root,
            evidence.counterfactual_segmentation,
            registry,
        )
        if (
            counterfactual.dtype != np.dtype("int32")
            or counterfactual.shape != expected_shape
            or counterfactual.shape != ordinary_segmentation.shape
        ):
            raise DatasetValidationError("counterfactual segmentation dtype or shape is invalid")
        observed_raw_ids = {int(value) for value in np.unique(counterfactual)}
        if not observed_raw_ids.issubset(raw_ids | {-1}):
            raise DatasetValidationError("counterfactual segmentation contains an unknown raw ID")
        ordinary_raw = np.full(ordinary_segmentation.shape, -1, dtype=np.int32)
        for opaque_id, raw_id in opaque_to_raw.items():
            ordinary_raw[ordinary_segmentation == labels[opaque_id]] = raw_id
        if np.any(counterfactual == oracle.candidate_occluder_raw_geom_id):
            raise DatasetValidationError(
                "counterfactual segmentation still contains the excluded occluder"
            )
        ordinary_occluder_mask = ordinary_raw == oracle.candidate_occluder_raw_geom_id
        changed_mask = counterfactual != ordinary_raw
        if not np.array_equal(changed_mask, ordinary_occluder_mask):
            raise DatasetValidationError(
                "counterfactual changes are not confined exactly to the occluder footprint"
            )
        reveal_mask = ordinary_occluder_mask & (
            counterfactual == oracle.candidate_occluded_raw_geom_id
        )
        count = int(np.count_nonzero(reveal_mask))
        if count != evidence.revealed_pixel_count:
            raise DatasetValidationError("counterfactual revealed-pixel count is inconsistent")
        if logical_array_hash(reveal_mask) != evidence.reveal_mask_logical_sha256:
            raise DatasetValidationError("counterfactual reveal-mask hash is inconsistent")
        if count > 0:
            supported_frames.append(frame_index)

    if not supported_frames:
        raise DatasetValidationError("counterfactual cross-check found no designated occlusion")
    if not isinstance(transition.occlusion, AvailableOcclusionAnnotation):
        raise DatasetValidationError("single-occluder occlusion must be available")
    designated = tuple(
        relation
        for relation in transition.occlusion.relations
        if relation.occluder_surface_id == occluder_id
        and relation.occluded_surface_id == occluded_id
    )
    if len(designated) != 1 or designated[0].frame_indices != tuple(supported_frames):
        raise DatasetValidationError(
            "designated counterfactual evidence disagrees with the complete boundary graph"
        )


def validate_dataset(root: Path) -> DatasetManifest:
    """Validate every declared artifact and cross-record invariant."""

    try:
        manifest_context = open_dataset_manifest(root)
        with manifest_context as (resolved_root, owned_manifest):
            manifest = DatasetManifest.model_validate_json(owned_manifest.payload)
    except UnsafeDatasetManifestError as error:
        raise DatasetValidationError(str(error)) from error
    except Exception as error:
        raise DatasetValidationError("dataset manifest failed schema validation") from error
    if compute_dataset_logical_hash(manifest) != manifest.dataset_logical_sha256:
        raise DatasetValidationError("dataset logical hash mismatch")
    source_provenance_sha256 = compute_source_provenance_hash(manifest.source_provenance)
    if source_provenance_sha256 != manifest.source_provenance_sha256:
        raise DatasetValidationError("source provenance hash mismatch")
    renderer_execution_provenance_sha256 = compute_renderer_execution_provenance_hash(
        manifest.renderer_provenance
    )
    if renderer_execution_provenance_sha256 != manifest.renderer_execution_provenance_sha256:
        raise DatasetValidationError("renderer/execution provenance hash mismatch")
    if (
        compute_content_provenance_binding(
            manifest.dataset_logical_sha256,
            manifest.source_provenance_sha256,
            manifest.renderer_execution_provenance_sha256,
        )
        != manifest.content_provenance_binding_sha256
    ):
        raise DatasetValidationError("content/provenance binding hash mismatch")

    registry = _ArtifactRegistry(resolved_root)

    appearance_registry_payload = _verify_json(
        resolved_root, manifest.appearance_registry_snapshot, registry
    )
    try:
        appearance_registry = AppearanceRegistry.model_validate_json(
            canonical_json_bytes(appearance_registry_payload)
        )
        validate_axis_isolation(appearance_registry)
    except Exception as error:
        raise DatasetValidationError("appearance registry snapshot is invalid") from error
    if appearance_registry_hash(appearance_registry) != manifest.appearance_registry_sha256:
        raise DatasetValidationError("appearance registry hash mismatch")
    try:
        selected_profile = profile_by_id(appearance_registry, manifest.appearance_profile_id)
    except ValueError as error:
        raise DatasetValidationError("selected appearance profile is absent") from error
    if appearance_profile_hash(selected_profile) != manifest.appearance_profile_sha256:
        raise DatasetValidationError("appearance profile hash mismatch")
    seed_registry_payload = _verify_json(
        resolved_root,
        manifest.evaluation_seed_registry_snapshot,
        registry,
    )
    try:
        seed_registry = EvaluationSeedRegistry.model_validate_json(
            canonical_json_bytes(seed_registry_payload)
        )
    except Exception as error:
        raise DatasetValidationError("evaluation seed registry snapshot is invalid") from error
    if seed_registry_hash(seed_registry) != manifest.evaluation_seed_registry_sha256:
        raise DatasetValidationError("evaluation seed registry hash mismatch")
    if (
        manifest.evaluation_seed_registry_snapshot.logical_sha256
        != manifest.evaluation_seed_registry_sha256
    ):
        raise DatasetValidationError("evaluation seed registry artifact identity mismatch")

    config_payload = _verify_json(resolved_root, manifest.resolved_config, registry)
    try:
        config = parse_config(config_payload)
    except Exception as error:
        raise DatasetValidationError("resolved configuration failed schema validation") from error
    if sha256_bytes(canonical_json_bytes(config)) != manifest.config_logical_sha256:
        raise DatasetValidationError("resolved configuration hash mismatch")
    if config.seed != manifest.root_seed:
        raise DatasetValidationError("manifest seed does not match resolved configuration")
    if config.appearance.registry_version != appearance_registry.registry_version:
        raise DatasetValidationError("appearance registry version is inconsistent")
    if config.appearance.profile_id != manifest.appearance_profile_id:
        raise DatasetValidationError("appearance profile selection is inconsistent")
    if config.scene_family != manifest.scene_family:
        raise DatasetValidationError("manifest scene family differs from resolved configuration")
    expected_raster_shape = (config.render.height, config.render.width)

    dataset_surface_ids: set[str] = set()
    for episode in manifest.episodes:
        if episode.episode_id != f"episode-{episode.episode_index:06d}":
            raise DatasetValidationError("episode identifier does not match its index")
        if episode.episode_seed != derive_seed(config.seed, f"episode:{episode.episode_index}"):
            raise DatasetValidationError("episode derived seed mismatch")
        transition_payload = _verify_json(resolved_root, episode.transition, registry)
        try:
            transition = TransitionRecord.model_validate_json(
                canonical_json_bytes(transition_payload)
            )
        except Exception as error:
            raise DatasetValidationError("transition failed schema validation") from error
        if transition.episode_id != episode.episode_id:
            raise DatasetValidationError("episode and transition identifiers differ")
        if compute_ecological_label_hash(transition) != transition.ecological_label_sha256:
            raise DatasetValidationError("ecological-label hash mismatch")
        if transition.ecological_label_sha256 != episode.ecological_label_sha256:
            raise DatasetValidationError("episode ecological-label hash mismatch")
        if not isinstance(transition.analytic_optical_transport, AvailableDenseOpticalTransport):
            raise DatasetValidationError("analytic optical transport must be available")
        if (
            transition.analytic_optical_transport.analytic_transport_sha256
            != episode.analytic_transport_sha256
        ):
            raise DatasetValidationError("episode analytic transport identity mismatch")
        if (
            transition.oriented_boundary_ownership.oriented_boundary_sha256
            != episode.oriented_boundary_sha256
        ):
            raise DatasetValidationError("episode oriented-boundary identity mismatch")
        if (
            transition.ecological_visibility_events.visibility_event_sha256
            != episode.visibility_event_sha256
        ):
            raise DatasetValidationError("episode visibility-event identity mismatch")
        instrumentation_payload = _verify_json(
            resolved_root,
            episode.privileged_instrumentation,
            registry,
        )
        try:
            instrumentation = parse_privileged_instrumentation_json(
                canonical_json_bytes(instrumentation_payload)
            )
        except Exception as error:
            raise DatasetValidationError("instrumentation failed schema validation") from error
        if instrumentation.episode_id != episode.episode_id:
            raise DatasetValidationError("instrumentation episode identifier mismatch")
        if (
            instrumentation.appearance.appearance_instance_sha256
            != episode.appearance_instance_sha256
        ):
            raise DatasetValidationError("episode appearance-instance identity mismatch")
        if instrumentation.scene_family != manifest.scene_family:
            raise DatasetValidationError("instrumentation scene family mismatch")
        if len(set(instrumentation.raw_geom_ids.values())) != len(instrumentation.raw_geom_ids):
            raise DatasetValidationError("raw MuJoCo geom identifiers must be unique")
        if set(instrumentation.raw_geom_ids) != set(instrumentation.raw_geom_world_positions):
            raise DatasetValidationError("raw geom coordinate records are incomplete")
        if set(instrumentation.raw_geom_ids) != set(instrumentation.raw_geom_compiled_sizes):
            raise DatasetValidationError("compiled geom size records are incomplete")
        if set(instrumentation.raw_geom_ids) != set(instrumentation.raw_geom_types):
            raise DatasetValidationError("compiled geom type records are incomplete")
        if set(instrumentation.raw_geom_ids) != set(
            instrumentation.raw_geom_world_rotations_row_major
        ):
            raise DatasetValidationError("compiled geom rotation records are incomplete")
        if set(instrumentation.raw_to_opaque_surface_ids.values()) != {
            surface.surface_id for surface in transition.surfaces
        }:
            raise DatasetValidationError("privileged surface remapping is incomplete")
        if set(instrumentation.raw_to_opaque_surface_ids) != {
            str(raw_id) for raw_id in instrumentation.raw_geom_ids.values()
        }:
            raise DatasetValidationError("privileged raw-ID remapping keys are inconsistent")
        ordinary_transition_bytes = canonical_json_bytes(transition_payload)
        forbidden_control_or_semantic_tokens = (
            b"scene_family",
            b"sampled_geometry",
            *(name.encode("utf-8") for name in instrumentation.raw_geom_ids),
        )
        if any(
            token in ordinary_transition_bytes for token in forbidden_control_or_semantic_tokens
        ):
            raise DatasetValidationError(
                "ordinary ecological transition leaks scene control or semantic apparatus names"
            )
        episode_surface_ids = {surface.surface_id for surface in transition.surfaces}
        duplicate_surface_ids = dataset_surface_ids & episode_surface_ids
        if duplicate_surface_ids:
            raise DatasetValidationError("surface identifiers must be disjoint across episodes")
        dataset_surface_ids.update(episode_surface_ids)

        if isinstance(instrumentation, SingleOccluderInstrumentation):
            if not isinstance(config, SingleOccluderConfig):
                raise DatasetValidationError(
                    "single-occluder instrumentation/configuration mismatch"
                )
            try:
                appearance = validate_appearance_instance(
                    instrumentation.appearance,
                    appearance_registry,
                    "single_occluder",
                    ("support_surface", "occluding_surface", "background_surface"),
                    episode.episode_seed,
                    config.seed,
                    seed_registry,
                )
            except Exception as error:
                raise DatasetValidationError("appearance instance failed validation") from error
            compiled_scene = compile_single_occluder_scene_contract(config, appearance)
            _require_compiled_apparatus_contract(instrumentation, compiled_scene)
            expected_scene_content_sha256 = _single_occluder_scene_content_hash(config)
            expected_analytic_transport = compute_single_occluder_analytic_transport(
                config, appearance
            )
            expected_boundary_visibility = compute_single_occluder_boundary_visibility(
                config, appearance
            )
        elif isinstance(instrumentation, CorridorInstrumentation):
            if not isinstance(config, CorridorConfig):
                raise DatasetValidationError("corridor instrumentation/configuration mismatch")
            _require_corridor_instrumentation(config, episode.episode_seed, instrumentation)
            try:
                appearance = validate_appearance_instance(
                    instrumentation.appearance,
                    appearance_registry,
                    "corridor",
                    CORRIDOR_SURFACE_NAMES,
                    episode.episode_seed,
                    config.seed,
                    seed_registry,
                )
            except Exception as error:
                raise DatasetValidationError("appearance instance failed validation") from error
            compiled_scene = compile_corridor_scene_contract(
                config, instrumentation.sampled_geometry, appearance
            )
            _require_compiled_apparatus_contract(instrumentation, compiled_scene)
            expected_scene_content_sha256 = _corridor_scene_content_hash(
                config,
                instrumentation,
            )
            expected_analytic_transport = compute_corridor_analytic_transport(
                config,
                instrumentation.sampled_geometry,
                appearance,
            )
            expected_boundary_visibility = compute_corridor_boundary_visibility(
                config,
                instrumentation.sampled_geometry,
                appearance,
            )
        else:
            raise DatasetValidationError("unsupported scene instrumentation")
        if episode.scene_content_sha256 != expected_scene_content_sha256:
            raise DatasetValidationError("episode scene-content hash mismatch")

        frame_arrays: list[
            tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]
        ] = []
        cameras: list[CameraInstrumentation] = []
        for frame in (transition.before, transition.after):
            if (frame.height, frame.width) != expected_raster_shape:
                raise DatasetValidationError(
                    "frame dimensions differ from the resolved render configuration"
                )
            rgb = _load_rgb(resolved_root, frame.rgb, registry)
            depth = _load_npy(resolved_root, frame.depth, registry)
            segmentation = _load_npy(resolved_root, frame.segmentation, registry)
            if rgb.dtype != np.dtype("uint8") or rgb.shape != (
                frame.height,
                frame.width,
                3,
            ):
                raise DatasetValidationError("RGB dtype or shape is invalid")
            if depth.dtype != np.dtype("float32") or depth.shape != (
                frame.height,
                frame.width,
            ):
                raise DatasetValidationError("depth dtype or shape is invalid")
            if not np.all(np.isfinite(depth)) or np.any(depth < 0.0):
                raise DatasetValidationError("depth values must be finite and non-negative")
            if segmentation.dtype != np.dtype("int32") or segmentation.shape != (
                frame.height,
                frame.width,
            ):
                raise DatasetValidationError("segmentation dtype or shape is invalid")
            if rgb.shape[:2] != depth.shape or depth.shape != segmentation.shape:
                raise DatasetValidationError("RGB, depth, and segmentation are misaligned")
            camera_payload = _verify_json(
                resolved_root,
                frame.camera_world_transform,
                registry,
            )
            try:
                camera = CameraInstrumentation.model_validate_json(
                    canonical_json_bytes(camera_payload)
                )
            except Exception as error:
                raise DatasetValidationError("camera failed schema validation") from error
            if camera.frame_index != frame.frame_index:
                raise DatasetValidationError("camera instrumentation frame mismatch")
            cameras.append(camera)
            frame_arrays.append((rgb, depth, segmentation))

        _require_camera_action_alignment(
            config,
            transition,
            (cameras[0], cameras[1]),
            instrumentation,
            compiled_scene,
        )
        observed_analytic_transport = _load_analytic_transport(
            resolved_root,
            transition,
            expected_raster_shape,
            registry,
        )
        _require_exact_transport_recomputation(
            observed_analytic_transport,
            expected_analytic_transport,
        )
        _require_boundary_and_event_recomputation(
            resolved_root,
            transition,
            instrumentation,
            expected_boundary_visibility,
            expected_raster_shape,
            registry,
        )
        if instrumentation.attachment_contract != _expected_attachment_contract(
            expected_boundary_visibility
        ):
            raise DatasetValidationError("attachment contract differs from compiled recomputation")
        if instrumentation.boundary_visibility_diagnostics != _expected_boundary_diagnostics(
            expected_boundary_visibility
        ):
            raise DatasetValidationError("boundary/event diagnostics differ from recomputation")

        if tuple(item[0].shape for item in frame_arrays) != (
            transition.before.rgb.shape,
            transition.after.rgb.shape,
        ):
            raise DatasetValidationError("frame shapes differ from transition metadata")
        if episode.rgb_logical_sha256 != (
            transition.before.rgb.logical_sha256,
            transition.after.rgb.logical_sha256,
        ):
            raise DatasetValidationError("episode RGB hashes differ from frame records")

        declared_labels = {surface.segmentation_label for surface in transition.surfaces}
        observed_labels: set[int] = set()
        for _, _, segmentation in frame_arrays:
            observed_labels.update(int(value) for value in np.unique(segmentation) if value != 0)
        if not observed_labels.issubset(declared_labels):
            raise DatasetValidationError("segmentation contains an undeclared surface label")
        if observed_labels != declared_labels:
            raise DatasetValidationError("a declared surface is absent from both frames")
        if isinstance(instrumentation, CorridorInstrumentation) and any(
            np.any(segmentation == 0) for _, _, segmentation in frame_arrays
        ):
            raise DatasetValidationError(
                "corridor optical field contains uncontrolled renderer background"
            )

        raw_surfaces = _raw_surface_records(transition, instrumentation)
        expected_boundary = _expected_oriented_boundary(
            expected_boundary_visibility,
            raw_surfaces,
            expected_raster_shape[1],
            expected_raster_shape[0],
        )
        occlusion_rule = (
            "oriented_boundary_ownership_with_counterfactual_crosscheck_v1"
            if isinstance(instrumentation, SingleOccluderInstrumentation)
            else "oriented_boundary_ownership_complete_v2"
        )
        if transition.occlusion != _expected_boundary_occlusion(
            expected_boundary,
            occlusion_rule,
        ):
            raise DatasetValidationError(
                "public occlusion graph is not the complete oriented-boundary relation set"
            )

        if isinstance(instrumentation, SingleOccluderInstrumentation):
            if not isinstance(config, SingleOccluderConfig):
                raise DatasetValidationError(
                    "single-occluder instrumentation/configuration mismatch"
                )
            _require_occlusion_oracle(
                resolved_root,
                transition,
                instrumentation,
                (frame_arrays[0][2], frame_arrays[1][2]),
                expected_raster_shape,
                registry,
            )
        else:
            _require_corridor_raw_segmentation(
                resolved_root,
                transition,
                instrumentation,
                (frame_arrays[0][2], frame_arrays[1][2]),
                expected_raster_shape,
                registry,
            )

        derived_visibility, derived_correspondence, derived_mask_changes = derive_visibility(
            frame_arrays[0][2], frame_arrays[1][2], transition.surfaces
        )
        if derived_visibility != transition.visibility_states:
            raise DatasetValidationError("visibility-state annotations are inconsistent")
        if derived_correspondence != transition.region_correspondence:
            raise DatasetValidationError("region-correspondence annotations are inconsistent")
        if derived_mask_changes != transition.region_mask_changes:
            raise DatasetValidationError("region mask-change annotations are inconsistent")
        derived_boundaries = (
            derive_boundary_structure(frame_arrays[0][2], transition.surfaces, 0),
            derive_boundary_structure(frame_arrays[1][2], transition.surfaces, 1),
        )
        if derived_boundaries != transition.boundary_structures:
            raise DatasetValidationError("boundary annotations are inconsistent")
        reconstructed_raw = (
            _reconstruct_raw_segmentation(transition, instrumentation, frame_arrays[0][2]),
            _reconstruct_raw_segmentation(transition, instrumentation, frame_arrays[1][2]),
        )
        expected_diagnostics = _expected_analytic_transport_diagnostics(
            expected_analytic_transport,
            reconstructed_raw,
        )
        if instrumentation.analytic_transport_diagnostics != expected_diagnostics:
            raise DatasetValidationError("analytic transport diagnostics are inconsistent")
    return manifest
