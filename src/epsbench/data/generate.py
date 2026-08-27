"""Deterministic dataset generation for the authorised Gate 0B scene families."""

from __future__ import annotations

import json
import os
import platform
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image

from epsbench import __version__
from epsbench.annotations import (
    AFTER_EVENT_CODE_DOMAIN,
    ANALYTIC_BOUNDARY_RULE,
    ANALYTIC_BOUNDARY_WIDTH_PIXELS,
    ANALYTIC_SURFACE_INTERSECTION_RULE,
    ANALYTIC_TRANSPORT_METHOD,
    ATTACHMENT_CONTACT_MANIFOLD_RULE,
    ATTACHMENT_EDGE_ASSOCIATION_RULE,
    ATTACHMENT_ENDPOINT_TIE_RULE,
    ATTACHMENT_MULTI_SURFACE_RULE,
    ATTACHMENT_PROJECTION_CONVENTION,
    ATTACHMENT_PUBLIC_CONTRACT_VERSION,
    ATTACHMENT_RULE,
    BEFORE_EVENT_CODE_DOMAIN,
    BOUNDARY_KIND_DOMAIN,
    COUNTERFACTUAL_CONTINUATION_RULE,
    COUNTERFACTUAL_TIE_RULE,
    EDGE_LATTICE_CONVENTION,
    FINITE_PLANE_EDGE_BINARY64_EPSILON,
    FINITE_PLANE_EDGE_COMPARISON_RULE,
    FINITE_PLANE_EDGE_MINIMUM_TOLERANCE_SCALE,
    FINITE_PLANE_EDGE_TOLERANCE_MULTIPLIER,
    FINITE_PLANE_EXTENT_RULE,
    FLOW_FIXED_POINT_SCALE,
    FLOW_QUANTISATION_ROUNDING,
    JUNCTION_AMBIGUITY_RULE,
    ORIENTED_BOUNDARY_METHOD,
    OWNER_SIDE_DOMAIN,
    RAY_DIRECTION_EPSILON,
    SILHOUETTE_RULE,
    SUPPORTED_CONTACT_MANIFOLD_TYPES,
    TARGET_VISIBILITY_RULE,
    VISIBILITY_EVENT_METHOD,
    VISIBILITY_MINIMUM_TOLERANCE_SCALE,
    VISIBILITY_RELATIVE_TOLERANCE,
    AnalyticTransportArrays,
    DirectionalTransportArrays,
    RawBoundaryVisibilityAnalysis,
    derive_boundary_structure,
    derive_visibility,
)
from epsbench.config import BenchmarkConfig, CorridorConfig, SingleOccluderConfig
from epsbench.data.identity import (
    compute_analytic_transport_hash,
    compute_boundary_numerical_contract_hash,
    compute_content_provenance_binding,
    compute_corridor_scene_content_hash,
    compute_dataset_logical_hash,
    compute_ecological_label_hash,
    compute_oriented_boundary_hash,
    compute_renderer_execution_provenance_hash,
    compute_single_occluder_scene_content_hash,
    compute_source_provenance_hash,
    compute_visibility_event_hash,
)
from epsbench.data.provenance import collect_source_provenance
from epsbench.schema import (
    Action,
    AnalyticBoundaryAmbiguityRule,
    AnalyticIntersectionVisibilityContract,
    AnalyticRendererFrameDiagnostic,
    AnalyticTransportDiagnostics,
    ArtifactRecord,
    AvailableDenseOpticalTransport,
    AvailableEcologicalVisibilityEvents,
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
    EpisodeManifest,
    FrameRecord,
    GenerationSeeds,
    Modality,
    OccludingVisibilityEventSummary,
    OcclusionFrameEvidence,
    OcclusionOracleEvidence,
    OcclusionRelation,
    OpticalTransportCoordinateConvention,
    OpticalTransportQuantisation,
    OrientedBoundaryElement,
    PrivilegedAttachmentContractEvidence,
    PrivilegedAttachmentPairEvidence,
    PrivilegedBoundaryElementEvidence,
    RawSegmentationFrameEvidence,
    RendererProvenance,
    SceneFamily,
    SingleOccluderInstrumentation,
    SurfaceReference,
    TransitionRecord,
    UnavailableComponentTopologyCapability,
    VisibilityEventCapabilities,
    WholeSurfaceEventKind,
    WholeSurfaceVisibilityEvent,
)
from epsbench.sim import (
    CORRIDOR_SURFACE_NAMES,
    corridor_generation_seeds,
    render_corridor_transition,
    render_single_occluder_transition,
    sample_corridor_geometry,
)
from epsbench.utils.canonical import (
    canonical_json_bytes,
    logical_array_hash,
    sha256_bytes,
    sha256_file,
    write_canonical_json,
)
from epsbench.utils.seeding import derive_seed, rng_for

_SURFACE_NAMES = ("support_surface", "occluding_surface", "background_surface")


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _array_artifact(
    path: Path,
    root: Path,
    array: np.ndarray[Any, Any],
    modality: Modality,
    media_type: str,
) -> ArtifactRecord:
    return ArtifactRecord(
        path=_relative(path, root),
        modality=modality,
        media_type=media_type,
        dtype=str(array.dtype),
        shape=tuple(array.shape),
        logical_sha256=logical_array_hash(array),
        file_sha256=sha256_file(path),
        byte_count=path.stat().st_size,
    )


def _json_artifact(
    path: Path,
    root: Path,
    payload: Any,
    modality: Modality,
    media_type: str,
) -> ArtifactRecord:
    logical_bytes = canonical_json_bytes(payload)
    return ArtifactRecord(
        path=_relative(path, root),
        modality=modality,
        media_type=media_type,
        dtype="json",
        shape=(len(logical_bytes),),
        logical_sha256=sha256_bytes(logical_bytes),
        file_sha256=sha256_file(path),
        byte_count=path.stat().st_size,
    )


def _surface_references(
    root_seed: int,
    episode_index: int,
) -> tuple[tuple[SurfaceReference, ...], dict[str, SurfaceReference]]:
    rng = rng_for(root_seed, f"surface-remapping:{episode_index}")
    permutation = rng.permutation(len(_SURFACE_NAMES))
    labels: set[int] = set()
    while len(labels) < len(_SURFACE_NAMES):
        labels.add(int(rng.integers(1, 2**31 - 1)))
    label_list = sorted(labels)
    by_name: dict[str, SurfaceReference] = {}
    ordered: list[SurfaceReference] = []
    for output_index, surface_index in enumerate(permutation.tolist()):
        name = _SURFACE_NAMES[surface_index]
        token = rng.bytes(8).hex()
        reference = SurfaceReference(
            surface_id=f"surface-{token}",
            segmentation_label=label_list[output_index],
        )
        by_name[name] = reference
        ordered.append(reference)
    return tuple(ordered), by_name


def _corridor_surface_references(
    remapping_seed: int,
) -> tuple[tuple[SurfaceReference, ...], dict[str, SurfaceReference]]:
    rng = np.random.default_rng(remapping_seed)
    permutation = rng.permutation(len(CORRIDOR_SURFACE_NAMES))
    labels: set[int] = set()
    while len(labels) < len(CORRIDOR_SURFACE_NAMES):
        labels.add(int(rng.integers(1, 2**31 - 1)))
    label_list = sorted(labels)
    by_name: dict[str, SurfaceReference] = {}
    ordered: list[SurfaceReference] = []
    for output_index, surface_index in enumerate(permutation.tolist()):
        name = CORRIDOR_SURFACE_NAMES[surface_index]
        reference = SurfaceReference(
            surface_id=f"surface-{rng.bytes(8).hex()}",
            segmentation_label=label_list[output_index],
        )
        by_name[name] = reference
        ordered.append(reference)
    return tuple(ordered), by_name


def _remap_segmentation(
    raw_segmentation: np.ndarray[Any, Any],
    raw_geom_ids: dict[str, int],
    references: dict[str, SurfaceReference],
) -> np.ndarray[Any, Any]:
    remapped = np.zeros(raw_segmentation.shape, dtype=np.int32)
    for name, raw_id in raw_geom_ids.items():
        remapped[raw_segmentation == raw_id] = references[name].segmentation_label
    return remapped


def _write_frame(
    root: Path,
    episode_directory: Path,
    frame_index: int,
    rgb: np.ndarray[Any, Any],
    depth: np.ndarray[Any, Any],
    segmentation: np.ndarray[Any, Any],
    camera: CameraInstrumentation,
) -> FrameRecord:
    stem = "before" if frame_index == 0 else "after"
    rgb_path = episode_directory / f"rgb_{stem}.png"
    depth_path = episode_directory / f"depth_{stem}.npy"
    segmentation_path = episode_directory / f"segmentation_{stem}.npy"
    camera_path = episode_directory / f"camera_{stem}.json"
    Image.fromarray(rgb, mode="RGB").save(rgb_path, compress_level=9, optimize=False)
    np.save(depth_path, depth, allow_pickle=False)
    np.save(segmentation_path, segmentation, allow_pickle=False)
    write_canonical_json(camera_path, camera)
    return FrameRecord(
        frame_index=frame_index,  # type: ignore[arg-type]
        width=int(rgb.shape[1]),
        height=int(rgb.shape[0]),
        rgb=_array_artifact(rgb_path, root, rgb, Modality.RGB, "image/png"),
        depth=_array_artifact(depth_path, root, depth, Modality.DEPTH, "application/x-npy"),
        segmentation=_array_artifact(
            segmentation_path,
            root,
            segmentation,
            Modality.SURFACE_REGIONS,
            "application/x-npy",
        ),
        camera_world_transform=_json_artifact(
            camera_path,
            root,
            camera,
            Modality.CAMERA_WORLD_TRANSFORM,
            "application/json",
        ),
    )


def _write_transport_direction(
    root: Path,
    episode_directory: Path,
    name: str,
    source_frame_index: int,
    target_frame_index: int,
    arrays: DirectionalTransportArrays,
) -> DirectionalOpticalTransport:
    vectors_path = episode_directory / f"analytic_transport_{name}_vectors_fixed.npy"
    validity_path = episode_directory / f"analytic_transport_{name}_validity.npy"
    reasons_path = episode_directory / f"analytic_transport_{name}_reasons.npy"
    np.save(vectors_path, arrays.vectors_fixed, allow_pickle=False)
    np.save(validity_path, arrays.validity, allow_pickle=False)
    np.save(reasons_path, arrays.reasons, allow_pickle=False)
    return DirectionalOpticalTransport(
        source_frame_index=source_frame_index,  # type: ignore[arg-type]
        target_frame_index=target_frame_index,  # type: ignore[arg-type]
        vectors_fixed=_array_artifact(
            vectors_path,
            root,
            arrays.vectors_fixed,
            Modality.ANALYTIC_OPTICAL_TRANSPORT,
            "application/x-npy",
        ),
        validity=_array_artifact(
            validity_path,
            root,
            arrays.validity,
            Modality.ANALYTIC_OPTICAL_TRANSPORT,
            "application/x-npy",
        ),
        reasons=_array_artifact(
            reasons_path,
            root,
            arrays.reasons,
            Modality.ANALYTIC_OPTICAL_TRANSPORT,
            "application/x-npy",
        ),
    )


def _write_analytic_transport(
    root: Path,
    episode_directory: Path,
    arrays: AnalyticTransportArrays,
) -> AvailableDenseOpticalTransport:
    transport = AvailableDenseOpticalTransport(
        status="available",
        method=ANALYTIC_TRANSPORT_METHOD,
        coordinate_convention=OpticalTransportCoordinateConvention(
            pixel_sample="centre_of_pixel",
            pixel_centre_x="column_plus_0.5",
            pixel_centre_y="row_plus_0.5",
            x_axis="increases_right",
            y_axis="increases_down",
            flow_definition="target_pixel_centre_minus_source_pixel_centre",
            units="image_pixels",
        ),
        quantisation=OpticalTransportQuantisation(
            dtype="int32",
            fixed_point_scale=FLOW_FIXED_POINT_SCALE,
            rounding=FLOW_QUANTISATION_ROUNDING,
        ),
        intersection_visibility=AnalyticIntersectionVisibilityContract(
            surface_intersection_rule=ANALYTIC_SURFACE_INTERSECTION_RULE,
            finite_plane_extent_rule=FINITE_PLANE_EXTENT_RULE,
            finite_plane_edge_comparison_rule=FINITE_PLANE_EDGE_COMPARISON_RULE,
            finite_plane_edge_binary64_epsilon=FINITE_PLANE_EDGE_BINARY64_EPSILON,
            finite_plane_edge_tolerance_multiplier=FINITE_PLANE_EDGE_TOLERANCE_MULTIPLIER,
            finite_plane_edge_minimum_tolerance_scale=(FINITE_PLANE_EDGE_MINIMUM_TOLERANCE_SCALE),
            target_visibility_rule=TARGET_VISIBILITY_RULE,
            visibility_relative_tolerance=VISIBILITY_RELATIVE_TOLERANCE,
            visibility_minimum_tolerance_scale=VISIBILITY_MINIMUM_TOLERANCE_SCALE,
            ray_direction_epsilon=RAY_DIRECTION_EPSILON,
        ),
        boundary_ambiguity=AnalyticBoundaryAmbiguityRule(
            rule=ANALYTIC_BOUNDARY_RULE,
            width_pixels=ANALYTIC_BOUNDARY_WIDTH_PIXELS,
            connectivity="four_neighbour",
            application="source_and_projected_target",
        ),
        reason_code_domain="analytic_transport_reason_codes_v1",
        forward=_write_transport_direction(
            root,
            episode_directory,
            "forward",
            0,
            1,
            arrays.forward,
        ),
        backward=_write_transport_direction(
            root,
            episode_directory,
            "backward",
            1,
            0,
            arrays.backward,
        ),
        analytic_transport_sha256="0" * 64,
    )
    return AvailableDenseOpticalTransport.model_validate(
        {
            **transport.model_dump(mode="python"),
            "analytic_transport_sha256": compute_analytic_transport_hash(transport),
        }
    )


def _surface_by_raw_id(
    raw_geom_ids: dict[str, int],
    references: dict[str, SurfaceReference],
) -> dict[int, SurfaceReference]:
    return {raw_id: references[name] for name, raw_id in raw_geom_ids.items()}


def _write_oriented_boundaries(
    analysis: RawBoundaryVisibilityAnalysis,
    raw_surfaces: dict[int, SurfaceReference],
    width: int,
    height: int,
) -> AvailableOrientedBoundaryOwnership:
    elements = tuple(
        OrientedBoundaryElement(
            frame_index=element.frame_index,  # type: ignore[arg-type]
            axis=BoundaryAxis(element.axis),
            row=element.row,
            column=element.column,
            negative_surface_id=(
                raw_surfaces[element.negative_raw_geom_id].surface_id
                if element.negative_raw_geom_id is not None
                else None
            ),
            positive_surface_id=(
                raw_surfaces[element.positive_raw_geom_id].surface_id
                if element.positive_raw_geom_id is not None
                else None
            ),
            kind=BoundaryKind(element.kind),
            owner_side=BoundaryOwnerSide(element.owner_side),
            owner_surface_id=(
                raw_surfaces[element.owner_raw_geom_id].surface_id
                if element.owner_raw_geom_id is not None
                else None
            ),
        )
        for element in analysis.boundary_elements
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


def _raw_ids_to_labels(
    raw_ids: np.ndarray[Any, Any],
    raw_surfaces: dict[int, SurfaceReference],
) -> np.ndarray[Any, Any]:
    labels = np.zeros(raw_ids.shape, dtype=np.int32)
    for raw_id, surface in raw_surfaces.items():
        labels[raw_ids == raw_id] = np.int32(surface.segmentation_label)
    return labels


def _write_event_direction(
    root: Path,
    episode_directory: Path,
    name: str,
    frame_index: int,
    codes: np.ndarray[Any, Any],
    affected_raw_ids: np.ndarray[Any, Any],
    owner_raw_ids: np.ndarray[Any, Any],
    raw_surfaces: dict[int, SurfaceReference],
) -> DirectionalVisibilityEventMap:
    affected_labels = _raw_ids_to_labels(affected_raw_ids, raw_surfaces)
    owner_labels = _raw_ids_to_labels(owner_raw_ids, raw_surfaces)
    codes_path = episode_directory / f"visibility_events_{name}_codes.npy"
    affected_path = episode_directory / f"visibility_events_{name}_affected_labels.npy"
    owner_path = episode_directory / f"visibility_events_{name}_owner_labels.npy"
    np.save(codes_path, codes, allow_pickle=False)
    np.save(affected_path, affected_labels, allow_pickle=False)
    np.save(owner_path, owner_labels, allow_pickle=False)
    return DirectionalVisibilityEventMap(
        frame_index=frame_index,  # type: ignore[arg-type]
        direction=("before_frame_fate" if frame_index == 0 else "after_frame_origin"),
        event_codes=_array_artifact(
            codes_path,
            root,
            codes,
            Modality.ECOLOGICAL_VISIBILITY_EVENTS,
            "application/x-npy",
        ),
        affected_surface_labels=_array_artifact(
            affected_path,
            root,
            affected_labels,
            Modality.ECOLOGICAL_VISIBILITY_EVENTS,
            "application/x-npy",
        ),
        owner_surface_labels=_array_artifact(
            owner_path,
            root,
            owner_labels,
            Modality.ECOLOGICAL_VISIBILITY_EVENTS,
            "application/x-npy",
        ),
    )


def _write_visibility_events(
    root: Path,
    episode_directory: Path,
    analysis: RawBoundaryVisibilityAnalysis,
    raw_surfaces: dict[int, SurfaceReference],
    boundary: AvailableOrientedBoundaryOwnership,
    transport: AvailableDenseOpticalTransport,
) -> AvailableEcologicalVisibilityEvents:
    events = AvailableEcologicalVisibilityEvents(
        status="available",
        method=VISIBILITY_EVENT_METHOD,
        capabilities=VisibilityEventCapabilities(
            transport_causal_pixel_events="available",
            whole_surface_events="available",
            component_topology=UnavailableComponentTopologyCapability(
                status="unavailable",
                reason_category="component_topology_oracle_not_defined_in_slice_4",
                reason="canonical Slice 4 does not define a component-topology oracle",
            ),
        ),
        before_event_code_domain=BEFORE_EVENT_CODE_DOMAIN,
        after_event_code_domain=AFTER_EVENT_CODE_DOMAIN,
        before_fate=_write_event_direction(
            root,
            episode_directory,
            "before_fate",
            0,
            analysis.before_fate_codes,
            analysis.before_affected_raw_geom_ids,
            analysis.before_owner_raw_geom_ids,
            raw_surfaces,
        ),
        after_origin=_write_event_direction(
            root,
            episode_directory,
            "after_origin",
            1,
            analysis.after_origin_codes,
            analysis.after_affected_raw_geom_ids,
            analysis.after_owner_raw_geom_ids,
            raw_surfaces,
        ),
        occluding_event_summaries=tuple(
            sorted(
                (
                    OccludingVisibilityEventSummary(
                        kind=summary.kind,  # type: ignore[arg-type]
                        affected_surface_id=raw_surfaces[summary.affected_raw_geom_id].surface_id,
                        owner_surface_id=raw_surfaces[summary.owner_raw_geom_id].surface_id,
                        pixel_count=summary.pixel_count,
                    )
                    for summary in analysis.event_summaries
                ),
                key=lambda item: (item.kind, item.affected_surface_id, item.owner_surface_id),
            )
        ),
        whole_surface_events=tuple(
            sorted(
                (
                    WholeSurfaceVisibilityEvent(
                        surface_id=raw_surfaces[item.raw_geom_id].surface_id,
                        kind=WholeSurfaceEventKind(item.kind),
                        before_visible_pixels=item.before_visible_pixels,
                        after_visible_pixels=item.after_visible_pixels,
                    )
                    for item in analysis.whole_surface_events
                ),
                key=lambda item: item.surface_id,
            )
        ),
        oriented_boundary_sha256=boundary.oriented_boundary_sha256,
        analytic_transport_sha256=transport.analytic_transport_sha256,
        visibility_event_sha256="0" * 64,
    )
    return events.model_copy(
        update={"visibility_event_sha256": compute_visibility_event_hash(events)}
    )


def _attachment_contract_evidence(
    analysis: RawBoundaryVisibilityAnalysis,
) -> PrivilegedAttachmentContractEvidence:
    contract = analysis.attachment_contract
    return PrivilegedAttachmentContractEvidence(
        method=contract.method,  # type: ignore[arg-type]
        contact_manifold_rule=contract.contact_manifold_rule,  # type: ignore[arg-type]
        supported_contact_manifold_types=contract.supported_contact_manifold_types,  # type: ignore[arg-type]
        projection_convention=contract.projection_convention,  # type: ignore[arg-type]
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


def _boundary_visibility_diagnostics(
    analysis: RawBoundaryVisibilityAnalysis,
) -> BoundaryVisibilityDiagnostics:
    kind_counts = {kind: 0 for kind in BoundaryKind}
    owner_counts = {side: 0 for side in BoundaryOwnerSide}
    evidence: list[PrivilegedBoundaryElementEvidence] = []
    for element in analysis.boundary_elements:
        kind = BoundaryKind(element.kind)
        side = BoundaryOwnerSide(element.owner_side)
        kind_counts[kind] += 1
        owner_counts[side] += 1
        evidence.append(
            PrivilegedBoundaryElementEvidence(
                frame_index=element.frame_index,  # type: ignore[arg-type]
                axis=BoundaryAxis(element.axis),
                row=element.row,
                column=element.column,
                negative_raw_geom_id=element.negative_raw_geom_id,
                positive_raw_geom_id=element.positive_raw_geom_id,
                kind=kind,
                owner_side=side,
                owner_raw_geom_id=element.owner_raw_geom_id,
                negative_counterfactual_next_raw_geom_id=(
                    element.negative_counterfactual_next_raw_geom_id
                ),
                positive_counterfactual_next_raw_geom_id=(
                    element.positive_counterfactual_next_raw_geom_id
                ),
                on_projected_attachment_locus=(element.on_projected_attachment_locus),
            )
        )
    return BoundaryVisibilityDiagnostics(
        boundary_method=ORIENTED_BOUNDARY_METHOD,
        counterfactual_continuation_rule=COUNTERFACTUAL_CONTINUATION_RULE,
        counterfactual_tie_rule=COUNTERFACTUAL_TIE_RULE,
        counterfactual_ray_direction_epsilon=RAY_DIRECTION_EPSILON,
        junction_ambiguity_rule=JUNCTION_AMBIGUITY_RULE,
        silhouette_rule=SILHOUETTE_RULE,
        visibility_event_method=VISIBILITY_EVENT_METHOD,
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


def _boundary_derived_occlusion(
    boundary: AvailableOrientedBoundaryOwnership,
    oracle_rule: str,
) -> AvailableOcclusionAnnotation:
    frames_by_pair: dict[tuple[str, str], set[int]] = {}
    for element in boundary.elements:
        if element.kind != BoundaryKind.OCCLUDING_CONTOUR:
            continue
        owner = element.owner_surface_id
        if owner is None:
            raise RuntimeError("occluding contour lacks a public owner")
        affected = (
            element.positive_surface_id
            if owner == element.negative_surface_id
            else element.negative_surface_id
        )
        if affected is None:
            raise RuntimeError("occluding contour lacks an affected surface")
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


def _directional_diagnostic(
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


def _analytic_transport_diagnostics(
    arrays: AnalyticTransportArrays,
    raw_before: np.ndarray[Any, Any],
    raw_after: np.ndarray[Any, Any],
) -> AnalyticTransportDiagnostics:
    frame_diagnostics: list[AnalyticRendererFrameDiagnostic] = []
    for frame_index, (assignment, boundary, rendered) in enumerate(
        (
            (
                arrays.before_surface_assignment,
                arrays.before_boundary_ambiguous,
                raw_before,
            ),
            (
                arrays.after_surface_assignment,
                arrays.after_boundary_ambiguous,
                raw_after,
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
        forward=_directional_diagnostic(arrays.forward),
        backward=_directional_diagnostic(arrays.backward),
    )


def _generate_single_occluder_episode(
    root: Path,
    config: SingleOccluderConfig,
    episode_index: int,
) -> EpisodeManifest:
    episode_id = f"episode-{episode_index:06d}"
    episode_seed = derive_seed(config.seed, f"episode:{episode_index}")
    episode_directory = root / "episodes" / episode_id
    episode_directory.mkdir(parents=True)
    rendered = render_single_occluder_transition(config)
    surfaces, references = _surface_references(config.seed, episode_index)
    before_segmentation = _remap_segmentation(
        rendered.before.raw_geom_segmentation,
        rendered.raw_geom_ids,
        references,
    )
    after_segmentation = _remap_segmentation(
        rendered.after.raw_geom_segmentation,
        rendered.raw_geom_ids,
        references,
    )
    camera_before = CameraInstrumentation(
        frame_index=0,
        camera_world_position=rendered.before.camera_world_position,
        camera_world_rotation_row_major=rendered.before.camera_world_rotation_row_major,
    )
    camera_after = CameraInstrumentation(
        frame_index=1,
        camera_world_position=rendered.after.camera_world_position,
        camera_world_rotation_row_major=rendered.after.camera_world_rotation_row_major,
    )
    before = _write_frame(
        root,
        episode_directory,
        0,
        rendered.before.rgb,
        rendered.before.depth,
        before_segmentation,
        camera_before,
    )
    after = _write_frame(
        root,
        episode_directory,
        1,
        rendered.after.rgb,
        rendered.after.depth,
        after_segmentation,
        camera_after,
    )
    visibility, correspondence, mask_changes = derive_visibility(
        before_segmentation,
        after_segmentation,
        surfaces,
    )
    boundaries = (
        derive_boundary_structure(before_segmentation, surfaces, 0),
        derive_boundary_structure(after_segmentation, surfaces, 1),
    )
    analytic_transport = _write_analytic_transport(
        root,
        episode_directory,
        rendered.analytic_transport,
    )
    raw_surfaces = _surface_by_raw_id(rendered.raw_geom_ids, references)
    oriented_boundaries = _write_oriented_boundaries(
        rendered.boundary_visibility,
        raw_surfaces,
        config.render.width,
        config.render.height,
    )
    visibility_events = _write_visibility_events(
        root,
        episode_directory,
        rendered.boundary_visibility,
        raw_surfaces,
        oriented_boundaries,
        analytic_transport,
    )
    occluder_raw_id = rendered.raw_geom_ids["occluding_surface"]
    occluded_raw_id = rendered.raw_geom_ids["background_surface"]
    occlusion_frame_evidence: list[OcclusionFrameEvidence] = []
    relation_frame_indices: list[int] = []
    for frame_index, frame in enumerate((rendered.before, rendered.after)):
        stem = "before" if frame_index == 0 else "after"
        counterfactual_path = episode_directory / f"counterfactual_segmentation_{stem}.npy"
        np.save(
            counterfactual_path,
            frame.counterfactual_raw_geom_segmentation,
            allow_pickle=False,
        )
        counterfactual_artifact = _array_artifact(
            counterfactual_path,
            root,
            frame.counterfactual_raw_geom_segmentation,
            Modality.PRIVILEGED_GENERATION_RECORDS,
            "application/x-npy",
        )
        reveal_mask = (frame.raw_geom_segmentation == occluder_raw_id) & (
            frame.counterfactual_raw_geom_segmentation == occluded_raw_id
        )
        revealed_pixel_count = int(np.count_nonzero(reveal_mask))
        if revealed_pixel_count > 0:
            relation_frame_indices.append(frame_index)
        occlusion_frame_evidence.append(
            OcclusionFrameEvidence(
                frame_index=frame_index,  # type: ignore[arg-type]
                counterfactual_segmentation=counterfactual_artifact,
                revealed_pixel_count=revealed_pixel_count,
                reveal_mask_logical_sha256=logical_array_hash(reveal_mask),
            )
        )
    if not relation_frame_indices:
        raise RuntimeError("counterfactual oracle found no foreground/background occlusion")
    transition = TransitionRecord(
        schema_version="0.1.0-dev.7",
        episode_id=episode_id,
        action=Action(**config.action.model_dump()),
        surfaces=surfaces,
        before=before,
        after=after,
        visibility_states=visibility,
        region_correspondence=correspondence,
        region_mask_changes=mask_changes,
        oriented_boundary_ownership=oriented_boundaries,
        ecological_visibility_events=visibility_events,
        occlusion=_boundary_derived_occlusion(
            oriented_boundaries,
            "oriented_boundary_ownership_with_counterfactual_crosscheck_v1",
        ),
        boundary_structures=boundaries,
        analytic_optical_transport=analytic_transport,
        ecological_label_sha256="0" * 64,
    )
    transition = TransitionRecord.model_validate(
        {
            **transition.model_dump(mode="python"),
            "ecological_label_sha256": compute_ecological_label_hash(transition),
        }
    )
    transition_path = episode_directory / "transition.json"
    write_canonical_json(transition_path, transition)

    instrumentation = SingleOccluderInstrumentation(
        schema_version="0.1.0-dev.7",
        scene_family=SceneFamily.SINGLE_OCCLUDER,
        episode_id=episode_id,
        appearance_variant=config.appearance.variant,
        raw_geom_ids=rendered.raw_geom_ids,
        raw_to_opaque_surface_ids={
            str(raw_id): references[name].surface_id
            for name, raw_id in rendered.raw_geom_ids.items()
        },
        raw_geom_world_positions=rendered.raw_geom_positions,
        raw_geom_compiled_sizes=rendered.raw_geom_compiled_sizes,
        raw_geom_types=rendered.raw_geom_types,  # type: ignore[arg-type]
        raw_geom_world_rotations_row_major=(rendered.raw_geom_world_rotations_row_major),
        occlusion_oracle=OcclusionOracleEvidence(
            rule="counterfactual_occluder_exclusion_v1",
            candidate_occluder_raw_geom_id=occluder_raw_id,
            candidate_occluded_raw_geom_id=occluded_raw_id,
            frames=tuple(occlusion_frame_evidence),  # type: ignore[arg-type]
        ),
        analytic_transport_diagnostics=_analytic_transport_diagnostics(
            rendered.analytic_transport,
            rendered.before.raw_geom_segmentation,
            rendered.after.raw_geom_segmentation,
        ),
        attachment_contract=_attachment_contract_evidence(rendered.boundary_visibility),
        boundary_visibility_diagnostics=_boundary_visibility_diagnostics(
            rendered.boundary_visibility
        ),
    )
    instrumentation_path = episode_directory / "instrumentation.json"
    write_canonical_json(instrumentation_path, instrumentation)
    return EpisodeManifest(
        episode_id=episode_id,
        episode_index=episode_index,
        episode_seed=episode_seed,
        transition=_json_artifact(
            transition_path,
            root,
            transition,
            Modality.TRANSITION_RECORD,
            "application/json",
        ),
        privileged_instrumentation=_json_artifact(
            instrumentation_path,
            root,
            instrumentation,
            Modality.PRIVILEGED_GENERATION_RECORDS,
            "application/json",
        ),
        scene_content_sha256=compute_single_occluder_scene_content_hash(config),
        ecological_label_sha256=transition.ecological_label_sha256,
        analytic_transport_sha256=analytic_transport.analytic_transport_sha256,
        oriented_boundary_sha256=oriented_boundaries.oriented_boundary_sha256,
        visibility_event_sha256=visibility_events.visibility_event_sha256,
        rgb_logical_sha256=(before.rgb.logical_sha256, after.rgb.logical_sha256),
    )


def _generate_corridor_episode(
    root: Path,
    config: CorridorConfig,
    episode_index: int,
) -> EpisodeManifest:
    episode_id = f"episode-{episode_index:06d}"
    episode_seed = derive_seed(config.seed, f"episode:{episode_index}")
    geometry_seed, remapping_seed, appearance_seed = corridor_generation_seeds(episode_seed)
    geometry = sample_corridor_geometry(config, episode_seed)
    episode_directory = root / "episodes" / episode_id
    episode_directory.mkdir(parents=True)
    rendered = render_corridor_transition(config, geometry, appearance_seed)
    surfaces, references = _corridor_surface_references(remapping_seed)
    before_segmentation = _remap_segmentation(
        rendered.before.raw_geom_segmentation,
        rendered.raw_geom_ids,
        references,
    )
    after_segmentation = _remap_segmentation(
        rendered.after.raw_geom_segmentation,
        rendered.raw_geom_ids,
        references,
    )
    camera_before = CameraInstrumentation(
        frame_index=0,
        camera_world_position=rendered.before.camera_world_position,
        camera_world_rotation_row_major=rendered.before.camera_world_rotation_row_major,
    )
    camera_after = CameraInstrumentation(
        frame_index=1,
        camera_world_position=rendered.after.camera_world_position,
        camera_world_rotation_row_major=rendered.after.camera_world_rotation_row_major,
    )
    before = _write_frame(
        root,
        episode_directory,
        0,
        rendered.before.rgb,
        rendered.before.depth,
        before_segmentation,
        camera_before,
    )
    after = _write_frame(
        root,
        episode_directory,
        1,
        rendered.after.rgb,
        rendered.after.depth,
        after_segmentation,
        camera_after,
    )
    visibility, correspondence, mask_changes = derive_visibility(
        before_segmentation,
        after_segmentation,
        surfaces,
    )
    boundaries = (
        derive_boundary_structure(before_segmentation, surfaces, 0),
        derive_boundary_structure(after_segmentation, surfaces, 1),
    )
    analytic_transport = _write_analytic_transport(
        root,
        episode_directory,
        rendered.analytic_transport,
    )
    raw_surfaces = _surface_by_raw_id(rendered.raw_geom_ids, references)
    oriented_boundaries = _write_oriented_boundaries(
        rendered.boundary_visibility,
        raw_surfaces,
        config.render.width,
        config.render.height,
    )
    visibility_events = _write_visibility_events(
        root,
        episode_directory,
        rendered.boundary_visibility,
        raw_surfaces,
        oriented_boundaries,
        analytic_transport,
    )
    transition = TransitionRecord(
        schema_version="0.1.0-dev.7",
        episode_id=episode_id,
        action=Action(**config.action.model_dump()),
        surfaces=surfaces,
        before=before,
        after=after,
        visibility_states=visibility,
        region_correspondence=correspondence,
        region_mask_changes=mask_changes,
        oriented_boundary_ownership=oriented_boundaries,
        ecological_visibility_events=visibility_events,
        occlusion=_boundary_derived_occlusion(
            oriented_boundaries,
            "oriented_boundary_ownership_complete_v2",
        ),
        boundary_structures=boundaries,
        analytic_optical_transport=analytic_transport,
        ecological_label_sha256="0" * 64,
    )
    transition = TransitionRecord.model_validate(
        {
            **transition.model_dump(mode="python"),
            "ecological_label_sha256": compute_ecological_label_hash(transition),
        }
    )
    transition_path = episode_directory / "transition.json"
    write_canonical_json(transition_path, transition)

    generation_seeds = GenerationSeeds(
        episode_seed=episode_seed,
        geometry_sampling_seed=geometry_seed,
        surface_remapping_seed=remapping_seed,
        appearance_seed=appearance_seed,
    )
    raw_segmentation_evidence: list[RawSegmentationFrameEvidence] = []
    for frame_index, frame in enumerate((rendered.before, rendered.after)):
        stem = "before" if frame_index == 0 else "after"
        raw_segmentation_path = episode_directory / f"raw_segmentation_{stem}.npy"
        np.save(raw_segmentation_path, frame.raw_geom_segmentation, allow_pickle=False)
        raw_segmentation_evidence.append(
            RawSegmentationFrameEvidence(
                frame_index=frame_index,  # type: ignore[arg-type]
                raw_segmentation=_array_artifact(
                    raw_segmentation_path,
                    root,
                    frame.raw_geom_segmentation,
                    Modality.PRIVILEGED_GENERATION_RECORDS,
                    "application/x-npy",
                ),
            )
        )
    instrumentation = CorridorInstrumentation(
        schema_version="0.1.0-dev.7",
        scene_family=SceneFamily.CORRIDOR,
        episode_id=episode_id,
        appearance_variant=config.appearance.variant,
        apparatus_surface_names=CORRIDOR_SURFACE_NAMES,
        raw_geom_ids=rendered.raw_geom_ids,
        raw_to_opaque_surface_ids={
            str(raw_id): references[name].surface_id
            for name, raw_id in rendered.raw_geom_ids.items()
        },
        raw_geom_world_positions=rendered.raw_geom_positions,
        raw_geom_compiled_sizes=rendered.raw_geom_compiled_sizes,
        raw_geom_types=rendered.raw_geom_types,  # type: ignore[arg-type]
        raw_geom_world_rotations_row_major=(rendered.raw_geom_world_rotations_row_major),
        sampled_geometry=geometry,
        camera_before=camera_before,
        camera_after=camera_after,
        generation_seeds=generation_seeds,
        raw_segmentation_frames=tuple(raw_segmentation_evidence),  # type: ignore[arg-type]
        geometry_sampling_rule="uniform_width_length_v1",
        appearance_rule="solid_colour_variant_v1",
        analytic_transport_diagnostics=_analytic_transport_diagnostics(
            rendered.analytic_transport,
            rendered.before.raw_geom_segmentation,
            rendered.after.raw_geom_segmentation,
        ),
        attachment_contract=_attachment_contract_evidence(rendered.boundary_visibility),
        boundary_visibility_diagnostics=_boundary_visibility_diagnostics(
            rendered.boundary_visibility
        ),
    )
    instrumentation_path = episode_directory / "instrumentation.json"
    write_canonical_json(instrumentation_path, instrumentation)
    scene_content_sha256 = compute_corridor_scene_content_hash(config, geometry)
    return EpisodeManifest(
        episode_id=episode_id,
        episode_index=episode_index,
        episode_seed=episode_seed,
        transition=_json_artifact(
            transition_path,
            root,
            transition,
            Modality.TRANSITION_RECORD,
            "application/json",
        ),
        privileged_instrumentation=_json_artifact(
            instrumentation_path,
            root,
            instrumentation,
            Modality.PRIVILEGED_GENERATION_RECORDS,
            "application/json",
        ),
        scene_content_sha256=scene_content_sha256,
        ecological_label_sha256=transition.ecological_label_sha256,
        analytic_transport_sha256=analytic_transport.analytic_transport_sha256,
        oriented_boundary_sha256=oriented_boundaries.oriented_boundary_sha256,
        visibility_event_sha256=visibility_events.visibility_event_sha256,
        rgb_logical_sha256=(before.rgb.logical_sha256, after.rgb.logical_sha256),
    )


def _generate_episode(
    root: Path,
    config: BenchmarkConfig,
    episode_index: int,
) -> EpisodeManifest:
    if isinstance(config, SingleOccluderConfig):
        return _generate_single_occluder_episode(root, config, episode_index)
    if isinstance(config, CorridorConfig):
        return _generate_corridor_episode(root, config, episode_index)
    raise TypeError(f"unsupported scene configuration: {type(config).__name__}")


def _renderer_provenance() -> RendererProvenance:
    configured_backend = os.environ.get("MUJOCO_GL")
    if configured_backend is None:
        configured_backend = "wgl-default" if platform.system() == "Windows" else "platform-default"
    return RendererProvenance(
        mujoco_version=mujoco.__version__,
        numpy_version=np.__version__,
        renderer="mujoco.Renderer",
        backend=configured_backend,
        operating_system=platform.system(),
    )


def generate_dataset(config: BenchmarkConfig, episodes: int, output: Path) -> DatasetManifest:
    """Generate a new dataset directory, refusing to overwrite existing content."""

    if episodes < 1:
        raise ValueError("episodes must be at least one")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    source_provenance = collect_source_provenance(Path.cwd())
    source_provenance_sha256 = compute_source_provenance_hash(source_provenance)
    output.mkdir(parents=True, exist_ok=True)
    resolved_config_path = output / "resolved_config.json"
    write_canonical_json(resolved_config_path, config)
    resolved_config_artifact = _json_artifact(
        resolved_config_path,
        output,
        config,
        Modality.PRIVILEGED_GENERATION_RECORDS,
        "application/json",
    )
    episode_manifests = tuple(
        _generate_episode(output, config, episode_index) for episode_index in range(episodes)
    )
    renderer_provenance = _renderer_provenance()
    renderer_execution_provenance_sha256 = compute_renderer_execution_provenance_hash(
        renderer_provenance
    )
    manifest = DatasetManifest(
        schema_version="0.1.0-dev.4",
        generator_version="0.1.0",
        scene_family=config.scene_family,
        root_seed=config.seed,
        config_logical_sha256=sha256_bytes(canonical_json_bytes(config)),
        appearance_variant=config.appearance.variant,
        resolved_config=resolved_config_artifact,
        renderer_provenance=renderer_provenance,
        renderer_execution_provenance_sha256=renderer_execution_provenance_sha256,
        episodes=episode_manifests,
        dataset_logical_sha256="0" * 64,
        source_provenance=source_provenance,
        source_provenance_sha256=source_provenance_sha256,
        content_provenance_binding_sha256="0" * 64,
    )
    dataset_logical_sha256 = compute_dataset_logical_hash(manifest)
    manifest = DatasetManifest.model_validate(
        {
            **manifest.model_dump(mode="python"),
            "dataset_logical_sha256": dataset_logical_sha256,
            "content_provenance_binding_sha256": compute_content_provenance_binding(
                dataset_logical_sha256,
                source_provenance_sha256,
                renderer_execution_provenance_sha256,
            ),
        }
    )
    write_canonical_json(output / "manifest.json", manifest)
    volatile_metadata = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "hostname": socket.gethostname(),
        "python": sys.version,
        "generator_version": __version__,
    }
    (output / "run.json").write_text(
        json.dumps(volatile_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
