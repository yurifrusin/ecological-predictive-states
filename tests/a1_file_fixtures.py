"""Synthetic file/schema bindings only: no renderer or empirical correctness claims."""

from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from pydantic import BaseModel, TypeAdapter

from epsbench.config import SingleOccluderConfig
from epsbench.data.dataset_identity import (
    compute_content_provenance_binding,
    compute_dataset_logical_hash,
    compute_renderer_execution_provenance_hash,
    compute_source_provenance_hash,
)
from epsbench.schema import (
    ArtifactRecord,
    AvailableEcologicalVisibilityEvents,
    AvailableOrientedBoundaryOwnership,
    CanonicalPairedEpisodeManifest,
    CanonicalPairedFrameRecord,
    CanonicalPairedOutputProvenance,
    DatasetManifest,
    Modality,
    RendererProvenance,
    SourceProvenance,
    TransitionRecord,
)
from epsbench.utils.canonical import canonical_json_bytes, logical_array_hash, sha256_bytes
from epsbench.utils.seeding import derive_seed

_Model = TypeVar("_Model", bound=BaseModel)


def _model(model: type[_Model], **value: Any) -> _Model:
    return model.model_validate_json(
        canonical_json_bytes(TypeAdapter(dict[str, Any]).dump_python(value, mode="json"))
    )


ZERO = "0" * 64
EPISODE = "episode-000000"


def _hash(value: BaseModel | dict[str, Any] | list[Any]) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def make_dataset(
    root: Path,
    config: SingleOccluderConfig,
    *,
    future_summary_delta: int = 0,
    episode_seed: int | None = None,
) -> Path:
    """Write handcrafted public transition files bound to one fixed config.

    Paired provenance carries required schema literals but explicitly synthetic
    source, backend and state. Native arrays and instrumentation are unusable as
    physical evidence; no rendering, dataset inspection or A1 launch occurs.
    future_summary_delta changes only a schema-valid inline future count, for
    projection-invariance checks; it does not describe a physical scene.
    """
    root.mkdir(parents=True, exist_ok=False)
    (root / EPISODE).mkdir()
    recorded_episode_seed = (
        derive_seed(config.seed, "episode:0") if episode_seed is None else episode_seed
    )

    def json_file(name: str, value: Any, modality: Modality) -> ArtifactRecord:
        logical = canonical_json_bytes(value)
        payload = logical + b"\n"
        (root / name).write_bytes(payload)
        return _model(
            ArtifactRecord,
            path=name,
            modality=modality,
            media_type="application/json",
            dtype="json",
            shape=(len(logical),),
            logical_sha256=sha256_bytes(logical),
            file_sha256=sha256_bytes(payload),
            byte_count=len(payload),
        )

    def array_file(name: str, array: Any, modality: Modality) -> ArtifactRecord:
        path = root / name
        np.save(path, array, allow_pickle=False)
        payload = path.read_bytes()
        return _model(
            ArtifactRecord,
            path=name,
            modality=modality,
            media_type="application/x-npy",
            dtype=str(array.dtype),
            shape=tuple(array.shape),
            logical_sha256=logical_array_hash(array),
            file_sha256=sha256_bytes(payload),
            byte_count=len(payload),
        )

    config_hash = _hash(config)
    fake = _hash({"synthetic_fixture": "a1_file_schema_v1", "config": config_hash})
    source = _model(
        SourceProvenance,
        git_repository=None,
        git_commit=None,
        git_dirty=None,
        dirty_diff_sha256=None,
        git_availability_status="unavailable",
        git_unavailable_reason="synthetic fixture; no checkout capture",
        uv_lock_sha256=fake,
        research_charter_sha256=fake,
        eps_bench_spec_sha256=fake,
        milestone_plan_sha256=fake,
        codex_handoff_sha256=fake,
        package_version="synthetic_fixture_v1",
        python_version="synthetic_not_captured",
    )
    renderer = _model(
        RendererProvenance,
        mujoco_version="synthetic_not_executed",
        numpy_version=np.__version__,
        renderer="mujoco.Renderer",
        backend="synthetic_handcrafted_no_renderer",
        operating_system="synthetic_fixture",
    )
    source_hash = compute_source_provenance_hash(source)
    renderer_hash = compute_renderer_execution_provenance_hash(renderer)
    first, second = "surface-" + config_hash[:16], "surface-" + config_hash[16:32]
    surfaces = [
        {"surface_id": first, "segmentation_label": 17},
        {"surface_id": second, "segmentation_label": 42},
    ]
    image = np.full((120, 160), 17, dtype=np.int32)
    image[:, 60:100] = 42
    frames, pair_hashes = [], []
    private = Modality.PRIVILEGED_GENERATION_RECORDS
    for index in (0, 1):
        prefix = f"{EPISODE}/"
        segmentation = array_file(
            f"{prefix}segmentation_{index}.npy", image, Modality.SURFACE_REGIONS
        )
        depth = array_file(
            f"{prefix}depth_{index}.npy", np.ones((120, 160), dtype=np.float32), Modality.DEPTH
        )
        rgb = array_file(
            f"{prefix}rgb_{index}.npy", np.zeros((120, 160, 3), dtype=np.uint8), Modality.RGB
        )
        camera = json_file(
            f"{prefix}camera_{index}.json",
            {
                "frame_index": index,
                "camera_world_position": [0.0, 0.0, 0.0],
                "camera_world_rotation_row_major": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            },
            Modality.CAMERA_WORLD_TRANSFORM,
        )
        native_id = array_file(
            f"{prefix}native_id_{index}.npy", np.zeros((120, 160, 3), dtype=np.uint8), private
        )
        native_depth = array_file(
            f"{prefix}native_depth_{index}.npy", np.zeros((120, 160), dtype=np.float32), private
        )
        state = json_file(
            f"{prefix}producer_state_{index}.json",
            {"synthetic": True, "purpose": "file/schema fixture only; no native capture"},
            private,
        )
        separate = "separate_draw_no_id_depth_correspondence"
        pair = _model(
            CanonicalPairedOutputProvenance,
            schema_version="canonical_paired_output_provenance/v1",
            association="shared_raster_id_depth",
            episode_id=EPISODE,
            frame_index=index,
            episode_seed=recorded_episode_seed,
            scene_family=config.scene_family,
            config_logical_sha256=config_hash,
            scene_content_sha256=fake,
            rgb_provenance={
                "producer": "ordinary_rgb",
                "association": separate,
                "artifact_logical_sha256": rgb.logical_sha256,
            },
            counterfactual_provenance={
                "producer": "single_occluder_counterfactual_segmentation",
                "association": separate,
                "artifact_logical_sha256": segmentation.logical_sha256,
            },
            depth_conversion="mujoco_3.12_float32_coefficients_float64_inverse_float32_output",
            orientation="native_bottom_up_then_common_vertical_flip_to_image_top_down",
            native_readback_atomicity="sequential_color_then_depth_not_hardware_atomic",
            native_id_rgb=native_id,
            native_depth_pre_metric=native_depth,
            producer_state=state,
            scene_map=(
                {"segid_plus_one": 1, "objid": 1, "objtype": 5},
                {"segid_plus_one": 2, "objid": 2, "objtype": 5},
            ),
            raw_to_opaque_surface_ids={"1": first, "2": second},
            opaque_surface_labels={first: 17, second: 42},
            canonical_segmentation_logical_sha256=segmentation.logical_sha256,
            canonical_depth_logical_sha256=depth.logical_sha256,
            canonical_segmentation_file_sha256=segmentation.file_sha256,
            canonical_depth_file_sha256=depth.file_sha256,
            near=0.1,
            far=10.0,
            source_provenance_sha256=source_hash,
            renderer_execution_provenance_sha256=renderer_hash,
            endpoint_logical_sha256=ZERO,
        )
        pair_domain = pair.model_dump(mode="json")
        pair_domain.pop("endpoint_logical_sha256")
        pair = pair.model_copy(update={"endpoint_logical_sha256": _hash(pair_domain)})
        pair_hashes.append(pair.endpoint_logical_sha256)
        frames.append(
            _model(
                CanonicalPairedFrameRecord,
                frame_index=index,
                width=160,
                height=120,
                rgb=rgb,
                depth=depth,
                segmentation=segmentation,
                camera_world_transform=camera,
                paired_output_provenance=json_file(f"{prefix}paired_{index}.json", pair, private),
            )
        )
    boundary = _model(
        AvailableOrientedBoundaryOwnership,
        status="available",
        method="analytic_oriented_boundary_ownership_v4",
        raster_width=160,
        raster_height=120,
        coordinate_convention={
            "version": "four_neighbour_sample_edge_lattice_v1",
            "horizontal_negative_sample": "pixel_centre_row_column_left",
            "horizontal_positive_sample": "pixel_centre_row_column_plus_1_right",
            "horizontal_shape": "height_by_width_minus_1",
            "vertical_negative_sample": "pixel_centre_row_column_top",
            "vertical_positive_sample": "pixel_centre_row_plus_1_column_bottom",
            "vertical_shape": "height_minus_1_by_width",
            "no_boundary_representation": "implicit_by_absent_sparse_record",
        },
        boundary_kind_domain="oriented_boundary_kind_domain_v1",
        owner_side_domain="oriented_boundary_owner_side_domain_v1",
        attachment_rule="projected_compiled_contact_locus_v3",
        attachment_public_contract_version="scene_attachment_public_contract_v4",
        attachment_contact_manifold_rule="compiled_axis_aligned_intersection_cell_v1",
        attachment_supported_contact_manifold_types=(
            "point",
            "axis_aligned_segment",
            "axis_aligned_rectangle",
            "axis_aligned_overlap_volume",
        ),
        attachment_projection_convention="analytic_pinhole_pixel_centre_v1",
        attachment_projection_in_front_rule="strict_forward_distance_greater_than_epsilon_v1",
        attachment_feasibility_rule="image_constraints_only_slack_strict_front_and_cell_bounds_exact_v1",
        attachment_edge_lattice_association_rule="sample_connection_segment_intersects_projected_contact_cell_v1",
        attachment_endpoint_tie_rule="inclusive_contact_endpoints_v1",
        attachment_multi_surface_rule="multi_surface_ambiguity_precedes_attachment_v1",
        numerical_contract_sha256=fake,
        counterfactual_continuation_rule="counterfactual_nearest_surface_continuation_v1",
        counterfactual_tie_rule="exactly_one_side_continues_v1",
        junction_ambiguity_rule="edge_incident_3x2_or_2x3_multi_assignment_v2",
        silhouette_rule="controlled_to_uncontrolled_side_owns_v1",
        elements=(
            {
                "frame_index": 0,
                "axis": "horizontal",
                "row": 0,
                "column": 59,
                "negative_surface_id": first,
                "positive_surface_id": second,
                "kind": "occluding_contour",
                "owner_side": "positive_axis_side",
                "owner_surface_id": second,
            },
            {
                "frame_index": 0,
                "axis": "horizontal",
                "row": 0,
                "column": 99,
                "negative_surface_id": second,
                "positive_surface_id": first,
                "kind": "occluding_contour",
                "owner_side": "negative_axis_side",
                "owner_surface_id": second,
            },
        ),
        oriented_boundary_sha256=ZERO,
    )
    boundary_domain = boundary.model_dump(mode="json")
    boundary_domain.pop("status")
    boundary_domain.pop("oriented_boundary_sha256")
    boundary_domain["frame_indices"] = [0, 1]
    boundary = boundary.model_copy(update={"oriented_boundary_sha256": _hash(boundary_domain)})
    directions = []
    for index, name in enumerate(("before_fate", "after_origin")):
        codes = np.zeros((120, 160), dtype=np.uint8)
        codes[:, 58:60] = 1
        directions.append(
            {
                "frame_index": index,
                "direction": "before_frame_fate" if index == 0 else "after_frame_origin",
                "event_codes": array_file(
                    f"{EPISODE}/{name}_codes.npy", codes, Modality.ECOLOGICAL_VISIBILITY_EVENTS
                ),
                "affected_surface_labels": array_file(
                    f"{EPISODE}/{name}_affected.npy", image, Modality.ECOLOGICAL_VISIBILITY_EVENTS
                ),
                "owner_surface_labels": array_file(
                    f"{EPISODE}/{name}_owner.npy",
                    np.zeros_like(image),
                    Modality.ECOLOGICAL_VISIBILITY_EVENTS,
                ),
            }
        )
    counts = {first: 14400, second: 4800}
    whole = [
        {
            "surface_id": identifier,
            "kind": "persistently_visible",
            "before_visible_pixels": counts[identifier],
            "after_visible_pixels": counts[identifier],
        }
        for identifier in sorted(counts)
    ]
    events = _model(
        AvailableEcologicalVisibilityEvents,
        status="available",
        method="analytic_transport_boundary_causal_events_v2",
        capabilities={
            "transport_causal_pixel_events": "available",
            "whole_surface_events": "available",
            "component_topology": {
                "status": "unavailable",
                "reason_category": "component_topology_oracle_not_defined_in_slice_4",
                "reason": "canonical Slice 4 does not define a component-topology oracle",
            },
        },
        before_event_code_domain="before_frame_fate_codes_v1",
        after_event_code_domain="after_frame_origin_codes_v1",
        before_fate=directions[0],
        after_origin=directions[1],
        occluding_event_summaries=(),
        whole_surface_events=whole,
        oriented_boundary_sha256=boundary.oriented_boundary_sha256,
        analytic_transport_sha256=fake,
        visibility_event_sha256=ZERO,
    )
    event_domain = events.model_dump(mode="json")
    event_domain.pop("status")
    event_domain.pop("visibility_event_sha256")
    for name in ("before_fate", "after_origin"):
        direction = event_domain[name]
        for field in ("event_codes", "affected_surface_labels", "owner_surface_labels"):
            direction[field + "_logical_sha256"] = direction.pop(field)["logical_sha256"]
    event_domain["frame_indices"] = [0, 1]
    events = events.model_copy(update={"visibility_event_sha256": _hash(event_domain)})
    transition = _model(
        TransitionRecord,
        schema_version="0.1.0-dev.9",
        episode_id=EPISODE,
        action=config.action.model_dump(mode="json"),
        surfaces=surfaces,
        before=frames[0],
        after=frames[1],
        visibility_states=[
            {
                "surface_id": identifier,
                "before_visible_pixels": count,
                "after_visible_pixels": count,
                "before_projected_image_fraction": count / 19200,
                "after_projected_image_fraction": count / 19200,
            }
            for identifier, count in counts.items()
        ],
        region_correspondence=[
            {
                "surface_id": identifier,
                "before_visible_pixels": count,
                "after_visible_pixels": count + future_summary_delta,
                "same_image_coordinate_overlap_pixels": count,
            }
            for identifier, count in counts.items()
        ],
        region_mask_changes=[
            {"surface_id": identifier, "change": "mask_unchanged", "affected_image_pixels": 0}
            for identifier in counts
        ],
        oriented_boundary_ownership=boundary,
        ecological_visibility_events=events,
        occlusion={
            "status": "available",
            "oracle_rule": "oriented_boundary_ownership_complete_v2",
            "relations": [],
        },
        boundary_structures=(
            {"frame_index": 0, "total_boundary_pixels": 0, "contacts": []},
            {"frame_index": 1, "total_boundary_pixels": 0, "contacts": []},
        ),
        analytic_optical_transport={
            "status": "unavailable",
            "reason_category": "analytic_transport_unavailable",
            "reason": "synthetic file fixture; no physical transport",
        },
        ecological_label_sha256=ZERO,
    )
    label_domain = transition.model_dump(mode="json")
    for field in ("episode_id", "before", "after", "ecological_label_sha256"):
        label_domain.pop(field)
    label_domain["segmentation_logical_sha256"] = [
        frame.segmentation.logical_sha256 for frame in frames
    ]
    label_domain["oriented_boundary_ownership"] = {
        **boundary_domain,
        "oriented_boundary_sha256": boundary.oriented_boundary_sha256,
    }
    label_domain["ecological_visibility_events"] = {
        **event_domain,
        "visibility_event_sha256": events.visibility_event_sha256,
    }
    transition = transition.model_copy(update={"ecological_label_sha256": _hash(label_domain)})
    transition = TransitionRecord.model_validate(transition.model_dump(mode="python"))
    episode = _model(
        CanonicalPairedEpisodeManifest,
        episode_id=EPISODE,
        episode_index=0,
        episode_seed=recorded_episode_seed,
        transition=json_file(f"{EPISODE}/transition.json", transition, Modality.TRANSITION_RECORD),
        privileged_instrumentation=json_file(
            f"{EPISODE}/instrumentation.json", {"synthetic": True}, private
        ),
        scene_content_sha256=fake,
        ecological_label_sha256=transition.ecological_label_sha256,
        analytic_transport_sha256=fake,
        oriented_boundary_sha256=boundary.oriented_boundary_sha256,
        visibility_event_sha256=events.visibility_event_sha256,
        appearance_instance_sha256=fake,
        rgb_logical_sha256=tuple(frame.rgb.logical_sha256 for frame in frames),
        paired_output_provenance_sha256=tuple(pair_hashes),
    )
    manifest = _model(
        DatasetManifest,
        schema_version="0.1.0-dev.11",
        generator_version="0.1.0",
        scene_family=config.scene_family,
        root_seed=config.seed,
        config_logical_sha256=config_hash,
        appearance_registry_sha256=fake,
        appearance_profile_id=config.appearance.profile_id,
        appearance_profile_sha256=fake,
        appearance_registry_snapshot=json_file(
            "appearances.json", {"synthetic": True}, Modality.APPEARANCE_CONTROL
        ),
        evaluation_seed_registry_sha256=fake,
        evaluation_seed_registry_snapshot=json_file(
            "seeds.json", {"synthetic": True}, Modality.APPEARANCE_CONTROL
        ),
        appearance_assignment_schedule_source="snapshotted_evaluation_seed_registry_v1",
        resolved_config=json_file("config.json", config, private),
        renderer_provenance=renderer,
        renderer_execution_provenance_sha256=renderer_hash,
        episodes=(episode,),
        dataset_logical_sha256=ZERO,
        source_provenance=source,
        source_provenance_sha256=source_hash,
        content_provenance_binding_sha256=ZERO,
    )
    dataset_hash = compute_dataset_logical_hash(manifest)
    manifest = manifest.model_copy(
        update={
            "dataset_logical_sha256": dataset_hash,
            "content_provenance_binding_sha256": compute_content_provenance_binding(
                dataset_hash, source_hash, renderer_hash
            ),
        }
    )
    (root / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    return root
