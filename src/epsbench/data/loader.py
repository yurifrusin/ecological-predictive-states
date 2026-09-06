"""Fail-closed typed access to dataset modalities."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt

from epsbench.appearance import (
    AppearanceInstanceRecord,
    SeedRegistryType,
    parse_seed_registry,
)
from epsbench.data.decoding import (
    decode_json_artifact,
    decode_npy_artifact,
    decode_rgb_artifact,
)
from epsbench.data.paths import (
    UnsafeOwnedFileError,
    open_dataset_manifest,
    open_owned_regular_file,
    sha256_open_file,
)
from epsbench.schema import (
    Action,
    AnalyticBoundaryAmbiguityRule,
    AnalyticIntersectionVisibilityContract,
    ArtifactRecord,
    AvailableDenseOpticalTransport,
    AvailableEcologicalVisibilityEvents,
    AvailableOrientedBoundaryOwnership,
    CameraInstrumentation,
    ComponentTopologyAnnotation,
    CorridorInstrumentation,
    CorridorSampledGeometry,
    DatasetManifest,
    EcologicalTransitionView,
    EpisodeManifest,
    FrameRecord,
    Modality,
    ModalityPermissionSet,
    OpticalTransportCoordinateConvention,
    PrivilegedInstrumentation,
    SceneFamily,
    TransitionRecord,
    parse_privileged_instrumentation_json,
)

_Decoded = TypeVar("_Decoded")


class PermissionDeniedError(PermissionError):
    """Raised before any unauthorised artifact is opened."""


@dataclass(frozen=True)
class LoadedAnalyticOpticalTransport:
    """Complete public runtime bundle; validity semantics always accompany vectors."""

    forward_vectors_fixed: npt.NDArray[np.int32]
    forward_validity: npt.NDArray[np.uint8]
    forward_reasons: npt.NDArray[np.uint8]
    backward_vectors_fixed: npt.NDArray[np.int32]
    backward_validity: npt.NDArray[np.uint8]
    backward_reasons: npt.NDArray[np.uint8]
    fixed_point_scale: int
    method: str
    coordinate_convention: OpticalTransportCoordinateConvention
    intersection_visibility: AnalyticIntersectionVisibilityContract
    boundary_ambiguity: AnalyticBoundaryAmbiguityRule
    analytic_transport_sha256: str

    @property
    def forward_flow_pixels(self) -> npt.NDArray[np.float64]:
        return self.forward_vectors_fixed.astype(np.float64) / self.fixed_point_scale

    @property
    def backward_flow_pixels(self) -> npt.NDArray[np.float64]:
        return self.backward_vectors_fixed.astype(np.float64) / self.fixed_point_scale


@dataclass(frozen=True)
class LoadedEcologicalVisibilityEvents:
    """Complete public event bundle with typed metadata and aligned dense maps."""

    annotation: AvailableEcologicalVisibilityEvents
    before_fate_codes: npt.NDArray[np.uint8]
    before_affected_surface_labels: npt.NDArray[np.int32]
    before_owner_surface_labels: npt.NDArray[np.int32]
    after_origin_codes: npt.NDArray[np.uint8]
    after_affected_surface_labels: npt.NDArray[np.int32]
    after_owner_surface_labels: npt.NDArray[np.int32]
    surface_ids_by_label: dict[int, str]


@dataclass(frozen=True)
class LoadedComponentTopology:
    annotation: ComponentTopologyAnnotation
    before_component_labels: npt.NDArray[np.int32]
    after_component_labels: npt.NDArray[np.int32]


class DatasetLoader:
    """Dataset reader that requires a declared permission set at construction."""

    def __init__(self, root: Path, permissions: ModalityPermissionSet) -> None:
        with open_dataset_manifest(root) as (resolved_root, owned_manifest):
            manifest = DatasetManifest.model_validate_json(owned_manifest.payload)
        self.root = resolved_root
        self.permissions = permissions
        self._manifest = manifest

    def _require(self, *modalities: Modality) -> None:
        denied = [modality for modality in modalities if not self.permissions.permits(modality)]
        if denied:
            names = ", ".join(modality.value for modality in denied)
            raise PermissionDeniedError(f"modality permission denied: {names}")

    def _decode_owned(
        self,
        record: ArtifactRecord,
        decoder: Callable[[bytes], _Decoded],
    ) -> _Decoded:
        try:
            with open_owned_regular_file(self.root, record.path) as owned:
                if owned.byte_count != record.byte_count:
                    raise ValueError(f"artifact byte count mismatch: {record.path}")
                if sha256_open_file(owned) != record.file_sha256:
                    raise ValueError(f"artifact file hash mismatch: {record.path}")
                return decoder(owned.payload)
        except UnsafeOwnedFileError as error:
            raise ValueError(str(error)) from error

    def _load_npy(self, record: ArtifactRecord) -> np.ndarray[Any, Any]:
        return self._decode_owned(record, lambda snapshot: decode_npy_artifact(snapshot, record))

    def _load_json(
        self,
        record: ArtifactRecord,
        decoder: Callable[[bytes], _Decoded],
    ) -> _Decoded:
        def verify_and_decode(snapshot: bytes) -> _Decoded:
            decode_json_artifact(snapshot, record)
            return decoder(snapshot)

        return self._decode_owned(record, verify_and_decode)

    def _episode(self, episode_index: int) -> EpisodeManifest:
        try:
            return next(
                episode
                for episode in self._manifest.episodes
                if episode.episode_index == episode_index
            )
        except StopIteration as error:
            raise IndexError(f"episode index not found: {episode_index}") from error

    def _transition(self, episode_index: int) -> TransitionRecord:
        episode = self._episode(episode_index)
        transition = self._load_json(
            episode.transition,
            TransitionRecord.model_validate_json,
        )
        has_topology = isinstance(
            transition.ecological_visibility_events.capabilities.component_topology,
            ComponentTopologyAnnotation,
        )
        if has_topology != (self._manifest.schema_version in {"0.1.0-dev.9", "0.1.0-dev.10"}):
            raise ValueError("dataset schema and component topology disagree")
        return transition

    @staticmethod
    def _frame(transition: TransitionRecord, frame_index: int) -> FrameRecord:
        if frame_index == 0:
            return transition.before
        if frame_index == 1:
            return transition.after
        raise IndexError(f"frame index must be 0 or 1, got {frame_index}")

    def read_action(self, episode_index: int) -> Action:
        self._require(Modality.EXECUTED_ACTION)
        return self._transition(episode_index).action

    def read_scene_family(self) -> SceneFamily:
        self._require(Modality.SCENE_FAMILY)
        return self._manifest.scene_family

    def read_dataset_manifest(self) -> DatasetManifest:
        """Return full control metadata only to explicitly privileged callers."""

        self._require(
            Modality.TRANSITION_RECORD,
            Modality.SCENE_FAMILY,
            Modality.PRIVILEGED_GENERATION_RECORDS,
            Modality.APPEARANCE_CONTROL,
        )
        return self._manifest.model_copy(deep=True)

    def read_ecological_transition(self, episode_index: int) -> EcologicalTransitionView:
        self._require_topology_if_present()
        self._require(
            Modality.EXECUTED_ACTION,
            Modality.SURFACE_REGIONS,
            Modality.VISIBILITY_FRACTIONS,
            Modality.REGION_CORRESPONDENCE,
            Modality.REGION_MASK_CHANGES,
            Modality.ECOLOGICAL_VISIBILITY_EVENTS,
            Modality.ORIENTED_BOUNDARY_OWNERSHIP,
            Modality.OCCLUSION_ANNOTATION,
            Modality.BOUNDARY_STRUCTURE,
            Modality.ANALYTIC_OPTICAL_TRANSPORT,
        )
        transition = self._transition(episode_index)
        return EcologicalTransitionView(
            episode_id=transition.episode_id,
            action=transition.action,
            surfaces=transition.surfaces,
            visibility_states=transition.visibility_states,
            region_correspondence=transition.region_correspondence,
            region_mask_changes=transition.region_mask_changes,
            oriented_boundary_ownership=transition.oriented_boundary_ownership,
            ecological_visibility_events=transition.ecological_visibility_events,
            occlusion=transition.occlusion,
            boundary_structures=transition.boundary_structures,
            analytic_optical_transport=transition.analytic_optical_transport,
            ecological_label_sha256=transition.ecological_label_sha256,
        )

    def read_oriented_boundaries(
        self,
        episode_index: int,
    ) -> AvailableOrientedBoundaryOwnership:
        self._require(Modality.ORIENTED_BOUNDARY_OWNERSHIP)
        return self._transition(episode_index).oriented_boundary_ownership.model_copy(deep=True)

    def read_ecological_visibility_events(
        self,
        episode_index: int,
    ) -> LoadedEcologicalVisibilityEvents:
        self._require(Modality.ECOLOGICAL_VISIBILITY_EVENTS)
        self._require_topology_if_present()
        transition = self._transition(episode_index)
        events = transition.ecological_visibility_events

        def load_uint8(record: ArtifactRecord) -> npt.NDArray[np.uint8]:
            return np.asarray(
                self._load_npy(record),
                dtype=np.uint8,
            )

        def load_int32(record: ArtifactRecord) -> npt.NDArray[np.int32]:
            return np.asarray(
                self._load_npy(record),
                dtype=np.int32,
            )

        return LoadedEcologicalVisibilityEvents(
            annotation=events.model_copy(deep=True),
            before_fate_codes=load_uint8(events.before_fate.event_codes),
            before_affected_surface_labels=load_int32(events.before_fate.affected_surface_labels),
            before_owner_surface_labels=load_int32(events.before_fate.owner_surface_labels),
            after_origin_codes=load_uint8(events.after_origin.event_codes),
            after_affected_surface_labels=load_int32(events.after_origin.affected_surface_labels),
            after_owner_surface_labels=load_int32(events.after_origin.owner_surface_labels),
            surface_ids_by_label={
                surface.segmentation_label: surface.surface_id for surface in transition.surfaces
            },
        )

    def _require_topology_if_present(self) -> None:
        if self._manifest.schema_version in {"0.1.0-dev.9", "0.1.0-dev.10"}:
            self._require(Modality.COMPONENT_TOPOLOGY)

    def read_component_topology(self, episode_index: int) -> LoadedComponentTopology:
        self._require(Modality.COMPONENT_TOPOLOGY)
        topology = self._transition(
            episode_index
        ).ecological_visibility_events.capabilities.component_topology
        if not isinstance(topology, ComponentTopologyAnnotation):
            raise ValueError("component topology was not generated for this legacy dataset")
        return LoadedComponentTopology(
            annotation=topology.model_copy(deep=True),
            before_component_labels=self._load_npy(topology.frames[0].component_labels),
            after_component_labels=self._load_npy(topology.frames[1].component_labels),
        )

    def read_analytic_optical_transport(
        self,
        episode_index: int,
    ) -> LoadedAnalyticOpticalTransport:
        self._require(Modality.ANALYTIC_OPTICAL_TRANSPORT)
        transport = self._transition(episode_index).analytic_optical_transport
        if not isinstance(transport, AvailableDenseOpticalTransport):
            raise ValueError(f"analytic optical transport is unavailable: {transport.reason}")

        def load_int32(record: ArtifactRecord) -> npt.NDArray[np.int32]:
            return np.asarray(
                self._load_npy(record),
                dtype=np.int32,
            )

        def load_uint8(record: ArtifactRecord) -> npt.NDArray[np.uint8]:
            return np.asarray(
                self._load_npy(record),
                dtype=np.uint8,
            )

        return LoadedAnalyticOpticalTransport(
            forward_vectors_fixed=load_int32(transport.forward.vectors_fixed),
            forward_validity=load_uint8(transport.forward.validity),
            forward_reasons=load_uint8(transport.forward.reasons),
            backward_vectors_fixed=load_int32(transport.backward.vectors_fixed),
            backward_validity=load_uint8(transport.backward.validity),
            backward_reasons=load_uint8(transport.backward.reasons),
            fixed_point_scale=transport.quantisation.fixed_point_scale,
            method=transport.method,
            coordinate_convention=transport.coordinate_convention,
            intersection_visibility=transport.intersection_visibility,
            boundary_ambiguity=transport.boundary_ambiguity,
            analytic_transport_sha256=transport.analytic_transport_sha256,
        )

    def read_rgb(self, episode_index: int, frame_index: int) -> npt.NDArray[np.uint8]:
        self._require(Modality.RGB)
        transition = self._transition(episode_index)
        frame = self._frame(transition, frame_index)

        return self._decode_owned(
            frame.rgb,
            lambda snapshot: decode_rgb_artifact(snapshot, frame.rgb),
        )

    def read_depth(self, episode_index: int, frame_index: int) -> npt.NDArray[np.float32]:
        self._require(Modality.DEPTH)
        transition = self._transition(episode_index)
        frame = self._frame(transition, frame_index)
        return np.asarray(self._load_npy(frame.depth), dtype=np.float32)

    def read_segmentation(self, episode_index: int, frame_index: int) -> npt.NDArray[np.int32]:
        self._require(Modality.SURFACE_REGIONS)
        transition = self._transition(episode_index)
        frame = self._frame(transition, frame_index)
        return np.asarray(self._load_npy(frame.segmentation), dtype=np.int32)

    def read_camera_world_transform(
        self, episode_index: int, frame_index: int
    ) -> CameraInstrumentation:
        self._require(Modality.CAMERA_WORLD_TRANSFORM)
        transition = self._transition(episode_index)
        frame = self._frame(transition, frame_index)
        return self._load_json(
            frame.camera_world_transform,
            CameraInstrumentation.model_validate_json,
        )

    def _instrumentation(self, episode_index: int) -> PrivilegedInstrumentation:
        episode = self._episode(episode_index)
        return self._load_json(
            episode.privileged_instrumentation,
            parse_privileged_instrumentation_json,
        )

    def read_appearance_control(self, episode_index: int) -> AppearanceInstanceRecord:
        """Return private appearance control only with both required permissions."""

        self._require(Modality.APPEARANCE_CONTROL, Modality.PRIVILEGED_GENERATION_RECORDS)
        return self._instrumentation(episode_index).appearance.model_copy(deep=True)

    def read_evaluation_seed_registry_snapshot(self) -> SeedRegistryType:
        """Return the dataset-bound assignment registry only with private permissions."""

        self._require(Modality.APPEARANCE_CONTROL, Modality.PRIVILEGED_GENERATION_RECORDS)
        artifact = self._manifest.evaluation_seed_registry_snapshot
        return self._load_json(
            artifact,
            lambda snapshot: parse_seed_registry(json.loads(snapshot.decode("utf-8"))),
        )

    def read_raw_mujoco_geom_ids(self, episode_index: int) -> dict[str, int]:
        self._require(Modality.MUJOCO_GEOM_IDS, Modality.PRIVILEGED_GENERATION_RECORDS)
        return dict(self._instrumentation(episode_index).raw_geom_ids)

    def read_raw_world_coordinates(
        self, episode_index: int
    ) -> dict[str, tuple[float, float, float]]:
        self._require(
            Modality.RAW_SIMULATOR_COORDINATES,
            Modality.PRIVILEGED_GENERATION_RECORDS,
        )
        return dict(self._instrumentation(episode_index).raw_geom_world_positions)

    def read_sampled_corridor_geometry(self, episode_index: int) -> CorridorSampledGeometry:
        self._require(
            Modality.SAMPLED_SCENE_GEOMETRY,
            Modality.PRIVILEGED_GENERATION_RECORDS,
        )
        instrumentation = self._instrumentation(episode_index)
        if not isinstance(instrumentation, CorridorInstrumentation):
            raise ValueError("sampled corridor geometry is unavailable for this scene family")
        return instrumentation.sampled_geometry

    def read_semantic_surface_names(self, episode_index: int) -> tuple[str, ...]:
        self._require(Modality.PRIVILEGED_GENERATION_RECORDS)
        instrumentation = self._instrumentation(episode_index)
        return tuple(instrumentation.raw_geom_ids)
