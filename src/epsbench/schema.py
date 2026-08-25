"""Public, strict schemas for EPS-Bench v0 data contracts."""

from __future__ import annotations

from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
SurfaceId = Annotated[str, StringConstraints(pattern=r"^surface-[0-9a-f]{16}$")]


class StrictModel(BaseModel):
    """Schema base with fail-closed handling of unknown fields."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class ModalityClass(StrEnum):
    SENSORY = "sensory"
    ECOLOGICAL_ORACLE = "ecological_oracle"
    METRIC_BASELINE = "metric_baseline"
    INSTRUMENTATION_ONLY = "instrumentation_only"


class Modality(StrEnum):
    RGB = "rgb"
    EXECUTED_ACTION = "executed_action"
    SURFACE_REGIONS = "surface_regions"
    BOUNDARY_STRUCTURE = "boundary_structure"
    VISIBILITY_FRACTIONS = "visibility_fractions"
    REGION_CORRESPONDENCE = "region_correspondence"
    OCCLUSION_RELATION = "occlusion_relation"
    VISIBILITY_EVENTS = "visibility_events"
    DEPTH = "depth"
    LOCAL_METRIC_ARRAYS = "local_metric_arrays"
    CAMERA_WORLD_TRANSFORM = "camera_world_transform"
    MUJOCO_GEOM_IDS = "mujoco_geom_ids"
    RAW_SIMULATOR_COORDINATES = "raw_simulator_coordinates"
    PRIVILEGED_GENERATION_RECORDS = "privileged_generation_records"

    @property
    def modality_class(self) -> ModalityClass:
        if self in {Modality.RGB, Modality.EXECUTED_ACTION}:
            return ModalityClass.SENSORY
        if self in {
            Modality.SURFACE_REGIONS,
            Modality.BOUNDARY_STRUCTURE,
            Modality.VISIBILITY_FRACTIONS,
            Modality.REGION_CORRESPONDENCE,
            Modality.OCCLUSION_RELATION,
            Modality.VISIBILITY_EVENTS,
        }:
            return ModalityClass.ECOLOGICAL_ORACLE
        if self in {Modality.DEPTH, Modality.LOCAL_METRIC_ARRAYS}:
            return ModalityClass.METRIC_BASELINE
        return ModalityClass.INSTRUMENTATION_ONLY


class ModalityPermissionSet(StrictModel):
    allowed: frozenset[Modality]

    def permits(self, modality: Modality) -> bool:
        return modality in self.allowed

    @classmethod
    def ecological_only(cls) -> ModalityPermissionSet:
        allowed = frozenset(
            modality
            for modality in Modality
            if modality.modality_class == ModalityClass.ECOLOGICAL_ORACLE
        )
        return cls(allowed=allowed | {Modality.EXECUTED_ACTION})

    @classmethod
    def all_modalities(cls) -> ModalityPermissionSet:
        return cls(allowed=frozenset(Modality))


class Action(StrictModel):
    name: Literal["lateral_right", "lateral_left"]
    delta_forward: float
    delta_lateral: float
    delta_yaw: float


class SurfaceReference(StrictModel):
    surface_id: SurfaceId
    segmentation_label: int = Field(gt=0, le=2**31 - 1)


class VisibilityState(StrictModel):
    surface_id: SurfaceId
    before_visible_pixels: int = Field(ge=0)
    after_visible_pixels: int = Field(ge=0)
    before_visible_fraction: float = Field(ge=0.0, le=1.0)
    after_visible_fraction: float = Field(ge=0.0, le=1.0)


class VisibilityEventKind(StrEnum):
    STABLE = "stable"
    ACCRETING = "accreting"
    DELETING = "deleting"
    APPEARING = "appearing"
    DISAPPEARING = "disappearing"


class VisibilityEvent(StrictModel):
    surface_id: SurfaceId
    event: VisibilityEventKind
    affected_pixels: int = Field(gt=0)


class RegionCorrespondence(StrictModel):
    surface_id: SurfaceId
    before_visible_pixels: int = Field(ge=0)
    after_visible_pixels: int = Field(ge=0)
    image_overlap_pixels: int = Field(ge=0)

    @model_validator(mode="after")
    def overlap_is_bounded(self) -> RegionCorrespondence:
        if self.image_overlap_pixels > min(self.before_visible_pixels, self.after_visible_pixels):
            raise ValueError("image overlap cannot exceed either visible region")
        return self


class BoundaryContact(StrictModel):
    first_surface_id: SurfaceId
    second_surface_id: SurfaceId
    pixel_count: int = Field(gt=0)

    @model_validator(mode="after")
    def surfaces_are_distinct(self) -> BoundaryContact:
        if self.first_surface_id == self.second_surface_id:
            raise ValueError("a boundary contact requires two distinct surfaces")
        return self


class BoundaryStructure(StrictModel):
    frame_index: Literal[0, 1]
    total_boundary_pixels: int = Field(ge=0)
    contacts: tuple[BoundaryContact, ...]


class OcclusionRelation(StrictModel):
    occluder_surface_id: SurfaceId
    occluded_surface_id: SurfaceId
    frame_indices: tuple[Literal[0, 1], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def relation_is_not_reflexive(self) -> OcclusionRelation:
        if self.occluder_surface_id == self.occluded_surface_id:
            raise ValueError("an occlusion relation cannot be reflexive")
        if len(set(self.frame_indices)) != len(self.frame_indices):
            raise ValueError("occlusion frame indices must be unique")
        return self


class ArtifactRecord(StrictModel):
    path: str
    modality: Modality
    media_type: str = Field(min_length=1)
    dtype: str = Field(min_length=1)
    shape: tuple[int, ...] = Field(min_length=1)
    logical_sha256: Sha256
    file_sha256: Sha256
    byte_count: int = Field(gt=0)

    @field_validator("path")
    @classmethod
    def path_is_safe_and_relative(cls, value: str) -> str:
        if "\\" in value:
            raise ValueError("artifact paths must use POSIX separators")
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or value in {"", "."}:
            raise ValueError("artifact path must be a safe dataset-relative path")
        return value

    @field_validator("shape")
    @classmethod
    def shape_is_positive(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(dimension <= 0 for dimension in value):
            raise ValueError("artifact dimensions must be positive")
        return value


class FrameRecord(StrictModel):
    frame_index: Literal[0, 1]
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    rgb: ArtifactRecord
    depth: ArtifactRecord
    segmentation: ArtifactRecord
    camera_world_transform: ArtifactRecord

    @model_validator(mode="after")
    def modalities_and_shapes_align(self) -> FrameRecord:
        expected_modalities = {
            "rgb": (self.rgb, Modality.RGB),
            "depth": (self.depth, Modality.DEPTH),
            "segmentation": (self.segmentation, Modality.SURFACE_REGIONS),
            "camera_world_transform": (
                self.camera_world_transform,
                Modality.CAMERA_WORLD_TRANSFORM,
            ),
        }
        for field_name, (artifact, modality) in expected_modalities.items():
            if artifact.modality != modality:
                raise ValueError(f"{field_name} artifact has the wrong modality")
        expected_prefix = (self.height, self.width)
        for artifact in (self.rgb, self.depth, self.segmentation):
            if artifact.shape[:2] != expected_prefix:
                raise ValueError("RGB, depth, and segmentation dimensions must align")
        return self


class UnavailableAnnotation(StrictModel):
    field: Literal["dense_optical_flow"]
    status: Literal["unavailable"]
    reason: str = Field(min_length=1)


class TransitionRecord(StrictModel):
    schema_version: Literal["0.1.0"]
    episode_id: str = Field(pattern=r"^episode-[0-9]{6}$")
    action: Action
    surfaces: tuple[SurfaceReference, ...] = Field(min_length=1)
    before: FrameRecord
    after: FrameRecord
    visibility_states: tuple[VisibilityState, ...] = Field(min_length=1)
    region_correspondence: tuple[RegionCorrespondence, ...] = Field(min_length=1)
    visibility_events: tuple[VisibilityEvent, ...] = Field(min_length=1)
    occlusion_relations: tuple[OcclusionRelation, ...] = Field(min_length=1)
    boundary_structures: tuple[BoundaryStructure, ...] = Field(min_length=2, max_length=2)
    dense_optical_flow: UnavailableAnnotation
    ecological_label_sha256: Sha256

    @model_validator(mode="after")
    def references_are_consistent(self) -> TransitionRecord:
        surface_ids = [surface.surface_id for surface in self.surfaces]
        labels = [surface.segmentation_label for surface in self.surfaces]
        if len(surface_ids) != len(set(surface_ids)):
            raise ValueError("surface identifiers must be unique")
        if len(labels) != len(set(labels)):
            raise ValueError("segmentation labels must be unique")
        known = set(surface_ids)
        for records in (self.visibility_states, self.region_correspondence):
            record_ids = [record.surface_id for record in records]
            if len(record_ids) != len(set(record_ids)):
                raise ValueError("per-surface annotation identifiers must be unique")
            if not set(record_ids).issubset(known):
                raise ValueError("per-surface annotation references an unknown surface")
            if set(record_ids) != known:
                raise ValueError("per-surface annotations must cover every declared surface")
        event_keys = [(record.surface_id, record.event) for record in self.visibility_events]
        if len(event_keys) != len(set(event_keys)):
            raise ValueError("visibility event records must be unique")
        if not {record.surface_id for record in self.visibility_events}.issubset(known):
            raise ValueError("visibility event references an unknown surface")
        if {record.surface_id for record in self.visibility_events} != known:
            raise ValueError("visibility events must cover every declared surface")
        for relation in self.occlusion_relations:
            if relation.occluder_surface_id not in known:
                raise ValueError("occlusion relation has an unknown occluder")
            if relation.occluded_surface_id not in known:
                raise ValueError("occlusion relation has an unknown occluded surface")
        for boundary in self.boundary_structures:
            for contact in boundary.contacts:
                if {contact.first_surface_id, contact.second_surface_id} - known:
                    raise ValueError("boundary contact references an unknown surface")
        if self.before.frame_index != 0 or self.after.frame_index != 1:
            raise ValueError("transition frames must be ordered before then after")
        if {item.frame_index for item in self.boundary_structures} != {0, 1}:
            raise ValueError("boundary structures must describe both frames")
        return self


class EcologicalTransitionView(StrictModel):
    """Permission-safe transition projection with no metric or instrumentation fields."""

    episode_id: str = Field(pattern=r"^episode-[0-9]{6}$")
    action: Action
    surfaces: tuple[SurfaceReference, ...]
    visibility_states: tuple[VisibilityState, ...]
    region_correspondence: tuple[RegionCorrespondence, ...]
    visibility_events: tuple[VisibilityEvent, ...]
    occlusion_relations: tuple[OcclusionRelation, ...]
    boundary_structures: tuple[BoundaryStructure, ...]
    dense_optical_flow: UnavailableAnnotation
    ecological_label_sha256: Sha256


class EpisodeManifest(StrictModel):
    episode_id: str = Field(pattern=r"^episode-[0-9]{6}$")
    episode_index: int = Field(ge=0)
    episode_seed: int = Field(ge=0)
    transition: ArtifactRecord
    privileged_instrumentation: ArtifactRecord
    ecological_label_sha256: Sha256
    rgb_logical_sha256: tuple[Sha256, Sha256]

    @model_validator(mode="after")
    def manifest_modalities_are_correct(self) -> EpisodeManifest:
        if self.transition.modality != Modality.VISIBILITY_EVENTS:
            raise ValueError("transition artifact must be ecological-oracle data")
        if self.privileged_instrumentation.modality != Modality.PRIVILEGED_GENERATION_RECORDS:
            raise ValueError("instrumentation artifact must be privileged")
        return self


class CameraInstrumentation(StrictModel):
    frame_index: Literal[0, 1]
    camera_world_position: tuple[float, float, float]
    camera_world_rotation_row_major: tuple[float, ...] = Field(min_length=9, max_length=9)


class PrivilegedInstrumentation(StrictModel):
    schema_version: Literal["0.1.0"]
    episode_id: str = Field(pattern=r"^episode-[0-9]{6}$")
    appearance_variant: Literal["base", "alternate"]
    raw_geom_ids: dict[str, int]
    raw_to_opaque_surface_ids: dict[str, SurfaceId]
    raw_geom_world_positions: dict[str, tuple[float, float, float]]


class RendererProvenance(StrictModel):
    mujoco_version: str
    numpy_version: str
    renderer: Literal["mujoco.Renderer"]
    backend: str
    operating_system: str


class DatasetManifest(StrictModel):
    schema_version: Literal["0.1.0"]
    generator_version: Literal["0.1.0"]
    root_seed: int = Field(ge=0)
    config_logical_sha256: Sha256
    appearance_variant: Literal["base", "alternate"]
    resolved_config: ArtifactRecord
    renderer_provenance: RendererProvenance
    episodes: tuple[EpisodeManifest, ...] = Field(min_length=1)
    dataset_logical_sha256: Sha256

    @model_validator(mode="after")
    def episodes_are_unique_and_ordered(self) -> DatasetManifest:
        if self.resolved_config.modality != Modality.PRIVILEGED_GENERATION_RECORDS:
            raise ValueError("resolved configuration must be privileged generation data")
        ids = [episode.episode_id for episode in self.episodes]
        indices = [episode.episode_index for episode in self.episodes]
        if len(ids) != len(set(ids)) or len(indices) != len(set(indices)):
            raise ValueError("episode identifiers and indices must be unique")
        if indices != sorted(indices):
            raise ValueError("episodes must be ordered by episode index")
        return self
