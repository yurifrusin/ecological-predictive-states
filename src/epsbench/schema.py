"""Public, strict schemas for EPS-Bench v0 data contracts."""

from __future__ import annotations

import math
import re
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Literal
from urllib.parse import urlsplit, urlunsplit

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    field_validator,
    model_validator,
)

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
GitCommit = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
SurfaceId = Annotated[str, StringConstraints(pattern=r"^surface-[0-9a-f]{16}$")]

LOCAL_REPOSITORY_REDACTION = "local-repository-redacted"
UNCLASSIFIED_REPOSITORY_REDACTION = "unclassified-repository-redacted"
_WINDOWS_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:")
_SCP_STYLE_ORIGIN = re.compile(r"^(?:[^@/:]+@)?(?P<host>[^@/:]+):(?P<path>.+)$")


def sanitize_git_repository(value: str) -> str:
    """Return a credential-free, non-local repository reference for serialization."""

    candidate = value.strip()
    lowered = candidate.lower()
    if candidate in {LOCAL_REPOSITORY_REDACTION, UNCLASSIFIED_REPOSITORY_REDACTION}:
        return candidate
    if (
        not candidate
        or lowered.startswith("file:")
        or _WINDOWS_DRIVE_PREFIX.match(candidate)
        or candidate.startswith(("/", "\\", "./", "../", "~"))
    ):
        return LOCAL_REPOSITORY_REDACTION

    scp_match = _SCP_STYLE_ORIGIN.fullmatch(candidate)
    if scp_match is not None and "://" not in candidate:
        host = scp_match.group("host")
        path = scp_match.group("path").lstrip("/")
        return f"ssh://{host}/{path}"

    try:
        parsed = urlsplit(candidate)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return UNCLASSIFIED_REPOSITORY_REDACTION
    if parsed.scheme not in {"http", "https", "ssh", "git"} or hostname is None:
        return UNCLASSIFIED_REPOSITORY_REDACTION
    host = f"[{hostname}]" if ":" in hostname else hostname
    netloc = f"{host}:{port}" if port is not None else host
    return urlunsplit((parsed.scheme.lower(), netloc, parsed.path, "", ""))


class StrictModel(BaseModel):
    """Schema base with fail-closed handling of unknown fields."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class SceneFamily(StrEnum):
    SINGLE_OCCLUDER = "single_occluder"
    CORRIDOR = "corridor"


class ModalityClass(StrEnum):
    SENSORY = "sensory"
    ECOLOGICAL_ORACLE = "ecological_oracle"
    METRIC_BASELINE = "metric_baseline"
    INSTRUMENTATION_ONLY = "instrumentation_only"
    CONTROL_METADATA = "control_metadata"


class Modality(StrEnum):
    RGB = "rgb"
    EXECUTED_ACTION = "executed_action"
    SURFACE_REGIONS = "surface_regions"
    BOUNDARY_STRUCTURE = "boundary_structure"
    VISIBILITY_FRACTIONS = "visibility_fractions"
    REGION_CORRESPONDENCE = "region_correspondence"
    REGION_MASK_CHANGES = "region_mask_changes"
    ECOLOGICAL_VISIBILITY_EVENTS = "ecological_visibility_events"
    OCCLUSION_ANNOTATION = "occlusion_annotation"
    ANALYTIC_OPTICAL_TRANSPORT = "analytic_optical_transport"
    DEPTH = "depth"
    LOCAL_METRIC_ARRAYS = "local_metric_arrays"
    CAMERA_WORLD_TRANSFORM = "camera_world_transform"
    MUJOCO_GEOM_IDS = "mujoco_geom_ids"
    RAW_SIMULATOR_COORDINATES = "raw_simulator_coordinates"
    SAMPLED_SCENE_GEOMETRY = "sampled_scene_geometry"
    PRIVILEGED_GENERATION_RECORDS = "privileged_generation_records"
    TRANSITION_RECORD = "transition_record"
    SCENE_FAMILY = "scene_family"

    @property
    def modality_class(self) -> ModalityClass:
        if self in {Modality.RGB, Modality.EXECUTED_ACTION}:
            return ModalityClass.SENSORY
        if self in {
            Modality.SURFACE_REGIONS,
            Modality.BOUNDARY_STRUCTURE,
            Modality.VISIBILITY_FRACTIONS,
            Modality.REGION_CORRESPONDENCE,
            Modality.REGION_MASK_CHANGES,
            Modality.ECOLOGICAL_VISIBILITY_EVENTS,
            Modality.OCCLUSION_ANNOTATION,
            Modality.ANALYTIC_OPTICAL_TRANSPORT,
        }:
            return ModalityClass.ECOLOGICAL_ORACLE
        if self in {Modality.DEPTH, Modality.LOCAL_METRIC_ARRAYS}:
            return ModalityClass.METRIC_BASELINE
        if self in {Modality.TRANSITION_RECORD, Modality.SCENE_FAMILY}:
            return ModalityClass.CONTROL_METADATA
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
    name: Literal["lateral_right", "lateral_left", "forward"]
    delta_forward: float
    delta_lateral: float
    delta_yaw: float

    @model_validator(mode="after")
    def supported_action_is_finite_and_axis_aligned(self) -> Action:
        values = (self.delta_forward, self.delta_lateral, self.delta_yaw)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("all action values must be finite")
        if self.delta_yaw != 0.0:
            raise ValueError("yaw motion is unsupported in this Gate 0B slice")
        if self.name == "forward":
            if self.delta_forward <= 0.0:
                raise ValueError("forward requires positive forward displacement")
            if self.delta_lateral != 0.0:
                raise ValueError("forward cannot include lateral displacement")
            return self
        if self.delta_forward != 0.0:
            raise ValueError("lateral actions cannot include forward displacement")
        if self.delta_lateral == 0.0:
            raise ValueError("lateral displacement must be non-zero")
        if self.name == "lateral_right" and self.delta_lateral < 0.0:
            raise ValueError("lateral_right requires positive displacement")
        if self.name == "lateral_left" and self.delta_lateral > 0.0:
            raise ValueError("lateral_left requires negative displacement")
        return self


class SurfaceReference(StrictModel):
    surface_id: SurfaceId
    segmentation_label: int = Field(gt=0, le=2**31 - 1)


class VisibilityState(StrictModel):
    surface_id: SurfaceId
    before_visible_pixels: int = Field(ge=0)
    after_visible_pixels: int = Field(ge=0)
    before_projected_image_fraction: float = Field(ge=0.0, le=1.0)
    after_projected_image_fraction: float = Field(ge=0.0, le=1.0)


class MaskChangeKind(StrEnum):
    GAINED_IMAGE_PIXELS = "gained_image_pixels"
    LOST_IMAGE_PIXELS = "lost_image_pixels"
    REGION_APPEARED = "region_appeared"
    REGION_DISAPPEARED = "region_disappeared"
    MASK_UNCHANGED = "mask_unchanged"


class RegionMaskChange(StrictModel):
    surface_id: SurfaceId
    change: MaskChangeKind
    affected_image_pixels: int = Field(ge=0)

    @model_validator(mode="after")
    def count_matches_change_kind(self) -> RegionMaskChange:
        if self.change == MaskChangeKind.MASK_UNCHANGED:
            if self.affected_image_pixels != 0:
                raise ValueError("unchanged masks must report zero affected image pixels")
        elif self.affected_image_pixels == 0:
            raise ValueError("changed masks must report positive affected image pixels")
        return self


class UnavailableEcologicalVisibilityEvents(StrictModel):
    status: Literal["unavailable"]
    reason_category: Literal["oriented_boundary_ownership_unavailable"]
    reason: str = Field(min_length=1)


class RegionCorrespondence(StrictModel):
    surface_id: SurfaceId
    before_visible_pixels: int = Field(ge=0)
    after_visible_pixels: int = Field(ge=0)
    same_image_coordinate_overlap_pixels: int = Field(ge=0)

    @model_validator(mode="after")
    def overlap_is_bounded(self) -> RegionCorrespondence:
        if self.same_image_coordinate_overlap_pixels > min(
            self.before_visible_pixels, self.after_visible_pixels
        ):
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
    frame_indices: tuple[Literal[0, 1], ...]

    @model_validator(mode="after")
    def relation_is_not_reflexive(self) -> OcclusionRelation:
        if self.occluder_surface_id == self.occluded_surface_id:
            raise ValueError("an occlusion relation cannot be reflexive")
        if len(set(self.frame_indices)) != len(self.frame_indices):
            raise ValueError("occlusion frame indices must be unique")
        return self


class AvailableOcclusionAnnotation(StrictModel):
    status: Literal["available"]
    oracle_rule: Literal["counterfactual_occluder_exclusion_v1"]
    relations: tuple[OcclusionRelation, ...]


class UnavailableOcclusionAnnotation(StrictModel):
    status: Literal["unavailable"]
    reason_category: Literal["oriented_corridor_occlusion_oracle_unavailable"]
    reason: str = Field(min_length=1)


OcclusionAnnotation = Annotated[
    AvailableOcclusionAnnotation | UnavailableOcclusionAnnotation,
    Field(discriminator="status"),
]


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
        if _WINDOWS_DRIVE_PREFIX.match(value):
            raise ValueError("artifact paths must not be Windows drive-qualified or drive-relative")
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
        expected = {
            "RGB": (self.rgb, "uint8", (self.height, self.width, 3)),
            "depth": (self.depth, "float32", (self.height, self.width)),
            "segmentation": (
                self.segmentation,
                "int32",
                (self.height, self.width),
            ),
        }
        for field_name, (artifact, dtype, shape) in expected.items():
            if artifact.dtype != dtype or artifact.shape != shape:
                raise ValueError(f"{field_name} artifact has an invalid dtype or shape")
        return self


class OpticalTransportCoordinateConvention(StrictModel):
    pixel_sample: Literal["centre_of_pixel"]
    pixel_centre_x: Literal["column_plus_0.5"]
    pixel_centre_y: Literal["row_plus_0.5"]
    x_axis: Literal["increases_right"]
    y_axis: Literal["increases_down"]
    flow_definition: Literal["target_pixel_centre_minus_source_pixel_centre"]
    units: Literal["image_pixels"]


class OpticalTransportQuantisation(StrictModel):
    dtype: Literal["int32"]
    fixed_point_scale: Literal[1024]
    rounding: Literal["nearest_ties_to_even"]


class AnalyticBoundaryAmbiguityRule(StrictModel):
    rule: Literal["four_neighbour_assignment_band_v1"]
    width_pixels: Literal[1]
    connectivity: Literal["four_neighbour"]
    application: Literal["source_and_projected_target"]


class DirectionalOpticalTransport(StrictModel):
    source_frame_index: Literal[0, 1]
    target_frame_index: Literal[0, 1]
    vectors_fixed: ArtifactRecord
    validity: ArtifactRecord
    reasons: ArtifactRecord

    @model_validator(mode="after")
    def artifacts_are_complete_and_aligned(self) -> DirectionalOpticalTransport:
        if self.source_frame_index == self.target_frame_index:
            raise ValueError("optical transport source and target frames must differ")
        artifacts = (self.vectors_fixed, self.validity, self.reasons)
        if any(artifact.modality != Modality.ANALYTIC_OPTICAL_TRANSPORT for artifact in artifacts):
            raise ValueError("all optical transport artifacts require the analytic modality")
        if self.vectors_fixed.dtype != "int32" or len(self.vectors_fixed.shape) != 3:
            raise ValueError("optical transport vectors must be a three-dimensional int32 array")
        height, width, components = self.vectors_fixed.shape
        if components != 2:
            raise ValueError("optical transport vectors must contain x/y components")
        expected_mask_shape = (height, width)
        if self.validity.dtype != "uint8" or self.validity.shape != expected_mask_shape:
            raise ValueError("optical transport validity mask has an invalid dtype or shape")
        if self.reasons.dtype != "uint8" or self.reasons.shape != expected_mask_shape:
            raise ValueError("optical transport reason mask has an invalid dtype or shape")
        return self


class AvailableDenseOpticalTransport(StrictModel):
    status: Literal["available"]
    method: Literal["analytic_static_scene_transport_v1"]
    coordinate_convention: OpticalTransportCoordinateConvention
    quantisation: OpticalTransportQuantisation
    boundary_ambiguity: AnalyticBoundaryAmbiguityRule
    reason_code_domain: Literal["analytic_transport_reason_codes_v1"]
    forward: DirectionalOpticalTransport
    backward: DirectionalOpticalTransport
    analytic_transport_sha256: Sha256

    @model_validator(mode="after")
    def directions_are_forward_then_backward(self) -> AvailableDenseOpticalTransport:
        if (self.forward.source_frame_index, self.forward.target_frame_index) != (0, 1):
            raise ValueError("forward optical transport must map frame 0 to frame 1")
        if (self.backward.source_frame_index, self.backward.target_frame_index) != (1, 0):
            raise ValueError("backward optical transport must map frame 1 to frame 0")
        if self.forward.vectors_fixed.shape != self.backward.vectors_fixed.shape:
            raise ValueError("forward and backward optical transport dimensions must match")
        return self


class UnavailableDenseOpticalTransport(StrictModel):
    status: Literal["unavailable"]
    reason_category: Literal["analytic_transport_unavailable"]
    reason: str = Field(min_length=1)


DenseOpticalTransport = Annotated[
    AvailableDenseOpticalTransport | UnavailableDenseOpticalTransport,
    Field(discriminator="status"),
]


class TransitionRecord(StrictModel):
    schema_version: Literal["0.1.0-dev.3"]
    episode_id: str = Field(pattern=r"^episode-[0-9]{6}$")
    action: Action
    surfaces: tuple[SurfaceReference, ...] = Field(min_length=1)
    before: FrameRecord
    after: FrameRecord
    visibility_states: tuple[VisibilityState, ...] = Field(min_length=1)
    region_correspondence: tuple[RegionCorrespondence, ...] = Field(min_length=1)
    region_mask_changes: tuple[RegionMaskChange, ...] = Field(min_length=1)
    ecological_visibility_events: UnavailableEcologicalVisibilityEvents
    occlusion: OcclusionAnnotation
    boundary_structures: tuple[BoundaryStructure, ...] = Field(min_length=2, max_length=2)
    analytic_optical_transport: DenseOpticalTransport
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
        mask_change_keys = [
            (record.surface_id, record.change) for record in self.region_mask_changes
        ]
        if len(mask_change_keys) != len(set(mask_change_keys)):
            raise ValueError("region mask-change records must be unique")
        if not {record.surface_id for record in self.region_mask_changes}.issubset(known):
            raise ValueError("region mask change references an unknown surface")
        if {record.surface_id for record in self.region_mask_changes} != known:
            raise ValueError("region mask changes must cover every declared surface")
        if isinstance(self.occlusion, AvailableOcclusionAnnotation):
            for relation in self.occlusion.relations:
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
        if isinstance(self.analytic_optical_transport, AvailableDenseOpticalTransport):
            expected_shape = (self.before.height, self.before.width, 2)
            if self.analytic_optical_transport.forward.vectors_fixed.shape != expected_shape:
                raise ValueError("analytic optical transport must align with both raster frames")
            if (self.before.width, self.before.height) != (self.after.width, self.after.height):
                raise ValueError("analytic optical transport requires aligned frame dimensions")
        return self


class EcologicalTransitionView(StrictModel):
    """Permission-safe transition projection with no metric or instrumentation fields."""

    episode_id: str = Field(pattern=r"^episode-[0-9]{6}$")
    action: Action
    surfaces: tuple[SurfaceReference, ...]
    visibility_states: tuple[VisibilityState, ...]
    region_correspondence: tuple[RegionCorrespondence, ...]
    region_mask_changes: tuple[RegionMaskChange, ...]
    ecological_visibility_events: UnavailableEcologicalVisibilityEvents
    occlusion: OcclusionAnnotation
    boundary_structures: tuple[BoundaryStructure, ...]
    analytic_optical_transport: DenseOpticalTransport
    ecological_label_sha256: Sha256


class EpisodeManifest(StrictModel):
    episode_id: str = Field(pattern=r"^episode-[0-9]{6}$")
    episode_index: int = Field(ge=0)
    episode_seed: int = Field(ge=0)
    transition: ArtifactRecord
    privileged_instrumentation: ArtifactRecord
    scene_content_sha256: Sha256
    ecological_label_sha256: Sha256
    analytic_transport_sha256: Sha256
    rgb_logical_sha256: tuple[Sha256, Sha256]

    @model_validator(mode="after")
    def manifest_modalities_are_correct(self) -> EpisodeManifest:
        if self.transition.modality != Modality.TRANSITION_RECORD:
            raise ValueError("transition artifact must be neutral control metadata")
        if self.privileged_instrumentation.modality != Modality.PRIVILEGED_GENERATION_RECORDS:
            raise ValueError("instrumentation artifact must be privileged")
        return self


class CameraInstrumentation(StrictModel):
    frame_index: Literal[0, 1]
    camera_world_position: tuple[float, float, float]
    camera_world_rotation_row_major: tuple[float, ...] = Field(min_length=9, max_length=9)

    @model_validator(mode="after")
    def pose_is_finite_and_rotation_is_orthonormal(self) -> CameraInstrumentation:
        if not all(math.isfinite(value) for value in self.camera_world_position):
            raise ValueError("camera world position must contain only finite values")
        if not all(math.isfinite(value) for value in self.camera_world_rotation_row_major):
            raise ValueError("camera rotation must contain only finite values")
        rotation = np.asarray(self.camera_world_rotation_row_major, dtype=np.float64).reshape(3, 3)
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6, rtol=0.0):
            raise ValueError("camera rotation must be approximately orthonormal")
        if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6, rel_tol=0.0):
            raise ValueError("camera rotation must be a proper rotation")
        return self


class OcclusionFrameEvidence(StrictModel):
    frame_index: Literal[0, 1]
    counterfactual_segmentation: ArtifactRecord
    revealed_pixel_count: int = Field(ge=0)
    reveal_mask_logical_sha256: Sha256

    @model_validator(mode="after")
    def artifact_is_privileged_segmentation(self) -> OcclusionFrameEvidence:
        artifact = self.counterfactual_segmentation
        if artifact.modality != Modality.PRIVILEGED_GENERATION_RECORDS:
            raise ValueError("counterfactual segmentation must be privileged")
        if artifact.dtype != "int32" or len(artifact.shape) != 2:
            raise ValueError("counterfactual segmentation must be a two-dimensional int32 array")
        return self


class OcclusionOracleEvidence(StrictModel):
    rule: Literal["counterfactual_occluder_exclusion_v1"]
    candidate_occluder_raw_geom_id: int = Field(ge=0)
    candidate_occluded_raw_geom_id: int = Field(ge=0)
    frames: tuple[OcclusionFrameEvidence, OcclusionFrameEvidence]

    @model_validator(mode="after")
    def candidate_and_frames_are_well_formed(self) -> OcclusionOracleEvidence:
        if self.candidate_occluder_raw_geom_id == self.candidate_occluded_raw_geom_id:
            raise ValueError("occlusion candidates must be distinct")
        if {frame.frame_index for frame in self.frames} != {0, 1}:
            raise ValueError("occlusion evidence must cover both frames")
        return self


class GenerationSeeds(StrictModel):
    episode_seed: int = Field(ge=0)
    geometry_sampling_seed: int = Field(ge=0)
    surface_remapping_seed: int = Field(ge=0)
    appearance_seed: int = Field(ge=0)


class RawSegmentationFrameEvidence(StrictModel):
    frame_index: Literal[0, 1]
    raw_segmentation: ArtifactRecord

    @model_validator(mode="after")
    def artifact_is_privileged_raw_segmentation(self) -> RawSegmentationFrameEvidence:
        artifact = self.raw_segmentation
        if artifact.modality != Modality.PRIVILEGED_GENERATION_RECORDS:
            raise ValueError("raw renderer segmentation must be privileged")
        if artifact.dtype != "int32" or len(artifact.shape) != 2:
            raise ValueError("raw renderer segmentation must be a two-dimensional int32 array")
        return self


class CorridorSampledGeometry(StrictModel):
    width: float = Field(ge=1.5, le=8.0)
    length: float = Field(ge=3.0, le=12.0)
    wall_height: float = Field(ge=2.0, le=8.0)
    camera_lateral_position: float
    camera_before_forward_position: float
    camera_after_forward_position: float
    camera_height: float = Field(gt=0.1)
    field_of_view_degrees: float = Field(gt=20.0, lt=80.0)

    @model_validator(mode="after")
    def geometry_and_camera_path_are_finite_and_legal(self) -> CorridorSampledGeometry:
        values = (
            self.width,
            self.length,
            self.wall_height,
            self.camera_lateral_position,
            self.camera_before_forward_position,
            self.camera_after_forward_position,
            self.camera_height,
            self.field_of_view_degrees,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("sampled corridor geometry must contain only finite values")
        if self.camera_before_forward_position <= 0.0:
            raise ValueError("corridor camera must start beyond the entry plane")
        if self.camera_after_forward_position <= self.camera_before_forward_position:
            raise ValueError("corridor camera must move strictly forward")
        if self.camera_after_forward_position >= self.length - 0.1:
            raise ValueError("corridor camera must remain clear of the end wall")
        if abs(self.camera_lateral_position) >= self.width / 2.0 - 0.1:
            raise ValueError("corridor camera must remain clear of the side walls")
        if self.camera_height >= self.wall_height:
            raise ValueError("corridor camera must remain below the wall height")
        return self


class AnalyticRendererFrameDiagnostic(StrictModel):
    frame_index: Literal[0, 1]
    compared_interior_pixels: int = Field(gt=0)
    agreeing_interior_pixels: int = Field(ge=0)
    interior_agreement_rate: float = Field(ge=0.0, le=1.0)
    excluded_analytic_boundary_pixels: int = Field(ge=0)

    @model_validator(mode="after")
    def agreement_rate_matches_counts(self) -> AnalyticRendererFrameDiagnostic:
        if self.agreeing_interior_pixels > self.compared_interior_pixels:
            raise ValueError("renderer agreement count cannot exceed compared pixels")
        expected = self.agreeing_interior_pixels / self.compared_interior_pixels
        if not math.isclose(self.interior_agreement_rate, expected, abs_tol=1e-15, rel_tol=0.0):
            raise ValueError("renderer agreement rate must equal the declared counts")
        return self


class DirectionalTransportDiagnostic(StrictModel):
    total_pixels: int = Field(gt=0)
    valid_transport_pixels: int = Field(ge=0)
    valid_transport_fraction: float = Field(ge=0.0, le=1.0)
    reason_code_counts: tuple[int, int, int, int, int]

    @model_validator(mode="after")
    def counts_and_fraction_are_consistent(self) -> DirectionalTransportDiagnostic:
        if any(count < 0 for count in self.reason_code_counts):
            raise ValueError("transport diagnostic reason counts must be non-negative")
        if sum(self.reason_code_counts) != self.total_pixels:
            raise ValueError("transport diagnostic reason counts must cover every pixel")
        if self.reason_code_counts[0] != self.valid_transport_pixels:
            raise ValueError("valid transport count must equal reason-code zero count")
        expected = self.valid_transport_pixels / self.total_pixels
        if not math.isclose(self.valid_transport_fraction, expected, abs_tol=1e-15, rel_tol=0.0):
            raise ValueError("valid transport fraction must equal the declared counts")
        return self


class AnalyticTransportDiagnostics(StrictModel):
    method: Literal["analytic_static_scene_transport_v1"]
    renderer_cross_check: Literal["non_authoritative_interior_segmentation_agreement_v1"]
    minimum_interior_agreement_rate: float = Field(ge=0.9, le=0.9)
    frames: tuple[AnalyticRendererFrameDiagnostic, AnalyticRendererFrameDiagnostic]
    forward: DirectionalTransportDiagnostic
    backward: DirectionalTransportDiagnostic

    @model_validator(mode="after")
    def frames_are_complete_and_above_threshold(self) -> AnalyticTransportDiagnostics:
        if {frame.frame_index for frame in self.frames} != {0, 1}:
            raise ValueError("analytic renderer diagnostics must cover both frames")
        if any(
            frame.interior_agreement_rate < self.minimum_interior_agreement_rate
            for frame in self.frames
        ):
            raise ValueError("broad analytic/renderer interior disagreement exceeds threshold")
        return self


class SingleOccluderInstrumentation(StrictModel):
    schema_version: Literal["0.1.0-dev.3"]
    scene_family: Literal[SceneFamily.SINGLE_OCCLUDER]
    episode_id: str = Field(pattern=r"^episode-[0-9]{6}$")
    appearance_variant: Literal["base", "alternate"]
    raw_geom_ids: dict[str, int]
    raw_to_opaque_surface_ids: dict[str, SurfaceId]
    raw_geom_world_positions: dict[str, tuple[float, float, float]]
    raw_geom_compiled_sizes: dict[str, tuple[float, float, float]]
    occlusion_oracle: OcclusionOracleEvidence
    analytic_transport_diagnostics: AnalyticTransportDiagnostics

    @model_validator(mode="after")
    def apparatus_mapping_is_exact_and_finite(self) -> SingleOccluderInstrumentation:
        expected_names = {"support_surface", "occluding_surface", "background_surface"}
        if set(self.raw_geom_ids) != expected_names:
            raise ValueError("raw geom IDs must describe exactly the single-occluder apparatus")
        raw_ids = set(self.raw_geom_ids.values())
        if len(raw_ids) != len(expected_names) or any(raw_id < 0 for raw_id in raw_ids):
            raise ValueError("apparatus raw geom IDs must be distinct and non-negative")
        if set(self.raw_geom_world_positions) != expected_names:
            raise ValueError("raw geom positions must describe exactly the apparatus surfaces")
        if set(self.raw_geom_compiled_sizes) != expected_names:
            raise ValueError("compiled geom sizes must describe exactly the apparatus surfaces")
        if set(self.raw_to_opaque_surface_ids) != {str(raw_id) for raw_id in raw_ids}:
            raise ValueError("raw-to-opaque mapping must cover exactly the apparatus raw IDs")
        if len(set(self.raw_to_opaque_surface_ids.values())) != len(expected_names):
            raise ValueError("raw-to-opaque apparatus mapping must be bijective")
        if not all(
            math.isfinite(coordinate)
            for position in self.raw_geom_world_positions.values()
            for coordinate in position
        ):
            raise ValueError("raw geom world positions must contain only finite values")
        if not all(
            math.isfinite(dimension) and dimension > 0.0
            for size in self.raw_geom_compiled_sizes.values()
            for dimension in size
        ):
            raise ValueError("compiled geom sizes must contain only finite positive values")
        return self


class CorridorInstrumentation(StrictModel):
    schema_version: Literal["0.1.0-dev.3"]
    scene_family: Literal[SceneFamily.CORRIDOR]
    episode_id: str = Field(pattern=r"^episode-[0-9]{6}$")
    appearance_variant: Literal["base", "alternate"]
    apparatus_surface_names: tuple[str, ...]
    raw_geom_ids: dict[str, int]
    raw_to_opaque_surface_ids: dict[str, SurfaceId]
    raw_geom_world_positions: dict[str, tuple[float, float, float]]
    raw_geom_compiled_sizes: dict[str, tuple[float, float, float]]
    sampled_geometry: CorridorSampledGeometry
    camera_before: CameraInstrumentation
    camera_after: CameraInstrumentation
    generation_seeds: GenerationSeeds
    raw_segmentation_frames: tuple[
        RawSegmentationFrameEvidence,
        RawSegmentationFrameEvidence,
    ]
    geometry_sampling_rule: Literal["uniform_width_length_v1"]
    appearance_rule: Literal["solid_colour_variant_v1"]
    analytic_transport_diagnostics: AnalyticTransportDiagnostics

    @model_validator(mode="after")
    def apparatus_mapping_and_evidence_are_exact(self) -> CorridorInstrumentation:
        expected_names = {
            "corridor_floor",
            "corridor_left_surface",
            "corridor_right_surface",
            "corridor_end_surface",
        }
        if set(self.apparatus_surface_names) != expected_names or len(
            self.apparatus_surface_names
        ) != len(expected_names):
            raise ValueError("corridor apparatus surface names must be exact and unique")
        if set(self.raw_geom_ids) != expected_names:
            raise ValueError("raw geom IDs must describe exactly the corridor apparatus")
        raw_ids = set(self.raw_geom_ids.values())
        if len(raw_ids) != len(expected_names) or any(raw_id < 0 for raw_id in raw_ids):
            raise ValueError("corridor raw geom IDs must be distinct and non-negative")
        if set(self.raw_geom_world_positions) != expected_names:
            raise ValueError("raw geom positions must describe exactly the corridor surfaces")
        if set(self.raw_geom_compiled_sizes) != expected_names:
            raise ValueError("compiled geom sizes must describe exactly the corridor surfaces")
        if set(self.raw_to_opaque_surface_ids) != {str(raw_id) for raw_id in raw_ids}:
            raise ValueError("corridor raw-to-opaque mapping must cover exactly the raw IDs")
        if len(set(self.raw_to_opaque_surface_ids.values())) != len(expected_names):
            raise ValueError("corridor raw-to-opaque mapping must be bijective")
        if not all(
            math.isfinite(coordinate)
            for position in self.raw_geom_world_positions.values()
            for coordinate in position
        ):
            raise ValueError("corridor raw geom positions must contain only finite values")
        if not all(
            math.isfinite(dimension) and dimension > 0.0
            for size in self.raw_geom_compiled_sizes.values()
            for dimension in size
        ):
            raise ValueError("corridor compiled geom sizes must contain finite positive values")
        if self.camera_before.frame_index != 0 or self.camera_after.frame_index != 1:
            raise ValueError("corridor camera evidence must cover ordered before/after frames")
        if {frame.frame_index for frame in self.raw_segmentation_frames} != {0, 1}:
            raise ValueError("corridor raw segmentation evidence must cover both frames")
        return self


PrivilegedInstrumentation = Annotated[
    SingleOccluderInstrumentation | CorridorInstrumentation,
    Field(discriminator="scene_family"),
]
PRIVILEGED_INSTRUMENTATION_ADAPTER: TypeAdapter[PrivilegedInstrumentation] = TypeAdapter(
    PrivilegedInstrumentation
)


def parse_privileged_instrumentation_json(
    payload: str | bytes | bytearray,
) -> PrivilegedInstrumentation:
    """Parse strict scene-specific instrumentation through its discriminator."""

    return PRIVILEGED_INSTRUMENTATION_ADAPTER.validate_json(payload)


class GitAvailabilityStatus(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class SourceProvenance(StrictModel):
    git_repository: str | None
    git_commit: GitCommit | None
    git_dirty: bool | None
    dirty_diff_sha256: Sha256 | None
    git_availability_status: GitAvailabilityStatus
    git_unavailable_reason: str | None
    uv_lock_sha256: Sha256
    research_charter_sha256: Sha256
    eps_bench_spec_sha256: Sha256
    milestone_plan_sha256: Sha256
    codex_handoff_sha256: Sha256
    package_version: str = Field(min_length=1)
    python_version: str = Field(min_length=1)

    @field_validator("git_repository")
    @classmethod
    def repository_reference_is_sanitized(cls, value: str | None) -> str | None:
        if value is not None and value != sanitize_git_repository(value):
            raise ValueError("Git repository reference must be sanitized before serialization")
        return value

    @model_validator(mode="after")
    def git_state_is_truthful(self) -> SourceProvenance:
        if self.git_availability_status == GitAvailabilityStatus.AVAILABLE:
            if self.git_repository is None or self.git_commit is None or self.git_dirty is None:
                raise ValueError(
                    "available Git provenance requires repository, commit, and dirty state"
                )
            if self.git_unavailable_reason is not None:
                raise ValueError("available Git provenance cannot have an unavailable reason")
            if self.git_dirty and self.dirty_diff_sha256 is None:
                raise ValueError("dirty Git provenance requires an exact dirty diff hash")
            if not self.git_dirty and self.dirty_diff_sha256 is not None:
                raise ValueError("clean Git provenance cannot report a dirty diff hash")
        else:
            if not self.git_unavailable_reason:
                raise ValueError("unavailable Git provenance requires a reason")
            if any(
                value is not None
                for value in (
                    self.git_repository,
                    self.git_commit,
                    self.git_dirty,
                    self.dirty_diff_sha256,
                )
            ):
                raise ValueError("unavailable Git provenance cannot imply a clean or known state")
        return self


class RendererProvenance(StrictModel):
    mujoco_version: str
    numpy_version: str
    renderer: Literal["mujoco.Renderer"]
    backend: str
    operating_system: str


class DatasetManifest(StrictModel):
    schema_version: Literal["0.1.0-dev.3"]
    generator_version: Literal["0.1.0"]
    scene_family: SceneFamily
    root_seed: int = Field(ge=0)
    config_logical_sha256: Sha256
    appearance_variant: Literal["base", "alternate"]
    resolved_config: ArtifactRecord
    renderer_provenance: RendererProvenance
    renderer_execution_provenance_sha256: Sha256
    episodes: tuple[EpisodeManifest, ...] = Field(min_length=1)
    dataset_logical_sha256: Sha256
    source_provenance: SourceProvenance
    source_provenance_sha256: Sha256
    content_provenance_binding_sha256: Sha256

    @model_validator(mode="after")
    def episodes_are_unique_and_ordered(self) -> DatasetManifest:
        if self.resolved_config.modality != Modality.PRIVILEGED_GENERATION_RECORDS:
            raise ValueError("resolved configuration must be privileged generation data")
        ids = [episode.episode_id for episode in self.episodes]
        indices = [episode.episode_index for episode in self.episodes]
        if len(ids) != len(set(ids)) or len(indices) != len(set(indices)):
            raise ValueError("episode identifiers and indices must be unique")
        if indices != list(range(len(indices))):
            raise ValueError("episode indices must be contiguous from zero")
        return self
