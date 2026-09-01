"""Strict procedural appearance profiles, identities, and episode instances."""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import numpy.typing as npt
import yaml
from PIL import Image
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from epsbench.utils.canonical import canonical_json_bytes, logical_array_hash, sha256_bytes
from epsbench.utils.seeding import derive_seed

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
RgbBytes = tuple[int, int, int]
TextureArray = npt.NDArray[np.uint8]

APPEARANCE_REGISTRY_VERSION = "appearance_candidate_registry_v1"
APPEARANCE_REVISION1_REGISTRY_VERSION = "appearance_candidate_registry_v2"
EVALUATION_SEED_REGISTRY_VERSION = "evaluation_seed_candidate_registry_v0"
QUALIFICATION_SEED_REGISTRY_VERSION = "appearance_revision1_qualification_seed_registry_v0"
FINAL_EVALUATION_SEED_REGISTRY_VERSION = (
    "appearance_benchmark_v0_evaluation_episode_seed_registry_v1"
)
APPEARANCE_GENERATOR_VERSION = "repository_procedural_texture_v1"
APPEARANCE_ASSIGNMENT_VERSION = "balanced_cyclic_permutation_v1"
LEGACY_ASSIGNMENT_VERSION = "fixed_semantic_regression_v1"
APPEARANCE_INSTANCE_VERSION: Literal["appearance_instance_v3"] = "appearance_instance_v3"
APPEARANCE_REVISION1_INSTANCE_VERSION: Literal["appearance_instance_v4"] = "appearance_instance_v4"
ASSIGNMENT_SCHEDULE_SOURCE: Literal["snapshotted_evaluation_seed_registry_v1"] = (
    "snapshotted_evaluation_seed_registry_v1"
)
REVISION1_ASSIGNMENT_SCHEDULE_SOURCE: Literal["snapshotted_revision_partition_seed_registry_v1"] = (
    "snapshotted_revision_partition_seed_registry_v1"
)
TEXTURE_RESOLUTION = 128

CANONICAL_PROFILE_IDS = (
    "legacy_solid_base_v1",
    "legacy_solid_alternate_v1",
    "balanced_solid_palette_v1",
    "balanced_checker_low_v1",
    "balanced_checker_high_v1",
    "balanced_stripes_low_v1",
    "balanced_stripes_high_oblique_v1",
    "balanced_illumination_left_v1",
    "balanced_illumination_right_dim_v1",
    "balanced_combined_stress_v1",
)

REVISION1_PROFILE_IDS = (
    "revision1_balanced_reference_v1",
    "revision1_colour_shift_v1",
    "revision1_checker_low_v1",
    "revision1_checker_high_v1",
    "revision1_stripes_low_v1",
    "revision1_illumination_shift_v1",
    "revision1_combined_stress_v1",
)

CANONICAL_DESIGN_SEEDS = (
    6594827050443514047,
    8414348828724831316,
    12608688243639701226,
    12500538448662680758,
    8773669028651797023,
    13662469654622758661,
    17452811338195253152,
    18258777022364122176,
)


class StrictAppearanceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class CandidateClass(StrEnum):
    LEGACY_REGRESSION_CONTROL = "legacy_regression_control"
    SINGLE_AXIS_CANDIDATE = "single_axis_candidate"
    COMBINED_STRESS_CANDIDATE = "combined_stress_candidate"


class AxisTag(StrEnum):
    CONTROL = "control"
    COLOUR = "colour"
    TEXTURE_FAMILY = "texture_family"
    TEXTURE_FREQUENCY = "texture_frequency"
    TEXTURE_ORIENTATION = "texture_orientation"
    ILLUMINATION = "illumination"
    COMBINED = "combined"


class StyleAssignmentRule(StrEnum):
    FIXED_SEMANTIC_REGRESSION = LEGACY_ASSIGNMENT_VERSION
    BALANCED_CYCLIC_PERMUTATION = APPEARANCE_ASSIGNMENT_VERSION


class TextureFamily(StrEnum):
    SOLID = "solid"
    CHECKER = "checker"
    STRIPES = "stripes"


class TextureOrientation(StrEnum):
    AXIS_X = "axis_x_0_degrees"
    DIAGONAL_DOWN = "diagonal_down_45_degrees"


class PaletteSlot(StrictAppearanceModel):
    slot_id: str = Field(pattern=r"^style-slot-[0-3]$")
    foreground_rgb: RgbBytes
    background_rgb: RgbBytes

    @field_validator("foreground_rgb", "background_rgb")
    @classmethod
    def rgb_is_byte_triplet(cls, value: RgbBytes) -> RgbBytes:
        if any(isinstance(channel, bool) or channel < 0 or channel > 255 for channel in value):
            raise ValueError("palette RGB channels must be integer bytes")
        return value


class PaletteDefinition(StrictAppearanceModel):
    single_occluder_slots: tuple[PaletteSlot, PaletteSlot, PaletteSlot]
    corridor_slots: tuple[PaletteSlot, PaletteSlot, PaletteSlot, PaletteSlot]

    @model_validator(mode="after")
    def slots_are_canonical(self) -> PaletteDefinition:
        for slots in (self.single_occluder_slots, self.corridor_slots):
            expected = tuple(f"style-slot-{index}" for index in range(len(slots)))
            if tuple(slot.slot_id for slot in slots) != expected:
                raise ValueError("palette slots must be uniquely ordered from style-slot-0")
        return self


class TextureDefinition(StrictAppearanceModel):
    family: TextureFamily
    resolution: Literal[128]
    cycles_per_tile: int = Field(ge=0, le=64)
    orientation: TextureOrientation
    phase_rule: Literal["namespaced_integer_phase_v1"]
    contrast_rule: Literal["exact_palette_slot_bytes_v1"]
    surface_repeat_uv_rule: Literal["mujoco_geom_local_uv_repeat_v1"]
    filtering: Literal["mujoco_linear_mipmap_linear_v1"]

    @model_validator(mode="after")
    def frequency_matches_family(self) -> TextureDefinition:
        if self.family == TextureFamily.SOLID and self.cycles_per_tile != 0:
            raise ValueError("solid textures require zero cycles per tile")
        if self.family != TextureFamily.SOLID and self.cycles_per_tile <= 0:
            raise ValueError("spatial textures require a positive frequency")
        return self


class MaterialDefinition(StrictAppearanceModel):
    specular: float = Field(ge=0.0, le=0.0)
    shininess: float = Field(ge=0.0, le=0.0)
    reflectance: float = Field(ge=0.0, le=0.0)
    texrepeat: tuple[float, float]
    texuniform: Literal[False]

    @model_validator(mode="after")
    def repeat_is_finite_positive(self) -> MaterialDefinition:
        if not all(math.isfinite(value) and value > 0.0 for value in self.texrepeat):
            raise ValueError("material texture repeat must be finite and positive")
        return self


class SceneLightDefinition(StrictAppearanceModel):
    direction: tuple[float, float, float]
    diffuse_intensity: float = Field(gt=0.0, le=1.0)

    @model_validator(mode="after")
    def direction_is_finite_nonzero(self) -> SceneLightDefinition:
        if not all(math.isfinite(value) for value in self.direction):
            raise ValueError("light direction must be finite")
        if math.sqrt(sum(value * value for value in self.direction)) <= 1e-12:
            raise ValueError("light direction must be non-zero")
        if not math.isfinite(self.diffuse_intensity):
            raise ValueError("light intensity must be finite")
        return self


class IlluminationDefinition(StrictAppearanceModel):
    single_occluder: SceneLightDefinition
    corridor: SceneLightDefinition
    directional: Literal[True]
    cast_shadows: Literal[False]
    ambient_rgb: tuple[float, float, float]
    specular_rgb: tuple[float, float, float]
    camera_exposure_rule: Literal["unchanged_renderer_default_v1"]

    @model_validator(mode="after")
    def non_directional_terms_are_explicit_and_finite(self) -> IlluminationDefinition:
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in self.ambient_rgb):
            raise ValueError("ambient illumination must be finite and bounded in [0, 1]")
        if self.specular_rgb != (0.0, 0.0, 0.0):
            raise ValueError("specular illumination must remain exactly zero")
        return self


class AdmissionThresholds(StrictAppearanceModel):
    changed_controlled_pixel_fraction_minimum: float = Field(ge=0.2, le=0.2)
    normalized_controlled_rgb_mad_minimum: float = Field(ge=0.025, le=0.025)
    visible_surface_mean_luminance_minimum: float = Field(ge=0.05, le=0.05)
    visible_surface_mean_luminance_maximum: float = Field(ge=0.95, le=0.95)
    textured_surface_minimum_pixels: Literal[100]
    textured_surface_luminance_std_minimum: float = Field(ge=0.025, le=0.025)
    frequency_partner_minimum_ratio: float = Field(ge=4.0, le=4.0)


class AppearanceProfile(StrictAppearanceModel):
    profile_id: str = Field(pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*_v[0-9]+$")
    profile_version: Literal["appearance_profile_v2", "appearance_profile_v3"]
    candidate_class: CandidateClass
    freeze_eligible: bool
    axis_tags: tuple[AxisTag, ...] = Field(min_length=1)
    matched_control_profile_id: str
    style_assignment_rule: StyleAssignmentRule
    palette: PaletteDefinition
    texture: TextureDefinition
    material: MaterialDefinition
    illumination: IlluminationDefinition
    non_degeneracy_thresholds: AdmissionThresholds

    @model_validator(mode="after")
    def intent_is_consistent(self) -> AppearanceProfile:
        if self.profile_version == "appearance_profile_v2" and self.illumination.ambient_rgb != (
            0.0,
            0.0,
            0.0,
        ):
            raise ValueError("appearance_profile_v2 requires exactly zero ambient illumination")
        if len(set(self.axis_tags)) != len(self.axis_tags):
            raise ValueError("appearance axis tags must be unique")
        if self.candidate_class == CandidateClass.LEGACY_REGRESSION_CONTROL:
            if self.freeze_eligible or self.axis_tags != (AxisTag.CONTROL,):
                raise ValueError("legacy controls must be non-freeze-eligible control profiles")
            if self.style_assignment_rule != StyleAssignmentRule.FIXED_SEMANTIC_REGRESSION:
                raise ValueError("legacy controls require the explicit fixed regression mapping")
        else:
            if not self.freeze_eligible:
                raise ValueError(
                    "balanced candidate profiles must remain freeze-eligible candidates"
                )
            if self.style_assignment_rule != StyleAssignmentRule.BALANCED_CYCLIC_PERMUTATION:
                raise ValueError("freeze-eligible candidates require balanced permutation")
            if AxisTag.CONTROL in self.axis_tags:
                raise ValueError("candidate profiles cannot claim the legacy control axis")
        if self.candidate_class == CandidateClass.COMBINED_STRESS_CANDIDATE:
            if AxisTag.COMBINED not in self.axis_tags or len(self.axis_tags) < 2:
                raise ValueError("combined profiles must explicitly declare every changed axis")
        elif AxisTag.COMBINED in self.axis_tags:
            raise ValueError("only combined candidates may declare the combined axis")
        return self


class AppearanceRegistry(StrictAppearanceModel):
    registry_version: Literal["appearance_candidate_registry_v1"]
    profiles: tuple[AppearanceProfile, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def profiles_are_unique_canonical_and_referentially_complete(
        self,
    ) -> AppearanceRegistry:
        ids = tuple(profile.profile_id for profile in self.profiles)
        if len(ids) != len(set(ids)):
            raise ValueError("appearance profile IDs must be unique")
        if ids != tuple(sorted(ids)):
            raise ValueError("appearance profiles must be ordered lexicographically")
        if ids != tuple(sorted(CANONICAL_PROFILE_IDS)):
            raise ValueError("appearance registry must contain the exact canonical profile set")
        known = set(ids)
        if any(profile.matched_control_profile_id not in known for profile in self.profiles):
            raise ValueError("matched appearance controls must name a registry profile")
        return self


class AppearanceRevision1Registry(StrictAppearanceModel):
    registry_version: Literal["appearance_candidate_registry_v2"]
    profiles: tuple[AppearanceProfile, ...] = Field(min_length=1)
    revision1_admission_profile_ids: tuple[str, str, str, str, str, str, str]

    @model_validator(mode="after")
    def profiles_are_exact_and_referentially_complete(self) -> AppearanceRevision1Registry:
        ids = tuple(profile.profile_id for profile in self.profiles)
        expected = tuple(sorted((*CANONICAL_PROFILE_IDS, *REVISION1_PROFILE_IDS)))
        if len(ids) != len(set(ids)) or ids != expected:
            raise ValueError("Revision 1 registry must contain the exact ordered profile set")
        if self.revision1_admission_profile_ids != REVISION1_PROFILE_IDS:
            raise ValueError("Revision 1 admission profile IDs are not exact")
        known = set(ids)
        if any(profile.matched_control_profile_id not in known for profile in self.profiles):
            raise ValueError("matched appearance controls must name a registry profile")
        if any(
            profile.profile_version != "appearance_profile_v3"
            for profile in self.profiles
            if profile.profile_id in REVISION1_PROFILE_IDS
        ):
            raise ValueError("Revision 1 profiles require appearance_profile_v3")
        return self


class EvaluationSeedRegistry(StrictAppearanceModel):
    registry_version: Literal["evaluation_seed_candidate_registry_v0"]
    root_seed: Literal[1729]
    derivation_rule: Literal["derive_seed_v1"]
    namespace: Literal["gate0b-slice5-appearance-candidate"]
    indices: tuple[int, int, int, int, int, int, int, int]
    candidate_episode_seeds: tuple[int, int, int, int, int, int, int, int]

    @model_validator(mode="after")
    def seeds_are_exact_and_prospective(self) -> EvaluationSeedRegistry:
        if self.indices != tuple(range(8)):
            raise ValueError("candidate seed indices must be exactly 0 through 7")
        expected = tuple(
            derive_seed(self.root_seed, f"{self.namespace}:{index}") for index in self.indices
        )
        if self.candidate_episode_seeds != expected:
            raise ValueError("candidate episode seeds differ from derive_seed_v1 recomputation")
        if len(set(self.candidate_episode_seeds)) != 8:
            raise ValueError("candidate episode seeds must be unique")
        return self


class QualificationSeedRegistry(StrictAppearanceModel):
    registry_version: Literal["appearance_revision1_qualification_seed_registry_v0"]
    root_seed: Literal[271828]
    derivation_rule: Literal["derive_seed_v1"]
    namespace: Literal["gate0b-appearance-revision1-qualification"]
    indices: tuple[int, int, int, int, int, int, int, int]
    candidate_episode_seeds: tuple[int, int, int, int, int, int, int, int]

    @model_validator(mode="after")
    def seeds_are_exact_untouched_and_disjoint(self) -> QualificationSeedRegistry:
        if self.indices != tuple(range(8)):
            raise ValueError("qualification seed indices must be exactly 0 through 7")
        expected = tuple(
            derive_seed(self.root_seed, f"{self.namespace}:{index}") for index in self.indices
        )
        if self.candidate_episode_seeds != expected:
            raise ValueError("qualification seeds differ from derive_seed_v1 recomputation")
        if len(set(self.candidate_episode_seeds)) != 8:
            raise ValueError("qualification seeds must be unique")
        if set(self.candidate_episode_seeds) & set(CANONICAL_DESIGN_SEEDS):
            raise ValueError("qualification seeds collide with canonical design seeds")
        return self


class FinalEvaluationSeedRegistry(StrictAppearanceModel):
    """Prospectively locked public episode roots for Appearance Benchmark v0."""

    registry_version: Literal["appearance_benchmark_v0_evaluation_episode_seed_registry_v1"]
    root_seed: Literal[314159]
    derivation_rule: Literal["derive_seed_v1"]
    namespace: Literal["gate0b-appearance-benchmark-v0-final-evaluation"]
    indices: tuple[int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int]
    candidate_episode_seeds: tuple[
        int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int
    ]

    @model_validator(mode="after")
    def roots_are_exact_disjoint_uint64_values(self) -> FinalEvaluationSeedRegistry:
        if self.indices != tuple(range(16)):
            raise ValueError("final evaluation indices must be exactly 0 through 15")
        if any(type(value) is not int or value < 0 or value >= 2**64 for value in self.indices):
            raise ValueError("final evaluation indices must be unsigned JSON integers")
        if any(
            type(value) is not int or value < 0 or value >= 2**64
            for value in self.candidate_episode_seeds
        ):
            raise ValueError("final evaluation roots must be unsigned 64-bit JSON integers")
        expected = tuple(
            derive_seed(self.root_seed, f"{self.namespace}:{index}") for index in self.indices
        )
        if self.candidate_episode_seeds != expected:
            raise ValueError("final evaluation roots differ from derive_seed_v1 recomputation")
        if len(set(self.candidate_episode_seeds)) != 16:
            raise ValueError("final evaluation roots must be unique")
        protected = set(CANONICAL_DESIGN_SEEDS) | {
            derive_seed(271828, f"gate0b-appearance-revision1-qualification:{index}")
            for index in range(8)
        }
        if set(self.candidate_episode_seeds) & protected:
            raise ValueError("final evaluation roots collide with protected apparatus roots")
        return self


AppearanceRegistryType = AppearanceRegistry | AppearanceRevision1Registry
SeedRegistryType = EvaluationSeedRegistry | QualificationSeedRegistry | FinalEvaluationSeedRegistry


class AppearanceSeeds(StrictAppearanceModel):
    appearance_base_seed: int = Field(ge=0)
    style_assignment_seed: int = Field(ge=0)
    texture_phase_seed: int = Field(ge=0)
    texture_slot_seed: int = Field(ge=0)
    illumination_seed: int = Field(ge=0)
    candidate_schedule_index: int | None = Field(default=None, ge=0, le=15)


class ProceduralTextureRecord(StrictAppearanceModel):
    semantic_surface_name: str = Field(min_length=1)
    style_slot_id: str = Field(pattern=r"^style-slot-[0-3]$")
    generator_version: Literal["repository_procedural_texture_v1"]
    family: TextureFamily
    resolution: Literal[128]
    cycles_per_tile: int = Field(ge=0, le=64)
    orientation: TextureOrientation
    phase_offset: int = Field(ge=0, lt=128)
    source_texture_logical_sha256: Sha256 | None

    @model_validator(mode="after")
    def texture_hash_presence_matches_family(self) -> ProceduralTextureRecord:
        if (self.family == TextureFamily.SOLID) != (self.source_texture_logical_sha256 is None):
            raise ValueError("only spatial textures have source-array identities")
        return self


class AppearanceInstanceRecord(StrictAppearanceModel):
    appearance_instance_version: Literal["appearance_instance_v3", "appearance_instance_v4"]
    registry_version: Literal[
        "appearance_candidate_registry_v1", "appearance_candidate_registry_v2"
    ]
    profile: AppearanceProfile
    appearance_registry_sha256: Sha256
    appearance_profile_sha256: Sha256
    appearance_instance_sha256: Sha256
    evaluation_seed_registry_version: Literal[
        "evaluation_seed_candidate_registry_v0",
        "appearance_revision1_qualification_seed_registry_v0",
        "appearance_benchmark_v0_evaluation_episode_seed_registry_v1",
    ]
    evaluation_seed_registry_sha256: Sha256
    assignment_schedule_source: Literal[
        "snapshotted_evaluation_seed_registry_v1",
        "snapshotted_revision_partition_seed_registry_v1",
    ]
    assignment_root_seed: int = Field(ge=0)
    assignment_schedule_posture: Literal[
        "candidate_registry_index",
        "non_candidate_seed_derived",
    ]
    seeds: AppearanceSeeds
    style_assignment: dict[str, str]
    textures: tuple[ProceduralTextureRecord, ...]
    asset_origin: Literal["repository_generated_procedural"]
    freeze_status: Literal["candidate_not_frozen"]

    @model_validator(mode="after")
    def assignment_and_textures_are_exact(self) -> AppearanceInstanceRecord:
        if self.registry_version == APPEARANCE_REGISTRY_VERSION:
            if (
                self.appearance_instance_version != APPEARANCE_INSTANCE_VERSION
                or self.evaluation_seed_registry_version != EVALUATION_SEED_REGISTRY_VERSION
                or self.assignment_schedule_source != ASSIGNMENT_SCHEDULE_SOURCE
            ):
                raise ValueError("canonical v1 appearance instance version binding is inconsistent")
        elif (
            self.appearance_instance_version != APPEARANCE_REVISION1_INSTANCE_VERSION
            or self.assignment_schedule_source != REVISION1_ASSIGNMENT_SCHEDULE_SOURCE
        ):
            raise ValueError("Revision 1 appearance instance version binding is inconsistent")
        expected_posture = (
            "candidate_registry_index"
            if self.seeds.candidate_schedule_index is not None
            else "non_candidate_seed_derived"
        )
        if self.assignment_schedule_posture != expected_posture:
            raise ValueError("appearance assignment schedule posture is inconsistent")
        if set(self.style_assignment) != {
            texture.semantic_surface_name for texture in self.textures
        }:
            raise ValueError(
                "appearance assignment and texture records must cover the same surfaces"
            )
        if len(self.textures) != len(self.style_assignment):
            raise ValueError("appearance texture records must be unique per controlled surface")
        for texture in self.textures:
            if self.style_assignment[texture.semantic_surface_name] != texture.style_slot_id:
                raise ValueError("texture style slot differs from the declared assignment")
        if self.profile.style_assignment_rule == StyleAssignmentRule.BALANCED_CYCLIC_PERMUTATION:
            if len(set(self.style_assignment.values())) != len(self.style_assignment):
                raise ValueError("balanced appearance assignments must be bijective")
        return self


@dataclass(frozen=True)
class AppearanceRenderPlan:
    record: AppearanceInstanceRecord
    rgba_by_surface: dict[str, str]
    material_by_surface: dict[str, str]
    asset_bytes: dict[str, bytes]
    asset_xml: str
    light_direction: str
    light_diffuse: str
    light_ambient: str


def parse_appearance_registry(payload: object) -> AppearanceRegistryType:
    if not isinstance(payload, dict):
        raise ValueError("appearance registry must be a mapping")
    version = payload.get("registry_version")
    model = (
        AppearanceRegistry
        if version == APPEARANCE_REGISTRY_VERSION
        else AppearanceRevision1Registry
        if version == APPEARANCE_REVISION1_REGISTRY_VERSION
        else None
    )
    if model is None:
        raise ValueError(f"unsupported appearance registry version: {version}")
    return model.model_validate_json(json.dumps(payload))


def load_appearance_registry_any(path: Path) -> AppearanceRegistryType:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("appearance registry must be a YAML mapping")
    return parse_appearance_registry(payload)


def load_appearance_registry(path: Path) -> AppearanceRegistry:
    registry = load_appearance_registry_any(path)
    if not isinstance(registry, AppearanceRegistry):
        raise ValueError("canonical appearance registry v1 required")
    return registry


def parse_seed_registry(payload: object) -> SeedRegistryType:
    if not isinstance(payload, dict):
        raise ValueError("appearance seed registry must be a mapping")
    version = payload.get("registry_version")
    model = (
        EvaluationSeedRegistry
        if version == EVALUATION_SEED_REGISTRY_VERSION
        else QualificationSeedRegistry
        if version == QUALIFICATION_SEED_REGISTRY_VERSION
        else FinalEvaluationSeedRegistry
        if version == FINAL_EVALUATION_SEED_REGISTRY_VERSION
        else None
    )
    if model is None:
        raise ValueError(f"unsupported appearance seed registry version: {version}")
    return model.model_validate_json(json.dumps(payload))


def load_seed_registry(path: Path) -> SeedRegistryType:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("appearance seed registry must be a YAML mapping")
    return parse_seed_registry(payload)


def load_evaluation_seed_registry(path: Path) -> EvaluationSeedRegistry:
    registry = load_seed_registry(path)
    if not isinstance(registry, EvaluationSeedRegistry):
        raise ValueError("canonical evaluation seed registry v0 required")
    return registry


def configured_appearance_render_plan(
    profile_id: str,
    scene_family: Literal["single_occluder", "corridor"],
    surface_names: tuple[str, ...],
    root_seed: int,
    episode_index: int = 0,
) -> AppearanceRenderPlan:
    """Resolve the repository registry selection used by direct simulation helpers."""

    repository_root = Path(__file__).resolve().parents[2]
    registry = load_appearance_registry(repository_root / "configs/appearance_candidates_v0.yaml")
    seed_registry = load_evaluation_seed_registry(
        repository_root / "configs/evaluation_seed_candidates_v0.yaml"
    )
    return resolve_appearance(
        registry,
        profile_id,
        scene_family,
        surface_names,
        derive_seed(root_seed, f"episode:{episode_index}"),
        root_seed,
        seed_registry,
    )


def appearance_registry_hash(registry: AppearanceRegistryType) -> str:
    return sha256_bytes(canonical_json_bytes(registry))


def appearance_profile_hash(profile: AppearanceProfile) -> str:
    return sha256_bytes(canonical_json_bytes(profile))


def seed_registry_hash(registry: SeedRegistryType) -> str:
    return sha256_bytes(canonical_json_bytes(registry))


def profile_by_id(registry: AppearanceRegistryType, profile_id: str) -> AppearanceProfile:
    try:
        return next(profile for profile in registry.profiles if profile.profile_id == profile_id)
    except StopIteration as error:
        raise ValueError(f"appearance profile is absent from registry: {profile_id}") from error


def appearance_seed_namespaces(episode_seed: int) -> AppearanceSeeds:
    base = derive_seed(episode_seed, "appearance-base")
    return AppearanceSeeds(
        appearance_base_seed=base,
        style_assignment_seed=derive_seed(base, "style-assignment"),
        texture_phase_seed=derive_seed(base, "texture-phase"),
        texture_slot_seed=derive_seed(base, "texture-slot"),
        illumination_seed=derive_seed(base, "illumination"),
        candidate_schedule_index=None,
    )


def candidate_schedule_index(root_seed: int, registry: SeedRegistryType) -> int | None:
    try:
        return registry.candidate_episode_seeds.index(root_seed)
    except ValueError:
        return None


def style_assignment(
    surface_names: tuple[str, ...],
    rule: StyleAssignmentRule,
    seeds: AppearanceSeeds,
) -> dict[str, str]:
    slot_ids = tuple(f"style-slot-{index}" for index in range(len(surface_names)))
    if rule == StyleAssignmentRule.FIXED_SEMANTIC_REGRESSION:
        return dict(zip(surface_names, slot_ids, strict=True))
    shift = (
        seeds.candidate_schedule_index % len(surface_names)
        if seeds.candidate_schedule_index is not None
        else seeds.style_assignment_seed % len(surface_names)
    )
    rotated = slot_ids[shift:] + slot_ids[:shift]
    return dict(zip(surface_names, rotated, strict=True))


def generate_texture(
    profile: AppearanceProfile,
    slot: PaletteSlot,
    phase_offset: int,
) -> TextureArray:
    """Generate the canonical raw 128x128x3 sRGB byte texture."""

    definition = profile.texture
    if definition.family == TextureFamily.SOLID:
        raise ValueError("solid profiles intentionally have no spatial source texture")
    rows, columns = np.indices((TEXTURE_RESOLUTION, TEXTURE_RESOLUTION), dtype=np.int64)
    if definition.family == TextureFamily.CHECKER:
        cell = TEXTURE_RESOLUTION // (2 * definition.cycles_per_tile)
        if cell <= 0 or TEXTURE_RESOLUTION % (2 * definition.cycles_per_tile) != 0:
            raise ValueError("checker frequency must divide the source texture exactly")
        pattern = ((columns + phase_offset) // cell + (rows + phase_offset) // cell) % 2
    else:
        if definition.orientation == TextureOrientation.AXIS_X:
            coordinate = columns
            span = TEXTURE_RESOLUTION
        elif definition.orientation == TextureOrientation.DIAGONAL_DOWN:
            coordinate = columns + rows
            span = 2 * TEXTURE_RESOLUTION
        else:  # pragma: no cover - strict enum is exhaustive
            raise ValueError("unsupported stripe orientation")
        pattern = ((coordinate + phase_offset) * 2 * definition.cycles_per_tile // span) % 2
    foreground = np.asarray(slot.foreground_rgb, dtype=np.uint8)
    background = np.asarray(slot.background_rgb, dtype=np.uint8)
    return np.asarray(np.where(pattern[..., None] == 0, foreground, background), dtype=np.uint8)


def _png_bytes(array: TextureArray) -> bytes:
    encoded = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(
        encoded,
        format="PNG",
        compress_level=9,
        optimize=False,
    )
    return encoded.getvalue()


def _instance_domain(record: AppearanceInstanceRecord) -> dict[str, Any]:
    return {
        "appearance_instance_version": record.appearance_instance_version,
        "appearance_profile_sha256": record.appearance_profile_sha256,
        "evaluation_seed_registry_version": record.evaluation_seed_registry_version,
        "evaluation_seed_registry_sha256": record.evaluation_seed_registry_sha256,
        "assignment_schedule_source": record.assignment_schedule_source,
        "assignment_root_seed": record.assignment_root_seed,
        "assignment_schedule_posture": record.assignment_schedule_posture,
        "appearance_base_seed": record.seeds.appearance_base_seed,
        "style_assignment_seed": record.seeds.style_assignment_seed,
        "texture_phase_seed": record.seeds.texture_phase_seed,
        "texture_slot_seed": record.seeds.texture_slot_seed,
        "illumination_seed": record.seeds.illumination_seed,
        "candidate_schedule_index": record.seeds.candidate_schedule_index,
        "style_assignment": dict(sorted(record.style_assignment.items())),
        "textures": [texture.model_dump(mode="json") for texture in record.textures],
        "palette": record.profile.palette.model_dump(mode="json"),
        "material": record.profile.material.model_dump(mode="json"),
        "illumination": record.profile.illumination.model_dump(mode="json"),
        "generator_version": APPEARANCE_GENERATOR_VERSION,
        "asset_origin": record.asset_origin,
    }


def appearance_instance_hash(record: AppearanceInstanceRecord) -> str:
    return sha256_bytes(canonical_json_bytes(_instance_domain(record)))


def resolve_appearance(
    registry: AppearanceRegistryType,
    profile_id: str,
    scene_family: Literal["single_occluder", "corridor"],
    surface_names: tuple[str, ...],
    episode_seed: int,
    root_seed: int,
    seed_registry: SeedRegistryType,
) -> AppearanceRenderPlan:
    profile = profile_by_id(registry, profile_id)
    seeds = appearance_seed_namespaces(episode_seed)
    schedule_index = candidate_schedule_index(root_seed, seed_registry)
    seeds = seeds.model_copy(update={"candidate_schedule_index": schedule_index})
    assignment = style_assignment(surface_names, profile.style_assignment_rule, seeds)
    slots = (
        profile.palette.single_occluder_slots
        if scene_family == "single_occluder"
        else profile.palette.corridor_slots
    )
    slots_by_id = {slot.slot_id: slot for slot in slots}
    texture_records: list[ProceduralTextureRecord] = []
    rgba_by_surface: dict[str, str] = {}
    material_by_surface: dict[str, str] = {}
    assets: dict[str, bytes] = {}
    asset_xml: list[str] = []
    for surface_index, surface_name in enumerate(surface_names):
        slot_id = assignment[surface_name]
        slot = slots_by_id[slot_id]
        phase_seed = derive_seed(
            seeds.texture_phase_seed,
            f"slot:{slot_id}:surface-index:{surface_index}",
        )
        phase_offset = phase_seed % TEXTURE_RESOLUTION
        source_hash: str | None = None
        if profile.texture.family == TextureFamily.SOLID:
            rgb = tuple(channel / 255.0 for channel in slot.foreground_rgb)
            rgba_by_surface[surface_name] = f"{rgb[0]} {rgb[1]} {rgb[2]} 1"
            material_by_surface[surface_name] = ""
        else:
            texture = generate_texture(profile, slot, phase_offset)
            source_hash = logical_array_hash(texture)
            asset_name = f"appearance-{surface_index}.png"
            texture_name = f"appearance_texture_{surface_index}"
            material_name = f"appearance_material_{surface_index}"
            assets[asset_name] = _png_bytes(texture)
            asset_xml.append(f'<texture name="{texture_name}" type="2d" file="{asset_name}"/>')
            repeat_x, repeat_y = profile.material.texrepeat
            asset_xml.append(
                f'<material name="{material_name}" texture="{texture_name}" '
                f'texrepeat="{repeat_x} {repeat_y}" texuniform="false" '
                'specular="0" shininess="0" reflectance="0"/>'
            )
            rgba_by_surface[surface_name] = "1 1 1 1"
            material_by_surface[surface_name] = f' material="{material_name}"'
        texture_records.append(
            ProceduralTextureRecord(
                semantic_surface_name=surface_name,
                style_slot_id=slot_id,
                generator_version="repository_procedural_texture_v1",
                family=profile.texture.family,
                resolution=128,
                cycles_per_tile=profile.texture.cycles_per_tile,
                orientation=profile.texture.orientation,
                phase_offset=phase_offset,
                source_texture_logical_sha256=source_hash,
            )
        )
    revision1 = registry.registry_version == APPEARANCE_REVISION1_REGISTRY_VERSION
    provisional = AppearanceInstanceRecord(
        appearance_instance_version=(
            APPEARANCE_REVISION1_INSTANCE_VERSION if revision1 else APPEARANCE_INSTANCE_VERSION
        ),
        registry_version=registry.registry_version,
        profile=profile,
        appearance_registry_sha256=appearance_registry_hash(registry),
        appearance_profile_sha256=appearance_profile_hash(profile),
        appearance_instance_sha256="0" * 64,
        evaluation_seed_registry_version=seed_registry.registry_version,
        evaluation_seed_registry_sha256=seed_registry_hash(seed_registry),
        assignment_schedule_source=(
            REVISION1_ASSIGNMENT_SCHEDULE_SOURCE if revision1 else ASSIGNMENT_SCHEDULE_SOURCE
        ),
        assignment_root_seed=root_seed,
        assignment_schedule_posture=(
            "candidate_registry_index"
            if schedule_index is not None
            else "non_candidate_seed_derived"
        ),
        seeds=seeds,
        style_assignment=assignment,
        textures=tuple(texture_records),
        asset_origin="repository_generated_procedural",
        freeze_status="candidate_not_frozen",
    )
    record = provisional.model_copy(
        update={"appearance_instance_sha256": appearance_instance_hash(provisional)}
    )
    light = (
        profile.illumination.single_occluder
        if scene_family == "single_occluder"
        else profile.illumination.corridor
    )
    return AppearanceRenderPlan(
        record=record,
        rgba_by_surface=rgba_by_surface,
        material_by_surface=material_by_surface,
        asset_bytes=assets,
        asset_xml="\n    ".join(asset_xml),
        light_direction=" ".join(str(value) for value in light.direction),
        light_diffuse=" ".join(str(light.diffuse_intensity) for _ in range(3)),
        light_ambient=" ".join(str(value) for value in profile.illumination.ambient_rgb),
    )


def validate_appearance_instance(
    record: AppearanceInstanceRecord,
    registry: AppearanceRegistryType,
    scene_family: Literal["single_occluder", "corridor"],
    surface_names: tuple[str, ...],
    episode_seed: int,
    root_seed: int,
    seed_registry: SeedRegistryType,
) -> AppearanceRenderPlan:
    expected = resolve_appearance(
        registry,
        record.profile.profile_id,
        scene_family,
        surface_names,
        episode_seed,
        root_seed,
        seed_registry,
    )
    if record != expected.record:
        raise ValueError("appearance instance differs from independent procedural recomputation")
    return expected


def validate_axis_isolation(registry: AppearanceRegistryType) -> None:
    """Require declared scientific axes to equal independently observed profile changes."""

    by_id = {profile.profile_id: profile for profile in registry.profiles}
    for profile in registry.profiles:
        AppearanceProfile.model_validate(profile.model_dump(mode="python"))
    canonical_spatial_introduction_baseline = by_id["balanced_checker_low_v1"].texture
    revision_spatial_introduction_baseline = (
        by_id["revision1_checker_low_v1"].texture
        if isinstance(registry, AppearanceRevision1Registry)
        else canonical_spatial_introduction_baseline
    )
    canonical_solid = by_id["balanced_solid_palette_v1"]

    def observed_axes(
        profile: AppearanceProfile,
        reference: AppearanceProfile,
    ) -> set[AxisTag]:
        spatial_introduction_baseline = (
            revision_spatial_introduction_baseline
            if profile.profile_id in REVISION1_PROFILE_IDS
            else canonical_spatial_introduction_baseline
        )
        if profile.material != reference.material:
            raise ValueError("appearance candidates cannot change the undeclared material domain")
        if profile.non_degeneracy_thresholds != reference.non_degeneracy_thresholds:
            raise ValueError("appearance candidates cannot change admission thresholds")
        texture = profile.texture.model_dump(mode="json")
        reference_texture = reference.texture.model_dump(mode="json")
        for key in (
            "family",
            "cycles_per_tile",
            "orientation",
        ):
            texture.pop(key)
            reference_texture.pop(key)
        if texture != reference_texture:
            raise ValueError("appearance candidates cannot change undeclared texture controls")
        assignment_changed = profile.style_assignment_rule != reference.style_assignment_rule
        permitted_assignment_transition = (
            reference.candidate_class == CandidateClass.LEGACY_REGRESSION_CONTROL
            and profile.candidate_class != CandidateClass.LEGACY_REGRESSION_CONTROL
            and reference.style_assignment_rule == StyleAssignmentRule.FIXED_SEMANTIC_REGRESSION
            and profile.style_assignment_rule == StyleAssignmentRule.BALANCED_CYCLIC_PERMUTATION
        )
        if assignment_changed and not permitted_assignment_transition:
            raise ValueError("appearance candidate changes the assignment domain unexpectedly")

        changed: set[AxisTag] = set()
        if profile.palette != reference.palette:
            changed.add(AxisTag.COLOUR)
        if profile.texture.family != reference.texture.family:
            changed.add(AxisTag.TEXTURE_FAMILY)
        both_spatial = (
            profile.texture.family != TextureFamily.SOLID
            and reference.texture.family != TextureFamily.SOLID
        )
        if both_spatial and profile.texture.cycles_per_tile != reference.texture.cycles_per_tile:
            changed.add(AxisTag.TEXTURE_FREQUENCY)
        elif (
            profile.texture.family != TextureFamily.SOLID
            and reference.texture.family == TextureFamily.SOLID
            and profile.texture.cycles_per_tile != spatial_introduction_baseline.cycles_per_tile
        ):
            changed.add(AxisTag.TEXTURE_FREQUENCY)
        either_spatial = (
            profile.texture.family != TextureFamily.SOLID
            or reference.texture.family != TextureFamily.SOLID
        )
        if either_spatial and profile.texture.orientation != reference.texture.orientation:
            changed.add(AxisTag.TEXTURE_ORIENTATION)
        elif (
            profile.texture.family != TextureFamily.SOLID
            and reference.texture.family == TextureFamily.SOLID
            and profile.texture.orientation != spatial_introduction_baseline.orientation
        ):
            changed.add(AxisTag.TEXTURE_ORIENTATION)
        if profile.illumination != reference.illumination:
            changed.add(AxisTag.ILLUMINATION)
        return changed

    scientific_tags = set(AxisTag) - {AxisTag.CONTROL, AxisTag.COMBINED}
    for profile in registry.profiles:
        if profile.candidate_class == CandidateClass.LEGACY_REGRESSION_CONTROL:
            if profile.matched_control_profile_id != profile.profile_id:
                raise ValueError("legacy controls must match themselves")
            continue
        reference = (
            canonical_solid
            if profile.profile_id in CANONICAL_PROFILE_IDS
            and profile.candidate_class == CandidateClass.COMBINED_STRESS_CANDIDATE
            else by_id[profile.matched_control_profile_id]
        )
        actual = observed_axes(profile, reference)
        declared = set(profile.axis_tags) & scientific_tags
        if profile.candidate_class == CandidateClass.SINGLE_AXIS_CANDIDATE:
            if len(declared) != 1 or declared != actual:
                raise ValueError(
                    f"single-axis declaration differs from actual changes: {profile.profile_id}"
                )
        elif AxisTag.COMBINED not in profile.axis_tags or len(actual) < 2 or declared != actual:
            raise ValueError(
                f"combined declaration differs from actual changes: {profile.profile_id}"
            )

    frequency_pairs = [("balanced_checker_low_v1", "balanced_checker_high_v1")]
    if isinstance(registry, AppearanceRevision1Registry):
        frequency_pairs.append(("revision1_checker_low_v1", "revision1_checker_high_v1"))
    for low_id, high_id in frequency_pairs:
        low = by_id[low_id]
        high = by_id[high_id]
        if high.texture.cycles_per_tile < 4 * low.texture.cycles_per_tile:
            raise ValueError("high-frequency checker is less than four times its low partner")


def assignment_balance(
    profile: AppearanceProfile,
    surface_names: tuple[str, ...],
    candidate_seeds: tuple[int, ...],
    seed_registry: SeedRegistryType,
) -> dict[str, dict[str, int]]:
    counts = {
        surface: {f"style-slot-{slot}": 0 for slot in range(len(surface_names))}
        for surface in surface_names
    }
    for candidate_seed in candidate_seeds:
        seeds = appearance_seed_namespaces(derive_seed(candidate_seed, "episode:0"))
        seeds = seeds.model_copy(
            update={
                "candidate_schedule_index": candidate_schedule_index(candidate_seed, seed_registry)
            }
        )
        mapping = style_assignment(surface_names, profile.style_assignment_rule, seeds)
        for surface, slot in mapping.items():
            counts[surface][slot] += 1
    return counts
