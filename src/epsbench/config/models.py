"""Typed, discriminated configuration for the authorised Gate 0B scene families."""

import math
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from epsbench.schema import SceneFamily


class StrictConfigModel(BaseModel):
    """Configuration base that rejects misspelled or unexpected fields."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class RenderConfig(StrictConfigModel):
    width: int = Field(ge=64, le=1024)
    height: int = Field(ge=64, le=1024)


class SingleOccluderCameraConfig(StrictConfigModel):
    before_lateral: float
    after_lateral: float
    forward: float
    height: float = Field(gt=0.1)
    field_of_view_degrees: float = Field(gt=10.0, lt=150.0)

    @model_validator(mode="after")
    def scalars_are_finite(self) -> "SingleOccluderCameraConfig":
        values = (
            self.before_lateral,
            self.after_lateral,
            self.forward,
            self.height,
            self.field_of_view_degrees,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("all camera configuration values must be finite")
        return self


class CorridorCameraConfig(StrictConfigModel):
    lateral_position: float
    starting_forward_position: float = Field(gt=0.0)
    height: float = Field(gt=0.1)
    field_of_view_degrees: float = Field(gt=20.0, lt=80.0)

    @model_validator(mode="after")
    def scalars_are_finite(self) -> "CorridorCameraConfig":
        values = (
            self.lateral_position,
            self.starting_forward_position,
            self.height,
            self.field_of_view_degrees,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("all corridor camera values must be finite")
        return self


class LateralActionConfig(StrictConfigModel):
    name: Literal["lateral_right", "lateral_left"]
    delta_forward: float
    delta_lateral: float
    delta_yaw: float

    @model_validator(mode="after")
    def supported_lateral_action_is_truthful(self) -> "LateralActionConfig":
        if not all(
            math.isfinite(value)
            for value in (self.delta_forward, self.delta_lateral, self.delta_yaw)
        ):
            raise ValueError("lateral action values must be finite")
        if self.delta_forward != 0.0 or self.delta_yaw != 0.0:
            raise ValueError("lateral actions require zero forward displacement and yaw")
        if self.delta_lateral == 0.0:
            raise ValueError("lateral displacement must be non-zero")
        if self.name == "lateral_right" and self.delta_lateral < 0.0:
            raise ValueError("lateral_right requires positive lateral displacement")
        if self.name == "lateral_left" and self.delta_lateral > 0.0:
            raise ValueError("lateral_left requires negative lateral displacement")
        return self


class ForwardActionConfig(StrictConfigModel):
    name: Literal["forward"]
    delta_forward: float = Field(gt=0.0)
    delta_lateral: float
    delta_yaw: float

    @model_validator(mode="after")
    def supported_forward_action_is_truthful(self) -> "ForwardActionConfig":
        if not all(
            math.isfinite(value)
            for value in (self.delta_forward, self.delta_lateral, self.delta_yaw)
        ):
            raise ValueError("forward action values must be finite")
        if self.delta_lateral != 0.0 or self.delta_yaw != 0.0:
            raise ValueError("forward actions require zero lateral displacement and yaw")
        return self


class AppearanceConfig(StrictConfigModel):
    registry_version: Literal["appearance_candidate_registry_v0"]
    profile_id: str = Field(pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*_v[0-9]+$")


class SampleRangeConfig(StrictConfigModel):
    minimum: float
    maximum: float

    @model_validator(mode="after")
    def range_is_finite_and_ordered(self) -> "SampleRangeConfig":
        if not math.isfinite(self.minimum) or not math.isfinite(self.maximum):
            raise ValueError("sample range bounds must be finite")
        if self.minimum <= 0.0 or self.maximum < self.minimum:
            raise ValueError("sample range must be positive and ordered")
        return self


class CorridorGeometryConfig(StrictConfigModel):
    width: SampleRangeConfig
    length: SampleRangeConfig
    wall_height: float = Field(ge=2.0, le=8.0)

    @model_validator(mode="after")
    def geometry_is_finite_and_bounded(self) -> "CorridorGeometryConfig":
        if not math.isfinite(self.wall_height):
            raise ValueError("corridor wall height must be finite")
        if self.width.minimum < 1.5 or self.width.maximum > 8.0:
            raise ValueError("corridor width range must remain within [1.5, 8.0]")
        if self.length.minimum < 3.0 or self.length.maximum > 12.0:
            raise ValueError("corridor length range must remain within [3.0, 12.0]")
        return self


class SingleOccluderConfig(StrictConfigModel):
    schema_version: Literal["0.1.0-dev.3"]
    scene_family: Literal[SceneFamily.SINGLE_OCCLUDER]
    seed: int = Field(ge=0, lt=2**64)
    render: RenderConfig
    camera: SingleOccluderCameraConfig
    action: LateralActionConfig
    appearance: AppearanceConfig

    @model_validator(mode="after")
    def action_matches_camera_motion(self) -> "SingleOccluderConfig":
        observed_delta = self.camera.after_lateral - self.camera.before_lateral
        if not math.isclose(
            observed_delta,
            self.action.delta_lateral,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("camera lateral displacement must equal the executed action")
        return self


class CorridorConfig(StrictConfigModel):
    schema_version: Literal["0.1.0-dev.3"]
    scene_family: Literal[SceneFamily.CORRIDOR]
    seed: int = Field(ge=0, lt=2**64)
    render: RenderConfig
    geometry: CorridorGeometryConfig
    camera: CorridorCameraConfig
    action: ForwardActionConfig
    appearance: AppearanceConfig

    @model_validator(mode="after")
    def camera_path_is_legal_for_every_sample(self) -> "CorridorConfig":
        wall_clearance = 0.1
        half_minimum_width = self.geometry.width.minimum / 2.0
        if abs(self.camera.lateral_position) >= half_minimum_width - wall_clearance:
            raise ValueError("corridor camera lateral position intersects a side wall")
        after_forward = self.camera.starting_forward_position + self.action.delta_forward
        if after_forward >= self.geometry.length.minimum - wall_clearance:
            raise ValueError("corridor camera action reaches or crosses the end wall")
        if self.camera.height >= self.geometry.wall_height:
            raise ValueError("corridor camera height must remain below the wall height")

        vertical_half_tangent = math.tan(math.radians(self.camera.field_of_view_degrees) / 2.0)
        highest_end_ray = (
            self.camera.height
            + (self.geometry.length.maximum - self.camera.starting_forward_position)
            * vertical_half_tangent
        )
        if highest_end_ray >= self.geometry.wall_height:
            raise ValueError(
                "corridor walls must cover the optical field; increase wall height or reduce "
                "length/field of view"
            )
        return self


BenchmarkConfig = Annotated[
    SingleOccluderConfig | CorridorConfig,
    Field(discriminator="scene_family"),
]
BENCHMARK_CONFIG_ADAPTER: TypeAdapter[BenchmarkConfig] = TypeAdapter(BenchmarkConfig)


def parse_config(payload: object) -> BenchmarkConfig:
    """Parse a strict scene-family-discriminated configuration payload."""

    return BENCHMARK_CONFIG_ADAPTER.validate_python(payload)


def load_config(path: Path) -> BenchmarkConfig:
    """Load a strict YAML configuration without applying implicit overrides."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("benchmark configuration must be a YAML mapping")
    return parse_config(payload)
