"""Typed configuration for the Milestone 0 vertical slice."""

import math
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictConfigModel(BaseModel):
    """Configuration base that rejects misspelled or unexpected fields."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class RenderConfig(StrictConfigModel):
    width: int = Field(ge=64, le=1024)
    height: int = Field(ge=64, le=1024)


class CameraConfig(StrictConfigModel):
    before_lateral: float
    after_lateral: float
    forward: float
    height: float = Field(gt=0.1)
    field_of_view_degrees: float = Field(gt=10.0, lt=150.0)

    @model_validator(mode="after")
    def scalars_are_finite(self) -> "CameraConfig":
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


class ActionConfig(StrictConfigModel):
    name: Literal["lateral_right", "lateral_left"]
    delta_forward: float
    delta_lateral: float
    delta_yaw: float

    @model_validator(mode="after")
    def supported_lateral_action_is_truthful(self) -> "ActionConfig":
        values = (self.delta_forward, self.delta_lateral, self.delta_yaw)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("all action values must be finite")
        if self.delta_forward != 0.0:
            raise ValueError("forward camera motion is not supported in this vertical slice")
        if self.delta_yaw != 0.0:
            raise ValueError("yaw camera motion is not supported in this vertical slice")
        if self.delta_lateral == 0.0:
            raise ValueError("lateral displacement must be non-zero")
        if self.name == "lateral_right" and self.delta_lateral < 0.0:
            raise ValueError("lateral_right requires positive lateral displacement")
        if self.name == "lateral_left" and self.delta_lateral > 0.0:
            raise ValueError("lateral_left requires negative lateral displacement")
        return self


class AppearanceConfig(StrictConfigModel):
    variant: Literal["base", "alternate"]


class BenchmarkConfig(StrictConfigModel):
    schema_version: Literal["0.1.0-dev.1"]
    seed: int = Field(ge=0, lt=2**63)
    render: RenderConfig
    camera: CameraConfig
    action: ActionConfig
    appearance: AppearanceConfig

    @model_validator(mode="after")
    def action_matches_camera_motion(self) -> "BenchmarkConfig":
        observed_delta = self.camera.after_lateral - self.camera.before_lateral
        if not math.isclose(
            observed_delta,
            self.action.delta_lateral,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("camera lateral displacement must equal the executed action")
        return self


def load_config(path: Path) -> BenchmarkConfig:
    """Load a strict YAML configuration without applying implicit overrides."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("benchmark configuration must be a YAML mapping")
    return BenchmarkConfig.model_validate(payload)
