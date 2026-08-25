"""Typed configuration for the Milestone 0 vertical slice."""

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


class ActionConfig(StrictConfigModel):
    name: Literal["lateral_right", "lateral_left"]
    delta_forward: float
    delta_lateral: float
    delta_yaw: float


class AppearanceConfig(StrictConfigModel):
    variant: Literal["base", "alternate"]


class BenchmarkConfig(StrictConfigModel):
    schema_version: Literal["0.1.0"]
    seed: int = Field(ge=0, lt=2**63)
    render: RenderConfig
    camera: CameraConfig
    action: ActionConfig
    appearance: AppearanceConfig

    @model_validator(mode="after")
    def action_matches_camera_motion(self) -> "BenchmarkConfig":
        observed_delta = self.camera.after_lateral - self.camera.before_lateral
        if abs(observed_delta - self.action.delta_lateral) > 1e-9:
            raise ValueError("camera lateral displacement must equal the executed action")
        return self


def load_config(path: Path) -> BenchmarkConfig:
    """Load a strict YAML configuration without applying implicit overrides."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("benchmark configuration must be a YAML mapping")
    return BenchmarkConfig.model_validate(payload)
