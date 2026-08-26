from pathlib import Path

import pytest
from pydantic import ValidationError

from epsbench.config import CorridorConfig, SingleOccluderConfig, load_config, parse_config


def test_checked_in_config_is_strict_and_valid() -> None:
    config = load_config(Path("configs/benchmark_v0.yaml"))
    assert isinstance(config, SingleOccluderConfig)
    assert config.camera.after_lateral - config.camera.before_lateral == pytest.approx(
        config.action.delta_lateral
    )


def test_checked_in_corridor_config_is_discriminated_and_valid() -> None:
    config = load_config(Path("configs/corridor_v0.yaml"))
    assert isinstance(config, CorridorConfig)
    assert config.action.name == "forward"
    assert config.camera.starting_forward_position + config.action.delta_forward < (
        config.geometry.length.minimum
    )


def test_unknown_configuration_field_is_rejected() -> None:
    config = load_config(Path("configs/benchmark_v0.yaml"))
    payload = config.model_dump(mode="python")
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        parse_config(payload)


@pytest.mark.parametrize(
    ("name", "delta_forward", "delta_lateral", "delta_yaw"),
    [
        ("lateral_right", 0.1, 0.7, 0.0),
        ("lateral_right", 0.0, 0.7, 0.1),
        ("lateral_right", 0.0, -0.7, 0.0),
        ("lateral_left", 0.0, 0.7, 0.0),
        ("lateral_right", 0.0, 0.0, 0.0),
        ("lateral_right", 0.0, float("nan"), 0.0),
        ("lateral_right", 0.0, float("inf"), 0.0),
        ("lateral_right", 0.0, float("-inf"), 0.0),
    ],
)
def test_unsupported_or_untruthful_actions_are_rejected(
    name: str,
    delta_forward: float,
    delta_lateral: float,
    delta_yaw: float,
) -> None:
    config = load_config(Path("configs/benchmark_v0.yaml"))
    payload = config.model_dump(mode="python")
    payload["action"] = {
        "name": name,
        "delta_forward": delta_forward,
        "delta_lateral": delta_lateral,
        "delta_yaw": delta_yaw,
    }
    with pytest.raises(ValidationError):
        parse_config(payload)


@pytest.mark.parametrize(
    ("delta_forward", "delta_lateral", "delta_yaw"),
    [
        (0.0, 0.0, 0.0),
        (-0.1, 0.0, 0.0),
        (0.7, 0.1, 0.0),
        (0.7, 0.0, 0.1),
        (float("nan"), 0.0, 0.0),
        (float("inf"), 0.0, 0.0),
    ],
)
def test_untruthful_forward_actions_are_rejected(
    delta_forward: float,
    delta_lateral: float,
    delta_yaw: float,
) -> None:
    config = load_config(Path("configs/corridor_v0.yaml"))
    payload = config.model_dump(mode="python")
    payload["action"] = {
        "name": "forward",
        "delta_forward": delta_forward,
        "delta_lateral": delta_lateral,
        "delta_yaw": delta_yaw,
    }
    with pytest.raises(ValidationError):
        parse_config(payload)


def test_scene_specific_configuration_fields_fail_closed() -> None:
    single = load_config(Path("configs/benchmark_v0.yaml")).model_dump(mode="python")
    corridor = load_config(Path("configs/corridor_v0.yaml")).model_dump(mode="python")

    single["geometry"] = corridor["geometry"]
    with pytest.raises(ValidationError):
        parse_config(single)

    corridor["camera"] = {
        "before_lateral": -0.35,
        "after_lateral": 0.35,
        "forward": -3.0,
        "height": 1.25,
        "field_of_view_degrees": 55.0,
    }
    with pytest.raises(ValidationError):
        parse_config(corridor)


def test_unknown_scene_family_fails() -> None:
    payload = load_config(Path("configs/corridor_v0.yaml")).model_dump(mode="python")
    payload["scene_family"] = "unknown"
    with pytest.raises(ValidationError):
        parse_config(payload)
