from pathlib import Path

import pytest
from pydantic import ValidationError

from epsbench.config import BenchmarkConfig, load_config


def test_checked_in_config_is_strict_and_valid() -> None:
    config = load_config(Path("configs/benchmark_v0.yaml"))
    assert isinstance(config, BenchmarkConfig)
    assert config.camera.after_lateral - config.camera.before_lateral == pytest.approx(
        config.action.delta_lateral
    )


def test_unknown_configuration_field_is_rejected() -> None:
    config = load_config(Path("configs/benchmark_v0.yaml"))
    payload = config.model_dump(mode="python")
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        BenchmarkConfig.model_validate(payload)


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
        BenchmarkConfig.model_validate(payload)
