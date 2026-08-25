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
