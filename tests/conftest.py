from pathlib import Path

import pytest

from epsbench.config import AppearanceConfig, BenchmarkConfig, load_config
from epsbench.data import generate_dataset


@pytest.fixture(scope="session")
def benchmark_config() -> BenchmarkConfig:
    return load_config(Path("configs/benchmark_v0.yaml"))


@pytest.fixture(scope="session")
def deterministic_datasets(
    tmp_path_factory: pytest.TempPathFactory,
    benchmark_config: BenchmarkConfig,
) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("determinism")
    first = root / "first"
    second = root / "second"
    generate_dataset(benchmark_config, 2, first)
    generate_dataset(benchmark_config, 2, second)
    return first, second


@pytest.fixture(scope="session")
def smoke_dataset(deterministic_datasets: tuple[Path, Path]) -> Path:
    return deterministic_datasets[0]


@pytest.fixture(scope="session")
def appearance_datasets(
    tmp_path_factory: pytest.TempPathFactory,
    benchmark_config: BenchmarkConfig,
) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("appearance")
    alternate = BenchmarkConfig.model_validate(
        {
            **benchmark_config.model_dump(mode="python"),
            "appearance": AppearanceConfig(variant="alternate").model_dump(mode="python"),
        }
    )
    base_path = root / "base"
    alternate_path = root / "alternate"
    generate_dataset(benchmark_config, 1, base_path)
    generate_dataset(alternate, 1, alternate_path)
    return base_path, alternate_path
