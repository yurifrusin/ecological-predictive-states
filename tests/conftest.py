from pathlib import Path

import pytest

from epsbench.config import (
    AppearanceConfig,
    CorridorConfig,
    SingleOccluderConfig,
    load_config,
)
from epsbench.data import generate_dataset


@pytest.fixture(scope="session")
def benchmark_config() -> SingleOccluderConfig:
    config = load_config(Path("configs/benchmark_v0.yaml"))
    assert isinstance(config, SingleOccluderConfig)
    return config


@pytest.fixture(scope="session")
def corridor_config() -> CorridorConfig:
    config = load_config(Path("configs/corridor_v0.yaml"))
    assert isinstance(config, CorridorConfig)
    return config


@pytest.fixture(scope="session")
def deterministic_datasets(
    tmp_path_factory: pytest.TempPathFactory,
    benchmark_config: SingleOccluderConfig,
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
    benchmark_config: SingleOccluderConfig,
) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("appearance")
    alternate = SingleOccluderConfig.model_validate(
        {
            **benchmark_config.model_dump(mode="python"),
            "appearance": AppearanceConfig(
                registry_version="appearance_candidate_registry_v1",
                profile_id="legacy_solid_alternate_v1",
            ).model_dump(mode="python"),
        }
    )
    base_path = root / "base"
    alternate_path = root / "alternate"
    generate_dataset(benchmark_config, 1, base_path)
    generate_dataset(alternate, 1, alternate_path)
    return base_path, alternate_path


@pytest.fixture(scope="session")
def deterministic_corridor_datasets(
    tmp_path_factory: pytest.TempPathFactory,
    corridor_config: CorridorConfig,
) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("corridor-determinism")
    first = root / "first"
    second = root / "second"
    generate_dataset(corridor_config, 2, first)
    generate_dataset(corridor_config, 2, second)
    return first, second


@pytest.fixture(scope="session")
def corridor_dataset(deterministic_corridor_datasets: tuple[Path, Path]) -> Path:
    return deterministic_corridor_datasets[0]


@pytest.fixture(scope="session")
def corridor_appearance_datasets(
    tmp_path_factory: pytest.TempPathFactory,
    corridor_config: CorridorConfig,
) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("corridor-appearance")
    alternate = CorridorConfig.model_validate(
        {
            **corridor_config.model_dump(mode="python"),
            "appearance": AppearanceConfig(
                registry_version="appearance_candidate_registry_v1",
                profile_id="legacy_solid_alternate_v1",
            ).model_dump(mode="python"),
        }
    )
    base_path = root / "base"
    alternate_path = root / "alternate"
    generate_dataset(corridor_config, 1, base_path)
    generate_dataset(alternate, 1, alternate_path)
    return base_path, alternate_path
