from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from epsbench.appearance import (
    AppearanceInstanceRecord,
    appearance_instance_hash,
    appearance_profile_hash,
    load_appearance_registry,
    load_evaluation_seed_registry,
)
from epsbench.config import load_config
from epsbench.data import DatasetValidationError, generate_dataset, validate_dataset
from epsbench.utils.canonical import canonical_json_bytes
from tests.dataset_mutations import _write_manifest, load_manifest, rewrite_json_artifact


@pytest.fixture(scope="module")
def textured_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("appearance-corruption") / "checker"
    registry = load_appearance_registry(Path("configs/appearance_candidates_v0.yaml"))
    seeds = load_evaluation_seed_registry(Path("configs/evaluation_seed_candidates_v0.yaml"))
    base = load_config(Path("configs/benchmark_v0.yaml"))
    config = type(base).model_validate(
        {
            **base.model_dump(mode="python"),
            "seed": seeds.candidate_episode_seeds[0],
            "appearance": {
                "registry_version": registry.registry_version,
                "profile_id": "balanced_checker_low_v1",
            },
        }
    )
    generate_dataset(
        config,
        1,
        output,
        appearance_registry=registry,
        seed_registry=seeds,
    )
    validate_dataset(output)
    return output


def _commit_fully_rehashed_appearance(
    root: Path,
    instrumentation_payload: dict[str, Any],
) -> None:
    manifest = load_manifest(root)
    episode = manifest.episodes[0]
    instrumentation_record = rewrite_json_artifact(
        root,
        episode.privileged_instrumentation,
        instrumentation_payload,
    )
    appearance_hash = instrumentation_payload["appearance"]["appearance_instance_sha256"]
    episodes = list(manifest.episodes)
    episodes[0] = episode.model_copy(
        update={
            "privileged_instrumentation": instrumentation_record,
            "appearance_instance_sha256": appearance_hash,
        }
    )
    _write_manifest(root, manifest, episodes)


@pytest.mark.parametrize(
    "corruption",
    [
        "profile_definition",
        "profile_hash",
        "texture_hash",
        "assignment",
        "seed_registry_binding",
        "light_record",
        "instance_identity",
    ],
)
def test_fully_rehashed_fabricated_appearance_records_are_rejected(
    textured_dataset: Path,
    tmp_path: Path,
    corruption: str,
) -> None:
    broken = tmp_path / corruption
    shutil.copytree(textured_dataset, broken)
    manifest = load_manifest(broken)
    instrumentation_path = broken / manifest.episodes[0].privileged_instrumentation.path
    instrumentation = json.loads(instrumentation_path.read_text(encoding="utf-8"))
    appearance = instrumentation["appearance"]
    if corruption in {"profile_definition", "light_record"}:
        light = appearance["profile"]["illumination"]["single_occluder"]
        light["diffuse_intensity"] = 0.8 if corruption == "light_record" else 0.85
        profile = appearance["profile"]
        appearance["appearance_profile_sha256"] = appearance_profile_hash(
            type(
                AppearanceInstanceRecord.model_validate_json(
                    canonical_json_bytes(appearance)
                ).profile
            ).model_validate_json(canonical_json_bytes(profile))
        )
    elif corruption == "profile_hash":
        appearance["appearance_profile_sha256"] = "0" * 64
    elif corruption == "texture_hash":
        appearance["textures"][0]["source_texture_logical_sha256"] = "0" * 64
    elif corruption == "assignment":
        names = list(appearance["style_assignment"])
        first_slot = appearance["style_assignment"][names[0]]
        second_slot = appearance["style_assignment"][names[1]]
        appearance["style_assignment"][names[0]] = second_slot
        appearance["style_assignment"][names[1]] = first_slot
        by_name = {item["semantic_surface_name"]: item for item in appearance["textures"]}
        by_name[names[0]]["style_slot_id"] = second_slot
        by_name[names[1]]["style_slot_id"] = first_slot
    elif corruption == "seed_registry_binding":
        appearance["evaluation_seed_registry_sha256"] = "0" * 64
    if corruption != "instance_identity":
        appearance["appearance_instance_sha256"] = "0" * 64
        record = AppearanceInstanceRecord.model_validate_json(canonical_json_bytes(appearance))
        appearance["appearance_instance_sha256"] = appearance_instance_hash(record)
    else:
        appearance["appearance_instance_sha256"] = "0" * 64
    _commit_fully_rehashed_appearance(broken, instrumentation)
    with pytest.raises(DatasetValidationError, match="appearance"):
        validate_dataset(broken)
