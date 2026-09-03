"""Materialise the prospective Appearance Benchmark Input Freeze v0 lock."""

from pathlib import Path

from epsbench.appearance import AppearanceRevision1Registry, load_appearance_registry_any
from epsbench.config import load_config
from epsbench.freeze import (
    benchmark_definition_hash,
    create_definition_lock_payload,
    load_benchmark_definition,
    load_final_evaluation_seeds,
)
from epsbench.utils.canonical import write_canonical_json

ROOT = Path(__file__).resolve().parents[1]
DEFINITION = ROOT / "configs/appearance_benchmark_freeze_v0.yaml"
SEEDS = ROOT / "configs/appearance_benchmark_v0_evaluation_episode_seeds.yaml"
REVISION = ROOT / "configs/appearance_candidates_revision1.yaml"
SINGLE = ROOT / "configs/benchmark_v0.yaml"
CORRIDOR = ROOT / "configs/corridor_v0.yaml"
LOCK = ROOT / "configs/appearance_benchmark_freeze_v0_lock.json"


def main() -> None:
    definition = load_benchmark_definition(DEFINITION)
    seeds = load_final_evaluation_seeds(SEEDS)
    revision = load_appearance_registry_any(REVISION)
    if not isinstance(revision, AppearanceRevision1Registry):
        raise ValueError("exact Revision 1 registry required")
    lock = create_definition_lock_payload(
        definition,
        seeds,
        revision,
        (load_config(SINGLE), load_config(CORRIDOR)),
    )
    write_canonical_json(LOCK, lock)
    print(f"benchmark_definition_sha256={benchmark_definition_hash(definition)}")
    print(f"profile_role_root_sha256={lock['profile_role_root_sha256']}")
    print(
        f"evaluation_episode_seed_registry_sha256={lock['evaluation_episode_seed_registry_sha256']}"
    )
    print(
        f"selected_matrix_membership_root_sha256={lock['selected_matrix_membership_root_sha256']}"
    )
    print(f"definition_lock_sha256={lock['definition_lock_sha256']}")


if __name__ == "__main__":
    main()
