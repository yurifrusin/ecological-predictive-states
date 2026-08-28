from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from epsbench.appearance import (
    APPEARANCE_INSTANCE_VERSION,
    APPEARANCE_REGISTRY_VERSION,
    CANONICAL_PROFILE_IDS,
    AppearanceInstanceRecord,
    AppearanceRegistry,
    AxisTag,
    EvaluationSeedRegistry,
    TextureFamily,
    appearance_instance_hash,
    appearance_profile_hash,
    appearance_seed_namespaces,
    assignment_balance,
    generate_texture,
    load_appearance_registry,
    load_evaluation_seed_registry,
    profile_by_id,
    resolve_appearance,
    seed_registry_hash,
    validate_axis_isolation,
)
from epsbench.sim import corridor_generation_seeds
from epsbench.sim.single_occluder import build_scene_xml
from epsbench.utils.seeding import derive_seed

REGISTRY_PATH = Path("configs/appearance_candidates_v0.yaml")
SEED_REGISTRY_PATH = Path("configs/evaluation_seed_candidates_v0.yaml")


@pytest.fixture(scope="module")
def registry() -> AppearanceRegistry:
    return load_appearance_registry(REGISTRY_PATH)


def test_registry_contains_the_exact_unique_canonical_profile_set(
    registry: AppearanceRegistry,
) -> None:
    ids = tuple(profile.profile_id for profile in registry.profiles)
    assert ids == tuple(sorted(CANONICAL_PROFILE_IDS))
    assert len(ids) == len(set(ids)) == 10


def test_renderer_filter_and_versioned_identity_contracts_are_truthful(
    registry: AppearanceRegistry,
) -> None:
    # EPS-ER11-0001
    assert registry.registry_version == APPEARANCE_REGISTRY_VERSION
    assert APPEARANCE_REGISTRY_VERSION == "appearance_candidate_registry_v1"
    assert APPEARANCE_INSTANCE_VERSION == "appearance_instance_v3"
    assert {profile.profile_version for profile in registry.profiles} == {"appearance_profile_v2"}
    assert {profile.texture.filtering for profile in registry.profiles} == {
        "mujoco_linear_mipmap_linear_v1"
    }


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("profiles", 0, "unknown"), True),
        (("profiles", 0, "palette", "single_occluder_slots", 0, "foreground_rgb", 0), 256),
        (("profiles", 0, "texture", "cycles_per_tile"), -1),
        (("profiles", 0, "illumination", "single_occluder", "diffuse_intensity"), float("nan")),
        (("profiles", 0, "axis_tags"), ["not_an_axis"]),
    ],
)
def test_registry_rejects_unknown_or_malformed_profile_fields(
    registry: AppearanceRegistry,
    path: tuple[str | int, ...],
    value: object,
) -> None:
    payload = registry.model_dump(mode="json")
    target: object = payload
    for component in path[:-1]:
        target = target[component]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]
    with pytest.raises(ValidationError):
        AppearanceRegistry.model_validate_json(json.dumps(payload))


def test_legacy_and_freeze_eligible_candidate_classes_are_disjoint(
    registry: AppearanceRegistry,
) -> None:
    legacy = [profile for profile in registry.profiles if profile.profile_id.startswith("legacy_")]
    candidates = [
        profile for profile in registry.profiles if not profile.profile_id.startswith("legacy_")
    ]
    assert legacy and all(not profile.freeze_eligible for profile in legacy)
    assert candidates and all(profile.freeze_eligible for profile in candidates)


def test_profile_hash_binds_label_defining_fields_only(registry: AppearanceRegistry) -> None:
    profile = profile_by_id(registry, "balanced_solid_palette_v1")
    changed_palette = profile.palette.model_copy(
        update={
            "single_occluder_slots": (
                profile.palette.single_occluder_slots[0].model_copy(
                    update={"foreground_rgb": (234, 70, 75)}
                ),
                *profile.palette.single_occluder_slots[1:],
            )
        }
    )
    changed = profile.model_copy(update={"palette": changed_palette})
    assert appearance_profile_hash(changed) != appearance_profile_hash(profile)
    assert "path" not in profile.model_dump(mode="json")
    assert "timestamp" not in profile.model_dump(mode="json")


def test_texture_arrays_phase_and_frequency_are_exact_and_deterministic(
    registry: AppearanceRegistry,
) -> None:
    low = profile_by_id(registry, "balanced_checker_low_v1")
    high = profile_by_id(registry, "balanced_checker_high_v1")
    slot = low.palette.single_occluder_slots[0]
    first = generate_texture(low, slot, 17)
    second = generate_texture(low, slot, 17)
    shifted = generate_texture(low, slot, 18)
    high_texture = generate_texture(high, slot, 17)
    assert first.dtype == np.uint8 and first.shape == (128, 128, 3)
    assert first.tobytes() == second.tobytes()
    assert first.tobytes() != shifted.tobytes()
    assert high.texture.cycles_per_tile == 4 * low.texture.cycles_per_tile
    assert np.count_nonzero(np.diff(high_texture[:, :, 0], axis=1)) >= 4 * np.count_nonzero(
        np.diff(first[:, :, 0], axis=1)
    )


def test_stripe_orientation_and_texture_record_are_identity_bound(
    registry: AppearanceRegistry,
) -> None:
    profile = profile_by_id(registry, "balanced_stripes_high_oblique_v1")
    slot = profile.palette.single_occluder_slots[0]
    texture = generate_texture(profile, slot, 0)
    assert profile.texture.family == TextureFamily.STRIPES
    assert not np.array_equal(texture[0, :, :], texture[1, :, :])
    seeds = load_evaluation_seed_registry(SEED_REGISTRY_PATH)
    plan = resolve_appearance(
        registry,
        profile.profile_id,
        "single_occluder",
        ("support_surface", "occluding_surface", "background_surface"),
        derive_seed(seeds.candidate_episode_seeds[0], "episode:0"),
        seeds.candidate_episode_seeds[0],
        seeds,
    )
    assert all(record.source_texture_logical_sha256 for record in plan.record.textures)


def test_seed_registry_is_recomputed_and_order_is_immutable() -> None:
    registry = load_evaluation_seed_registry(SEED_REGISTRY_PATH)
    expected = tuple(
        derive_seed(registry.root_seed, f"{registry.namespace}:{index}") for index in range(8)
    )
    assert registry.candidate_episode_seeds == expected
    payload = registry.model_dump(mode="json")
    payload["candidate_episode_seeds"] = list(reversed(expected))
    with pytest.raises(ValidationError):
        type(registry).model_validate_json(json.dumps(payload))


def test_candidate_and_non_candidate_schedule_postures_reconstruct_exactly(
    registry: AppearanceRegistry,
) -> None:
    seeds = load_evaluation_seed_registry(SEED_REGISTRY_PATH)
    surface_names = ("support_surface", "occluding_surface", "background_surface")
    candidate_root = seeds.candidate_episode_seeds[1]
    candidate = resolve_appearance(
        registry,
        "balanced_solid_palette_v1",
        "single_occluder",
        surface_names,
        derive_seed(candidate_root, "episode:0"),
        candidate_root,
        seeds,
    ).record
    assert candidate.seeds.candidate_schedule_index == 1
    assert candidate.assignment_schedule_posture == "candidate_registry_index"
    assert candidate.style_assignment == {
        "support_surface": "style-slot-1",
        "occluding_surface": "style-slot-2",
        "background_surface": "style-slot-0",
    }

    non_candidate_root = 42
    assert non_candidate_root not in seeds.candidate_episode_seeds
    non_candidate = resolve_appearance(
        registry,
        "balanced_solid_palette_v1",
        "single_occluder",
        surface_names,
        derive_seed(non_candidate_root, "episode:0"),
        non_candidate_root,
        seeds,
    ).record
    assert non_candidate.seeds.candidate_schedule_index is None
    assert non_candidate.assignment_schedule_posture == "non_candidate_seed_derived"
    assert non_candidate.assignment_root_seed == non_candidate_root


def test_seed_registry_schedule_mutations_change_identity_and_appearance_instance(
    registry: AppearanceRegistry,
) -> None:
    seeds = load_evaluation_seed_registry(SEED_REGISTRY_PATH)
    root_seed = seeds.candidate_episode_seeds[0]
    surface_names = ("support_surface", "occluding_surface", "background_surface")

    def resolve(candidate_registry: EvaluationSeedRegistry) -> AppearanceInstanceRecord:
        return resolve_appearance(
            registry,
            "balanced_solid_palette_v1",
            "single_occluder",
            surface_names,
            derive_seed(root_seed, "episode:0"),
            root_seed,
            candidate_registry,
        ).record

    original = resolve(seeds)
    variants = (
        seeds.model_copy(update={"candidate_episode_seeds": seeds.candidate_episode_seeds[:-1]}),
        seeds.model_copy(
            update={"candidate_episode_seeds": tuple(reversed(seeds.candidate_episode_seeds))}
        ),
        seeds.model_copy(
            update={"candidate_episode_seeds": (999, *seeds.candidate_episode_seeds[1:])}
        ),
    )
    for changed in variants:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            changed_record = resolve(changed)
            assert seed_registry_hash(changed) != seed_registry_hash(seeds)
            assert appearance_instance_hash(changed_record) != appearance_instance_hash(original)


def test_appearance_namespace_is_independent_of_geometry_and_remapping() -> None:
    episode_seed = derive_seed(1729, "episode:0")
    geometry, remapping, appearance = corridor_generation_seeds(episode_seed)
    namespaces = appearance_seed_namespaces(episode_seed)
    assert len({geometry, remapping, appearance}) == 3
    assert namespaces.appearance_base_seed == appearance
    assert namespaces.style_assignment_seed not in {geometry, remapping}


def test_balanced_assignments_hold_for_both_scene_families(
    registry: AppearanceRegistry,
) -> None:
    seeds = load_evaluation_seed_registry(SEED_REGISTRY_PATH)
    profile = profile_by_id(registry, "balanced_solid_palette_v1")
    single = assignment_balance(profile, ("a", "b", "c"), seeds.candidate_episode_seeds, seeds)
    corridor = assignment_balance(
        profile, ("a", "b", "c", "d"), seeds.candidate_episode_seeds, seeds
    )
    assert all(max(counts.values()) - min(counts.values()) <= 1 for counts in single.values())
    assert all(set(counts.values()) == {2} for counts in corridor.values())


def test_axis_isolation_is_validated_as_exact_profile_domains(
    registry: AppearanceRegistry,
) -> None:
    validate_axis_isolation(registry)
    profile = profile_by_id(registry, "balanced_illumination_left_v1")
    malformed = profile.model_copy(
        update={
            "texture": profile.texture.model_copy(update={"cycles_per_tile": 4}),
        }
    )
    profiles = tuple(
        malformed if item.profile_id == profile.profile_id else item for item in registry.profiles
    )
    altered = registry.model_copy(update={"profiles": profiles})
    with pytest.raises(ValueError, match="solid textures"):
        validate_axis_isolation(altered)


def test_axis_isolation_rejects_a_false_extra_axis(
    registry: AppearanceRegistry,
) -> None:
    # EPS-ER11-0008
    profile = profile_by_id(registry, "balanced_illumination_left_v1")
    malformed = profile.model_copy(
        update={"axis_tags": (AxisTag.ILLUMINATION, AxisTag.TEXTURE_FAMILY)}
    )
    altered = registry.model_copy(
        update={
            "profiles": tuple(
                malformed if item.profile_id == profile.profile_id else item
                for item in registry.profiles
            )
        }
    )
    with pytest.raises(ValueError, match="single-axis declaration"):
        validate_axis_isolation(altered)


def test_axis_isolation_rejects_an_undeclared_illumination_change(
    registry: AppearanceRegistry,
) -> None:
    profile = profile_by_id(registry, "balanced_checker_low_v1")
    illumination = profile_by_id(registry, "balanced_illumination_left_v1").illumination
    malformed = profile.model_copy(update={"illumination": illumination})
    altered = registry.model_copy(
        update={
            "profiles": tuple(
                malformed if item.profile_id == profile.profile_id else item
                for item in registry.profiles
            )
        }
    )
    with pytest.raises(ValueError, match="single-axis declaration"):
        validate_axis_isolation(altered)


def test_render_xml_disables_unstated_headlight_and_specular_defaults(
    registry: AppearanceRegistry,
) -> None:
    from epsbench.config import load_config

    config = load_config(Path("configs/benchmark_v0.yaml"))
    seeds = load_evaluation_seed_registry(SEED_REGISTRY_PATH)
    plan = resolve_appearance(
        registry,
        "balanced_checker_low_v1",
        "single_occluder",
        ("support_surface", "occluding_surface", "background_surface"),
        derive_seed(seeds.candidate_episode_seeds[0], "episode:0"),
        seeds.candidate_episode_seeds[0],
        seeds,
    )
    xml = build_scene_xml(config, plan)  # type: ignore[arg-type]
    assert 'headlight ambient="0 0 0" diffuse="0 0 0"' in xml
    assert 'specular="0 0 0" active="0"' in xml
    assert 'specular="0" shininess="0" reflectance="0"' in xml
