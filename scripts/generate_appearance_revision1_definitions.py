"""Materialise the prospectively locked Revision 1 YAML definitions."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from epsbench.appearance import REVISION1_PROFILE_IDS
from epsbench.utils.seeding import derive_seed


class _NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


ROOT = Path(__file__).resolve().parents[1]
BASELINE_REGISTRY = ROOT / "configs/appearance_candidates_v0.yaml"
REVISION_REGISTRY = ROOT / "configs/appearance_candidates_revision1.yaml"
QUALIFICATION_REGISTRY = ROOT / "configs/appearance_revision1_qualification_seeds_v0.yaml"

REFERENCE_PALETTE = (
    ((60, 105, 150), (210, 200, 130)),
    ((145, 75, 75), (190, 220, 180)),
    ((65, 125, 90), (225, 170, 200)),
    ((100, 80, 150), (210, 195, 150)),
)
COLOUR_SHIFT_PALETTE = (
    ((150, 85, 70), (175, 215, 225)),
    ((65, 120, 145), (230, 190, 140)),
    ((125, 80, 155), (190, 220, 150)),
    ((75, 125, 75), (225, 175, 205)),
)


def _palette(values: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]) -> dict[str, Any]:
    slots = [
        {
            "slot_id": f"style-slot-{index}",
            "foreground_rgb": list(foreground),
            "background_rgb": list(background),
        }
        for index, (foreground, background) in enumerate(values)
    ]
    return {
        "single_occluder_slots": deepcopy(slots[:3]),
        "corridor_slots": deepcopy(slots),
    }


def _illumination(*, shifted: bool) -> dict[str, Any]:
    if shifted:
        single_direction = [-0.35, 0.35, -1.0]
        corridor_direction = [-0.35, 0.2, -1.0]
        diffuse = 0.55
        ambient = [0.4, 0.4, 0.4]
    else:
        single_direction = [0.2, 0.5, -1.0]
        corridor_direction = [0.0, 0.25, -1.0]
        diffuse = 0.65
        ambient = [0.35, 0.35, 0.35]
    return {
        "single_occluder": {
            "direction": single_direction,
            "diffuse_intensity": diffuse,
        },
        "corridor": {
            "direction": corridor_direction,
            "diffuse_intensity": diffuse,
        },
        "directional": True,
        "cast_shadows": False,
        "ambient_rgb": ambient,
        "specular_rgb": [0.0, 0.0, 0.0],
        "camera_exposure_rule": "unchanged_renderer_default_v1",
    }


def _profile(
    baseline_profile: dict[str, Any],
    *,
    profile_id: str,
    candidate_class: str,
    axis_tags: list[str],
    matched_control: str,
    palette: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
    family: str,
    cycles: int,
    shifted_illumination: bool,
) -> dict[str, Any]:
    texture = deepcopy(baseline_profile["texture"])
    texture.update(
        {
            "family": family,
            "cycles_per_tile": cycles,
            "orientation": "axis_x_0_degrees",
        }
    )
    return {
        "profile_id": profile_id,
        "profile_version": "appearance_profile_v3",
        "candidate_class": candidate_class,
        "freeze_eligible": True,
        "axis_tags": axis_tags,
        "matched_control_profile_id": matched_control,
        "style_assignment_rule": "balanced_cyclic_permutation_v1",
        "palette": _palette(palette),
        "texture": texture,
        "material": deepcopy(baseline_profile["material"]),
        "illumination": _illumination(shifted=shifted_illumination),
        "non_degeneracy_thresholds": deepcopy(baseline_profile["non_degeneracy_thresholds"]),
    }


def _revision_profiles(baseline_profile: dict[str, Any]) -> list[dict[str, Any]]:
    reference = "revision1_balanced_reference_v1"
    checker_low = "revision1_checker_low_v1"
    return [
        _profile(
            baseline_profile,
            profile_id=reference,
            candidate_class="combined_stress_candidate",
            axis_tags=["colour", "illumination", "combined"],
            matched_control="legacy_solid_base_v1",
            palette=REFERENCE_PALETTE,
            family="solid",
            cycles=0,
            shifted_illumination=False,
        ),
        _profile(
            baseline_profile,
            profile_id="revision1_colour_shift_v1",
            candidate_class="single_axis_candidate",
            axis_tags=["colour"],
            matched_control=reference,
            palette=COLOUR_SHIFT_PALETTE,
            family="solid",
            cycles=0,
            shifted_illumination=False,
        ),
        _profile(
            baseline_profile,
            profile_id=checker_low,
            candidate_class="single_axis_candidate",
            axis_tags=["texture_family"],
            matched_control=reference,
            palette=REFERENCE_PALETTE,
            family="checker",
            cycles=1,
            shifted_illumination=False,
        ),
        _profile(
            baseline_profile,
            profile_id="revision1_checker_high_v1",
            candidate_class="single_axis_candidate",
            axis_tags=["texture_frequency"],
            matched_control=checker_low,
            palette=REFERENCE_PALETTE,
            family="checker",
            cycles=4,
            shifted_illumination=False,
        ),
        _profile(
            baseline_profile,
            profile_id="revision1_stripes_low_v1",
            candidate_class="single_axis_candidate",
            axis_tags=["texture_family"],
            matched_control=checker_low,
            palette=REFERENCE_PALETTE,
            family="stripes",
            cycles=1,
            shifted_illumination=False,
        ),
        _profile(
            baseline_profile,
            profile_id="revision1_illumination_shift_v1",
            candidate_class="single_axis_candidate",
            axis_tags=["illumination"],
            matched_control=reference,
            palette=REFERENCE_PALETTE,
            family="solid",
            cycles=0,
            shifted_illumination=True,
        ),
        _profile(
            baseline_profile,
            profile_id="revision1_combined_stress_v1",
            candidate_class="combined_stress_candidate",
            axis_tags=[
                "colour",
                "texture_family",
                "texture_frequency",
                "illumination",
                "combined",
            ],
            matched_control=reference,
            palette=COLOUR_SHIFT_PALETTE,
            family="checker",
            cycles=4,
            shifted_illumination=True,
        ),
    ]


def main() -> None:
    baseline = yaml.safe_load(BASELINE_REGISTRY.read_text(encoding="utf-8"))
    baseline_profiles = deepcopy(baseline["profiles"])
    baseline_profile = next(
        profile
        for profile in baseline_profiles
        if profile["profile_id"] == "balanced_solid_palette_v1"
    )
    profiles = sorted(
        [*baseline_profiles, *_revision_profiles(baseline_profile)],
        key=lambda profile: profile["profile_id"],
    )
    revision = {
        "registry_version": "appearance_candidate_registry_v2",
        "profiles": profiles,
        "revision1_admission_profile_ids": list(REVISION1_PROFILE_IDS),
    }
    REVISION_REGISTRY.write_text(
        yaml.dump(revision, Dumper=_NoAliasDumper, sort_keys=False),
        encoding="utf-8",
    )

    namespace = "gate0b-appearance-revision1-qualification"
    indices = list(range(8))
    qualification = {
        "registry_version": "appearance_revision1_qualification_seed_registry_v0",
        "root_seed": 271828,
        "derivation_rule": "derive_seed_v1",
        "namespace": namespace,
        "indices": indices,
        "candidate_episode_seeds": [
            derive_seed(271828, f"{namespace}:{index}") for index in indices
        ],
    }
    QUALIFICATION_REGISTRY.write_text(
        yaml.dump(qualification, Dumper=_NoAliasDumper, sort_keys=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
