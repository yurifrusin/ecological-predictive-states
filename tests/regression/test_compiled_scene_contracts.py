from __future__ import annotations

import pytest

from epsbench.config import CorridorConfig, SingleOccluderConfig
from epsbench.data.identity import (
    corridor_scene_content_domain,
    single_occluder_scene_content_domain,
)
from epsbench.sim import (
    compile_corridor_scene_contract,
    compile_single_occluder_scene_contract,
    sample_corridor_geometry,
)
from epsbench.utils.seeding import derive_seed


def test_single_scene_content_constants_match_compiled_mujoco_contract(
    benchmark_config: SingleOccluderConfig,
) -> None:
    domain = single_occluder_scene_content_domain(benchmark_config)
    compiled = compile_single_occluder_scene_contract(benchmark_config)
    assert domain["surfaces"] == {
        "support_surface": {
            "type": "plane",
            "position": list(compiled.raw_geom_world_positions["support_surface"]),
            "size": list(compiled.raw_geom_compiled_sizes["support_surface"]),
        },
        "background_surface": {
            "type": "box",
            "position": list(compiled.raw_geom_world_positions["background_surface"]),
            "half_size": list(compiled.raw_geom_compiled_sizes["background_surface"]),
        },
        "occluding_surface": {
            "type": "box",
            "position": list(compiled.raw_geom_world_positions["occluding_surface"]),
            "half_size": list(compiled.raw_geom_compiled_sizes["occluding_surface"]),
        },
    }
    assert domain["camera"]["before_position"] == list(compiled.camera_world_position)
    assert domain["camera"]["field_of_view_degrees"] == (compiled.camera_field_of_view_degrees)
    assert domain["camera"]["orientation_rule"] == "xyaxes_1_0_0_0_0.16_1"
    assert compiled.camera_world_rotation_row_major == pytest.approx(
        (
            1.0,
            0.0,
            0.0,
            0.0,
            0.15799050110667284,
            -0.9874406319167053,
            0.0,
            0.9874406319167053,
            0.15799050110667284,
        )
    )


def test_corridor_scene_content_constants_match_compiled_mujoco_contract(
    corridor_config: CorridorConfig,
) -> None:
    episode_seed = derive_seed(corridor_config.seed, "episode:0")
    geometry = sample_corridor_geometry(corridor_config, episode_seed)
    compiled = compile_corridor_scene_contract(corridor_config, geometry, appearance_seed=0)
    domain = corridor_scene_content_domain(corridor_config, geometry)
    surface_domain = domain["surfaces"]
    half_width = surface_domain["width"] / 2.0
    half_length = surface_domain["length"] / 2.0
    half_height = surface_domain["wall_height"] / 2.0
    wall_thickness = surface_domain["wall_thickness"]
    floor_thickness = surface_domain["floor_thickness"]
    assert set(surface_domain["surface_names"]) == set(compiled.raw_geom_ids)
    assert compiled.raw_geom_world_positions == {
        "corridor_floor": (0.0, half_length, -floor_thickness),
        "corridor_left_surface": (-half_width, half_length, half_height),
        "corridor_right_surface": (half_width, half_length, half_height),
        "corridor_end_surface": (0.0, surface_domain["length"], half_height),
    }
    assert compiled.raw_geom_compiled_sizes == {
        "corridor_floor": (half_width, half_length, floor_thickness),
        "corridor_left_surface": (wall_thickness, half_length, half_height),
        "corridor_right_surface": (wall_thickness, half_length, half_height),
        "corridor_end_surface": (half_width, wall_thickness, half_height),
    }
    assert domain["camera"]["before_position"] == list(compiled.camera_world_position)
    assert domain["camera"]["field_of_view_degrees"] == (compiled.camera_field_of_view_degrees)
    assert domain["camera"]["orientation_rule"] == "xyaxes_1_0_0_0_0_1"
    assert compiled.camera_world_rotation_row_major == pytest.approx(
        (1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0),
        abs=1e-15,
    )
