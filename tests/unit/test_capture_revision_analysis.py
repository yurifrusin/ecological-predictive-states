from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from epsbench.diagnostics.revision_analysis import (
    RevisionAnalysisError,
    analytic_boundary_band,
    analytic_geometry_maps,
    analyze_capture_results,
    analyze_pose,
    compare_modalities,
    decode_encoded_segmentation,
    depth_residual_summary,
    exact_array_comparison,
)

POSE = {"camera_position": [0, 0, 0], "camera_rotation": np.eye(3), "fovy_degrees": 90.0}
PLANE = {
    "finite_planes": [
        {
            "raw_geom_id": 7,
            "center": [0, 0, -2],
            "axis_u": [1, 0, 0],
            "axis_v": [0, 1, 0],
            "half_extents": [10, 10],
        }
    ]
}


def test_plane_camera_axis_depth_is_not_euclidean_ray_length() -> None:
    maps = analytic_geometry_maps(PLANE, POSE, 1, 2)
    assert np.allclose(maps["camera_axis_depth"], 2.0)
    assert np.all(maps["ray_distance"] > 2.0)


def test_nonidentity_compiled_rotation_uses_local_to_world_transpose() -> None:
    # MuJoCo xmat maps world row vectors to local via ``world @ R``.
    camera = {
        "camera_position": [0, 0, 0],
        "camera_rotation": [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
        "fovy_degrees": 30.0,
    }
    geometry = {
        "finite_planes": [
            {
                "raw_geom_id": 4,
                "center": [2, 0, 0],
                "rotation": [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
                "half_extents": [10, 10],
            }
        ]
    }
    maps = analytic_geometry_maps(geometry, camera, 1, 1)
    assert maps["raw_geom_ids"].item() == 4
    assert maps["camera_axis_depth"].item() == pytest.approx(2.0)


def test_oriented_box_and_corner_rays_produce_hits_and_no_hits() -> None:
    geometry = {
        "oriented_boxes": [
            {
                "raw_geom_id": 11,
                "center": [0, 0, -3],
                "rotation": np.eye(3),
                "half_extents": [1, 1, 1],
            }
        ]
    }
    maps = analytic_geometry_maps(geometry, POSE, 3, 3)
    assert maps["raw_geom_ids"][1, 1] == 11
    assert maps["camera_axis_depth"][1, 1] == pytest.approx(2.0)
    # Narrow box misses the extreme rays; corner tracing has one extra row/column.
    assert maps["raw_geom_ids"][0, 0] == -1
    corners = analytic_geometry_maps(geometry, POSE, 3, 3, corners=True)
    assert corners["raw_geom_ids"].shape == (4, 4)


def test_four_neighbour_boundary_includes_both_sides_and_no_hit_interface() -> None:
    values = np.array([[1, 1, -1], [1, 2, -1]], dtype=np.int32)
    assert analytic_boundary_band(values).tolist() == [[False, True, True], [True, True, True]]


def test_exact_comparison_requires_dtype_shape_and_values() -> None:
    a = np.array([[1]], dtype=np.int32)
    assert not exact_array_comparison(a, a.astype(np.int64))["exact_equal"]
    assert not exact_array_comparison(a, np.array([1], dtype=np.int32))["exact_equal"]
    assert not exact_array_comparison(a, np.array([[2]], dtype=np.int32))["exact_equal"]


def test_empty_summary_is_finite_json_with_nulls_not_nan() -> None:
    summary = depth_residual_summary(np.array([1.0]), np.array([False]))
    encoded = json.dumps(summary, allow_nan=False)
    assert json.loads(encoded)["median_abs"] is None


def test_decode_map_and_vertical_orientation_and_bad_schema() -> None:
    encoded = np.array([[[2, 0, 0]], [[3, 0, 0]]], dtype=np.uint8)  # segids 1, 2
    pairs, _ = decode_encoded_segmentation(encoded, [(1, 10, 5), (2, 11, 5)], flip_vertical=True)
    assert pairs[:, 0, 0].tolist() == [11, 10]
    with pytest.raises(RevisionAnalysisError):
        decode_encoded_segmentation(encoded.astype(np.int16), [], flip_vertical=False)
    with pytest.raises(RevisionAnalysisError, match="absent from the scene map"):
        decode_encoded_segmentation(encoded, [(1, 10, 5)], flip_vertical=False)


def _pose_record(depth: np.ndarray) -> dict[str, object]:
    encoded = np.array([[[2, 0, 0]]], dtype=np.uint8)
    pairs = np.array([[[7, 5]]], dtype=np.int32)
    return {
        "pose_facts": POSE,
        "geometry_facts": PLANE,
        "modalities": {
            "rgb": np.zeros((1, 1, 3), dtype=np.uint8),
            "depth": depth,
            "segmentation": {
                "encoded_rgb": encoded,
                "decoded_pairs": pairs,
                "raw_geom_ids": np.array([[7]], dtype=np.int32),
                "segid_map": [(1, 7, 5)],
                "readback_requires_vertical_flip": False,
                "geom_objtype": 5,
            },
        },
    }


def test_pose_results_keep_outlier_and_all_seed_style_counts() -> None:
    record = _pose_record(np.array([[2.2]], dtype=np.float32))
    result = analyze_pose(record)
    assert result["depth_residual_outside_boundary"]["count"] == 1
    assert result["depth_residual_outside_boundary"]["above_scales"]["1e-01"] == 1
    assert result["raw_id_disagreement_counts"] == {"interior": 0, "boundary": 0, "no_hit": 0}


def test_fixed_diagnostic_scale_is_strict_with_float32_rounding() -> None:
    # 2.1 is stored below the exact 0.1 residual scale in binary32; scale rules
    # remain strict, rather than being changed to satisfy a rounded expectation.
    result = analyze_pose(_pose_record(np.array([[2.1]], dtype=np.float32)))
    assert result["depth_residual_outside_boundary"]["above_scales"]["1e-01"] == 0


def test_pose_rejects_wrong_persisted_array_contracts() -> None:
    record = _pose_record(np.array([[2.0]], dtype=np.float64))
    with pytest.raises(RevisionAnalysisError, match="float32"):
        analyze_pose(record)
    record = _pose_record(np.array([[2.0]], dtype=np.float32))
    record["modalities"]["segmentation"]["readback_requires_vertical_flip"] = 0  # type: ignore[index]
    with pytest.raises(RevisionAnalysisError, match="must be a bool"):
        analyze_pose(record)


def test_compare_modalities_reports_changes_without_declaring_ground_truth() -> None:
    first, second = (
        _pose_record(np.array([[2.0]], dtype=np.float32))["modalities"],
        _pose_record(np.array([[2.5]], dtype=np.float32))["modalities"],
    )
    second["rgb"] = np.ones((1, 1, 3), dtype=np.uint8)  # type: ignore[index]
    comparison = compare_modalities(first, second)  # type: ignore[arg-type]
    assert comparison["rgb_changed_count"] == 1
    assert comparison["depth_changed_count"] == 1


def test_matrix_retains_every_seed_result_including_an_outlier() -> None:
    cells = []
    for episode, depth in ((0, 2.0), (1, 2.2)):
        record = _pose_record(np.array([[depth]], dtype=np.float32))
        record["pose_name"] = "before"
        cells.append(
            {
                "identity": {
                    "family": "corridor",
                    "episode_index": episode,
                    "backend": "wgl",
                    "policy": "joint4",
                },
                "poses": [record],
            }
        )
    report = analyze_capture_results(cells)
    assert len(report["cells"]) == 2
    assert (
        report["cells"][1]["analysis"]["depth_residual_outside_boundary"]["above_scales"]["1e-01"]
        == 1
    )


def test_vectorized_geometry_matches_canonical_cpu_intersections_for_real_configured_families() -> (
    None
):
    """Compile only; this regression deliberately never constructs a Renderer."""
    import mujoco

    from epsbench.annotations.optical_transport import (
        AnalyticCamera,
        _nearest_controlled_intersections,
        pixel_rays_world,
    )
    from epsbench.appearance import configured_appearance_render_plan
    from epsbench.config import load_config
    from epsbench.sim.corridor import (
        CORRIDOR_SURFACE_NAMES,
        build_corridor_scene_xml,
        sample_corridor_geometry,
    )
    from epsbench.sim.single_occluder import SINGLE_OCCLUDER_SURFACE_NAMES, build_scene_xml
    from epsbench.utils.seeding import derive_seed

    root = Path(__file__).resolve().parents[2]

    def geometry_facts(model: object, data: object, names: tuple[str, ...]) -> dict[str, object]:
        planes: list[dict[str, object]] = []
        boxes: list[dict[str, object]] = []
        for name in names:
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            item = {
                "raw_geom_id": int(geom_id),
                "center": np.asarray(data.geom_xpos[geom_id], dtype=np.float64),
                "rotation": np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3),
                "half_extents": np.asarray(model.geom_size[geom_id], dtype=np.float64),
            }
            if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
                planes.append(item)
            elif int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_BOX):
                boxes.append(item)
            else:
                raise AssertionError(f"unexpected controlled geom type: {name}")
        return {"finite_planes": planes, "oriented_boxes": boxes}

    families = (
        ("corridor", "corridor_v0.yaml", CORRIDOR_SURFACE_NAMES),
        ("single_occluder", "benchmark_v0.yaml", SINGLE_OCCLUDER_SURFACE_NAMES),
    )
    for family, filename, names in families:
        config = load_config(root / "configs" / filename)
        for episode_index in (0, 3):
            episode_seed = derive_seed(1729, f"episode:{episode_index}")
            appearance = configured_appearance_render_plan(
                config.appearance.profile_id, family, names, episode_seed
            )
            if family == "corridor":
                sampled = sample_corridor_geometry(config, episode_seed)
                xml = build_corridor_scene_xml(config, sampled, appearance)
                pose_axis = 1
                pose_values = (
                    sampled.camera_before_forward_position,
                    sampled.camera_after_forward_position,
                )
            else:
                xml = build_scene_xml(config, appearance)
                pose_axis = 0
                pose_values = (config.camera.before_lateral, config.camera.after_lateral)
            model = mujoco.MjModel.from_xml_string(xml, appearance.asset_bytes)
            data = mujoco.MjData(model)
            camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "monocular_camera")
            controlled = tuple(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in names
            )
            for pose_value in pose_values:
                model.cam_pos[camera_id, pose_axis] = pose_value
                mujoco.mj_forward(model, data)
                camera = AnalyticCamera(
                    tuple(float(x) for x in data.cam_xpos[camera_id]),
                    tuple(float(x) for x in data.cam_xmat[camera_id].reshape(-1)),
                    float(model.cam_fovy[camera_id]),
                )
                directions = pixel_rays_world(config.render.width, config.render.height, camera)
                expected_distance, expected_ids = _nearest_controlled_intersections(
                    model, data, controlled, np.asarray(camera.world_position), directions
                )
                pose = {
                    "camera_position": camera.world_position,
                    "camera_rotation": camera.world_rotation_row_major,
                    "fovy_degrees": camera.vertical_field_of_view_degrees,
                }
                actual = analytic_geometry_maps(
                    geometry_facts(model, data, names),
                    pose,
                    config.render.height,
                    config.render.width,
                )
                assert np.array_equal(actual["raw_geom_ids"], expected_ids)
                rotation = np.asarray(camera.world_rotation_row_major).reshape(3, 3)
                hit = expected_ids >= 0
                safe_distance = np.where(hit, expected_distance, 0.0)
                expected_depth = -((directions * safe_distance[..., None]) @ rotation)[..., 2]
                assert np.allclose(
                    actual["camera_axis_depth"][hit], expected_depth[hit], rtol=0.0, atol=1e-12
                )
                assert np.all(np.isnan(actual["camera_axis_depth"][~hit]))
