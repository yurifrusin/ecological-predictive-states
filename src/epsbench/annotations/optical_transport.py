"""Geometry-derived image-plane transport for controlled static MuJoCo scenes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Final

import mujoco
import numpy as np
import numpy.typing as npt

ANALYTIC_TRANSPORT_METHOD: Final = "analytic_static_scene_transport_v2"
ANALYTIC_BOUNDARY_RULE: Final = "four_neighbour_assignment_band_v1"
ANALYTIC_BOUNDARY_WIDTH_PIXELS: Final = 1
ANALYTIC_SURFACE_INTERSECTION_RULE: Final = "compiled_plane_and_oriented_box_nearest_hit_v2"
FINITE_PLANE_EXTENT_RULE: Final = "finite_plane_visual_extent_v1"
TARGET_VISIBILITY_RULE: Final = "same_surface_point_nearest_hit_v1"
FLOW_FIXED_POINT_SCALE: Final = 1024
FLOW_QUANTISATION_ROUNDING: Final = "nearest_ties_to_even"
VISIBILITY_RELATIVE_TOLERANCE: Final = 1e-7
VISIBILITY_MINIMUM_TOLERANCE_SCALE: Final = 1.0
RAY_DIRECTION_EPSILON: Final = 1e-12

FloatArray = npt.NDArray[np.float64]
Int32Array = npt.NDArray[np.int32]
UInt8Array = npt.NDArray[np.uint8]
BoolArray = npt.NDArray[np.bool_]


class TransportReasonCode(IntEnum):
    """Canonical uint8 validity/reason domain."""

    VALID_TRANSPORT = 0
    NO_CONTROLLED_SOURCE_SURFACE = 1
    TARGET_OUT_OF_FRAME = 2
    OCCLUDED_AT_TARGET = 3
    ANALYTIC_BOUNDARY_AMBIGUOUS = 4


@dataclass(frozen=True)
class AnalyticCamera:
    """Compiled camera pose and vertical field of view used only during computation."""

    world_position: tuple[float, float, float]
    world_rotation_row_major: tuple[float, ...]
    vertical_field_of_view_degrees: float


@dataclass(frozen=True)
class DirectionalTransportArrays:
    """One exact source-to-target transport field."""

    vectors_fixed: Int32Array
    validity: UInt8Array
    reasons: UInt8Array


@dataclass(frozen=True)
class AnalyticTransportArrays:
    """Forward/backward transport plus privileged analytic assignment diagnostics."""

    forward: DirectionalTransportArrays
    backward: DirectionalTransportArrays
    before_surface_assignment: Int32Array
    after_surface_assignment: Int32Array
    before_boundary_ambiguous: BoolArray
    after_boundary_ambiguous: BoolArray


def focal_scales_from_vertical_fov(
    width: int,
    height: int,
    vertical_field_of_view_degrees: float,
) -> tuple[float, float]:
    """Return horizontal/vertical focal scales in pixels from aspect ratio and vertical FOV."""

    vertical_half_tangent = math.tan(math.radians(vertical_field_of_view_degrees) / 2.0)
    aspect_ratio = width / height
    horizontal_half_tangent = aspect_ratio * vertical_half_tangent
    focal_x = width / (2.0 * horizontal_half_tangent)
    focal_y = height / (2.0 * vertical_half_tangent)
    return focal_x, focal_y


def pixel_rays_world(width: int, height: int, camera: AnalyticCamera) -> FloatArray:
    """Construct unit world-space rays through every centre-of-pixel sample."""

    focal_x, focal_y = focal_scales_from_vertical_fov(
        width,
        height,
        camera.vertical_field_of_view_degrees,
    )
    rows, columns = np.indices((height, width), dtype=np.float64)
    centre_x = width / 2.0
    centre_y = height / 2.0
    directions_camera = np.stack(
        (
            (columns + 0.5 - centre_x) / focal_x,
            -(rows + 0.5 - centre_y) / focal_y,
            -np.ones((height, width), dtype=np.float64),
        ),
        axis=-1,
    )
    directions_camera /= np.linalg.norm(directions_camera, axis=-1, keepdims=True)
    rotation = np.asarray(camera.world_rotation_row_major, dtype=np.float64).reshape(3, 3)
    return np.asarray(directions_camera @ rotation.T, dtype=np.float64)


def project_world_points(
    world_points: FloatArray,
    width: int,
    height: int,
    camera: AnalyticCamera,
) -> tuple[FloatArray, FloatArray, BoolArray]:
    """Project world points to continuous image coordinates under the declared convention."""

    focal_x, focal_y = focal_scales_from_vertical_fov(
        width,
        height,
        camera.vertical_field_of_view_degrees,
    )
    position = np.asarray(camera.world_position, dtype=np.float64)
    rotation = np.asarray(camera.world_rotation_row_major, dtype=np.float64).reshape(3, 3)
    camera_points = np.asarray((world_points - position) @ rotation, dtype=np.float64)
    forward_distance = -camera_points[..., 2]
    in_front = forward_distance > RAY_DIRECTION_EPSILON
    safe_distance = np.where(in_front, forward_distance, 1.0)
    x = width / 2.0 + focal_x * camera_points[..., 0] / safe_distance
    y = height / 2.0 - focal_y * camera_points[..., 1] / safe_distance
    in_frame = in_front & (x >= 0.0) & (x < width) & (y >= 0.0) & (y < height)
    return x, y, np.asarray(in_frame, dtype=np.bool_)


def _nearest_controlled_intersections(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controlled_geom_ids: tuple[int, ...],
    ray_origin: FloatArray,
    ray_directions: FloatArray,
) -> tuple[FloatArray, Int32Array]:
    """Intersect rays with compiled plane/box apparatus using explicit analytic geometry."""

    nearest_distance = np.full(ray_directions.shape[:-1], np.inf, dtype=np.float64)
    nearest_geom = np.full(ray_directions.shape[:-1], -1, dtype=np.int32)
    for geom_id in sorted(controlled_geom_ids):
        geom_type = int(model.geom_type[geom_id])
        geom_position = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
        geom_rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        local_origin = np.asarray((ray_origin - geom_position) @ geom_rotation, dtype=np.float64)
        local_directions = np.asarray(ray_directions @ geom_rotation, dtype=np.float64)

        if geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
            denominator = local_directions[..., 2]
            non_parallel = np.abs(denominator) > RAY_DIRECTION_EPSILON
            candidate = np.where(non_parallel, -local_origin[2] / denominator, np.inf)
            positive = np.isfinite(candidate) & (candidate > RAY_DIRECTION_EPSILON)
            finite_candidate = np.where(positive, candidate, 0.0)
            local_hit_x = local_origin[0] + finite_candidate * local_directions[..., 0]
            local_hit_y = local_origin[1] + finite_candidate * local_directions[..., 1]
            visual_half_extent = np.asarray(model.geom_size[geom_id, :2], dtype=np.float64)
            inside_visual_extent = (np.abs(local_hit_x) <= visual_half_extent[0]) & (
                np.abs(local_hit_y) <= visual_half_extent[1]
            )
            candidate = np.where(positive & inside_visual_extent, candidate, np.inf)
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            half_size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
            parallel = np.abs(local_directions) <= RAY_DIRECTION_EPSILON
            parallel_outside = parallel & (np.abs(local_origin) > half_size)
            safe_directions = np.where(parallel, 1.0, local_directions)
            first = (-half_size - local_origin) / safe_directions
            second = (half_size - local_origin) / safe_directions
            near_components = np.where(parallel, -np.inf, np.minimum(first, second))
            far_components = np.where(parallel, np.inf, np.maximum(first, second))
            near = np.max(near_components, axis=-1)
            far = np.min(far_components, axis=-1)
            candidate = np.where(near > RAY_DIRECTION_EPSILON, near, far)
            hit = (
                ~np.any(parallel_outside, axis=-1)
                & (far >= np.maximum(near, RAY_DIRECTION_EPSILON))
                & (candidate > RAY_DIRECTION_EPSILON)
            )
            candidate = np.where(hit, candidate, np.inf)
        else:
            raise ValueError(
                "analytic transport supports only controlled MuJoCo plane and box geoms; "
                f"geom {geom_id} has type {geom_type}"
            )

        nearer = candidate < nearest_distance
        nearest_distance = np.where(nearer, candidate, nearest_distance)
        nearest_geom = np.where(nearer, np.int32(geom_id), nearest_geom)
    return nearest_distance, np.asarray(nearest_geom, dtype=np.int32)


def analytic_boundary_ambiguity(surface_assignment: Int32Array) -> BoolArray:
    """Mark both pixels adjoining any four-neighbour analytic assignment discontinuity."""

    if surface_assignment.ndim != 2:
        raise ValueError("analytic surface assignment must be a two-dimensional array")
    ambiguous = np.zeros(surface_assignment.shape, dtype=np.bool_)
    horizontal_change = surface_assignment[:, :-1] != surface_assignment[:, 1:]
    ambiguous[:, :-1] |= horizontal_change
    ambiguous[:, 1:] |= horizontal_change
    vertical_change = surface_assignment[:-1, :] != surface_assignment[1:, :]
    ambiguous[:-1, :] |= vertical_change
    ambiguous[1:, :] |= vertical_change
    return ambiguous


def _directional_transport(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controlled_geom_ids: tuple[int, ...],
    width: int,
    height: int,
    source_camera: AnalyticCamera,
    target_camera: AnalyticCamera,
    source_assignment: Int32Array,
    source_distance: FloatArray,
    source_boundary: BoolArray,
    target_boundary: BoolArray,
) -> DirectionalTransportArrays:
    source_directions = pixel_rays_world(width, height, source_camera)
    source_origin = np.asarray(source_camera.world_position, dtype=np.float64)
    safe_source_distance = np.where(np.isfinite(source_distance), source_distance, 1.0)
    world_points = source_origin + source_directions * safe_source_distance[..., None]
    target_x, target_y, target_in_frame = project_world_points(
        world_points,
        width,
        height,
        target_camera,
    )

    target_origin = np.asarray(target_camera.world_position, dtype=np.float64)
    target_vectors = world_points - target_origin
    point_distances = np.linalg.norm(target_vectors, axis=-1)
    safe_point_distances = np.where(
        point_distances > RAY_DIRECTION_EPSILON,
        point_distances,
        1.0,
    )
    target_directions = target_vectors / safe_point_distances[..., None]
    target_hit_distance, target_hit_assignment = _nearest_controlled_intersections(
        model,
        data,
        controlled_geom_ids,
        target_origin,
        target_directions,
    )
    visibility_tolerance = VISIBILITY_RELATIVE_TOLERANCE * np.maximum(
        VISIBILITY_MINIMUM_TOLERANCE_SCALE,
        point_distances,
    )
    same_surface_point_visible = (
        (target_hit_assignment == source_assignment)
        & np.isfinite(target_hit_distance)
        & (np.abs(target_hit_distance - point_distances) <= visibility_tolerance)
    )

    target_rows = np.clip(np.floor(target_y).astype(np.int64), 0, height - 1)
    target_columns = np.clip(np.floor(target_x).astype(np.int64), 0, width - 1)
    projected_target_boundary = target_boundary[target_rows, target_columns]
    boundary_ambiguous = source_boundary | (target_in_frame & projected_target_boundary)

    has_source = source_assignment >= 0
    reasons = np.full(
        (height, width),
        int(TransportReasonCode.NO_CONTROLLED_SOURCE_SURFACE),
        dtype=np.uint8,
    )
    reasons[has_source & ~target_in_frame] = int(TransportReasonCode.TARGET_OUT_OF_FRAME)
    transport_candidates = has_source & target_in_frame
    reasons[transport_candidates & boundary_ambiguous] = int(
        TransportReasonCode.ANALYTIC_BOUNDARY_AMBIGUOUS
    )
    interior_candidates = transport_candidates & ~boundary_ambiguous
    reasons[interior_candidates & ~same_surface_point_visible] = int(
        TransportReasonCode.OCCLUDED_AT_TARGET
    )
    valid = interior_candidates & same_surface_point_visible
    reasons[valid] = int(TransportReasonCode.VALID_TRANSPORT)

    rows, columns = np.indices((height, width), dtype=np.float64)
    source_x = columns + 0.5
    source_y = rows + 0.5
    flow_pixels = np.stack((target_x - source_x, target_y - source_y), axis=-1)
    vectors_fixed = np.zeros((height, width, 2), dtype=np.int32)
    quantised = np.rint(flow_pixels * FLOW_FIXED_POINT_SCALE).astype(np.int64)
    if np.any(quantised[valid] > np.iinfo(np.int32).max) or np.any(
        quantised[valid] < np.iinfo(np.int32).min
    ):
        raise OverflowError("analytic transport exceeds the canonical int32 fixed-point range")
    vectors_fixed[valid] = quantised[valid].astype(np.int32)
    return DirectionalTransportArrays(
        vectors_fixed=vectors_fixed,
        validity=np.asarray(valid, dtype=np.uint8),
        reasons=reasons,
    )


def compute_analytic_transport(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controlled_geom_ids: tuple[int, ...],
    width: int,
    height: int,
    before_camera: AnalyticCamera,
    after_camera: AnalyticCamera,
) -> AnalyticTransportArrays:
    """Compute exact forward/backward static-surface transport from compiled geometry."""

    if not controlled_geom_ids or len(set(controlled_geom_ids)) != len(controlled_geom_ids):
        raise ValueError("controlled geom identifiers must be non-empty and unique")
    mujoco.mj_forward(model, data)
    before_directions = pixel_rays_world(width, height, before_camera)
    after_directions = pixel_rays_world(width, height, after_camera)
    before_distance, before_assignment = _nearest_controlled_intersections(
        model,
        data,
        controlled_geom_ids,
        np.asarray(before_camera.world_position, dtype=np.float64),
        before_directions,
    )
    after_distance, after_assignment = _nearest_controlled_intersections(
        model,
        data,
        controlled_geom_ids,
        np.asarray(after_camera.world_position, dtype=np.float64),
        after_directions,
    )
    before_boundary = analytic_boundary_ambiguity(before_assignment)
    after_boundary = analytic_boundary_ambiguity(after_assignment)
    forward = _directional_transport(
        model,
        data,
        controlled_geom_ids,
        width,
        height,
        before_camera,
        after_camera,
        before_assignment,
        before_distance,
        before_boundary,
        after_boundary,
    )
    backward = _directional_transport(
        model,
        data,
        controlled_geom_ids,
        width,
        height,
        after_camera,
        before_camera,
        after_assignment,
        after_distance,
        after_boundary,
        before_boundary,
    )
    return AnalyticTransportArrays(
        forward=forward,
        backward=backward,
        before_surface_assignment=before_assignment,
        after_surface_assignment=after_assignment,
        before_boundary_ambiguous=before_boundary,
        after_boundary_ambiguous=after_boundary,
    )
