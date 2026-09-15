"""CPU-only instruments for the bounded capture-contract revision study.

This module deliberately has no renderer imports.  Geometry is a privileged
instrument, supplied as the compiled finite-plane/oriented-box facts recorded
with a capture receipt; it is not a label source for ordinary model paths.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np


class RevisionAnalysisError(ValueError):
    """A malformed receipt/array is an integrity stop, not an empirical result."""


SCALES = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
_EPS = 1e-12


def _array(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if ndim is not None and array.ndim != ndim:
        raise RevisionAnalysisError(f"{name} must have {ndim} dimensions")
    if not np.all(np.isfinite(array)):
        raise RevisionAnalysisError(f"{name} contains non-finite values")
    return array


def _vector(value: Any, *, name: str) -> np.ndarray:
    array = _array(value, name=name, ndim=1).astype(np.float64, copy=False)
    if array.shape != (3,):
        raise RevisionAnalysisError(f"{name} must be a length-three vector")
    return array


def _rotation(value: Any, *, name: str) -> np.ndarray:
    array = _array(value, name=name).astype(np.float64, copy=False)
    if array.size != 9:
        raise RevisionAnalysisError(f"{name} must contain nine values")
    rotation = array.reshape(3, 3)
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-9):
        raise RevisionAnalysisError(f"{name} must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, rtol=0.0, atol=1e-9):
        raise RevisionAnalysisError(f"{name} must be a proper rotation")
    return rotation


def _field(mapping: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    raise RevisionAnalysisError("missing required geometry fact: " + names[0])


def _raw_id(item: Mapping[str, Any]) -> int:
    value = _field(item, "raw_geom_id", "geom_id", "id")
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise RevisionAnalysisError("geometry raw id must be a non-negative integer")
    return int(value)


def _plane_basis(item: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return centre, unit u/v axes and half extents for one finite plane."""
    centre = _vector(_field(item, "center", "point", "position"), name="plane centre")
    half = np.asarray(_field(item, "half_extents", "half_sizes"), dtype=np.float64).reshape(-1)
    if half.size < 2 or np.any(~np.isfinite(half[:2])) or np.any(half[:2] <= 0):
        raise RevisionAnalysisError("plane half_extents must contain two positive values")
    if "axis_u" in item and "axis_v" in item:
        u = _vector(item["axis_u"], name="plane axis_u")
        v = _vector(item["axis_v"], name="plane axis_v")
    elif "rotation" in item or "world_rotation" in item:
        rotation = _rotation(
            item.get("rotation", item.get("world_rotation")), name="plane rotation"
        )
        # MuJoCo compiled xmat: columns are local axes in world coordinates.
        u, v = rotation[:, 0], rotation[:, 1]
    else:
        normal = _vector(_field(item, "normal"), name="plane normal")
        normal /= np.linalg.norm(normal)
        helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(helper, normal)
        v = np.cross(normal, u)
    for axis, name in ((u, "plane axis_u"), (v, "plane axis_v")):
        norm = np.linalg.norm(axis)
        if not np.isfinite(norm) or norm <= _EPS:
            raise RevisionAnalysisError(f"{name} is degenerate")
    u /= np.linalg.norm(u)
    v -= u * np.dot(u, v)
    vnorm = np.linalg.norm(v)
    if vnorm <= _EPS:
        raise RevisionAnalysisError("plane axes are collinear")
    return centre, u, v / vnorm, half[:2]


def _plane_intersections(
    origin: np.ndarray, directions: np.ndarray, item: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized finite-plane intersections; ``directions`` is N by 3."""
    centre, u, v, half = _plane_basis(item)
    rotation = np.column_stack((u, v, np.cross(u, v)))
    local_origin = (origin - centre) @ rotation
    local_directions = directions @ rotation
    denom = local_directions[:, 2]
    distance = np.full(len(directions), np.inf, dtype=np.float64)
    non_parallel = np.abs(denom) > _EPS
    distance[non_parallel] = -local_origin[2] / denom[non_parallel]
    finite_distance = np.where(np.isfinite(distance), distance, 0.0)
    local_x = local_origin[0] + finite_distance * local_directions[:, 0]
    local_y = local_origin[1] + finite_distance * local_directions[:, 1]
    tolerance = 16 * np.finfo(np.float64).eps * np.maximum(1.0, half)
    valid = (
        (distance > _EPS)
        & (np.abs(local_x) <= half[0] + tolerance[0])
        & (np.abs(local_y) <= half[1] + tolerance[1])
    )
    distance[~valid] = np.inf
    safe_distance = np.where(np.isfinite(distance), distance, 0.0)
    return distance, origin + safe_distance[:, None] * directions


def _box_intersections(
    origin: np.ndarray, directions: np.ndarray, item: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized oriented-box slab intersections using compiled xmat convention."""
    centre = _vector(_field(item, "center", "position"), name="box centre")
    rotation = _rotation(_field(item, "rotation", "world_rotation"), name="box rotation")
    half = _vector(_field(item, "half_extents", "half_sizes"), name="box half_extents")
    if np.any(half <= 0):
        raise RevisionAnalysisError("box half_extents must be positive")
    local_origin = (origin - centre) @ rotation
    local_directions = directions @ rotation
    parallel = np.abs(local_directions) <= _EPS
    outside_parallel = parallel & (np.abs(local_origin) > half)
    safe_directions = np.where(parallel, 1.0, local_directions)
    first = (-half - local_origin) / safe_directions
    second = (half - local_origin) / safe_directions
    t_min = np.max(np.where(parallel, -np.inf, np.minimum(first, second)), axis=1)
    t_max = np.min(np.where(parallel, np.inf, np.maximum(first, second)), axis=1)
    distance = np.where(t_min > _EPS, t_min, t_max)
    valid = (
        ~np.any(outside_parallel, axis=1) & (t_max >= np.maximum(t_min, _EPS)) & (distance > _EPS)
    )
    distance = np.where(valid, distance, np.inf)
    safe_distance = np.where(np.isfinite(distance), distance, 0.0)
    return distance, origin + safe_distance[:, None] * directions


def _ray_directions(
    pose: Mapping[str, Any], height: int, width: int, *, corners: bool
) -> np.ndarray:
    if height <= 0 or width <= 0:
        raise RevisionAnalysisError("image dimensions must be positive")
    fovy = float(_field(pose, "fovy_degrees", "camera_field_of_view_degrees"))
    if not np.isfinite(fovy) or not 0 < fovy < 180:
        raise RevisionAnalysisError("camera FOV must be in (0, 180) degrees")
    rotation = _rotation(
        _field(pose, "camera_rotation", "camera_world_rotation_row_major"), name="camera rotation"
    )
    x = (
        np.arange(width + int(corners), dtype=np.float64) * 2 + (0 if corners else 1) - width
    ) / width
    y = (
        height - (np.arange(height + int(corners), dtype=np.float64) * 2 + (0 if corners else 1))
    ) / height
    xx, yy = np.meshgrid(x * (width / height), y)
    tangent = np.tan(np.deg2rad(fovy) / 2.0)
    local = np.stack((xx * tangent, yy * tangent, -np.ones_like(xx)), axis=-1)
    # xmat is world->camera for row-vector coordinates; ray is local->world.
    directions = local @ rotation.T
    return np.asarray(
        directions / np.linalg.norm(directions, axis=-1, keepdims=True),
        dtype=np.float64,
    )


def analytic_geometry_maps(
    geometry: Mapping[str, Any],
    pose: Mapping[str, Any],
    height: int,
    width: int,
    *,
    corners: bool = False,
) -> dict[str, np.ndarray]:
    """Trace centre or grid-corner rays against recorded finite controlled geometry."""
    origin = _vector(
        _field(pose, "camera_position", "camera_world_position"), name="camera position"
    )
    directions = _ray_directions(pose, height, width, corners=corners)
    assignment = np.full(directions.shape[:2], -1, dtype=np.int32)
    ray_distance = np.full(directions.shape[:2], np.nan, dtype=np.float64)
    axis_depth = np.full(directions.shape[:2], np.nan, dtype=np.float64)
    rotation = _rotation(
        _field(pose, "camera_rotation", "camera_world_rotation_row_major"), name="camera rotation"
    )
    items: list[tuple[str, Mapping[str, Any]]] = []
    items += [("plane", x) for x in geometry.get("finite_planes", ())]
    items += [("box", x) for x in geometry.get("oriented_boxes", ())]
    if not items:
        raise RevisionAnalysisError("geometry has no finite planes or oriented boxes")
    ids = [_raw_id(item) for _, item in items if isinstance(item, Mapping)]
    if len(ids) != len(items) or len(set(ids)) != len(ids):
        raise RevisionAnalysisError("controlled geometry raw ids must be unique")
    flat_directions = directions.reshape(-1, 3)
    best_distance = np.full(len(flat_directions), np.inf, dtype=np.float64)
    best_id = np.full(len(flat_directions), -1, dtype=np.int32)
    best_point = np.full((len(flat_directions), 3), np.nan, dtype=np.float64)
    for kind, item in items:
        if not isinstance(item, Mapping):
            raise RevisionAnalysisError("geometry member is not an object")
        distance, points = (
            _plane_intersections(origin, flat_directions, item)
            if kind == "plane"
            else _box_intersections(origin, flat_directions, item)
        )
        geom_id = _raw_id(item)
        replace = (distance < best_distance) | ((distance == best_distance) & (geom_id < best_id))
        best_distance[replace], best_id[replace], best_point[replace] = (
            distance[replace],
            geom_id,
            points[replace],
        )
    valid = np.isfinite(best_distance)
    assignment[...] = best_id.reshape(assignment.shape)
    ray_distance[...] = best_distance.reshape(ray_distance.shape)
    depths = -((best_point - origin) @ rotation)[:, 2]
    axis_depth[...] = np.where(valid, depths, np.nan).reshape(axis_depth.shape)
    return {
        "raw_geom_ids": assignment,
        "ray_distance": ray_distance,
        "camera_axis_depth": axis_depth,
    }


def analytic_boundary_band(assignments: np.ndarray) -> np.ndarray:
    assignments = np.asarray(assignments)
    if assignments.ndim != 2:
        raise RevisionAnalysisError("analytic assignments must be HxW")
    band = np.zeros(assignments.shape, dtype=bool)
    changed = assignments[:, 1:] != assignments[:, :-1]
    band[:, 1:] |= changed
    band[:, :-1] |= changed
    changed = assignments[1:, :] != assignments[:-1, :]
    band[1:, :] |= changed
    band[:-1, :] |= changed
    return band


def decode_encoded_segmentation(
    encoded_rgb: np.ndarray, segid_map: Any, *, flip_vertical: bool
) -> tuple[np.ndarray, np.ndarray]:
    encoded = np.asarray(encoded_rgb)
    if encoded.dtype != np.uint8 or encoded.ndim != 3 or encoded.shape[-1] != 3:
        raise RevisionAnalysisError("encoded RGB must be HxWx3 uint8")
    triples = scene_map_triples(segid_map)
    packed = (
        encoded[..., 0].astype(np.int64)
        + 256 * encoded[..., 1].astype(np.int64)
        + 65536 * encoded[..., 2].astype(np.int64)
    )
    pairs = np.full((*packed.shape, 2), -1, dtype=np.int32)
    for segid, objid, objtype in triples:
        pairs[packed == segid + 1] = (objid, objtype)
    if np.any((packed != 0) & (pairs[..., 0] == -1)):
        raise RevisionAnalysisError(
            "encoded RGB contains a non-background segid absent from the scene map"
        )
    if flip_vertical:
        pairs = np.flipud(pairs)
    return pairs, packed


def scene_map_triples(value: Any) -> tuple[tuple[int, int, int], ...]:
    rows: list[tuple[int, int, int]] = []
    iterable: Iterable[Any]
    if isinstance(value, Mapping):
        iterable = ((key, *pair) for key, pair in value.items())
    else:
        iterable = value
    try:
        for row in iterable:
            if isinstance(row, Mapping):
                row = (row["segid"], row["objid"], row["objtype"])
            if len(row) != 3 or not all(isinstance(x, (int, np.integer)) for x in row):
                raise RevisionAnalysisError(
                    "scene map rows must be integer segid/object-id/object-type triples"
                )
            rows.append((int(row[0]), int(row[1]), int(row[2])))
    except TypeError as exc:
        raise RevisionAnalysisError("scene map is invalid") from exc
    if len({x[0] for x in rows}) != len(rows):
        raise RevisionAnalysisError("scene map has duplicate segids")
    return tuple(sorted(rows))


def canonical_array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(
        np.asarray(array).astype(np.asarray(array).dtype.newbyteorder("<"), copy=False)
    )
    header = f"{value.dtype.str}|{value.shape}|C|".encode("ascii")
    return hashlib.sha256(header + value.tobytes()).hexdigest()


def exact_array_comparison(reference: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
    a, b = np.asarray(reference), np.asarray(observed)
    shape_equal, dtype_equal = a.shape == b.shape, a.dtype == b.dtype
    equal = bool(shape_equal and dtype_equal and np.array_equal(a, b))
    return {
        "shape_equal": shape_equal,
        "dtype_equal": dtype_equal,
        "value_equal": equal,
        "exact_equal": equal,
        "reference_sha256": canonical_array_hash(a),
        "observed_sha256": canonical_array_hash(b),
    }


def _empty_summary(scales: Sequence[float] = SCALES) -> dict[str, Any]:
    return {
        "count": 0,
        "signed_mean": None,
        "absolute_mean": None,
        "median_abs": None,
        "p95_abs": None,
        "p99_abs": None,
        "max": None,
        "above_scales": {f"{scale:.0e}": 0 for scale in scales},
    }


def depth_residual_summary(
    residual: np.ndarray, mask: np.ndarray | None = None, *, scales: Sequence[float] = SCALES
) -> dict[str, Any]:
    values = np.asarray(residual, dtype=np.float64)
    if mask is not None:
        values = values[np.asarray(mask, dtype=bool)]
    else:
        values = values.ravel()
    if not np.all(np.isfinite(values)):
        raise RevisionAnalysisError("depth residual selection contains non-finite values")
    if not values.size:
        return _empty_summary(scales)
    absolute = np.abs(values)
    return {
        "count": int(values.size),
        "signed_mean": float(np.mean(values)),
        "absolute_mean": float(np.mean(absolute)),
        "median_abs": float(np.quantile(absolute, 0.5, method="linear")),
        "p95_abs": float(np.quantile(absolute, 0.95, method="linear")),
        "p99_abs": float(np.quantile(absolute, 0.99, method="linear")),
        "max": float(np.max(absolute)),
        "above_scales": {
            f"{scale:.0e}": int(np.count_nonzero(absolute > scale)) for scale in scales
        },
    }


def analyze_pose(pose_record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one saved pose and return masks/results without changing membership."""
    pose, geometry, modalities = (
        pose_record["pose_facts"],
        pose_record["geometry_facts"],
        pose_record["modalities"],
    )
    rgb = np.asarray(modalities["rgb"])
    saved_depth = np.asarray(modalities["depth"])
    raw = np.asarray(modalities["segmentation"]["raw_geom_ids"])
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise RevisionAnalysisError("RGB must be HxWx3 uint8")
    if (
        saved_depth.dtype != np.float32
        or saved_depth.ndim != 2
        or not np.all(np.isfinite(saved_depth))
        or np.any(saved_depth <= 0)
    ):
        raise RevisionAnalysisError("depth must be finite positive HxW float32")
    if (
        raw.dtype != np.int32
        or raw.ndim != 2
        or raw.shape != saved_depth.shape
        or rgb.shape[:2] != saved_depth.shape
    ):
        raise RevisionAnalysisError(
            "raw geometry IDs must be HxW int32 with RGB/depth-matching shape"
        )
    depth = saved_depth.astype(np.float64)
    segmentation = modalities["segmentation"]
    if not isinstance(segmentation.get("readback_requires_vertical_flip"), bool):
        raise RevisionAnalysisError("readback_requires_vertical_flip must be a bool")
    if int(segmentation.get("geom_objtype", 5)) != 5:
        raise RevisionAnalysisError("geom_objtype must be the fixed MuJoCo geom object type 5")
    pairs, _ = decode_encoded_segmentation(
        segmentation["encoded_rgb"],
        segmentation["segid_map"],
        flip_vertical=segmentation["readback_requires_vertical_flip"],
    )
    saved_pairs = np.asarray(segmentation["decoded_pairs"])
    if (
        saved_pairs.dtype != np.int32
        or saved_pairs.ndim != 3
        or saved_pairs.shape != (*depth.shape, 2)
    ):
        raise RevisionAnalysisError("decoded object/type pairs must be HxWx2 int32")
    if not exact_array_comparison(saved_pairs, pairs)["exact_equal"]:
        raise RevisionAnalysisError(
            "saved encoded RGB independently decodes differently from saved pairs"
        )
    geom_objtype = 5
    decoded_raw = np.where(pairs[..., 1] == geom_objtype, pairs[..., 0], -1).astype(raw.dtype)
    if not exact_array_comparison(raw, decoded_raw)["exact_equal"]:
        raise RevisionAnalysisError(
            "saved raw geometry IDs disagree with independently decoded RGB"
        )
    centre = analytic_geometry_maps(geometry, pose, *depth.shape)
    declared_ids = [
        _raw_id(item)
        for collection in (geometry.get("finite_planes", ()), geometry.get("oriented_boxes", ()))
        for item in collection
        if isinstance(item, Mapping)
    ]
    if not np.all(np.isin(raw, np.asarray([-1, *declared_ids], dtype=raw.dtype))):
        raise RevisionAnalysisError("saved raw IDs include an undeclared controlled geometry")
    boundary = analytic_boundary_band(centre["raw_geom_ids"])
    controlled = centre["raw_geom_ids"] >= 0
    no_hit = ~controlled
    disagreement = raw != centre["raw_geom_ids"]
    corners = analytic_geometry_maps(geometry, pose, *depth.shape, corners=True)
    corner_assignments = np.stack(
        (
            corners["raw_geom_ids"][:-1, :-1],
            corners["raw_geom_ids"][:-1, 1:],
            corners["raw_geom_ids"][1:, :-1],
            corners["raw_geom_ids"][1:, 1:],
        ),
        axis=-1,
    )
    corner_depths = np.stack(
        (
            corners["camera_axis_depth"][:-1, :-1],
            corners["camera_axis_depth"][:-1, 1:],
            corners["camera_axis_depth"][1:, :-1],
            corners["camera_axis_depth"][1:, 1:],
        ),
        axis=-1,
    )
    ambiguous = boundary | np.any(corner_assignments != centre["raw_geom_ids"][..., None], axis=-1)
    residual = depth - centre["camera_axis_depth"]
    finite = controlled & np.isfinite(residual)
    non_ambiguous = finite & ~ambiguous
    footprint_depths = np.concatenate(
        (centre["camera_axis_depth"][..., None], corner_depths), axis=-1
    )
    footprint_finite = np.isfinite(footprint_depths)
    has_footprint_depth = np.any(footprint_finite, axis=-1)
    lo = np.where(
        has_footprint_depth,
        np.min(np.where(footprint_finite, footprint_depths, np.inf), axis=-1),
        np.nan,
    )
    hi = np.where(
        has_footprint_depth,
        np.max(np.where(footprint_finite, footprint_depths, -np.inf), axis=-1),
        np.nan,
    )
    guard = 1e-6 * np.maximum(1.0, np.abs(centre["camera_axis_depth"]))
    out_of_range = non_ambiguous & ((depth < lo - guard) | (depth > hi + guard))
    named_ids = {
        str(item.get("name")): int(item["raw_geom_id"])
        for group in ("finite_planes", "oriented_boxes")
        for item in geometry.get(group, ())
        if isinstance(item, Mapping) and "name" in item and "raw_geom_id" in item
    }
    opposite_wall = None
    if {"corridor_left_surface", "corridor_right_surface"} <= set(named_ids):
        left, right = named_ids["corridor_left_surface"], named_ids["corridor_right_surface"]
        impossible = ((centre["raw_geom_ids"] == left) & (raw == right)) | (
            (centre["raw_geom_ids"] == right) & (raw == left)
        )
        opposite_wall = {
            "rule": "corridor_opposite_side_wall_sufficient_exclusion_v1",
            "excluded_disagreement_count": int(np.count_nonzero(impossible)),
            "excluded_disagreement_mask": impossible,
        }
    return {
        "analytic_raw_geom_ids": centre["raw_geom_ids"],
        "analytic_camera_axis_depth": centre["camera_axis_depth"],
        "analytic_ray_distance": centre["ray_distance"],
        "analytic_boundary_band": boundary,
        "analytic_no_hit": no_hit,
        "corner_raw_geom_ids": corner_assignments,
        "corner_camera_axis_depth": corner_depths,
        "sampled_footprint_ambiguity": ambiguous,
        "depth_outside_sampled_range": out_of_range,
        "raw_id_disagreement": disagreement,
        "raw_id_disagreement_counts": {
            "interior": int(np.count_nonzero(disagreement & controlled & ~boundary)),
            "boundary": int(np.count_nonzero(disagreement & controlled & boundary)),
            "no_hit": int(np.count_nonzero(disagreement & no_hit)),
        },
        "raw_id_disagreement_category_definition": (
            "controlled_hit_interior/control_hit_boundary/no_controlled_hit"
        ),
        "depth_residual": residual,
        "depth_residual_inside_boundary": depth_residual_summary(residual, finite & boundary),
        "depth_residual_outside_boundary": depth_residual_summary(residual, finite & ~boundary),
        "sampled_range_outside_count": int(np.count_nonzero(out_of_range)),
        "corridor_opposite_wall_sufficient_exclusion": opposite_wall,
    }


def compare_modalities(reference: Mapping[str, Any], observed: Mapping[str, Any]) -> dict[str, Any]:
    """Exact cross-policy/backend comparison; neither side is a ground truth oracle."""
    result: dict[str, Any] = {
        "rgb": exact_array_comparison(reference["rgb"], observed["rgb"]),
        "depth": exact_array_comparison(reference["depth"], observed["depth"]),
    }
    for key in ("encoded_rgb", "decoded_pairs", "raw_geom_ids"):
        first = np.asarray(reference["segmentation"][key])
        second = np.asarray(observed["segmentation"][key])
        result[key] = exact_array_comparison(first, second)
        if first.shape != second.shape:
            result[f"{key}_changed_pixel_count"] = None
        elif first.ndim == 3:
            result[f"{key}_changed_pixel_count"] = int(
                np.count_nonzero(np.any(first != second, axis=-1))
            )
        else:
            result[f"{key}_changed_pixel_count"] = int(np.count_nonzero(first != second))
    result["scene_map_equal"] = scene_map_triples(
        reference["segmentation"]["segid_map"]
    ) == scene_map_triples(observed["segmentation"]["segid_map"])
    rgb_a, rgb_b = np.asarray(reference["rgb"]), np.asarray(observed["rgb"])
    depth_a, depth_b = (
        np.asarray(reference["depth"], dtype=np.float64),
        np.asarray(observed["depth"], dtype=np.float64),
    )
    if rgb_a.shape == rgb_b.shape:
        result["rgb_changed_count"] = int(np.count_nonzero(np.any(rgb_a != rgb_b, axis=-1)))
    else:
        result["rgb_changed_count"] = None
    if (
        depth_a.shape == depth_b.shape
        and np.all(np.isfinite(depth_a))
        and np.all(np.isfinite(depth_b))
    ):
        result["depth_changed_count"] = int(np.count_nonzero(depth_a != depth_b))
        result["depth_difference"] = depth_residual_summary(depth_b - depth_a)
    else:
        result["depth_changed_count"], result["depth_difference"] = None, None
    return result


def analyze_capture_results(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Analyze every saved cell and predeclared comparable pair.

    ``cells`` are loader results with ``identity`` and ``poses`` from the
    revision capture receipt.  The return is intentionally per cell/pose and
    per pair: no aggregation can conceal an individual seed, policy or backend.
    """
    individual: list[dict[str, Any]] = []
    indexed: dict[tuple[Any, ...], tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    for cell in cells:
        identity = cell.get("identity")
        poses = cell.get("poses")
        if not isinstance(identity, Mapping) or not isinstance(poses, Sequence):
            raise RevisionAnalysisError("cell result must contain identity and poses")
        required = ("family", "episode_index", "backend", "policy")
        if any(key not in identity for key in required):
            raise RevisionAnalysisError("cell identity is incomplete")
        for pose in poses:
            if not isinstance(pose, Mapping) or "pose_name" not in pose:
                raise RevisionAnalysisError("cell pose is incomplete")
            key = (
                identity["family"],
                identity["episode_index"],
                identity["backend"],
                identity["policy"],
                pose["pose_name"],
            )
            if key in indexed:
                raise RevisionAnalysisError("duplicate cell/pose result")
            indexed[key] = (identity, pose)
            individual.append(
                {
                    "identity": dict(identity),
                    "pose_name": pose["pose_name"],
                    "analysis": analyze_pose(pose),
                }
            )
    comparisons: list[dict[str, Any]] = []
    for key, (_identity, pose) in indexed.items():
        family, episode, backend, policy, pose_name = key
        # All comparable endpoints are retained; absent fixed cells simply have
        # no entry, rather than being silently imputed or retried.
        for other_backend in ("wgl", "osmesa"):
            other_key = (family, episode, other_backend, policy, pose_name)
            if backend < other_backend and other_key in indexed:
                comparisons.append(
                    {
                        "kind": "same_policy_cross_backend",
                        "reference": key,
                        "observed": other_key,
                        "metrics": compare_modalities(
                            pose["modalities"], indexed[other_key][1]["modalities"]
                        ),
                    }
                )
        for other_policy in ("joint4", "hybrid", "joint0"):
            other_key = (family, episode, backend, other_policy, pose_name)
            if policy < other_policy and other_key in indexed:
                comparisons.append(
                    {
                        "kind": "same_backend_cross_policy",
                        "reference": key,
                        "observed": other_key,
                        "metrics": compare_modalities(
                            pose["modalities"], indexed[other_key][1]["modalities"]
                        ),
                    }
                )
    return {
        "schema_version": "capture_revision_analysis/v1",
        "cells": individual,
        "comparisons": comparisons,
    }
