"""Privileged, native-free geometry check for one fixed prospective fixture table.

This is not a renderer, sequence loader, causal state, or native qualification.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]
TOLERANCE = 1e-10


@dataclass(frozen=True)
class Camera:
    width: int = 160
    height: int = 120
    forward: float = -3.0
    elevation: float = 0.5
    up_y: float = 0.16
    fovy: float = 60.0

    @property
    def basis(self) -> FloatArray:
        up = np.array([0.0, self.up_y, 1.0])
        up /= np.linalg.norm(up)
        right = np.array([1.0, 0.0, 0.0])
        return np.stack((right, up, -np.cross(right, up)))

    def origin(self, lateral: float) -> FloatArray:
        return np.array([lateral, self.forward, self.elevation])

    def rays(self) -> FloatArray:
        row, col = np.indices((self.height, self.width))
        scale = math.tan(math.radians(self.fovy) / 2)
        x = (2 * (col + 0.5) / self.width - 1) * scale * self.width / self.height
        y = (1 - 2 * (row + 0.5) / self.height) * scale
        return np.stack((x, y, np.ones_like(x)), axis=-1) @ self.basis

    def project(self, points: FloatArray, lateral: float) -> tuple[FloatArray, BoolArray]:
        local = (points - self.origin(lateral)) @ self.basis.T
        depth = local[..., 2]
        scale = math.tan(math.radians(self.fovy) / 2)
        safe = np.where(depth > 0, depth, 1.0)
        x = local[..., 0] / safe / (scale * self.width / self.height)
        y = local[..., 1] / safe / scale
        pixel = np.stack(((x + 1) * self.width / 2 - 0.5, (1 - y) * self.height / 2 - 0.5), axis=-1)
        inside = (depth > 0) & (x >= -1) & (x < 1) & (y > -1) & (y <= 1)
        return pixel, inside


@dataclass(frozen=True)
class Box:
    lower: tuple[float, float, float]
    upper: tuple[float, float, float]


def box_distance(origin: FloatArray, rays: FloatArray, box: Box) -> FloatArray:
    """Slab intersections include all faces; closed edges, positive entry/exit."""
    near = np.full(rays.shape[:-1], -np.inf)
    far = np.full(rays.shape[:-1], np.inf)
    possible = np.ones(rays.shape[:-1], dtype=bool)
    for axis in range(3):
        component = rays[..., axis]
        parallel = component == 0
        possible &= ~parallel | (
            (origin[axis] >= box.lower[axis]) & (origin[axis] <= box.upper[axis])
        )
        divisor = np.where(parallel, 1.0, component)
        a = (box.lower[axis] - origin[axis]) / divisor
        b = (box.upper[axis] - origin[axis]) / divisor
        near = np.maximum(near, np.where(parallel, -np.inf, np.minimum(a, b)))
        far = np.minimum(far, np.where(parallel, np.inf, np.maximum(a, b)))
    distance = np.where(near > 0, near, far)
    return np.where(possible & (far >= near) & (distance > 0), distance, np.inf)


def plane_distance(
    origin: FloatArray, rays: FloatArray, half_extent: tuple[float, float]
) -> FloatArray:
    divisor = rays[..., 2]
    distance = -origin[2] / np.where(divisor == 0, 1.0, divisor)
    point = origin + distance[..., None] * rays
    valid = (divisor != 0) & (distance > 0)
    valid &= np.abs(point[..., 0]) <= half_extent[0]
    valid &= np.abs(point[..., 1]) <= half_extent[1]
    return np.where(valid, distance, np.inf)


@dataclass(frozen=True)
class Scene:
    camera: Camera
    occluder: Box
    background: Box
    support: tuple[float, float] = (4.0, 7.0)

    def intersect(self, origin: FloatArray, rays: FloatArray) -> tuple[IntArray, FloatArray]:
        distances = np.stack(
            (
                plane_distance(origin, rays, self.support),
                box_distance(origin, rays, self.occluder),
                box_distance(origin, rays, self.background),
            ),
            axis=-1,
        )
        winner = np.argmin(distances, axis=-1)
        distance = np.min(distances, axis=-1)
        labels = np.where(np.isfinite(distance), winner + 1, 0).astype(np.int64)
        return labels, distance

    def frame(self, lateral: float) -> tuple[IntArray, FloatArray]:
        origin = self.camera.origin(lateral)
        rays = self.camera.rays()
        labels, distance = self.intersect(origin, rays)
        points = origin + np.where(np.isfinite(distance), distance, 0)[..., None] * rays
        return labels, points

    def transport(self, before: float, after: float) -> tuple[IntArray, BoolArray]:
        labels, points = self.frame(before)
        pixel, inside = self.camera.project(points, after)
        origin = self.camera.origin(after)
        rays = points - origin
        target_labels, distance = self.intersect(origin, rays)
        # Parametric distance one means the same source point is first-visible.
        usable = (labels != 0) & inside & (labels == target_labels)
        usable &= np.abs(distance - 1) <= TOLERANCE
        row, col = np.indices(labels.shape)
        vectors = np.rint((pixel - np.stack((col, row), axis=-1)) * 1024).astype(np.int64)
        return np.where(usable[..., None], vectors, 0), usable


def equal_transport(a: tuple[IntArray, BoolArray], b: tuple[IntArray, BoolArray]) -> bool:
    return all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def check_candidate(path: Path) -> dict[str, Any]:
    """Evaluate only the prospectively recorded candidate; report failed relations."""
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    camera_config = config["camera"]
    camera = Camera(
        camera_config["width"],
        camera_config["height_pixels"],
        camera_config["forward"],
        camera_config["height"],
        camera_config["up_y"],
        camera_config["fovy_degrees"],
    )
    occluder = Box(tuple(config["occluder"]["lower"]), tuple(config["occluder"]["upper"]))
    yz = config["background_yz"]
    reports = []
    failures: list[str] = []
    for pair in config["pairs"]:
        poses = pair["poses"]
        decision = pair["decision"]
        scenes = [
            Scene(
                camera,
                occluder,
                Box((bounds[0], *yz["lower"]), (bounds[1], *yz["upper"])),
                tuple(config["support"]["half_extent"]),
            )
            for bounds in pair["background_x"]
        ]
        frames = [[scene.frame(pose)[0] for pose in poses] for scene in scenes]
        transports = [[scene.transport(a, b) for a, b in pairwise(poses)] for scene in scenes]
        current_equal = np.array_equal(frames[0][decision], frames[1][decision])
        history_equal = all(
            np.array_equal(a, b)
            for a, b in zip(frames[0][: decision + 1], frames[1][: decision + 1], strict=True)
        )
        flow_history_equal = all(
            equal_transport(a, b)
            for a, b in zip(transports[0][:decision], transports[1][:decision], strict=True)
        )
        target_difference = int(np.count_nonzero((frames[0][-1] == 3) != (frames[1][-1] == 3)))
        counts = [[int(np.count_nonzero(frame == 3)) for frame in member] for member in frames]
        support = [
            [int(np.count_nonzero((frames[m][i] == 3) & item[1])) for i, item in enumerate(member)]
            for m, member in enumerate(transports)
        ]
        relations = {
            "current_labels_equal": bool(current_equal),
            "target_masks_different": target_difference > 0,
            "background_previously_observed": all(
                any(n > 0 for n in c[: decision + 1]) for c in counts
            ),
        }
        if pair["pair"] == 1:
            relations["earlier_labels_distinct"] = not history_equal
            relations["reset_labels_equal"] = np.array_equal(frames[0][1], frames[1][1])
            relations["reset_transport_equal"] = equal_transport(transports[0][1], transports[1][1])
            for token in (1, 2, 3):
                relations[f"reset_support_token_{token}"] = all(
                    np.any((frames[m][1] == token) & transports[m][1][1]) for m in (0, 1)
                )
        elif pair["pair"] == 2:
            relations["earlier_labels_distinct"] = not history_equal
            relations["two_hidden_frames"] = all(c[2] == c[3] == 0 for c in counts)
            relations["observed_first_two_frames"] = all(c[0] > 0 and c[1] > 0 for c in counts)
            relations["usable_background_before_hiding"] = all(s[0] > 0 for s in support)
            relations["target_reveals_background"] = all(c[-1] > 0 for c in counts)
            relations["nonzero_signed_path"] = poses[-1] - poses[1] == 0.25
        else:
            relations["full_label_history_equal"] = history_equal
            relations["full_transport_history_equal"] = flow_history_equal
        failures.extend(
            f"pair{pair['pair']}:{name}" for name, value in relations.items() if not value
        )
        reports.append(
            {
                "pair": pair["pair"],
                "relations": {k: bool(v) for k, v in relations.items()},
                "background_pixel_counts": counts,
                "usable_background_counts": support,
                "target_symmetric_difference": target_difference,
                "signed_actions": [b - a for a, b in pairwise(poses)],
            }
        )
    return {
        "status": "ANALYTIC_FAILED" if failures else "ANALYTIC_RELATIONS_SATISFIED",
        "native_rgb_equality": "UNPROVEN",
        "native_state_equality": "UNPROVEN",
        "pose_count": sum(2 * len(p["poses"]) for p in config["pairs"]),
        "failures": failures,
        "pairs": reports,
    }
