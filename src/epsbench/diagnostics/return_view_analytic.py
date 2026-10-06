"""Privileged fixed analytic producer; importing never evaluates the candidate."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np
import numpy.typing as npt

from epsbench.diagnostics.boundary_observation import VisibleRaster
from epsbench.diagnostics.causal_history_fixture import Box, Camera, Scene
from epsbench.diagnostics.return_view_core import Observation, observe
from epsbench.utils.canonical import canonical_json_bytes

FIXED: dict[str, Any] = {
    "schema": "return-view-analytic-config-v1",
    "width": 160,
    "height": 120,
    "camera": [-4, 0.625, 0.125, 60],
    "support": [5, 9],
    "plane_z": 0,
    "occluder": [[-0.875, 1, 0], [1.125, 1.125, 2.25]],
    "background_lower": [-1.875, 3, 0],
    "background_upper": [[0.75, 3.125, 2.5], [1.125, 3.125, 2.5]],
    "poses": [2.5, 0, 0.25, 2.5],
    "executed": ["-5/2", "1/4"],
    "announced": "9/4",
    "members": ["rv-v1-a", "rv-v1-b"],
    "seeds": [202610062101, 202610062102],
    "decision": 2,
    "target": 3,
    "target_raw_label": 3,
    "provenance": "analytic-pixel-centre-slab-v1;tolerance=1e-10;native-unproven",
}
CONFIG_BYTES = canonical_json_bytes(FIXED)
CONFIG_SHA256 = hashlib.sha256(CONFIG_BYTES).hexdigest()


def validate_config(value: Any) -> None:
    if canonical_json_bytes(value) != CONFIG_BYTES:
        raise ValueError("only the prospectively committed exact configuration is supported")


@dataclass(frozen=True)
class Frame:
    raw_labels: npt.NDArray[np.int64]
    world_points: npt.NDArray[np.float64]
    observation: Observation


class Producer:
    def __init__(self, config: Any) -> None:
        validate_config(config)
        self.calls: set[tuple[int, int]] = set()
        self.scenes = tuple(
            Scene(
                Camera(160, 120, -4, 0.625, 0.125, 60),
                Box((-0.875, 1, 0), (1.125, 1.125, 2.25)),
                Box((-1.875, 3, 0), (upper, 3.125, 2.5)),
                (5, 9),
            )
            for upper in (0.75, 1.125)
        )
        self.mappings = []
        for seed in FIXED["seeds"]:
            rng = random.Random("return-view-mapping-v1:" + str(seed))
            labels = rng.sample(range(1, 1000), 3)
            opaque = rng.sample(range(1, 2**63), 3)
            self.mappings.append(
                {
                    raw: (label, f"surface-{token:016x}")
                    for raw, label, token in zip((1, 2, 3), labels, opaque, strict=True)
                }
            )

    def frame(self, member: int, index: int) -> Frame:
        if (
            type(member) is not int
            or member not in (0, 1)
            or type(index) is not int
            or index not in range(4)
        ):
            raise ValueError("fixed membership and pose index required")
        if (member, index) in self.calls:
            raise RuntimeError("pose evaluation may not repeat")
        self.calls.add((member, index))
        raw, points = self.scenes[member].frame(FIXED["poses"][index])
        return Frame(raw, points, adapt(raw, index, self.mappings[member]))


def adapt(
    raw: npt.NDArray[np.int64], index: int, mapping: dict[int, tuple[int, str]]
) -> Observation:
    if raw.dtype != np.int64 or raw.ndim != 2 or not set(np.unique(raw)) <= {0, 1, 2, 3}:
        raise ValueError("unsupported raw labels")
    if set(mapping) != {1, 2, 3} or len({v[0] for v in mapping.values()}) != 3:
        raise ValueError("explicit bijective positive adapter mapping required")
    if any(type(v[0]) is not int or not 0 < v[0] < 2**31 for v in mapping.values()):
        raise ValueError("zero reserved for unassociated sentinel")
    out = np.zeros(raw.shape, dtype=np.int32)
    identities = []
    for label, (positive, token) in mapping.items():
        out[raw == label] = positive
        if np.any(raw == label):
            identities.append((positive, token))
    return observe(VisibleRaster(index, out, tuple(identities)))


EXECUTED = (Fraction(-5, 2), Fraction(1, 4))
ANNOUNCED = Fraction(9, 4)
