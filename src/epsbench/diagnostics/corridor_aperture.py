"""Fixed nine-box aperture apparatus; source acceptance is not launch authority."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from fractions import Fraction as Q
from typing import Any

VERSION = "corridor-aperture-box-capability-v1"
WIDTH, HEIGHT = 128, 96
POSES = (2, 4, 6)
ACTIONS = ((2, 0, 0), (2, 0, 0))
Point = tuple[Q, Q, Q]


@dataclass(frozen=True)
class Box:
    name: str
    lower: Point
    upper: Point


def box(name: str, lower: tuple[str, str, str], upper: tuple[str, str, str]) -> Box:
    return Box(name, tuple(Q(v) for v in lower), tuple(Q(v) for v in upper))  # type: ignore[arg-type]


BOXES = (
    box("floor", ("-2", "0", "-1/10"), ("2", "11", "0")),
    box("left", ("-41/20", "0", "0"), ("-39/20", "11", "8")),
    box("end", ("-2", "219/20", "0"), ("2", "221/20", "8")),
    box("right_bottom", ("39/20", "0", "0"), ("41/20", "11", "4/5")),
    box("right_top", ("39/20", "0", "6/5"), ("41/20", "11", "8")),
    box("right_front", ("39/20", "0", "4/5"), ("41/20", "7", "6/5")),
    box("right_pier", ("39/20", "38/5", "4/5"), ("41/20", "42/5", "6/5")),
    box("right_back", ("39/20", "89/10", "4/5"), ("41/20", "11", "6/5")),
    box("target", ("3", "199/20", "9/10"), ("61/20", "201/20", "11/10")),
)
NAMES = tuple(b.name for b in BOXES)
TARGET = "target"  # Privileged evaluator designation, never a public feature.


def encode(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def configuration() -> dict[str, Any]:
    return {
        "version": VERSION,
        "boxes": [[b.name, list(map(str, b.lower)), list(map(str, b.upper))] for b in BOXES],
        "poses": list(POSES),
        "actions": [list(a) for a in ACTIONS],
        "raster": [WIDTH, HEIGHT],
        "vertical_half_tangent": "3/4",
        "fixed_colour": ["3/5", "3/5", "3/5", "1"],
        "raw_identity": "whole_box_not_face",
        "compiled_coordinate_tolerance": "1/1000000000000",
        "draw_coordinate_tolerance": "1/500000",
        "projection_coefficient_tolerance": "1/500000",
        "near_far": "measured_extent_scaled_and_target_unclipped",
        "orientation": "top_down_half_offset_centres",
        "launch": "HELD_DEFAULT",
    }


def config_root() -> str:
    return digest(encode(configuration()))


def index(value: int) -> None:
    if type(value) is not int or value not in range(3):
        raise ValueError("exactly three consecutive aperture views required")


def scene_xml() -> str:
    """Pure fixed constructor; no native import, assets or caller XML injection."""
    geoms = []
    for b in BOXES:
        center = [(a + z) / 2 for a, z in zip(b.lower, b.upper, strict=True)]
        half = [(z - a) / 2 for a, z in zip(b.lower, b.upper, strict=True)]
        pos, size = (" ".join(repr(float(v)) for v in values) for values in (center, half))
        geoms.append(
            f'<geom name="{b.name}" type="box" pos="{pos}" size="{size}" rgba="0.6 0.6 0.6 1"/>'
        )
    fovy = math.degrees(2 * math.atan(0.75))
    return f'''<mujoco model="{VERSION}"><compiler angle="radian"/>
<visual><global offwidth="128" offheight="96"/><quality offsamples="0" shadowsize="0"/>
<map znear="0.01" zfar="30"/></visual><worldbody>
{"".join(geoms)}<camera name="aperture_camera" pos="0 2 1" xyaxes="1 0 0 0 0 1"
fovy="{fovy!r}"/></worldbody></mujoco>'''


def opaque_mapping(nonce: bytes, raw_ids: tuple[int, ...]) -> tuple[tuple[int, str], ...]:
    """External per-episode random nonce; no allocation occurs in source checks."""
    if type(nonce) is not bytes or len(nonce) != 32 or len(raw_ids) != 9:
        raise ValueError("private 32-byte episode nonce and nine raw IDs required")
    if any(type(v) is not int or not 0 <= v < 2**31 for v in raw_ids) or len(set(raw_ids)) != 9:
        raise ValueError("unique nonnegative raw IDs required")
    pairs = tuple(
        (
            raw,
            "surface-" + digest(VERSION.encode() + b"\0" + nonce + raw.to_bytes(8, "little"))[:16],
        )
        for raw in raw_ids
    )
    if len({t for _, t in pairs}) != 9:
        raise ValueError("opaque collision; retain failure, never redraw")
    return pairs


def validate_mapping(pairs: tuple[tuple[int, str], ...]) -> None:
    if (
        type(pairs) is not tuple
        or len(pairs) != 9
        or any(type(p) is not tuple or len(p) != 2 for p in pairs)
        or any(
            type(r) is not int
            or not 0 <= r < 2**31
            or type(t) is not str
            or re.fullmatch(r"surface-[0-9a-f]{16}", t) is None
            for r, t in pairs
        )
        or len({r for r, _ in pairs}) != 9
        or len({t for _, t in pairs}) != 9
    ):
        raise ValueError("exact bijective privileged mapping required")
