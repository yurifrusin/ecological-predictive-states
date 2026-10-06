"""Development-only paired appearance contracts; importing this module never imports native SDKs."""

from __future__ import annotations

import copy
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np
import numpy.typing as npt
import yaml
from PIL import Image

from epsbench.diagnostics.boundary_observation import BoundaryObservationView, VisibleRaster
from epsbench.schema import Modality, ModalityPermissionSet
from epsbench.utils.canonical import canonical_json_bytes, logical_array_hash, sha256_bytes
from epsbench.utils.seeding import derive_seed, rng_for

VERSION = "paired-appearance-development-v1"
BASE = "0125c510b9d2b1e4760dcf1b987d52638bd9e18e"
HEIGHT, WIDTH = 120, 160
MIB = 1024**2
ENDPOINT_CAP, SHARED_CAP, TOTAL_CAP = MIB, 2 * MIB, 64 * MIB
ORIENTATION = "native_bottom_up_then_common_vertical_flip_to_image_top_down"
SDK_RENDERER_SHA256 = "c193df6a8b8cc1659819abd0ded5c17dab0e3e64edb7af6a1e8d249bb4a4a548"
FAMILIES: tuple[Literal["single_occluder", "corridor"], ...] = ("single_occluder", "corridor")
APPEARANCES = ("dev_pair_solid_v1", "dev_pair_brick_v1")
SURFACES = {
    "single_occluder": ("support_surface", "occluding_surface", "background_surface"),
    "corridor": (
        "corridor_floor",
        "corridor_left_surface",
        "corridor_right_surface",
        "corridor_end_surface",
    ),
}
PALETTE = (
    ((204, 132, 72), (54, 88, 124)),
    ((82, 184, 164), (132, 48, 94)),
    ((164, 104, 202), (44, 132, 76)),
    ((202, 184, 84), (84, 64, 154)),
)
Array = npt.NDArray[Any]
Family = Literal["single_occluder", "corridor"]

# Exact values are authority, not defaults that a caller may tune.
FIXED: dict[str, Any] = {
    "schema": VERSION,
    "base": BASE,
    "purpose": "development_only",
    "width": WIDTH,
    "height": HEIGHT,
    "roots": [2026100601, 2026100602],
    "appearances": list(APPEARANCES),
    "repeats": 2,
    "single": {
        "support_size": [4.0, 7.0, 0.1],
        "background_position": [0.0, 2.5, 1.05],
        "background_size": [2.2, 0.05, 1.05],
        "occluder_position": [0.0, 0.8, 0.9],
        "occluder_size": [0.55, 0.05, 0.9],
        "poses": [-0.35, 0.35],
        "camera_forward": -3.0,
        "height": 1.25,
        "fovy": 55.0,
        "xyaxes": [1.0, 0.0, 0.0, 0.0, 0.16, 1.0],
        "delta": [0.0, 0.7, 0.0],
    },
    "corridor": {
        "width": 3.0,
        "length": 6.0,
        "wall_height": 5.0,
        "wall_thickness": 0.05,
        "floor_thickness": 0.05,
        "poses": [0.5, 1.2],
        "camera_lateral": 0.0,
        "height": 1.25,
        "fovy": 55.0,
        "xyaxes": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "delta": [0.7, 0.0, 0.0],
    },
    "visual": {
        "ambient": [0.1, 0.1, 0.1],
        "diffuse": 0.7,
        "directions": [[0.2, 0.5, -1.0], [0.0, 0.25, -1.0]],
        "positions": [[-1.0, -2.0, 5.0], [0.0, -1.0, 6.0]],
        "znear": 0.01,
        "zfar": [20.0, 30.0],
        "texrepeat": [1.0, 1.0],
        "texuniform": False,
        "specular": 0.0,
        "shininess": 0.0,
        "reflectance": 0.0,
        "shadows": False,
        "fog": False,
        "haze": False,
        "offSamples": 0,
        "uv": "mujoco_geom_local_uv_repeat_v1",
        "filtering": "mujoco_linear_mipmap_linear_v1",
    },
    "palette": [[list(a), list(b)] for a, b in PALETTE],
    "texture": {
        "resolution": 128,
        "generator": "staggered_brick_128_v1",
        "background_rule": "row%16<4 OR (column+16*((row//16)%2))%32<8",
    },
    "thresholds": {
        "changed_fraction": 0.2,
        "normalized_mad": 0.025,
        "interior_pixels": 100,
        "luminance_std": 0.025,
        "luminance_mean": [0.05, 0.95],
        "neighbour_difference": 0.05,
        "neighbour_fraction": 0.02,
        "luminance_coefficients": [0.2126, 0.7152, 0.0722],
    },
    "limits": {
        "endpoint": ENDPOINT_CAP,
        "shared": SHARED_CAP,
        "total": TOTAL_CAP,
        "snapshots": 4,
        "snapshot": 65536,
        "other_metadata": 2 * 65536,
    },
}


def fixed_config(payload: bytes) -> dict[str, Any]:
    if len(payload) > 16384:
        raise ValueError("config bound")
    value = strict_json(payload)
    if canonical_json_bytes(value) != canonical_json_bytes(FIXED):
        raise ValueError("exact prospective configuration required; no substitution")
    return copy.deepcopy(value)


def config_bytes() -> bytes:
    return canonical_json_bytes(FIXED)


def contexts() -> tuple[tuple[Family, str, int], ...]:
    return tuple(
        (family, appearance, repeat)
        for family in FAMILIES
        for appearance in APPEARANCES
        for repeat in range(2)
    )


def seed_domain(family: Family) -> dict[str, int]:
    root = FIXED["roots"][FAMILIES.index(family)]
    episode = derive_seed(root, VERSION + ":episode:0")
    return {
        "root": root,
        "episode": episode,
        **{
            name: derive_seed(episode, VERSION + ":" + name)
            for name in ("geometry", "opaque", "style", "texture")
        },
    }


def slots(family: Family) -> tuple[int, ...]:
    return tuple(
        int(v)
        for v in rng_for(seed_domain(family)["style"], VERSION).permutation(len(SURFACES[family]))
    )


def mapping(family: Family, raw_ids: tuple[int, ...]) -> tuple[tuple[int, int, str], ...]:
    if (
        len(raw_ids) != len(SURFACES[family])
        or len(set(raw_ids)) != len(raw_ids)
        or any(type(v) is not int or v < 0 for v in raw_ids)
    ):
        raise ValueError("complete unique controlled geom IDs required")
    rng = rng_for(seed_domain(family)["opaque"], VERSION)
    labels = [int(v) + 1 for v in rng.permutation(len(raw_ids))]
    tokens = ["surface-" + bytes(rng.bytes(8)).hex() for _ in raw_ids]
    if len(set(tokens)) != len(tokens):
        raise ValueError("opaque collision; no replacement")
    return tuple(zip(raw_ids, labels, tokens, strict=True))


def brick(slot: int) -> Array:
    rows, cols = np.indices((128, 128))
    background = (rows % 16 < 4) | ((cols + 16 * ((rows // 16) % 2)) % 32 < 8)
    fg, bg = PALETTE[slot]
    return np.asarray(np.where(background[..., None], bg, fg), dtype=np.uint8)


@dataclass(frozen=True)
class DevelopmentVisualPlan:
    rgba_by_surface: dict[str, str]
    material_by_surface: dict[str, str]
    asset_bytes: dict[str, bytes]
    asset_xml: str
    light_direction: str
    light_diffuse: str
    light_ambient: str
    record: dict[str, Any]


def visual_plan(family: Family, appearance: str) -> DevelopmentVisualPlan:
    if family not in FAMILIES or appearance not in APPEARANCES:
        raise ValueError("fixed development appearance/family required")
    colours, materials, assets, xml, asset_hashes = {}, {}, {}, [], {}
    assignment = slots(family)
    for i, (name, slot) in enumerate(zip(SURFACES[family], assignment, strict=True)):
        if appearance == APPEARANCES[0]:
            colours[name] = " ".join(str(v / 255.0) for v in PALETTE[slot][0]) + " 1"
            materials[name] = f' material="dev_material_{i}"'
            xml.append(
                f'<material name="dev_material_{i}" rgba="1 1 1 1" '
                'specular="0" shininess="0" reflectance="0"/>'
            )
        else:
            array = brick(slot)
            stream = io.BytesIO()
            Image.fromarray(array).save(stream, format="PNG", compress_level=9, optimize=False)
            filename = f"dev-brick-{i}.png"
            assets[filename] = stream.getvalue()
            asset_hashes[filename] = {
                "file": sha256_bytes(assets[filename]),
                "array": logical_array_hash(array),
            }
            xml.extend(
                (
                    f'<texture name="dev_texture_{i}" type="2d" file="{filename}"/>',
                    f'<material name="dev_material_{i}" texture="dev_texture_{i}" '
                    'texrepeat="1 1" texuniform="false" '
                    'specular="0" shininess="0" reflectance="0"/>',
                )
            )
            colours[name] = "1 1 1 1"
            materials[name] = f' material="dev_material_{i}"'
    direction = FIXED["visual"]["directions"][FAMILIES.index(family)]
    record = {
        "schema": VERSION + ":appearance",
        "purpose": "development_only",
        "family": family,
        "appearance": appearance,
        "config_sha256": sha256_bytes(config_bytes()),
        "seeds": seed_domain(family),
        "slots": assignment,
        "assets": asset_hashes,
        "visual": FIXED["visual"],
        "palette": FIXED["palette"],
        "texture": FIXED["texture"] if assets else {"generator": "solid_foreground_v1"},
    }
    return DevelopmentVisualPlan(
        colours,
        materials,
        assets,
        "\n".join(xml),
        " ".join(map(str, direction)),
        ".7 .7 .7",
        ".1 .1 .1",
        json.loads(canonical_json_bytes(record)),
    )


PROTECTED_SEEDS = (
    "configs/evaluation_seed_candidates_v0.yaml",
    "configs/appearance_revision1_qualification_seeds_v0.yaml",
    "configs/appearance_benchmark_v0_evaluation_episode_seeds.yaml",
    "configs/development/causal_history_fixture_design_v1.json",
    "configs/development/return_view_analytic_v1.json",
)
PROTECTED_APPEARANCES = (
    "configs/appearance_candidates_v0.yaml",
    "configs/appearance_candidates_revision1.yaml",
)


def protection_check(repository: Path) -> dict[str, Any]:
    """Included public commitments only; no historical/private recovery completeness claim."""
    protected: set[int] = set()
    roots: list[dict[str, str]] = []

    def visit(value: Any, seed_context: bool = False) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, seed_context or "seed" in key)
        elif isinstance(value, list):
            for item in value:
                visit(item, seed_context)
        elif seed_context and type(value) is int:
            protected.add(value)

    for filename in PROTECTED_SEEDS:
        raw = (repository / filename).read_bytes()
        roots.append({"path": filename, "sha256": sha256_bytes(raw)})
        visit(yaml.safe_load(raw))
    # Existing public generator namespaces, plus the fresh diagnostic namespaces.
    namespaces = (
        "episode:0",
        "geometry-sampling",
        "surface-remapping",
        "surface-remapping:0",
        "causal-history-sequence-v1:opaque-surface-remapping",
        "appearance-base",
        "style-assignment",
        "texture-phase",
        "texture-slot",
        "illumination",
        *(VERSION + ":" + n for n in ("episode:0", "geometry", "opaque", "style", "texture")),
    )
    protected |= {derive_seed(seed, ns) for seed in tuple(protected) for ns in namespaces}
    # Explicit episode0 appearance chain and per-slot phase inputs used by included producers.
    episode_inputs = tuple(protected)
    for seed in episode_inputs:
        base = derive_seed(seed, "appearance-base")
        protected.add(base)
        for name in ("style-assignment", "texture-phase", "texture-slot", "illumination"):
            derived = derive_seed(base, name)
            protected.add(derived)
            if name == "texture-phase":
                protected.update(
                    derive_seed(derived, f"slot:style-slot-{slot}:surface-index:{index}")
                    for slot in range(4)
                    for index in range(4)
                )
    fresh = {v for family in FAMILIES for v in seed_domain(family).values()}
    fresh.update(
        derive_seed(seed_domain(family)[name], VERSION)
        for family in FAMILIES
        for name in ("style", "opaque")
    )
    if fresh & protected:
        raise ValueError("protected seed collision; fixed candidate rejected")
    for filename in PROTECTED_APPEARANCES:
        raw = (repository / filename).read_bytes()
        roots.append({"path": filename, "sha256": sha256_bytes(raw)})
        for profile in yaml.safe_load(raw)["profiles"]:
            old_palette = profile["palette"]
            old_pairs = {
                tuple(slot["foreground_rgb"]) + tuple(slot["background_rgb"])
                for values in old_palette.values()
                for slot in values
            }
            new_pairs = {fg + bg for fg, bg in PALETTE}
            if old_pairs & new_pairs:
                raise ValueError("protected palette-slot alias")
            # A solid alias can have different hidden background bytes.
            if profile["texture"]["family"] == "solid" and {
                tuple(s["foreground_rgb"]) for values in old_palette.values() for s in values
            } & {fg for fg, _ in PALETTE}:
                raise ValueError("protected solid colour alias")
            if profile["texture"]["family"] not in ("solid", "checker", "stripes"):
                raise ValueError("unsupported protected texture definition")
            # Our staggered rectangular brick is structurally neither canonical checker nor stripe.
            for slot in range(4):
                pixels = brick(slot)
                if not (
                    np.any(pixels[5, 0] != pixels[5, 9])
                    and np.any(pixels[5, 0] != pixels[21, 0])
                    and np.array_equal(pixels[0], pixels[16])
                ):
                    raise ValueError("brick structural signature differs")
    return {
        "schema": VERSION + ":protection",
        "inputs": roots,
        "fresh_seed_values": sorted(fresh),
        "included_protected_seed_values": len(protected),
        "coverage": (
            "enumerated public seed commitments, declared namespaces, episode0 appearance "
            "chain and four-slot phase inputs only"
        ),
        "historical_limit": (
            "not complete historical/private seed recovery; 18/21 limitation preserved"
        ),
        "appearance_check": (
            "all included profile definitions; palette slot/solid colour alias "
            "denial and distinct brick structural signature; no old assets rendered"
        ),
    }


def owned(array: Array) -> Array:
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def derive_pair(
    native_id: Array,
    native_depth: Array,
    scene_map: tuple[tuple[int, int, int], ...],
    near: float,
    far: float,
) -> tuple[Array, Array]:
    """Independent no-SDK reconstruction of top-down retained MuJoCo 3.12 paired terms."""
    if (
        native_id.dtype != np.uint8
        or native_id.shape != (HEIGHT, WIDTH, 3)
        or native_depth.dtype != np.float32
        or native_depth.shape != (HEIGHT, WIDTH)
    ):
        raise ValueError("native terms shape/dtype")
    if (
        not np.isfinite(native_depth).all()
        or np.any((native_depth < 0) | (native_depth > 1))
        or not np.isfinite([near, far]).all()
        or not 0 < near < far
    ):
        raise ValueError("depth/clipping range")
    entries = {}
    for segid, raw, objtype in scene_map:
        if (
            any(type(v) is not int for v in (segid, raw, objtype))
            or not 0 < segid <= 0xFFFFFF
            or not 0 <= raw < 2**31
            or objtype != 5
            or segid in entries
        ):
            raise ValueError("strict geom scene map")
        entries[segid] = raw
    code = native_id.astype(np.uint32)
    ids = code[..., 0] + 256 * code[..., 1] + 65536 * code[..., 2]
    if set(np.unique(ids)) - {0} - set(entries):
        raise ValueError("unmapped native ID")
    raw_labels = np.full(ids.shape, -1, dtype=np.int32)
    for segid, raw in entries.items():
        raw_labels[ids == segid] = raw
    zfar, znear = np.float32(far), np.float32(near)
    c = np.float32(-0.5) * (-(zfar + znear) / (zfar - znear)) - np.float32(0.5)
    d = np.float32(-0.5) * (-(np.float32(2) * zfar * znear) / (zfar - znear))
    metric = (d / (native_depth.astype(np.float64) + c)).astype(np.float32)
    return raw_labels, metric


@dataclass(frozen=True)
class Frame:
    family: Family
    appearance: str
    index: int
    rgb: Array
    native_id: Array
    native_depth: Array
    raw: Array
    depth: Array
    opaque: Array
    controlled: Array
    horizontal: Array
    vertical: Array
    evidence: dict[str, Any]

    def __post_init__(self) -> None:
        if (
            self.family not in FAMILIES
            or self.appearance not in APPEARANCES
            or type(self.index) is not int
            or self.index not in (0, 1)
        ):
            raise ValueError("fixed frame identity")
        specs = {
            "rgb": (np.dtype("uint8"), (HEIGHT, WIDTH, 3)),
            "native_id": (np.dtype("uint8"), (HEIGHT, WIDTH, 3)),
            "native_depth": (np.dtype("float32"), (HEIGHT, WIDTH)),
            "raw": (np.dtype("int32"), (HEIGHT, WIDTH)),
            "depth": (np.dtype("float32"), (HEIGHT, WIDTH)),
            "opaque": (np.dtype("int32"), (HEIGHT, WIDTH)),
            "controlled": (np.dtype("bool"), (HEIGHT, WIDTH)),
            "horizontal": (np.dtype("bool"), (HEIGHT, WIDTH - 1)),
            "vertical": (np.dtype("bool"), (HEIGHT - 1, WIDTH)),
        }
        for name, (dtype, shape) in specs.items():
            a = getattr(self, name)
            if type(a) is not np.ndarray or a.dtype != dtype or a.shape != shape:
                raise ValueError("frame shape/dtype: " + name)
            object.__setattr__(self, name, owned(a))
        object.__setattr__(self, "evidence", json.loads(canonical_json_bytes(self.evidence)))


def validate_frame(frame: Frame) -> None:
    e = frame.evidence
    required = {
        "schema",
        "source_head",
        "source_tree",
        "config_sha256",
        "appearance_record",
        "scene_xml_sha256",
        "compiled",
        "camera",
        "action",
        "runtime",
        "mapping",
        "scene_map",
        "near",
        "far",
        "orientation",
        "paired_stable",
        "rgb_stable",
        "rgb_material",
        "paired_material",
        "rgb_provenance",
    }
    if (
        set(e) != required
        or e["schema"] != VERSION + ":endpoint"
        or e["config_sha256"] != sha256_bytes(config_bytes())
        or e["appearance_record"] != visual_plan(frame.family, frame.appearance).record
    ):
        raise ValueError("endpoint evidence fields/config/appearance")
    for key in ("source_head", "source_tree"):
        if type(e[key]) is not str or re.fullmatch(r"[0-9a-f]{40}", e[key]) is None:
            raise ValueError("source binding")
    if (
        re.fullmatch(r"[0-9a-f]{64}", e["scene_xml_sha256"]) is None
        or e["orientation"] != ORIENTATION
        or e["rgb_provenance"]
        != {
            "producer": "mujoco.Renderer.render",
            "sdk_renderer_sha256": SDK_RENDERER_SHA256,
            "operation": "separate_ordinary_rgb_draw_before_owned_id_depth",
            "same_draw_as_pair": False,
            "index": frame.index,
        }
    ):
        raise ValueError("scene/orientation/separate RGB provenance")
    validate_registration(e)
    validate_compiled(frame.family, e["compiled"])
    validate_camera(frame.family, frame.index, e["camera"], e["action"])
    raw_ids = tuple(e["compiled"]["raw_geom_ids"][name] for name in SURFACES[frame.family])
    remap = mapping(frame.family, raw_ids)
    if e["mapping"] != [list(v) for v in remap]:
        raise ValueError("independent opaque mapping")
    raw, depth = derive_pair(
        frame.native_id,
        frame.native_depth,
        tuple(tuple(v) for v in e["scene_map"]),
        e["near"],
        e["far"],
    )
    if not np.array_equal(raw, frame.raw) or not np.array_equal(depth, frame.depth):
        raise ValueError("paired derivation")
    opaque = np.zeros(raw.shape, dtype=np.int32)
    if set(int(v) for v in np.unique(raw)) - {-1} - set(raw_ids):
        raise ValueError("uncontrolled visible raw geom")
    for raw_id, label, _ in remap:
        opaque[raw == raw_id] = label
    if (
        not np.array_equal(opaque, frame.opaque)
        or not np.array_equal(opaque > 0, frame.controlled)
        or not np.array_equal(opaque[:, :-1] != opaque[:, 1:], frame.horizontal)
        or not np.array_equal(opaque[:-1] != opaque[1:], frame.vertical)
    ):
        raise ValueError("mask/lattice derivation")
    if len(canonical_json_bytes(e)) > 65536:
        raise ValueError("endpoint scientific metadata cap")


def identities(frame: Frame) -> tuple[tuple[int, str], ...]:
    visible = set(int(v) for v in np.unique(frame.opaque)) - {0}
    return tuple(
        (label, token) for _, label, token in frame.evidence["mapping"] if label in visible
    )


class FrameProvider(Protocol):
    def frame(self, index: int) -> Frame: ...


@dataclass(frozen=True)
class ObservationView:
    provider: FrameProvider
    permissions: ModalityPermissionSet
    decision_index: int

    def __post_init__(self) -> None:
        if (
            type(self.permissions) is not ModalityPermissionSet
            or type(self.decision_index) is not int
            or self.decision_index not in (0, 1)
        ):
            raise ValueError("typed permissions and decision index")
        object.__setattr__(self, "permissions", self.permissions.model_copy(deep=True))

    def _read(self, index: int, modality: Modality) -> Frame:
        if type(index) is not int or index not in (0, 1):
            raise ValueError("endpoint index")
        if index > self.decision_index or not self.permissions.permits(modality):
            raise PermissionError("future/modality denied before provider access")
        frame = self.provider.frame(index)
        if type(frame) is not Frame or frame.index != index:
            raise ValueError("provider chronology")
        return frame

    def raster(self, index: int) -> VisibleRaster:
        f = self._read(index, Modality.SURFACE_REGIONS)
        return VisibleRaster(index, f.opaque, identities(f))

    def rgb(self, index: int) -> Array:
        return owned(self._read(index, Modality.RGB).rgb)

    def metric_depth(self, index: int) -> Array:
        return owned(self._read(index, Modality.DEPTH).depth)

    def instrumentation(self, index: int) -> dict[str, Any]:
        # Generation metadata includes poses/raw IDs/assets; never an ecological read.
        return copy.deepcopy(self._read(index, Modality.PRIVILEGED_GENERATION_RECORDS).evidence)

    def boundaries(self, index: int) -> bytes:
        return (
            BoundaryObservationView(self, self.permissions, self.decision_index)
            .observe(index)
            .canonical_bytes()
        )


def interior(mask: Array) -> Array:
    # Centre plus four neighbours; outer image border is excluded.
    padded = np.pad(mask, 1, constant_values=False)
    return cast(
        Array,
        np.logical_and.reduce(
            [
                padded[r : r + HEIGHT, c : c + WIDTH]
                for r, c in ((1, 1), (0, 1), (2, 1), (1, 0), (1, 2))
            ]
        ),
    )


def comparison(solid: Frame, texture: Frame) -> dict[str, Any]:
    validate_frame(solid)
    validate_frame(texture)
    if (solid.family, solid.index, solid.appearance, texture.appearance) != (
        texture.family,
        texture.index,
        APPEARANCES[0],
        APPEARANCES[1],
    ):
        raise ValueError("matched appearance pair")
    if any(
        solid.evidence[k] != texture.evidence[k]
        for k in (
            "source_head",
            "source_tree",
            "compiled",
            "camera",
            "action",
            "runtime",
            "mapping",
        )
    ):
        raise ValueError("matched structure/provenance differs")
    invariant = all(
        np.array_equal(getattr(solid, n), getattr(texture, n))
        for n in ("raw", "depth", "opaque", "controlled", "horizontal", "vertical")
    )
    if not invariant:
        return {"status": "FAIL", "reason": "appearance invariance"}
    controlled = solid.controlled
    denominator = int(controlled.sum())
    if denominator == 0:
        return {"status": "FAIL", "reason": "empty controlled coverage"}
    difference = np.abs(solid.rgb.astype(np.float64) - texture.rgb.astype(np.float64)) / 255.0
    changed = float(np.any(difference > 0, axis=2)[controlled].mean())
    mad = float(difference[controlled].mean())
    luminance = (texture.rgb.astype(np.float64) / 255.0) @ np.asarray([0.2126, 0.7152, 0.0722])
    surfaces = []
    passed = changed >= 0.2 and mad >= 0.025
    for _, label, token in solid.evidence["mapping"]:
        mask = interior(controlled & (solid.opaque == label))
        count = int(mask.sum())
        mean, std = (
            (float(luminance[mask].mean()), float(luminance[mask].std(ddof=0)))
            if count
            else (0.0, 0.0)
        )
        h = mask[:, :-1] & mask[:, 1:]
        v = mask[:-1] & mask[1:]
        pairs = int(h.sum() + v.sum())
        contrast = int(
            ((np.abs(luminance[:, :-1] - luminance[:, 1:]) >= 0.05) & h).sum()
            + ((np.abs(luminance[:-1] - luminance[1:]) >= 0.05) & v).sum()
        )
        fraction = contrast / pairs if pairs else 0.0
        ok = (
            count >= 100
            and 0.05 <= mean <= 0.95
            and std >= 0.025
            and pairs > 0
            and fraction >= 0.02
        )
        passed &= ok
        surfaces.append(
            {
                "surface": token,
                "interior_pixels": count,
                "mean": mean,
                "std": std,
                "pairs": pairs,
                "contrast_fraction": fraction,
                "pass": ok,
            }
        )
    return {
        "status": "PASS" if passed else "FAIL",
        "changed_fraction": changed,
        "normalized_mad": mad,
        "surfaces": surfaces,
    }


def repeat_equal(a: Frame, b: Frame) -> bool:
    validate_frame(a)
    validate_frame(b)
    return (
        (a.family, a.appearance, a.index) == (b.family, b.appearance, b.index)
        and a.evidence == b.evidence
        and all(np.array_equal(getattr(a, n), getattr(b, n)) for n in ARRAY_NAMES)
    )


ARRAY_NAMES = (
    "rgb",
    "native_id",
    "native_depth",
    "raw",
    "depth",
    "opaque",
    "controlled",
    "horizontal",
    "vertical",
)


def encode(frame: Frame) -> bytes:
    validate_frame(frame)
    metadata = canonical_json_bytes(
        {
            "schema": VERSION + ":frame",
            "family": frame.family,
            "appearance": frame.appearance,
            "index": frame.index,
            "evidence": frame.evidence,
            "hashes": {n: logical_array_hash(getattr(frame, n)) for n in ARRAY_NAMES},
        }
    )
    stream = io.BytesIO()
    np.savez(
        stream,
        metadata=np.frombuffer(metadata, dtype=np.uint8),
        **{n: getattr(frame, n) for n in ARRAY_NAMES},
    )
    data = stream.getvalue()
    if len(data) > ENDPOINT_CAP:
        raise ValueError("endpoint encoding cap")
    return data


def decode(data: bytes) -> Frame:
    if len(data) > ENDPOINT_CAP:
        raise ValueError("endpoint decoding cap")
    import zipfile

    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if (
            len(archive.infolist()) != 10
            or sum(v.file_size for v in archive.infolist()) > ENDPOINT_CAP
        ):
            raise ValueError("expanded endpoint cap")
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        if set(archive.files) != {"metadata", *ARRAY_NAMES}:
            raise ValueError("endpoint archive fields")
        metadata_array = archive["metadata"]
        if (
            metadata_array.dtype != np.uint8
            or metadata_array.ndim != 1
            or metadata_array.nbytes > 65536
        ):
            raise ValueError("metadata shape/cap")
        meta = strict_json(metadata_array.tobytes())
        if (
            set(meta) != {"schema", "family", "appearance", "index", "evidence", "hashes"}
            or meta["schema"] != VERSION + ":frame"
        ):
            raise ValueError("frame envelope")
        frame = Frame(
            meta["family"],
            meta["appearance"],
            meta["index"],
            **{n: archive[n] for n in ARRAY_NAMES},
            evidence=meta["evidence"],
        )
    validate_frame(frame)
    if meta["hashes"] != {n: logical_array_hash(getattr(frame, n)) for n in ARRAY_NAMES}:
        raise ValueError("scientific array hash corruption")
    return frame


def inspect(frame: Frame) -> bytes:
    """Small RGB/lattice overlay, returned as bytes; no filesystem or native action."""
    validate_frame(frame)
    rgb = frame.rgb.copy()
    edge = np.zeros((HEIGHT, WIDTH), dtype=np.bool_)
    edge[:, :-1] |= frame.horizontal
    edge[:-1] |= frame.vertical
    rgb[edge] = [255, 255, 255]
    stream = io.BytesIO()
    Image.fromarray(rgb).save(stream, format="PNG")
    return stream.getvalue()


def retention_bound() -> dict[str, int]:
    arrays = (
        HEIGHT * WIDTH * (3 + 3 + 4 + 4 + 4 + 4 + 1) + HEIGHT * (WIDTH - 1) + (HEIGHT - 1) * WIDTH
    )
    endpoint = arrays + HEIGHT * WIDTH * (3 + 4) + 4 * 65536 + 2 * 65536
    original = 16 * ENDPOINT_CAP + SHARED_CAP
    total = original + 2 * original + MIB
    if endpoint > ENDPOINT_CAP or total > 55 * MIB or total > TOTAL_CAP:
        raise ValueError("prospective retained application bound")
    return {
        "array_bytes": arrays,
        "endpoint_reserved_bytes": endpoint,
        "original_cap": original,
        "total_with_two_exports_and_host_metadata": total,
        "application_cap": TOTAL_CAP,
    }


REGISTRATION_KEYS = (
    "rect",
    "ngeom",
    "scene_map",
    "scene_geometry",
    "scene_cameras",
    "projection_matrix_float32",
    "modelview_matrix_float32",
    "readPixelFormat",
    "readDepthMap",
    "read_buffer",
    "draw_buffer",
    "pack_alignment",
    "pack_row_length",
    "pack_skip_rows",
    "pack_skip_pixels",
    "pixel_pack_buffer_binding",
    "clip_origin",
    "clip_depth_mode",
    "framewidth",
    "stereo",
    "rnd_depth",
    "mjr_currentBuffer",
    "offSamples",
    "offWidth",
    "offHeight",
    "query_bindings_restored",
    "near",
    "far",
    "extent",
    "context_runtime",
    "offscreen_attachments",
)


def validate_registration(evidence: dict[str, Any]) -> None:
    rgb, pair = evidence["rgb_stable"], evidence["paired_stable"]
    # Exclusions are only segmentation/ID flags; the RGB operation necessarily differs there.
    if set(rgb) != {*REGISTRATION_KEYS, "scene_flags", "segment_enabled", "idcolor_enabled"} or set(
        pair
    ) != set(rgb):
        raise ValueError("closed actual RGB/pair state fields")
    if any(rgb[k] != pair[k] for k in REGISTRATION_KEYS):
        raise ValueError("actual RGB/pair registration drift")
    if (
        rgb["segment_enabled"] is not False
        or rgb["idcolor_enabled"] is not False
        or pair["segment_enabled"] is not True
        or pair["idcolor_enabled"] is not True
    ):
        raise ValueError("ordinary versus paired flags")
    if (
        pair["rect"] != [0, 0, WIDTH, HEIGHT]
        or pair["offSamples"] != 0
        or pair["rnd_depth"] is not False
        or pair["offWidth"] != WIDTH
        or pair["offHeight"] != HEIGHT
    ):
        raise ValueError("registered raster/sample convention")
    for key in ("projection_matrix_float32", "modelview_matrix_float32"):
        values = np.asarray(pair[key])
        if values.shape != (16,) or not np.isfinite(values).all():
            raise ValueError("actual projection/modelview")
    validate_producer_state(pair)
    if (
        len(rgb["scene_flags"]) != 11
        or any(
            rgb["scene_flags"][i] != pair["scene_flags"][i] for i in range(11) if i not in (8, 9)
        )
        or rgb["scene_flags"][8:10] != [0, 0]
    ):
        raise ValueError("closed ordinary/pair flag differences")
    if evidence["rgb_material"] != evidence["paired_material"] or set(evidence["rgb_material"]) != {
        "model",
        "scene_geoms",
        "scene_lights",
    }:
        raise ValueError("actual material/asset/light drift")
    if evidence["runtime"] != pair["context_runtime"]:
        raise ValueError("actual runtime binding")
    expected_map = [
        {"segid_plus_one": segid, "objid": raw, "objtype": kind}
        for segid, raw, kind in evidence["scene_map"]
    ]
    if pair["scene_map"] != expected_map or (pair["near"], pair["far"]) != (
        evidence["near"],
        evidence["far"],
    ):
        raise ValueError("native state/array producer terms")


def expected_geometry(family: Family) -> dict[str, tuple[list[float], list[float], str]]:
    if family == "single_occluder":
        return {
            "support_surface": ([0.0, 0.0, 0.0], [4.0, 7.0, 0.1], "plane"),
            "occluding_surface": ([0.0, 0.8, 0.9], [0.55, 0.05, 0.9], "box"),
            "background_surface": ([0.0, 2.5, 1.05], [2.2, 0.05, 1.05], "box"),
        }
    return {
        "corridor_floor": ([0.0, 3.0, -0.05], [1.5, 3.0, 0.05], "box"),
        "corridor_left_surface": ([-1.5, 3.0, 2.5], [0.05, 3.0, 2.5], "box"),
        "corridor_right_surface": ([1.5, 3.0, 2.5], [0.05, 3.0, 2.5], "box"),
        "corridor_end_surface": ([0.0, 6.0, 2.5], [1.5, 0.05, 2.5], "box"),
    }


def validate_compiled(family: Family, compiled: dict[str, Any]) -> None:
    expected = expected_geometry(family)
    for name, (pos, size, kind) in expected.items():
        if (
            compiled["raw_geom_world_positions"][name] != pos
            or compiled["raw_geom_compiled_sizes"][name] != size
            or compiled["raw_geom_types"][name] != kind
            or compiled["raw_geom_world_rotations_row_major"][name]
            != [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        ):
            raise ValueError("fixed compiled geometry diverged")
    if (
        set(compiled["raw_geom_ids"]) != set(expected)
        or compiled["camera_field_of_view_degrees"] != 55.0
    ):
        raise ValueError("compiled inventory/calibration")


def validate_xml(family: Family, appearance: str, xml: str) -> None:
    """Check builder constants/visual plan BEFORE compilation, without native imports."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml)
    geoms = root.findall("worldbody/geom")
    if len(geoms) != len(SURFACES[family]):
        raise ValueError("fixed builder geom inventory")
    plan = visual_plan(family, appearance)

    def vector(text: str) -> list[float]:
        return [float(v) for v in text.split()]

    for geom in geoms:
        name = geom.attrib["name"]
        pos, size, kind = expected_geometry(family)[name]
        if (
            vector(geom.get("pos", "0 0 0")) != pos
            or vector(geom.attrib["size"]) != size
            or geom.attrib["type"] != kind
            or vector(geom.attrib["rgba"]) != vector(plan.rgba_by_surface[name])
        ):
            raise ValueError("fixed builder geometry/colour")
        expected_material = (
            plan.material_by_surface[name].split('"')[1] if plan.material_by_surface[name] else None
        )
        if geom.get("material") != expected_material:
            raise ValueError("fixed material attachment")
    camera = root.find("worldbody/camera")
    if camera is None or camera.get("name") != "monocular_camera":
        raise ValueError("fixed builder camera")
    fixed = FIXED["single" if family == "single_occluder" else "corridor"]
    pos = [fixed["poses"][0], -3.0, 1.25] if family == "single_occluder" else [0.0, 0.5, 1.25]
    if (
        vector(camera.attrib["pos"]) != pos
        or vector(camera.attrib["xyaxes"]) != fixed["xyaxes"]
        or float(camera.attrib["fovy"]) != 55.0
    ):
        raise ValueError("fixed camera pose/calibration")
    light = root.find("worldbody/light")
    if (
        light is None
        or light.get("directional") != "true"
        or light.get("castshadow") != "false"
        or vector(light.attrib["dir"]) != vector(plan.light_direction)
        or vector(light.attrib["ambient"]) != [0.1, 0.1, 0.1]
        or vector(light.attrib["diffuse"]) != [0.7, 0.7, 0.7]
        or vector(light.attrib["specular"]) != [0.0, 0.0, 0.0]
        or vector(light.attrib["pos"]) != FIXED["visual"]["positions"][FAMILIES.index(family)]
    ):
        raise ValueError("fixed builder light")
    actual = root.find("asset")
    wanted = ET.fromstring("<asset>" + plan.asset_xml + "</asset>")
    if actual is None or [(v.tag, v.attrib) for v in actual] != [(v.tag, v.attrib) for v in wanted]:
        raise ValueError("exact texture/material asset definitions")
    visual = root.find("visual")
    if visual is None:
        raise ValueError("visual definitions")
    global_ = visual.find("global")
    map_ = visual.find("map")
    head = visual.find("headlight")
    quality = visual.find("quality")
    if (
        global_ is None
        or global_.attrib != {"offwidth": "160", "offheight": "120"}
        or map_ is None
        or float(map_.attrib["znear"]) != 0.01
        or float(map_.attrib["zfar"]) != FIXED["visual"]["zfar"][FAMILIES.index(family)]
        or quality is None
        or quality.attrib != {"shadowsize": "0"}
        or head is None
        or head.attrib
        != {"ambient": "0 0 0", "diffuse": "0 0 0", "specular": "0 0 0", "active": "0"}
    ):
        raise ValueError("fixed visual clipping/headlight/storage")


def validate_camera(
    family: Family, index: int, camera: dict[str, Any], action: list[float]
) -> None:
    fixed = FIXED["single" if family == "single_occluder" else "corridor"]
    position = (
        [fixed["poses"][index], -3.0, 1.25]
        if family == "single_occluder"
        else [0.0, fixed["poses"][index], 1.25]
    )
    x = np.asarray(fixed["xyaxes"][:3])
    y = np.asarray(fixed["xyaxes"][3:])
    y = y / np.linalg.norm(y)
    rotation = np.column_stack((x, y, np.cross(x, y))).reshape(-1)
    if (
        set(camera) != {"world_position", "rotation_row_major", "fovy"}
        or camera["world_position"] != position
        or camera["fovy"] != 55.0
        or action != fixed["delta"]
        or not np.allclose(camera["rotation_row_major"], rotation, rtol=0.0, atol=1e-14)
    ):
        raise ValueError("fixed actual camera/action diverged")


def validate_producer_state(state: dict[str, Any]) -> None:
    """Replay pinned facts without SDK imports; actual review also uses the original validator."""
    exact = {
        "rect": [0, 0, WIDTH, HEIGHT],
        "readPixelFormat": 6407,
        "readDepthMap": 1,
        "read_buffer": 36064,
        "draw_buffer": 36064,
        "pack_alignment": 1,
        "pack_row_length": 0,
        "pack_skip_rows": 0,
        "pack_skip_pixels": 0,
        "pixel_pack_buffer_binding": 0,
        "clip_origin": 36001,
        "clip_depth_mode": 37727,
        "framewidth": 0.0,
        "stereo": 0,
        "rnd_depth": False,
        "segment_enabled": True,
        "idcolor_enabled": True,
        "mjr_currentBuffer": 1,
        "offSamples": 0,
        "offWidth": WIDTH,
        "offHeight": HEIGHT,
        "query_bindings_restored": True,
    }
    if any(type(state[k]) is not type(v) or state[k] != v for k, v in exact.items()):
        raise ValueError("pinned stable producer facts")
    if state["offscreen_attachments"] != {
        "offFBO": {
            "present": True,
            "draw_framebuffer_samples": 0,
            "color0": {
                "object_type": 36161,
                "component_type": 35863,
                "internal_format": 32856,
                "samples": 0,
            },
            "depth": {
                "object_type": 36161,
                "component_type": 5126,
                "internal_format": 36013,
                "samples": 0,
            },
        },
        "offFBO_r": {"present": False},
    }:
        raise ValueError("pinned attachment storage")
    flags = state["scene_flags"]
    if (
        type(flags) is not list
        or len(flags) != 11
        or any(type(v) is not int for v in flags)
        or any(flags[i] for i in (0, 5, 6, 7))
        or flags[8:10] != [1, 1]
    ):
        raise ValueError("pinned scene flag settings")
    cameras = state["scene_cameras"]
    camera_fields = {
        "pos",
        "forward",
        "up",
        "frustum_near",
        "frustum_far",
        "frustum_top",
        "frustum_bottom",
        "frustum_center",
        "frustum_width",
        "orthographic",
    }
    if type(cameras) is not list or len(cameras) != 2:
        raise ValueError("retained cameras")
    for camera in cameras:
        if set(camera) != camera_fields or camera["orthographic"] != 0:
            raise ValueError("pinned camera fields")
        for name in ("pos", "forward", "up"):
            vector = np.asarray(camera[name])
            if vector.shape != (3,) or not np.isfinite(vector).all():
                raise ValueError("camera vectors")
        if (
            not np.isfinite([camera[k] for k in camera_fields - {"pos", "forward", "up"}]).all()
            or not 0 < camera["frustum_near"] < camera["frustum_far"]
        ):
            raise ValueError("camera frustum")
    geometry = state["scene_geometry"]
    fields = {"type", "objid", "objtype", "segid", "category", "dataid", "pos", "mat", "size"}
    if type(geometry) is not list or not geometry or state["ngeom"] != len(geometry):
        raise ValueError("retained scene geometry")
    for geom in geometry:
        if set(geom) != fields or any(
            type(geom[k]) is not int for k in fields - {"pos", "mat", "size"}
        ):
            raise ValueError("geometry fields")
        for name, length in (("pos", 3), ("mat", 9), ("size", 3)):
            vector = np.asarray(geom[name])
            if vector.shape != (length,) or not np.isfinite(vector).all():
                raise ValueError("geometry vector")
    if state["scene_map"] != [
        {"segid_plus_one": g["segid"] + 1, "objid": g["objid"], "objtype": g["objtype"]}
        for g in geometry
        if g["segid"] != -1
    ]:
        raise ValueError("scene map/geometry association")
    if (
        not np.isfinite([state[k] for k in ("near", "far", "extent")]).all()
        or not 0 < state["near"] < state["far"]
        or state["extent"] <= 0
    ):
        raise ValueError("native clipping")
    runtime = {
        "actual_backend": "osmesa",
        "context_module": "mujoco.osmesa",
        "requested_offsamples": 0,
        "actual_offsamples": 0,
        "model_offsamples": 0,
        "width": WIDTH,
        "height": HEIGHT,
        "sample_buffers": 0,
        "samples": 0,
        "mjr_off_width": WIDTH,
        "mjr_off_height": HEIGHT,
        "color_storage_dimensions": [WIDTH, HEIGHT],
        "depth_storage_dimensions": [WIDTH, HEIGHT],
        "attachment_format": 32856,
        "attachment_component_type": 35863,
        "gl_samples": 0,
    }
    actual = state["context_runtime"]
    if (
        set(actual) != {*runtime, "gl_vendor", "gl_renderer", "gl_version"}
        or any(actual[k] != v for k, v in runtime.items())
        or any(
            type(actual[k]) is not str or not actual[k]
            for k in ("gl_vendor", "gl_renderer", "gl_version")
        )
    ):
        raise ValueError("pinned retained runtime")


def assess(frames: dict[tuple[Family, str, int], tuple[Frame, Frame]]) -> dict[str, Any]:
    """Finite full-matrix analysis only; failures are never dropped or pooled away."""
    if set(frames) != set(contexts()):
        return {"status": "INCONCLUSIVE", "reason": "missing/extra fixed context evidence"}
    cells: list[dict[str, Any]] = []
    try:
        for family, appearance, repeat in contexts():
            pair = frames[(family, appearance, repeat)]
            if len(pair) != 2:
                raise ValueError("endpoint coverage")
            for index, frame in enumerate(pair):
                if (frame.family, frame.appearance, frame.index) != (family, appearance, index):
                    raise ValueError("matrix/frame membership")
                validate_frame(frame)
        for family in FAMILIES:
            for appearance in APPEARANCES:
                if not all(
                    repeat_equal(a, b)
                    for a, b in zip(
                        frames[(family, appearance, 0)],
                        frames[(family, appearance, 1)],
                        strict=True,
                    )
                ):
                    return {"status": "FAIL", "reason": "exact regeneration mismatch"}
            for repeat in range(2):
                for index in range(2):
                    result = comparison(
                        frames[(family, APPEARANCES[0], repeat)][index],
                        frames[(family, APPEARANCES[1], repeat)][index],
                    )
                    cells.append(
                        {"family": family, "repeat": repeat, "endpoint": index, "result": result}
                    )
        return {
            "status": "PASS" if all(c["result"]["status"] == "PASS" for c in cells) else "FAIL",
            "cells": cells,
            "contexts": 8,
            "endpoints": 16,
            "phase_gate_effect": "NONE",
            "claims": (
                "finite apparatus consistency only; RGB separate, metric accuracy unqualified"
            ),
        }
    except (ValueError, KeyError, TypeError) as error:
        return {
            "status": "INCONCLUSIVE",
            "reason": str(error),
            "completed_analysis_prefix": cells,
            "phase_gate_effect": "NONE",
        }


def strict_json(data: bytes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise ValueError("nonfinite JSON constant: " + value)

    return json.loads(data, object_pairs_hook=pairs, parse_constant=reject)
