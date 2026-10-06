"""Closed one-pose illumination/sampling discriminator; no SDK access on import.

This apparatus module is privileged instrumentation, never an ecological loader.
"""

from __future__ import annotations

import copy
import io
import math
import os
import platform
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np
from pydantic import Field, field_validator, model_validator

from epsbench.diagnostics import paired_appearance as p
from epsbench.diagnostics.appearance_study import CONTRAST_V1, PAIRED_V1
from epsbench.diagnostics.paired_appearance_execution import (
    SPECS as SPECS,
)
from epsbench.diagnostics.paired_appearance_execution import (
    exclusive_file,
    safe_path,
)
from epsbench.diagnostics.paired_appearance_runtime import Closed, Hex40, Hex64, Image, Token
from epsbench.utils.canonical import canonical_json_bytes, logical_array_hash, sha256_bytes
from epsbench.utils.seeding import derive_seed, rng_for

SCHEMA = "renderer-illumination-sampling-v1"
CONFIG_PATH = "configs/development/renderer_illumination_sampling_v1.json"
VERSION: Literal["renderer_illumination_sampling_execution_v1"] = (
    "renderer_illumination_sampling_execution_v1"
)
NATIVE = "renderer_illumination_sampling_native_v1"
DUMMY = "renderer_illumination_sampling_dummy_v1"
APPEARANCES = ("dev_renderer_solid_v1", "dev_renderer_brick_v1")
ARMS = ("original_default", "unit_default", "original_nearest", "unit_nearest")
ROOT = 2026100605
SOURCE_URL: Literal["https://github.com/yurifrusin/ecological-predictive-states.git"] = (
    "https://github.com/yurifrusin/ecological-predictive-states.git"
)
ENV = "EPS_RENDERER_DISCRIMINATOR_BINDING"
FIXED = {
    "schema": SCHEMA,
    "root": ROOT,
    "purpose": "exposed_development_diagnostic_only",
    "appearances": list(APPEARANCES),
    "arms": list(ARMS),
    "repeats": 2,
    "pose": [0.0, 1.2, 1.25],
    "fovy": 55.0,
    "xyaxes": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    "width": 160,
    "height": 120,
    "geometry": CONTRAST_V1.fixed["corridor"],
    "visual": CONTRAST_V1.fixed["visual"],
    "thresholds": CONTRAST_V1.fixed["thresholds"],
    "illumination": {"original": [[0.1] * 3, [0.7] * 3], "unit": [[1.0] * 3, [0.0] * 3]},
    "sampling": {"default": [9987, 9729, 0, 1000], "nearest": [9728, 9728, 0, 0]},
    "asset_file": (
        "src/epsbench/diagnostics/source_assets/appearance_contrast_calibration_v1/brick.png"
    ),
    "asset_sha256": CONTRAST_V1.asset_hashes[0],
    "asset_pixel_sha256": CONTRAST_V1.asset_pixel_hashes[0],
    "absolute_native_uv_transfer_qualification": "UNRESOLVED",
    "limits": {
        "endpoint": p.MIB,
        "shared": 2 * p.MIB,
        "report": p.MIB,
        "original": 18 * p.MIB,
        "inclusive_copies": 2,
        "host": p.MIB,
        "total": 64 * p.MIB,
        "seconds": 300,
    },
}
# Caller dictionaries and replacements never confer authority. Canonical bytes are source authority.
CONFIG_BYTES = canonical_json_bytes(FIXED) + b"\n"


class Identity(Closed):
    arm: Literal["original_default", "unit_default", "original_nearest", "unit_nearest"]
    appearance: Literal["dev_renderer_solid_v1", "dev_renderer_brick_v1"]
    repeat: Literal[0, 1]

    @field_validator("repeat", mode="before")
    @classmethod
    def strict_repeat(cls, value: object) -> object:
        if type(value) is not int or value not in (0, 1):
            raise ValueError("integer repeated-capture ordinal required")
        return value


MEMBERS = tuple(
    Identity.model_validate({"arm": arm, "appearance": appearance, "repeat": repeat})
    for arm in ARMS
    for appearance in APPEARANCES
    for repeat in (0, 1)
)


def config(payload: bytes) -> dict[str, Any]:
    if type(payload) is not bytes or payload != CONFIG_BYTES:
        raise ValueError("exact discriminator source configuration required")
    return cast(dict[str, Any], p.strict_json(payload))


def seeds() -> dict[str, int]:
    episode = derive_seed(ROOT, SCHEMA + ":episode:0")
    return {
        "root": ROOT,
        "episode": episode,
        **{
            n: derive_seed(episode, SCHEMA + ":" + n)
            for n in ("geometry", "opaque", "style", "texture")
        },
    }


def remapping(raw_ids: tuple[int, ...]) -> tuple[tuple[int, int, str], ...]:
    if (
        len(raw_ids) != 4
        or len(set(raw_ids)) != 4
        or any(type(v) is not int or v < 0 for v in raw_ids)
    ):
        raise ValueError("four controlled raw identities required")
    rng = rng_for(seeds()["opaque"], SCHEMA)
    labels = [int(v) + 1 for v in rng.permutation(4)]
    tokens = ["surface-" + bytes(rng.bytes(8)).hex() for _ in raw_ids]
    if len(set(tokens)) != 4:
        raise ValueError("opaque collision, no replacement")
    return tuple(zip(raw_ids, labels, tokens, strict=True))


def protection(repository: Path) -> dict[str, Any]:
    # Reuse all existing final/public protections, with no generic waiver.
    previous = p.protection_check(repository, CONTRAST_V1)
    fresh = set(seeds().values()) | {derive_seed(seeds()[n], SCHEMA) for n in ("opaque", "style")}
    import yaml

    protected: set[int] = set()

    def visit(value: Any, selected: bool = False) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, selected or "seed" in key)
        elif isinstance(value, list):
            for item in value:
                visit(item, selected)
        elif selected and type(value) is int:
            protected.add(value)

    for filename in p.PROTECTED_SEEDS:
        visit(yaml.safe_load((repository / filename).read_bytes()))
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
        *(SCHEMA + ":" + n for n in ("episode:0", "geometry", "opaque", "style", "texture")),
    )
    protected |= {derive_seed(v, n) for v in tuple(protected) for n in namespaces}
    for value in tuple(protected):
        base = derive_seed(value, "appearance-base")
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
    old_configs = []
    for study in (PAIRED_V1, CONTRAST_V1):
        if (repository / study.config_path).read_bytes() != study.config_file_bytes:
            raise ValueError("protected appearance source configuration changed")
        old_configs.append({"path": study.config_path, "sha256": study.config_file_hash})
        protected.update(v for f in p.FAMILIES for v in p.seed_domain(f, study).values())
        protected.update(
            derive_seed(p.seed_domain(f, study)[n], study.schema)
            for f in p.FAMILIES
            for n in ("opaque", "style")
        )
        if set(APPEARANCES) & set(study.appearances):
            raise ValueError("protected profile identity collision")
        p.canonical_assets(study)
    if fresh & protected:
        raise ValueError("protected seed collision; fixed candidate rejected without replacement")
    # The only exception is this exact already-exposed canonical development source file.
    raw = (repository / FIXED["asset_file"]).read_bytes()
    if sha256_bytes(raw) != CONTRAST_V1.asset_hashes[0]:
        raise ValueError("exact shared development asset required")
    return {
        "schema": SCHEMA + ":protection",
        "previous": previous,
        "previous_configs": old_configs,
        "fresh_seed_values": sorted(fresh),
        "protected_count": len(protected),
        "reuse_exception": {
            "path": FIXED["asset_file"],
            "file": CONTRAST_V1.asset_hashes[0],
            "pixels": CONTRAST_V1.asset_pixel_hashes[0],
        },
        "historical_limit": "18/21; no complete historical/private recovery claim",
    }


def scene_xml() -> tuple[str, dict[str, bytes]]:
    """One immutable model includes four textured and four solid materials, one light."""
    assets = p.canonical_assets(CONTRAST_V1)
    parts = []
    for i in range(4):
        parts.append(f'<texture name="dev_texture_{i}" type="2d" file="dev-brick-{i}.png"/>')
        parts.append(
            f'<material name="dev_material_{i}" texture="dev_texture_{i}" '
            'texrepeat="1 1" texuniform="false" specular="0" shininess="0" reflectance="0"/>'
        )
    for i in range(4, 8):
        parts.append(
            f'<material name="dev_material_{i}" rgba="1 1 1 1" '
            'specular="0" shininess="0" reflectance="0"/>'
        )
    geoms = []
    for i, (name, (position, size, _)) in enumerate(p.expected_geometry("corridor").items()):
        geoms.append(
            f'<geom name="{name}" type="box" pos="{" ".join(map(str, position))}" '
            f'size="{" ".join(map(str, size))}" rgba="1 1 1 1" material="dev_material_{i}"/>'
        )
    xml = (
        '<mujoco model="eps_renderer_discriminator_v1"><compiler angle="radian"/>'
        '<option gravity="0 0 -9.81" timestep="0.01"/><visual>'
        '<global offwidth="160" offheight="120"/>'
        '<quality shadowsize="0" offsamples="0"/><map znear="0.01" zfar="30"/>'
        '<headlight ambient="0 0 0" diffuse="0 0 0" specular="0 0 0" active="0"/>'
        "</visual><asset>" + "".join(parts) + "</asset><worldbody>"
        '<light name="key" directional="true" castshadow="false" pos="0 -1 6" '
        'dir="0 0.25 -1" diffuse="0.7 0.7 0.7" ambient="0.1 0.1 0.1" specular="0 0 0"/>'
        + "".join(geoms)
        + '<camera name="monocular_camera" pos="0 1.2 1.25" '
        'xyaxes="1 0 0 0 0 1" fovy="55"/></worldbody></mujoco>'
    )
    return xml, {f"dev-brick-{i}.png": assets[0] for i in range(4)}


class Preparation(Closed):
    version: Literal["renderer_illumination_sampling_execution_v1"] = VERSION
    source_url: Literal["https://github.com/yurifrusin/ecological-predictive-states.git"] = (
        SOURCE_URL
    )
    source_head: Hex40
    source_tree: Hex40
    config_file: Hex64
    asset_root: Hex64
    protection_root: Hex64
    membership_root: Hex64
    scene_root: Hex64

    @model_validator(mode="after")
    def fixed_source(self) -> Preparation:
        if (
            self.config_file != sha256_bytes(CONFIG_BYTES)
            or self.membership_root
            != sha256_bytes(canonical_json_bytes([m.model_dump() for m in MEMBERS]))
            or self.scene_root != sha256_bytes(scene_xml()[0].encode())
            or self.asset_root
            != sha256_bytes(
                canonical_json_bytes(
                    [CONTRAST_V1.asset_hashes[0], CONTRAST_V1.asset_pixel_hashes[0]]
                )
            )
        ):
            raise ValueError("mixed discriminator source/config/assets/membership denied")
        return self


class Binding(Closed):
    preparation: Preparation
    purpose: Literal[
        "renderer_illumination_sampling_native_v1", "renderer_illumination_sampling_dummy_v1"
    ]
    image: Image
    output_id: Token
    token: Token
    policy: Literal["2g_2cpu_64pid_300s_64m_v1"] = "2g_2cpu_64pid_300s_64m_v1"

    @property
    def root(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.model_dump(mode="json")))


class Decision(Closed):
    binding_root: Hex64
    purpose: Literal[
        "renderer_illumination_sampling_native_v1", "renderer_illumination_sampling_dummy_v1"
    ]
    approved: Literal[True]
    authorization: str = Field(min_length=1, max_length=4096)

    @field_validator("approved", mode="before")
    @classmethod
    def strict_approval(cls, value: object) -> object:
        if value is not True:
            raise ValueError("approval must be strict boolean true")
        return value


def preparation(repository: Path, head: str, tree: str) -> Preparation:
    config((repository / CONFIG_PATH).read_bytes())
    return Preparation(
        source_head=head,
        source_tree=tree,
        config_file=sha256_bytes(CONFIG_BYTES),
        asset_root=sha256_bytes(
            canonical_json_bytes([CONTRAST_V1.asset_hashes[0], CONTRAST_V1.asset_pixel_hashes[0]])
        ),
        protection_root=sha256_bytes(canonical_json_bytes(protection(repository))),
        membership_root=sha256_bytes(canonical_json_bytes([m.model_dump() for m in MEMBERS])),
        scene_root=sha256_bytes(scene_xml()[0].encode()),
    )


def require_decision(binding: Binding, decision: Decision) -> None:
    if (
        type(binding) is not Binding
        or type(decision) is not Decision
        or decision.approved is not True
        or decision.binding_root != binding.root
        or decision.purpose != binding.purpose
        or not decision.authorization.strip()
    ):
        raise PermissionError("separate exact-source/image/purpose launch decision required")


def environment_binding() -> Binding:
    raw = os.environ.get(ENV, "").encode()
    if not raw or len(raw) > 16384:
        raise PermissionError("bounded discriminator binding required")
    return Binding.model_validate(p.strict_json(raw))


def candidate_runtime() -> bool:
    """Closed resource facts; no SDK access. Host launch authority is separately consumed."""
    try:
        binding = environment_binding()
        if (
            binding.purpose != NATIVE
            or platform.system() != "Linux"
            or not Path("/.dockerenv").is_file()
            or any(
                os.environ.get(k)
                for k in (
                    "EPS_A1_RUNTIME",
                    "EPS_CAUSAL_RUNTIME",
                    "WSL_INTEROP",
                    "WSL_DISTRO_NAME",
                    "EPS_PAIRED_APPEARANCE_RUNTIME",
                )
            )
            or any(
                os.environ.get(k) != v
                for k, v in {
                    "EPS_RENDERER_DISCRIMINATOR_RUNTIME": "docker_candidate_v1",
                    "EPS_RENDERER_DISCRIMINATOR_ROOT": binding.root,
                    "EPS_RENDERER_DISCRIMINATOR_IMAGE": binding.image,
                    "MUJOCO_GL": "osmesa",
                    "PYOPENGL_PLATFORM": "osmesa",
                    "LP_NUM_THREADS": "2",
                    "OMP_NUM_THREADS": "2",
                }.items()
            )
        ):
            return False
        affinity, filesystem = "sched_getaffinity", "statvfs"
        root = Path("/sys/fs/cgroup")
        cpu = (root / "cpu.max").read_text().split()
        return (
            (root / "memory.max").read_text().strip() == str(2 * 1024**3)
            and (root / "memory.swap.max").read_text().strip() == "0"
            and (root / "pids.max").read_text().strip() == "64"
            and len(cpu) == 2
            and int(cpu[1]) > 0
            and int(cpu[0]) == 2 * int(cpu[1])
            and sorted(getattr(os, affinity)(0)) == [0, 1]
            and all(
                getattr(os, filesystem)(path).f_blocks * getattr(os, filesystem)(path).f_frsize
                == size * p.MIB
                for path, size in (("/tmp", 8), ("/dev/shm", 1))
            )
        )
    except (ValueError, OSError, AttributeError, TypeError):
        return False


def require_native(repository: Path, binding: Binding) -> None:
    if type(binding) is not Binding or binding.purpose != NATIVE:
        raise PermissionError("discriminator native binding required before SDK access")
    if environment_binding() != binding or not candidate_runtime():
        raise PermissionError("unsupported discriminator runtime before SDK access")
    if (
        preparation(repository, binding.preparation.source_head, binding.preparation.source_tree)
        != binding.preparation
    ):
        raise PermissionError("changed source/config/assets/protections before SDK access")
    for filename, expected in (
        ("/preparation/renderer.json", binding.preparation.model_dump(mode="json")),
    ):
        raw = Path(filename).read_bytes()
        if len(raw) > 16384 or p.strict_json(raw) != expected:
            raise PermissionError("prepared immutable image manifest differs")
    anchor = Path("/output/consumed.json")
    if anchor.stat().st_size > 16384:
        raise PermissionError("consumed decision bound")
    consumed = p.strict_json(anchor.read_bytes())
    if (
        set(consumed) != {"binding", "decision", "binding_root"}
        or consumed["binding"] != binding.model_dump(mode="json")
        or consumed["binding_root"] != binding.root
    ):
        raise PermissionError("consumed decision source/image differs")
    require_decision(binding, Decision.model_validate(consumed["decision"]))


PARAMS = ("min", "mag", "base", "max")


class Sampler(Protocol):
    def bindings(self) -> tuple[int, int, int]: ...
    def restore_bindings(self, saved: tuple[int, int, int]) -> None: ...
    def query(self) -> dict[str, Any]: ...
    def set_parameters(self, index: int, values: tuple[int, int, int, int]) -> None: ...


def validate_sampler(state: dict[str, Any], setting: str) -> None:
    if set(state) != {
        "active_unit",
        "active_binding",
        "unit0_binding",
        "sampler_binding",
        "texture_matrix",
        "texture_env_mode",
        "textures",
    }:
        raise ValueError("closed sampler fields")
    if any(
        type(state[k]) is not int or state[k] < 0
        for k in ("active_unit", "active_binding", "unit0_binding")
    ):
        raise ValueError("actual texture bindings unavailable")
    if (
        state["sampler_binding"] != 0
        or type(state["sampler_binding"]) is not int
        or state["texture_env_mode"] != 8448
        or state["texture_matrix"] != np.eye(4, dtype=np.float32).reshape(-1).tolist()
    ):
        raise ValueError("actual unit0 sampler/environment/matrix differs")
    textures = state["textures"]
    if (
        type(textures) is not list
        or len(textures) != 4
        or len({v["object"] for v in textures}) != 4
    ):
        raise ValueError("complete owned texture inventory")
    expected = FIXED["sampling"][setting]
    for tex in textures:
        for key in (
            "index",
            "target",
            "object",
            "width",
            "height",
            "internal_format",
            "wrap_s",
            "wrap_t",
            *PARAMS,
        ):
            if type(tex.get(key)) is not int:
                raise ValueError("strict actual sampler integer fields")
    for index, tex in enumerate(textures):
        if (
            set(tex)
            != {
                "index",
                "target",
                "object",
                "width",
                "height",
                "internal_format",
                "colorspace",
                "wrap_s",
                "wrap_t",
                "pixels",
                "min_lod",
                "max_lod",
                "lod_bias",
                "compare_mode",
                "anisotropy",
                *PARAMS,
            }
            or tex["index"] != index
            or type(tex["object"]) is not int
            or tex["object"] <= 0
            or (
                tex["target"],
                tex["width"],
                tex["height"],
                tex["internal_format"],
                tex["colorspace"],
                tex["wrap_s"],
                tex["wrap_t"],
                tex["pixels"],
            )
            != (3553, 128, 128, 32849, "linear", 10497, 10497, CONTRAST_V1.asset_pixel_hashes[0])
            or (
                tex["min_lod"],
                tex["max_lod"],
                tex["lod_bias"],
                tex["compare_mode"],
                tex["anisotropy"],
            )
            != (-1000.0, 1000.0, 0.0, 0, 1.0)
            or [tex[k] for k in PARAMS] != expected
        ):
            raise ValueError("actual owned texture setting/content mismatch")


@contextmanager
def sampler_override(sampler: Sampler, setting: str, record: dict[str, Any]) -> Iterator[None]:
    """Restore every owned override and both unit bindings even after query/set/draw faults."""
    if setting not in ("default", "nearest"):
        raise ValueError("closed sampling factor")
    saved = sampler.bindings()
    originals: list[tuple[int, int, int, int]] = []
    try:
        baseline = sampler.query()
        validate_sampler(baseline, "default")
        record["original"] = baseline
        originals = [tuple(tex[k] for k in PARAMS) for tex in baseline["textures"]]
        for index in range(4):
            sampler.set_parameters(index, tuple(FIXED["sampling"][setting]))
        record["selected"] = sampler.query()
        validate_sampler(record["selected"], setting)
        yield
    finally:
        faults = []
        for index, values in enumerate(originals):
            try:
                sampler.set_parameters(index, values)
            except BaseException as exc:
                faults.append(type(exc).__name__)
        try:
            sampler.restore_bindings(saved)
            restored = sampler.query()
            validate_sampler(restored, "default")
            if sampler.bindings() != saved:
                raise ValueError("texture binding restoration mismatch")
            record["restored"] = restored
        except BaseException as exc:
            faults.append(type(exc).__name__)
        record["restoration"] = "RESTORED" if not faults else "FAILED_CLOSED"
        if faults:
            raise ValueError("sampler restoration failure: " + ",".join(faults))


class NativeSampler:
    """Context-local owned GL texture access, constructed only after admission."""

    def __init__(self, renderer: Any, repository: Path, binding: Binding):
        require_native(repository, binding)
        import importlib

        self.gl, self.renderer = importlib.import_module("OpenGL.GL"), renderer
        con = renderer._mjr_context
        if int(con.ntexture) != 4 or tuple(int(v) for v in con.textureType[:4]) != (0,) * 4:
            raise ValueError("exact owned 2D texture inventory")
        self.objects = tuple(int(v) for v in con.texture[:4])
        if len(set(self.objects)) != 4 or any(v <= 0 for v in self.objects):
            raise ValueError("owned texture objects unavailable")

    def bindings(self) -> tuple[int, int, int]:
        self.renderer._gl_context.make_current()
        gl = self.gl
        active = int(gl.glGetIntegerv(gl.GL_ACTIVE_TEXTURE))
        binding = int(gl.glGetIntegerv(gl.GL_TEXTURE_BINDING_2D))
        try:
            gl.glActiveTexture(gl.GL_TEXTURE0)
            zero = int(gl.glGetIntegerv(gl.GL_TEXTURE_BINDING_2D))
        finally:
            gl.glActiveTexture(active)
        return active, binding, zero

    def restore_bindings(self, saved: tuple[int, int, int]) -> None:
        gl = self.gl
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, saved[2])
        gl.glActiveTexture(saved[0])
        gl.glBindTexture(gl.GL_TEXTURE_2D, saved[1])

    def set_parameters(self, index: int, values: tuple[int, int, int, int]) -> None:
        if type(index) is not int or index not in range(4):
            raise ValueError("owned texture index required")
        saved = self.bindings()
        gl = self.gl
        try:
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.objects[index])
            for name, value in zip(
                (
                    gl.GL_TEXTURE_MIN_FILTER,
                    gl.GL_TEXTURE_MAG_FILTER,
                    gl.GL_TEXTURE_BASE_LEVEL,
                    gl.GL_TEXTURE_MAX_LEVEL,
                ),
                values,
                strict=True,
            ):
                gl.glTexParameteri(gl.GL_TEXTURE_2D, name, value)
        finally:
            self.restore_bindings(saved)

    def query(self) -> dict[str, Any]:
        saved = self.bindings()
        gl = self.gl
        result: dict[str, Any] = {
            "active_unit": saved[0],
            "active_binding": saved[1],
            "unit0_binding": saved[2],
            "textures": [],
        }
        try:
            gl.glActiveTexture(gl.GL_TEXTURE0)
            result["sampler_binding"] = int(gl.glGetIntegerv(gl.GL_SAMPLER_BINDING))
            result["texture_env_mode"] = int(
                gl.glGetTexEnviv(gl.GL_TEXTURE_ENV, gl.GL_TEXTURE_ENV_MODE)
            )
            result["texture_matrix"] = (
                np.asarray(gl.glGetFloatv(gl.GL_TEXTURE_MATRIX), dtype=np.float32)
                .reshape(-1)
                .tolist()
            )
            # A separate sampler object would supersede the texture parameters.
            if int(gl.glGetIntegerv(gl.GL_SAMPLER_BINDING)) != 0:
                raise ValueError("external sampler object denied")
            for index, obj in enumerate(self.objects):
                if not gl.glIsTexture(obj):
                    raise ValueError("owned texture disappeared")
                gl.glBindTexture(gl.GL_TEXTURE_2D, obj)
                texture = {"index": index, "target": 3553, "object": obj, "colorspace": "linear"}
                for key, name in zip(
                    (*PARAMS, "wrap_s", "wrap_t"),
                    (
                        gl.GL_TEXTURE_MIN_FILTER,
                        gl.GL_TEXTURE_MAG_FILTER,
                        gl.GL_TEXTURE_BASE_LEVEL,
                        gl.GL_TEXTURE_MAX_LEVEL,
                        gl.GL_TEXTURE_WRAP_S,
                        gl.GL_TEXTURE_WRAP_T,
                    ),
                    strict=True,
                ):
                    texture[key] = int(gl.glGetTexParameteriv(gl.GL_TEXTURE_2D, name))
                for key, name in (
                    ("min_lod", gl.GL_TEXTURE_MIN_LOD),
                    ("max_lod", gl.GL_TEXTURE_MAX_LOD),
                    ("lod_bias", gl.GL_TEXTURE_LOD_BIAS),
                    ("anisotropy", 34046),
                ):
                    texture[key] = float(gl.glGetTexParameterfv(gl.GL_TEXTURE_2D, name))
                texture["compare_mode"] = int(
                    gl.glGetTexParameteriv(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_COMPARE_MODE)
                )
                for key, name in (
                    ("width", gl.GL_TEXTURE_WIDTH),
                    ("height", gl.GL_TEXTURE_HEIGHT),
                    ("internal_format", gl.GL_TEXTURE_INTERNAL_FORMAT),
                ):
                    texture[key] = int(gl.glGetTexLevelParameteriv(gl.GL_TEXTURE_2D, 0, name))
                if (texture["width"], texture["height"]) != (128, 128):
                    raise ValueError("owned base-level dimensions differ")
                # Width384 bytes is divisible by every supported pack alignment. No pack mutation.
                if any(
                    int(gl.glGetIntegerv(name)) != value
                    for name, value in (
                        (gl.GL_PACK_ROW_LENGTH, 0),
                        (gl.GL_PACK_SKIP_ROWS, 0),
                        (gl.GL_PACK_SKIP_PIXELS, 0),
                        (gl.GL_PIXEL_PACK_BUFFER_BINDING, 0),
                    )
                ):
                    raise ValueError("texture read requires qualified pack state")
                if int(gl.glGetIntegerv(gl.GL_PACK_ALIGNMENT)) not in (1, 2, 4, 8):
                    raise ValueError("unsupported texture read alignment")
                pixels = np.empty((128, 128, 3), dtype=np.uint8)
                gl.glGetTexImage(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, pixels)
                texture["pixels"] = logical_array_hash(pixels)
                result["textures"].append(texture)
        finally:
            self.restore_bindings(saved)
        if self.bindings() != saved:
            raise ValueError("sampler query binding drift")
        return result


def expected_material(identity: Identity, compiled: dict[str, Any]) -> dict[str, Any]:
    # The four brick materials remain present for solids in the one compiled model.
    result = p.expected_material("corridor", CONTRAST_V1.appearances[1], compiled, CONTRAST_V1)
    model = result["model"]
    for key in (
        "mat_rgba",
        "mat_texrepeat",
        "mat_texuniform",
        "mat_emission",
        "mat_specular",
        "mat_shininess",
        "mat_reflectance",
    ):
        model[key] += copy.deepcopy(model[key])
    model["mat_texid"] += [[-1] * 10 for _ in range(4)]
    if identity.appearance == APPEARANCES[0]:
        model["geom_rgba"] = [p.float32_values([224 / 255] * 3 + [1.0]) for _ in range(4)]
        model["geom_matid"] = [4 + i for i in model["geom_matid"]]
        for geom in result["scene_geoms"]:
            geom["matid"] += 4
            geom["texid"] = -1
            geom["rgba"] = p.float32_values([224 / 255] * 3 + [1.0])
    ambient, diffuse = FIXED["illumination"][
        "unit" if identity.arm.startswith("unit") else "original"
    ]
    model["light_ambient"] = [p.float32_values(ambient)]
    model["light_diffuse"] = [p.float32_values(diffuse)]
    result["scene_lights"][0]["ambient"] = p.float32_values(ambient)
    result["scene_lights"][0]["diffuse"] = p.float32_values(diffuse)
    return result


@dataclass(frozen=True)
class Endpoint:
    identity: Identity
    arrays: dict[str, p.Array]
    evidence: dict[str, Any]

    def __post_init__(self) -> None:
        if type(self.identity) is not Identity or self.identity not in MEMBERS:
            raise ValueError("exact endpoint membership")
        if set(self.arrays) != set(SPECS):
            raise ValueError("closed endpoint array inventory")
        copied = {}
        for name, (dtype, shape) in SPECS.items():
            array = self.arrays[name]
            if (
                type(array) is not np.ndarray
                or array.dtype != np.dtype(dtype)
                or array.shape != shape
            ):
                raise ValueError("endpoint shape/dtype: " + name)
            copied[name] = p.owned(array)
        object.__setattr__(self, "arrays", copied)
        object.__setattr__(self, "evidence", p.strict_json(canonical_json_bytes(self.evidence)))


def validate_endpoint(endpoint: Endpoint, binding: Binding) -> None:
    if type(endpoint) is not Endpoint or type(binding) is not Binding:
        raise ValueError("typed discriminator endpoint/binding required")
    if set(endpoint.arrays) != set(SPECS):
        raise ValueError("closed current array inventory")
    for name, (dtype, shape) in SPECS.items():
        array = endpoint.arrays[name]
        if type(array) is not np.ndarray or array.dtype != np.dtype(dtype) or array.shape != shape:
            raise ValueError("current endpoint shape/dtype")
    e = endpoint.evidence
    required = {
        "schema",
        "source_head",
        "source_tree",
        "binding_root",
        "config_sha256",
        "scene_xml_sha256",
        "compiled",
        "camera",
        "mapping",
        "scene_map",
        "near",
        "far",
        "orientation",
        "runtime",
        "rgb_stable",
        "paired_stable",
        "rgb_material",
        "paired_material",
        "rgb_provenance",
        "sampler",
        "model_restoration",
    }
    if (
        set(e) != required
        or e["schema"] != SCHEMA + ":endpoint"
        or e["source_head"] != binding.preparation.source_head
        or e["source_tree"] != binding.preparation.source_tree
        or e["binding_root"] != binding.root
        or e["config_sha256"] != binding.preparation.config_file
        or e["scene_xml_sha256"] != binding.preparation.scene_root
        or e["orientation"] != p.ORIENTATION
    ):
        raise ValueError("closed endpoint provenance/config/source binding")
    restoration = e["model_restoration"]
    keys = ("geom_matid", "geom_rgba", "light_ambient", "light_diffuse")
    baseline = expected_material(MEMBERS[2], e["compiled"])["model"]
    expected_restore = {key: baseline[key] for key in keys}
    if restoration != {"before": expected_restore, "after": expected_restore}:
        raise ValueError("fixed owned model baseline/restoration differs")
    p.validate_registration(e)
    p.validate_compiled("corridor", e["compiled"])
    camera = {
        "world_position": [0.0, 1.2, 1.25],
        "rotation_row_major": [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0],
        "fovy": 55.0,
    }
    if (
        e["camera"] != camera
        or e["compiled"]["camera_world_position"] != camera["world_position"]
        or e["compiled"]["camera_world_rotation_row_major"] != camera["rotation_row_major"]
    ):
        raise ValueError("fixed second corridor pose; no motion")
    state = e["paired_stable"]
    if state["scene_geometry"] != p.expected_draw_geometry(e["compiled"]) or state[
        "scene_cameras"
    ] != p.expected_scene_cameras(camera, e["near"], e["far"]):
        raise ValueError("fixed actual draw geometry/camera")
    if (e["near"], e["far"]) != (float(np.float32(0.01)) * state["extent"], 30.0 * state["extent"]):
        raise ValueError("fixed clipping")
    if canonical_json_bytes(e["paired_material"]) != canonical_json_bytes(
        expected_material(endpoint.identity, e["compiled"])
    ):
        raise ValueError("exact actual material/light/asset intervention")
    if e["rgb_provenance"] != {
        "producer": "mujoco.Renderer.render",
        "sdk_renderer_sha256": p.SDK_RENDERER_SHA256,
        "operation": "separate_ordinary_rgb_draw_before_owned_id_depth",
        "same_draw_as_pair": False,
    }:
        raise ValueError("ordinary RGB producer identity")
    sampler = e["sampler"]
    if (
        set(sampler)
        != {
            "original",
            "selected",
            "rgb_before",
            "rgb_after",
            "pair_before",
            "pair_after",
            "restored",
            "restoration",
        }
        or sampler["restoration"] != "RESTORED"
    ):
        raise ValueError("complete actual sampler/restoration evidence")
    setting = "nearest" if endpoint.identity.arm.endswith("nearest") else "default"
    objects = None
    for key in (
        "original",
        "selected",
        "rgb_before",
        "rgb_after",
        "pair_before",
        "pair_after",
        "restored",
    ):
        validate_sampler(sampler[key], "default" if key in ("original", "restored") else setting)
        current = [tex["object"] for tex in sampler[key]["textures"]]
        if objects is not None and objects != current:
            raise ValueError("owned texture identity drift")
        objects = current
    if any(
        sampler["original"][k] != sampler["restored"][k]
        for k in ("active_unit", "active_binding", "unit0_binding")
    ):
        raise ValueError("retained actual sampler bindings not restored")
    raw_ids = tuple(e["compiled"]["raw_geom_ids"][name] for name in p.SURFACES["corridor"])
    mapping = remapping(raw_ids)
    if e["mapping"] != [list(v) for v in mapping]:
        raise ValueError("independent episode opaque mapping")
    arrays = endpoint.arrays
    raw, depth = p.derive_pair(
        arrays["native_id"],
        arrays["native_depth"],
        tuple(tuple(v) for v in e["scene_map"]),
        e["near"],
        e["far"],
    )
    opaque = np.zeros_like(raw)
    for raw_id, label, _ in mapping:
        opaque[raw == raw_id] = label
    derived = {
        "raw": raw,
        "depth": depth,
        "opaque": opaque,
        "controlled": opaque > 0,
        "horizontal": opaque[:, :-1] != opaque[:, 1:],
        "vertical": opaque[:-1] != opaque[1:],
    }
    if set(int(v) for v in np.unique(raw)) - {-1} - set(raw_ids) or any(
        not np.array_equal(arrays[k], v) for k, v in derived.items()
    ):
        raise ValueError("retained native reconstruction/mask/lattice")
    if len(canonical_json_bytes(e)) > 49152:
        raise ValueError("endpoint scientific metadata cap")


def ideal_gain(normal: tuple[int, int, int], unit: bool) -> float:
    incoming = np.asarray([0.0, -0.25, 1.0]) / math.sqrt(1.0625)
    return 1.0 if unit else 0.1 + 0.7 * max(0.0, float(np.dot(normal, incoming)))


def box_uv(local: tuple[float, float, float], axis: int) -> tuple[float, float]:
    x, y, z = local
    if axis == 2:
        return (x + 1) / 2, (1 - y) / 2
    if axis == 0:
        return (y + 1) / 2, (1 - z) / 2
    if axis == 1:
        return (x + 1) / 2, (1 - z) / 2
    raise ValueError("box face axis")


def nearest_texel(uv: tuple[float, float]) -> tuple[int, int] | None:
    """Exact mathematical boundary ties only; no native-precision margin is claimed."""
    scaled = np.asarray(uv) * 128
    if not np.isfinite(scaled).all():
        raise ValueError("finite source UV")
    if any(float(v).is_integer() for v in scaled):
        return None
    return math.floor(scaled[1]) % 128, math.floor(scaled[0]) % 128


def ideal_rays() -> dict[str, p.Array]:
    """All19200 fixed pixel-centre rays. Eligibility has no native precision claim."""
    owner = np.full((120, 160), -1, dtype=np.int32)
    texel = np.full((120, 160), -1, dtype=np.int32)
    gain = np.zeros((120, 160), dtype=np.float64)
    category = np.zeros((120, 160), dtype=np.uint8)
    origin = np.asarray([0.0, 1.2, 1.25])
    scale = 2 * math.tan(math.radians(55) / 2) / 120
    boxes = list(p.expected_geometry("corridor").values())
    pixels = p.brick(0, CONTRAST_V1)
    for row in range(120):
        for column in range(160):
            ray = np.asarray([(column + 0.5 - 80) * scale, 1.0, -(row + 0.5 - 60) * scale])
            hits = []
            for index, (position, size, _) in enumerate(boxes):
                centre, half = np.asarray(position), np.asarray(size)
                for axis in range(3):
                    if ray[axis] == 0:
                        continue
                    for sign in (-1, 1):
                        distance = (centre[axis] + sign * half[axis] - origin[axis]) / ray[axis]
                        if distance <= 0:
                            continue
                        local = (origin + distance * ray - centre) / half
                        # This coordinate was solved algebraically, not estimated by reconstruction.
                        local[axis] = sign
                        if np.all(np.abs(local) <= 1):
                            hits.append((distance, index, axis, sign, local))
            if not hits:
                continue
            hits.sort(key=lambda hit: hit[0])
            distance, index, axis, sign, local = hits[0]
            owner[row, column] = index
            if len(hits) > 1 and hits[1][0] == distance:
                category[row, column] = 2
                continue
            uv = box_uv((float(local[0]), float(local[1]), float(local[2])), axis)
            tex = nearest_texel(uv)
            if tex is None:
                category[row, column] = 2
                continue
            normal = cast(tuple[int, int, int], tuple(sign if a == axis else 0 for a in range(3)))
            gain[row, column] = ideal_gain(normal, False)
            texel[row, column] = int(pixels[tex][0])
            category[row, column] = 1
    return {"owner": owner, "texel": texel, "gain": gain, "category": category}


def residual(values: p.Array) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite diagnostic residual")
    return {
        "denominator": int(values.size),
        "mean": float(values.mean()) if values.size else None,
        "mean_absolute": float(np.abs(values).mean()) if values.size else None,
        "minimum": float(values.min()) if values.size else None,
        "maximum": float(values.max()) if values.size else None,
        "rms": float(np.sqrt(np.mean(values**2))) if values.size else None,
    }


def suitability(solid: Endpoint, brick: Endpoint) -> dict[str, Any]:
    mask = solid.arrays["controlled"]
    count = int(mask.sum())
    difference = (
        np.abs(solid.arrays["rgb"].astype(np.float64) - brick.arrays["rgb"].astype(np.float64))
        / 255
    )
    changed = float(np.any(difference > 0, axis=2)[mask].mean()) if count else 0.0
    mad = float(difference[mask].mean()) if count else 0.0
    luminance = (brick.arrays["rgb"].astype(np.float64) / 255) @ np.asarray(
        [0.2126, 0.7152, 0.0722]
    )
    surfaces = []
    for _, label, token in solid.evidence["mapping"]:
        interior = p.interior(mask & (solid.arrays["opaque"] == label))
        n = int(interior.sum())
        mean, std = (
            (float(luminance[interior].mean()), float(luminance[interior].std()))
            if n
            else (0.0, 0.0)
        )
        h, v = interior[:, :-1] & interior[:, 1:], interior[:-1] & interior[1:]
        gradients = np.concatenate(
            (np.abs(np.diff(luminance, axis=1))[h], np.abs(np.diff(luminance, axis=0))[v])
        )
        pairs = int(gradients.size)
        contrast = int((gradients >= 0.05).sum())
        fraction = contrast / pairs if pairs else 0.0
        suitable = (
            n >= 100 and 0.05 <= mean <= 0.95 and std >= 0.025 and pairs > 0 and fraction >= 0.02
        )
        surfaces.append(
            {
                "surface": token,
                "interior_pixels": n,
                "mean": mean,
                "std": std,
                "pairs": pairs,
                "contrast_pairs": contrast,
                "contrast_fraction": fraction,
                "pass": suitable,
                "normalized_gradients": residual(gradients / mean)
                if mean
                else residual(np.empty(0)),
                "normalization_denominator": mean,
                "clipped_pixels": int(
                    np.any((brick.arrays["rgb"] == 0) | (brick.arrays["rgb"] == 255), axis=2)[
                        interior
                    ].sum()
                ),
            }
        )
    return {
        "status": "PASS"
        if count and changed >= 0.2 and mad >= 0.025 and all(v["pass"] for v in surfaces)
        else "FAIL",
        "controlled_pixels": count,
        "changed_fraction": changed,
        "normalized_mad": mad,
        "surfaces": surfaces,
    }


def validate_relations(endpoints: list[Endpoint]) -> None:
    """Only acquisition validity; never suitability, direct controls or ideal residuals."""
    if not endpoints or tuple(e.identity for e in endpoints) != MEMBERS[: len(endpoints)]:
        raise ValueError("ordered prefix membership required")
    invariant_names = tuple(n for n in SPECS if n != "rgb")
    first = endpoints[0]
    for endpoint in endpoints[1:]:
        if any(not np.array_equal(first.arrays[n], endpoint.arrays[n]) for n in invariant_names):
            raise ValueError("cross-arm paired ID/depth/mask/lattice differs")
        if any(
            first.evidence[k] != endpoint.evidence[k]
            for k in ("compiled", "camera", "runtime", "mapping", "scene_map", "near", "far")
        ):
            raise ValueError("cross-arm provenance/geometry differs")
        if any(
            first.evidence["paired_stable"][k] != endpoint.evidence["paired_stable"][k]
            for k in (*p.REGISTRATION_KEYS, "scene_flags")
        ):
            raise ValueError("cross-arm registered state differs")
    for offset in range(0, len(endpoints) - 1, 2):
        a, b = endpoints[offset : offset + 2]
        if a.evidence != b.evidence or any(
            not np.array_equal(a.arrays[n], b.arrays[n]) for n in SPECS
        ):
            raise ValueError("exact repeated-capture relation differs")


def interpret_suitability(passes: list[str]) -> str:
    if passes != [arm for arm in ARMS if arm in passes]:
        raise ValueError("ordered unique observed passing arms required")
    if not passes:
        return "no tested arm meets all fixed texture criteria"
    observed = "fixed texture criteria met by: " + ", ".join(passes)
    if passes == ["unit_nearest"]:
        return (
            observed
            + "; joint dependence under this fixed criterion; no physical interaction proof"
        )
    return (
        observed + "; observed suitability does not isolate causes or grant readiness; "
        "absolute transfer remains unresolved"
    )


def assess(endpoints: list[Endpoint], binding: Binding) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "INCONCLUSIVE",
        "endpoints": len(endpoints),
        "planned": 16,
        "absolute_native_uv_transfer_qualification": "UNRESOLVED",
    }
    try:
        if len(endpoints) != 16 or tuple(e.identity for e in endpoints) != MEMBERS:
            raise ValueError("complete ordered membership required")
        for endpoint in endpoints:
            validate_endpoint(endpoint, binding)
        validate_relations(endpoints)
        arms = {
            arm: suitability(endpoints[i * 4], endpoints[i * 4 + 2]) for i, arm in enumerate(ARMS)
        }
        controls = []
        comparisons = []
        for left, right, factor in (
            (0, 1, "illumination_default"),
            (2, 3, "illumination_nearest"),
            (0, 2, "sampling_original"),
            (1, 3, "sampling_unit"),
        ):
            a, b = endpoints[left * 4], endpoints[right * 4]
            equal = np.array_equal(a.arrays["rgb"], b.arrays["rgb"])
            if factor.startswith("sampling"):
                controls.append(
                    {
                        "relation": factor,
                        "status": "CONTROL_SUPPORTED" if equal else "CONTROL_NOT_SUPPORTED",
                        "whole_mask_rgb_residual": residual(
                            (b.arrays["rgb"].astype(float) - a.arrays["rgb"].astype(float))[
                                a.arrays["controlled"]
                            ]
                        ),
                    }
                )
            surfaces = []
            for sa, sb in zip(
                arms[ARMS[left]]["surfaces"], arms[ARMS[right]]["surfaces"], strict=True
            ):
                surfaces.append(
                    {
                        "surface": sa["surface"],
                        **{k: sb[k] - sa[k] for k in ("mean", "std", "contrast_fraction")},
                        "from_pairs": sa["pairs"],
                        "to_pairs": sb["pairs"],
                    }
                )
            comparisons.append(
                {
                    "factor": factor,
                    "from": ARMS[left],
                    "to": ARMS[right],
                    "surface_score_differences": surfaces,
                }
            )
        ideal = ideal_rays()
        diagnostics = []
        for endpoint in endpoints[::2]:
            i = MEMBERS.index(endpoint.identity)
            unit = endpoint.identity.arm.startswith("unit")
            solid = endpoint.identity.appearance == APPEARANCES[0]
            surfaces = []
            for index, (_, label, token) in enumerate(endpoint.evidence["mapping"]):
                observed = endpoint.arrays["opaque"] == label
                mathematical = ideal["owner"] == index
                eligible = observed & mathematical & (ideal["category"] == 1)
                predicted = (np.full((120, 160), 224) if solid else ideal["texel"]) * (
                    1 if unit else ideal["gain"]
                )
                values = endpoint.arrays["rgb"].astype(float) - predicted[..., None]
                surfaces.append(
                    {
                        "surface": token,
                        "whole_mask_pixels": int(observed.sum()),
                        "mathematical_pixels": int(mathematical.sum()),
                        "eligible": int(eligible.sum()),
                        "tie": int((mathematical & (ideal["category"] == 2)).sum()),
                        "rejected": int((observed & ~eligible).sum()),
                        "mask_disagreement": int((observed ^ mathematical).sum()),
                        "nearest_ideal_residual": residual(values[eligible])
                        if solid or endpoint.identity.arm.endswith("nearest")
                        else None,
                    }
                )
            diagnostics.append({"member": i, "qualification": "UNRESOLVED", "surfaces": surfaces})
        transfer = []
        for original, unit_arm in ((0, 1), (2, 3)):
            for appearance_index in (0, 2):
                a, b = (
                    endpoints[original * 4 + appearance_index],
                    endpoints[unit_arm * 4 + appearance_index],
                )
                for index, (_, label, token) in enumerate(a.evidence["mapping"]):
                    mask = a.arrays["opaque"] == label
                    normal = ((0, 0, 1), (1, 0, 0), (-1, 0, 0), (0, -1, 0))[index]
                    gain = ideal_gain(normal, False)
                    values = a.arrays["rgb"].astype(float) - gain * b.arrays["rgb"].astype(float)
                    transfer.append(
                        {
                            "sampling": "default" if original == 0 else "nearest",
                            "appearance": a.identity.appearance,
                            "surface": token,
                            "ideal_gain": gain,
                            "qualification": "UNRESOLVED",
                            "whole_mask_rgb_residual": residual(values[mask]),
                        }
                    )
        supported = all(v["status"] == "CONTROL_SUPPORTED" for v in controls)
        passes = [arm for arm in ARMS if arms[arm]["status"] == "PASS"]
        interpretation = interpret_suitability(passes)
        result.update(
            status="DIAGNOSTIC_COMPLETE",
            arms=arms,
            solid_sampling_controls=controls,
            sampling_only_explanation="SUPPORTED_CONTROL"
            if supported
            else "UNQUALIFIED_CONTROL_NOT_SUPPORTED",
            empirical_factor_comparisons=comparisons,
            ideal_diagnostics=diagnostics,
            mathematical_rays={
                "total": 19200,
                "eligible": int((ideal["category"] == 1).sum()),
                "tie": int((ideal["category"] == 2).sum()),
                "rejected": int((ideal["category"] == 0).sum()),
            },
            illumination_transfer=transfer,
            interpretation=interpretation,
            limitations=[
                "one exposed development pose",
                "nearest may alias",
                "no radiometric/UV precision qualification",
                "no shader-cause, sequence, model or gate admission",
            ],
        )
    except (ValueError, KeyError, TypeError) as exc:
        result["reason"] = str(exc)
    if len(canonical_json_bytes(result)) > p.MIB:
        raise ValueError("bounded report capacity")
    return result


STAGES = (
    "endpoint_attempt",
    "rgb_attempt",
    "rgb_read_complete",
    "rgb_state_complete",
    "paired_draw_input",
    "paired_draw_attempt",
    "paired_draw_complete",
    "paired_draw_output",
    "paired_read_input",
    "paired_read_attempt",
    "paired_read_complete",
    "paired_read_output",
    "sampler_complete",
    "endpoint_complete",
)


METADATA_STAGES = (
    "rgb_state_complete",
    "paired_draw_input",
    "paired_draw_output",
    "paired_read_input",
    "paired_read_output",
    "sampler_complete",
)
COMPONENT_CAPS = {
    **{
        name + ".bin": int(np.prod(shape)) * np.dtype(dtype).itemsize
        for name, (dtype, shape) in SPECS.items()
    },
    "read_id.bin": 57600,
    "read_depth.bin": 76800,
    **{
        stage + ".json": 16384 if stage == "sampler_complete" else 65536
        for stage in METADATA_STAGES
    },
    "sampler_failure.json": 16384,
    "frame.json": 65536,
}


def component_cap(name: str) -> int:
    if type(name) is not str:
        raise ValueError("closed physical namespace")
    shared = {"consumed.json": 16384, "report.json": p.MIB, "terminal.json": 16384}
    if name in shared:
        return shared[name]
    if re.fullmatch(r"events/[0-9]{4}\.json", name):
        if int(name[7:11]) < 16 * len(STAGES):
            return 4096
    match = re.fullmatch(r"endpoints/e([0-9]{2})/([a-z_]+\.(?:bin|json))", name)
    if match and int(match[1]) < 16 and match[2] in COMPONENT_CAPS:
        return COMPONENT_CAPS[match[2]]
    raise ValueError("closed physical namespace/ordinal")


def stage_paths(stage: str, ordinal: int) -> list[str]:
    prefix = f"endpoints/e{ordinal:02d}/"
    if stage in METADATA_STAGES:
        return [prefix + stage + ".json"]
    if stage == "rgb_read_complete":
        return [prefix + "rgb.bin"]
    if stage == "paired_read_complete":
        return [prefix + "read_id.bin", prefix + "read_depth.bin"]
    if stage == "endpoint_complete":
        return [*(prefix + n + ".bin" for n in SPECS if n != "rgb"), prefix + "frame.json"]
    return []


def validate_ref(ref: Any, expected_path: str) -> None:
    if (
        type(ref) is not dict
        or set(ref) != {"path", "bytes", "sha256"}
        or type(ref["path"]) is not str
        or ref["path"] != expected_path
        or type(ref["bytes"]) is not int
        or not 0 <= ref["bytes"] <= component_cap(expected_path)
        or type(ref["sha256"]) is not str
        or re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]) is None
    ):
        raise ValueError("closed typed component reference")


def physical_inventory(root: Path) -> dict[str, int]:
    files: dict[str, int] = {}
    endpoints, shared = [0] * 16, 0
    for path in root.rglob("*"):
        safe_path(path)
        name = path.relative_to(root).as_posix()
        if path.is_dir():
            if (
                name not in ("events", "endpoints")
                and re.fullmatch(r"endpoints/e(?:0[0-9]|1[0-5])", name) is None
            ):
                raise ValueError("closed physical directory namespace")
            continue
        if not path.is_file():
            raise ValueError("non-file physical evidence")
        size = path.stat().st_size
        if size > component_cap(name):
            raise ValueError("physical component cap")
        files[name] = size
        if name.startswith("endpoints/"):
            endpoints[int(name.split("/")[1][1:])] += size
        else:
            shared += size
    if any(n > p.MIB for n in endpoints) or shared > 2 * p.MIB or sum(files.values()) > 18 * p.MIB:
        raise ValueError("physical endpoint/shared/original cap")
    return files


class Sink:
    """Immediate exclusive component retention in one consumed, bounded owned namespace."""

    def __init__(self, root: Path, binding: Binding, decision: Decision):
        require_decision(binding, decision)
        safe_path(root)
        if binding.purpose == NATIVE:
            if (
                root != Path("/output")
                or not candidate_runtime()
                or environment_binding() != binding
            ):
                raise PermissionError("native sink requires exact future owned-container runtime")
        if root.exists():
            if binding.purpose != NATIVE or not root.is_dir() or any(root.iterdir()):
                raise ValueError("owned attempt collision; no resume/replacement")
        else:
            root.mkdir()
        self.root, self.binding = root, binding
        self.started = time.monotonic()
        self.used, self.shared, self.ordinal, self.stage = [0] * 16, 0, 0, 0
        self.sequence, self.failed = 0, False
        self.native_claimed = False
        self.counts = {s: 0 for s in STAGES if s.endswith("attempt") or s.endswith("complete")}
        self.put(
            "consumed.json",
            canonical_json_bytes(
                {
                    "binding": binding.model_dump(mode="json"),
                    "decision": decision.model_dump(mode="json"),
                    "binding_root": binding.root,
                }
            ),
        )

    def put(self, name: str, payload: bytes) -> dict[str, Any]:
        if self.failed:
            raise OSError("poisoned sink; no writes")
        try:
            if type(payload) is not bytes or len(payload) > component_cap(name):
                raise ValueError("component cap")
            if name.startswith("endpoints/"):
                ordinal = int(name.split("/")[1][1:])
                if ordinal not in range(16) or self.used[ordinal] + len(payload) > p.MIB:
                    raise ValueError("endpoint cap")
                self.used[ordinal] += len(payload)
            else:
                cap = 2 * p.MIB if name == "terminal.json" else 2 * p.MIB - 16384
                if self.shared + len(payload) > cap or (
                    name == "report.json" and len(payload) > p.MIB
                ):
                    raise ValueError("shared/report cap")
                self.shared += len(payload)
            path = self.root / name
            safe_path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            exclusive_file(path, payload)
            return {"path": name, "bytes": len(payload), "sha256": sha256_bytes(payload)}
        except BaseException:
            self.failed = True
            raise

    def progress(self, stage: str, ordinal: int, value: Any) -> None:
        if (
            self.failed
            or type(ordinal) is not int
            or ordinal != self.ordinal
            or ordinal not in range(16)
            or stage != STAGES[self.stage]
        ):
            raise ValueError("closed immediate callback order")
        refs = []
        prefix = f"endpoints/e{ordinal:02d}/"
        if stage in self.counts:
            self.counts[stage] += 1
        if stage in (
            "rgb_state_complete",
            "paired_draw_input",
            "paired_draw_output",
            "paired_read_input",
            "paired_read_output",
            "sampler_complete",
        ):
            if type(value) is not bytes or len(value) > (
                16384 if stage == "sampler_complete" else 65536
            ):
                raise ValueError("immediate metadata cap")
            p.strict_json(value)
            refs.append(self.put(prefix + stage + ".json", value))
        elif stage == "rgb_read_complete":
            if (
                type(value) is not np.ndarray
                or value.shape != (120, 160, 3)
                or value.dtype != np.uint8
            ):
                raise ValueError("RGB read shape")
            refs.append(self.put(prefix + "rgb.bin", value.tobytes()))
        elif stage == "paired_read_complete":
            if (
                type(value) is not tuple
                or tuple(map(len, value)) != (57600, 76800)
                or any(type(v) is not bytes for v in value)
            ):
                raise ValueError("paired read lengths")
            for name, payload in zip(("read_id", "read_depth"), value, strict=True):
                refs.append(self.put(prefix + name + ".bin", payload))
        elif stage == "endpoint_complete":
            if type(value) is not Endpoint or value.identity != MEMBERS[ordinal]:
                raise ValueError("endpoint membership")
            validate_endpoint(value, self.binding)
            arrays = {}
            for name, array in value.arrays.items():
                payload = array.tobytes()
                arrays[name] = (
                    {
                        "path": prefix + name + ".bin",
                        "bytes": len(payload),
                        "sha256": sha256_bytes(payload),
                    }
                    if name == "rgb"
                    else self.put(prefix + name + ".bin", payload)
                )
            refs.extend(value for name, value in arrays.items() if name != "rgb")
            refs.append(
                self.put(
                    prefix + "frame.json",
                    canonical_json_bytes(
                        {
                            "identity": value.identity.model_dump(),
                            "evidence": value.evidence,
                            "arrays": arrays,
                        }
                    ),
                )
            )
        elif value is not None:
            raise ValueError("payload-free attempt stage")
        event = canonical_json_bytes(
            {
                "sequence": self.sequence,
                "ordinal": ordinal,
                "stage": stage,
                "binding_root": self.binding.root,
                "refs": refs,
            }
        )
        if len(event) > 4096:
            raise ValueError("event cap")
        self.put(f"events/{self.sequence:04d}.json", event)
        self.sequence += 1
        self.stage += 1
        if stage == "endpoint_complete":
            self.ordinal += 1
            self.stage = 0


def replay(root: Path, binding: Binding) -> list[Endpoint]:
    """Strict prefix replay and immediate producer reconstruction; never resumes a run."""
    safe_path(root)
    physical = physical_inventory(root)
    consumed_raw = (root / "consumed.json").read_bytes()
    if len(consumed_raw) > 16384:
        raise ValueError("consumed cap")
    anchor = p.strict_json(consumed_raw)
    if (
        set(anchor) != {"binding", "decision", "binding_root"}
        or anchor["binding"] != binding.model_dump(mode="json")
        or anchor["binding_root"] != binding.root
    ):
        raise ValueError("replay consumed binding")
    require_decision(binding, Decision.model_validate(anchor["decision"]))
    completed = []
    ordinal, stage = 0, 0
    seen: set[str] = {"consumed.json"}
    for sequence, path in enumerate(sorted((root / "events").glob("*.json"))):
        if path.name != f"{sequence:04d}.json" or path.stat().st_size > 4096:
            raise ValueError("bounded contiguous event log")
        event = p.strict_json(path.read_bytes())
        if (
            type(event) is not dict
            or set(event) != {"sequence", "ordinal", "stage", "binding_root", "refs"}
            or type(event["sequence"]) is not int
            or type(event["ordinal"]) is not int
            or type(event["stage"]) is not str
            or event["stage"] not in STAGES
            or type(event["binding_root"]) is not str
            or type(event["refs"]) is not list
            or ordinal >= 16
            or (event["sequence"], event["ordinal"], event["stage"], event["binding_root"])
            != (sequence, ordinal, STAGES[stage], binding.root)
        ):
            raise ValueError("ordered typed replay events")
        current_stage = event["stage"]
        expected_paths = stage_paths(current_stage, ordinal)
        if len(event["refs"]) != len(expected_paths):
            raise ValueError("closed event component inventory")
        for ref, expected_path in zip(event["refs"], expected_paths, strict=True):
            validate_ref(ref, expected_path)
            if expected_path.endswith(".bin") and ref["bytes"] != component_cap(expected_path):
                raise ValueError("completed binary read length")
        payloads = {}
        for ref in event["refs"]:
            name = ref["path"]
            if name in seen:
                raise ValueError("duplicate immediate component ref")
            file = root / name
            safe_path(file)
            if file.stat().st_size != ref["bytes"]:
                raise ValueError("component length")
            raw = file.read_bytes()
            if sha256_bytes(raw) != ref["sha256"]:
                raise ValueError("component corruption")
            payloads[name.split("/")[-1]] = raw
            seen.add(name)
        seen.add("events/" + path.name)
        if event["stage"] == "endpoint_complete":
            meta = p.strict_json(payloads["frame.json"])
            if (
                set(meta) != {"identity", "evidence", "arrays"}
                or Identity.model_validate(meta["identity"]) != MEMBERS[ordinal]
                or set(meta["arrays"]) != set(SPECS)
            ):
                raise ValueError("retained endpoint schema")
            arrays = {}
            for name, (dtype, shape) in SPECS.items():
                ref = meta["arrays"][name]
                expected_path = f"endpoints/e{ordinal:02d}/{name}.bin"
                validate_ref(ref, expected_path)
                if (
                    ref["path"] != expected_path
                    or ref["bytes"] != np.prod(shape) * np.dtype(dtype).itemsize
                    or expected_path not in seen
                ):
                    raise ValueError("retained array identity/length")
                raw = (root / expected_path).read_bytes()
                if sha256_bytes(raw) != ref["sha256"]:
                    raise ValueError("retained array hash")
                arrays[name] = np.frombuffer(raw, dtype=dtype).reshape(shape)
            endpoint = Endpoint(MEMBERS[ordinal], arrays, meta["evidence"])
            validate_endpoint(endpoint, binding)
            for name, transformed in (
                ("id", endpoint.arrays["native_id"]),
                ("depth", endpoint.arrays["native_depth"]),
            ):
                if (
                    np.flipud(transformed).tobytes()
                    != (root / f"endpoints/e{ordinal:02d}/read_{name}.bin").read_bytes()
                ):
                    raise ValueError(
                        "immediate native raw read differs from retained transformed arrays"
                    )
            if (
                p.strict_json(
                    (root / f"endpoints/e{ordinal:02d}/sampler_complete.json").read_bytes()
                )
                != endpoint.evidence["sampler"]
            ):
                raise ValueError("immediate sampler evidence differs")
            rgb_meta = p.strict_json(
                (root / f"endpoints/e{ordinal:02d}/rgb_state_complete.json").read_bytes()
            )
            if rgb_meta != {
                "stable": endpoint.evidence["rgb_stable"],
                "material": endpoint.evidence["rgb_material"],
            }:
                raise ValueError("immediate RGB state differs")
            snapshots = [
                p.strict_json((root / f"endpoints/e{ordinal:02d}/{name}.json").read_bytes())
                for name in (
                    "paired_draw_input",
                    "paired_draw_output",
                    "paired_read_input",
                    "paired_read_output",
                )
            ]
            excluded = {
                "projection_matrix_float32",
                "modelview_matrix_float32",
                "clip_origin",
                "clip_depth_mode",
            }
            if (
                {k: v for k, v in snapshots[0].items() if k not in excluded}
                != {k: v for k, v in snapshots[1].items() if k not in excluded}
                or snapshots[1] != snapshots[2]
                or snapshots[2] != snapshots[3]
            ):
                raise ValueError("immediate paired producer snapshot drift")
            stable = {k: snapshots[1][k] for k in endpoint.evidence["paired_stable"]}
            attachments = copy.deepcopy(stable["offscreen_attachments"])
            attachments["offFBO"].pop("framebuffer", None)
            for role in ("color0", "depth"):
                attachments["offFBO"][role].pop("object_name", None)
            stable["offscreen_attachments"] = attachments
            if stable != endpoint.evidence["paired_stable"]:
                raise ValueError("retained paired state differs from immediate callback")
            completed.append(endpoint)
            ordinal += 1
            stage = 0
        else:
            stage += 1
    # Only current callback components may be orphaned; all physical bytes were charged above.
    orphans = set(physical) - seen - {"report.json", "terminal.json"}
    allowed_orphans = set(stage_paths(STAGES[stage], ordinal)) if ordinal < 16 else set()
    if ordinal < 16:
        allowed_orphans.add(f"endpoints/e{ordinal:02d}/sampler_failure.json")
    if not orphans <= allowed_orphans:
        raise ValueError("unsupported orphan evidence")
    if "terminal.json" in physical:
        terminal = p.strict_json((root / "terminal.json").read_bytes())
        counts = {s for s in STAGES if s.endswith("attempt") or s.endswith("complete")}
        if (
            type(terminal) is not dict
            or set(terminal) != {"status", "binding_root", "completed", "counts", "report"}
            or terminal["status"] not in ("INCONCLUSIVE", "DIAGNOSTIC_COMPLETE")
            or terminal["binding_root"] != binding.root
            or type(terminal["completed"]) is not int
            or not 0 <= terminal["completed"] <= len(completed)
            or type(terminal["counts"]) is not dict
            or set(terminal["counts"]) != counts
            or any(type(v) is not int or not 0 <= v <= 16 for v in terminal["counts"].values())
        ):
            raise ValueError("closed typed terminal receipt")
        validate_ref(terminal["report"], "report.json")
        report = (root / "report.json").read_bytes()
        if (
            len(report) != terminal["report"]["bytes"]
            or sha256_bytes(report) != terminal["report"]["sha256"]
        ):
            raise ValueError("terminal report corruption")
    return completed


def inspect(endpoint: Endpoint, binding: Binding) -> bytes:
    validate_endpoint(endpoint, binding)
    from PIL import Image as PILImage

    stream = io.BytesIO()
    PILImage.fromarray(endpoint.arrays["rgb"]).save(stream, format="PNG")
    return stream.getvalue()


def retention_bound() -> dict[str, int]:
    old = p.retention_bound()
    endpoint = old["endpoint_reserved_bytes"] + 16384
    if endpoint > p.MIB:
        raise ValueError("discriminator metadata exceeds endpoint cap")
    return {
        **old,
        "endpoint_reserved_bytes": endpoint,
        "sampler_metadata_cap": 16384,
        "report_cap": p.MIB,
        "shared_cap": 2 * p.MIB,
        "events_max_bytes": 16 * len(STAGES) * 4096,
    }


class NativeCapture:
    """One live model/context, sixteen repeated captures. Constructor remains HELD."""

    def __init__(self, repository: Path, binding: Binding, sink: Sink):
        if (
            type(sink) is not Sink
            or sink.binding != binding
            or sink.ordinal != 0
            or sink.native_claimed
        ):
            raise PermissionError("owned bound fresh sink before SDK access")
        require_native(repository, binding)
        sink.native_claimed = True
        import importlib
        import inspect as source_inspect

        import mujoco

        from epsbench.diagnostics.paired_appearance_native import _material_state
        from epsbench.sim.canonical_paired import CanonicalPairedRenderer
        from epsbench.sim.compiled import extract_compiled_scene_contract

        classic = importlib.import_module("mujoco.rendering.classic.renderer")
        native = importlib.import_module("mujoco._render")
        renderer_file = Path(classic.__file__ or "")
        if (
            mujoco.Renderer is not classic.Renderer
            or mujoco.mjr_render is not native.mjr_render
            or mujoco.mjr_readPixels is not native.mjr_readPixels
            or sha256_bytes(renderer_file.read_bytes()) != p.SDK_RENDERER_SHA256
            or Path(source_inspect.getsourcefile(mujoco.Renderer.render) or "").resolve()
            != renderer_file.resolve()
        ):
            raise ValueError("pinned ordinary RGB SDK implementation")
        self.binding, self.sink, self.mujoco = binding, sink, mujoco
        self.next_index, self.failed = 0, False
        self.renderer: Any = None
        self.material_state = _material_state
        xml, assets = scene_xml()
        self.model = mujoco.MjModel.from_xml_string(xml, assets)
        self.data = mujoco.MjData(self.model)
        self.camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "monocular_camera"
        )
        compiled = extract_compiled_scene_contract(
            self.model, self.data, p.SURFACES["corridor"], "monocular_camera"
        )
        self.compiled = p.strict_json(canonical_json_bytes(asdict(compiled)))
        p.validate_compiled("corridor", self.compiled)
        self.mapping = remapping(
            tuple(compiled.raw_geom_ids[name] for name in p.SURFACES["corridor"])
        )
        try:
            self.renderer = mujoco.Renderer(self.model, height=120, width=160)
            self.paired = CanonicalPairedRenderer(
                self.renderer, progress_observer=self._paired_progress
            )
            self.sampler = NativeSampler(self.renderer, repository, binding)
        except BaseException:
            self.close()
            raise

    def _emit(self, stage: str, value: Any) -> None:
        self.sink.progress(stage, self.next_index, value)

    def _paired_progress(self, stage: str, value: Any) -> None:
        self._emit("paired_" + stage, value)

    def capture(self, ordinal: int) -> Endpoint:
        from epsbench.sim.canonical_paired import observe_canonical_paired_state, stable_state

        if (
            self.failed
            or type(ordinal) is not int
            or ordinal != self.next_index
            or ordinal not in range(16)
            or self.renderer is None
        ):
            raise ValueError("ordered live discriminator; failure/close terminal")
        identity = MEMBERS[ordinal]
        original_matid = self.model.geom_matid.copy()
        original_rgba = self.model.geom_rgba.copy()
        original_ambient = self.model.light_ambient.copy()
        original_diffuse = self.model.light_diffuse.copy()
        sampler: dict[str, Any] = {}
        try:
            self._emit("endpoint_attempt", None)
            solid = identity.appearance == APPEARANCES[0]
            for index, (raw, _, _) in enumerate(self.mapping):
                self.model.geom_matid[raw] = index + 4 if solid else index
                self.model.geom_rgba[raw] = [224 / 255] * 3 + [1] if solid else [1] * 4
            ambient, diffuse = FIXED["illumination"][
                "unit" if identity.arm.startswith("unit") else "original"
            ]
            self.model.light_ambient[0] = ambient
            self.model.light_diffuse[0] = diffuse
            self.mujoco.mj_forward(self.model, self.data)
            self.renderer.update_scene(self.data, camera=self.camera_id)
            for flag in (
                self.mujoco.mjtRndFlag.mjRND_SHADOW,
                self.mujoco.mjtRndFlag.mjRND_FOG,
                self.mujoco.mjtRndFlag.mjRND_HAZE,
            ):
                self.renderer.scene.flags[flag] = False
            setting = "nearest" if identity.arm.endswith("nearest") else "default"
            with sampler_override(self.sampler, setting, sampler):
                sampler["rgb_before"] = self.sampler.query()
                validate_sampler(sampler["rgb_before"], setting)
                self._emit("rgb_attempt", None)
                rgb = np.asarray(self.renderer.render(), dtype=np.uint8).copy()
                self._emit("rgb_read_complete", rgb)
                sampler["rgb_after"] = self.sampler.query()
                validate_sampler(sampler["rgb_after"], setting)
                rgb_state = stable_state(observe_canonical_paired_state(self.renderer))
                material = self.material_state(self.model, self.renderer)
                self._emit(
                    "rgb_state_complete",
                    canonical_json_bytes({"stable": rgb_state, "material": material}),
                )
                sampler["pair_before"] = self.sampler.query()
                validate_sampler(sampler["pair_before"], setting)
                pair = self.paired.capture()
                sampler["pair_after"] = self.sampler.query()
                validate_sampler(sampler["pair_after"], setting)
                paired_material = self.material_state(self.model, self.renderer)
            self._emit("sampler_complete", canonical_json_bytes(sampler))
            opaque = np.zeros((120, 160), dtype=np.int32)
            for raw, label, _ in self.mapping:
                opaque[pair.raw_geom_segmentation == raw] = label
            e = {
                "schema": SCHEMA + ":endpoint",
                "source_head": self.binding.preparation.source_head,
                "source_tree": self.binding.preparation.source_tree,
                "binding_root": self.binding.root,
                "config_sha256": self.binding.preparation.config_file,
                "scene_xml_sha256": self.binding.preparation.scene_root,
                "compiled": self.compiled,
                "camera": {
                    "world_position": self.data.cam_xpos[self.camera_id].tolist(),
                    "rotation_row_major": self.data.cam_xmat[self.camera_id].tolist(),
                    "fovy": float(self.model.cam_fovy[self.camera_id]),
                },
                "mapping": [list(v) for v in self.mapping],
                "scene_map": [[v.segid_plus_one, v.objid, v.objtype] for v in pair.scene_map],
                "near": pair.near,
                "far": pair.far,
                "orientation": p.ORIENTATION,
                "runtime": pair.stable_state["context_runtime"],
                "rgb_stable": rgb_state,
                "paired_stable": dict(pair.stable_state),
                "rgb_material": material,
                "paired_material": paired_material,
                "sampler": sampler,
                "rgb_provenance": {
                    "producer": "mujoco.Renderer.render",
                    "sdk_renderer_sha256": p.SDK_RENDERER_SHA256,
                    "operation": "separate_ordinary_rgb_draw_before_owned_id_depth",
                    "same_draw_as_pair": False,
                },
            }
            endpoint = Endpoint(
                identity,
                {
                    "rgb": rgb,
                    "native_id": pair.native_id_rgb,
                    "native_depth": pair.native_depth_pre_metric,
                    "raw": pair.raw_geom_segmentation,
                    "depth": pair.depth,
                    "opaque": opaque,
                    "controlled": opaque > 0,
                    "horizontal": opaque[:, :-1] != opaque[:, 1:],
                    "vertical": opaque[:-1] != opaque[1:],
                },
                e,
            )
            self.model.geom_matid[:] = original_matid
            self.model.geom_rgba[:] = original_rgba
            self.model.light_ambient[:] = original_ambient
            self.model.light_diffuse[:] = original_diffuse
            endpoint.evidence["model_restoration"] = {
                "before": {
                    "geom_matid": original_matid.tolist(),
                    "geom_rgba": original_rgba.tolist(),
                    "light_ambient": original_ambient.tolist(),
                    "light_diffuse": original_diffuse.tolist(),
                },
                "after": {
                    "geom_matid": self.model.geom_matid.tolist(),
                    "geom_rgba": self.model.geom_rgba.tolist(),
                    "light_ambient": self.model.light_ambient.tolist(),
                    "light_diffuse": self.model.light_diffuse.tolist(),
                },
            }
            validate_endpoint(endpoint, self.binding)
            self._emit("endpoint_complete", endpoint)
            self.next_index += 1
            return endpoint
        except BaseException:
            self.failed = True
            if sampler and not self.sink.failed:
                self.sink.put(
                    f"endpoints/e{ordinal:02d}/sampler_failure.json", canonical_json_bytes(sampler)
                )
            raise
        finally:
            self.model.geom_matid[:] = original_matid
            self.model.geom_rgba[:] = original_rgba
            self.model.light_ambient[:] = original_ambient
            self.model.light_diffuse[:] = original_diffuse
            if (
                not np.array_equal(self.model.geom_matid, original_matid)
                or not np.array_equal(self.model.geom_rgba, original_rgba)
                or not np.array_equal(self.model.light_ambient, original_ambient)
                or not np.array_equal(self.model.light_diffuse, original_diffuse)
            ):
                self.failed = True
                raise ValueError("owned model intervention restoration failed")

    def close(self) -> None:
        try:
            if self.renderer is not None:
                self.renderer.close()
        finally:
            self.renderer, self.failed = None, True


def _drive(capture: Any, sink: Sink, deadline: float | None = None) -> dict[str, Any]:
    """Internal finite loop. Valid suitability failures never truncate the factorial."""
    start = sink.started
    work_deadline = start + 300
    endpoints = []
    failure = None
    try:
        if deadline is not None:
            if type(deadline) is not float or not math.isfinite(deadline):
                raise ValueError("finite shortening-only deadline required")
            work_deadline = min(work_deadline, deadline)
        for ordinal in range(16):
            if time.monotonic() >= work_deadline:
                raise TimeoutError("inclusive 300s slot")
            endpoint = capture.capture(ordinal)
            validate_endpoint(endpoint, sink.binding)
            endpoints.append(endpoint)
            validate_relations(endpoints)
    except BaseException as exc:
        failure = type(exc).__name__
    finally:
        try:
            capture.close()
        except BaseException as exc:
            failure = "close:" + type(exc).__name__
    result = (
        assess(endpoints, sink.binding)
        if failure is None
        else {
            "status": "INCONCLUSIVE",
            "endpoints": len(endpoints),
            "planned": 16,
            "failure": failure,
            "absolute_native_uv_transfer_qualification": "UNRESOLVED",
        }
    )
    if time.monotonic() >= work_deadline:
        result["status"], result["failure"] = "INCONCLUSIVE", "inclusive deadline"
    if not sink.failed:
        report = sink.put("report.json", canonical_json_bytes(result))
        sink.put(
            "terminal.json",
            canonical_json_bytes(
                {
                    "status": result["status"],
                    "binding_root": sink.binding.root,
                    "completed": len(endpoints),
                    "counts": sink.counts,
                    "report": report,
                }
            ),
        )
    return result
