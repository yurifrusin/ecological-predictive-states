"""Lazy capture source only. No driver, qualification, launch authority or native work on import."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from epsbench.diagnostics.paired_appearance import (
    FIXED,
    HEIGHT,
    ORIENTATION,
    SDK_RENDERER_SHA256,
    SURFACES,
    VERSION,
    WIDTH,
    Family,
    Frame,
    config_bytes,
    fixed_config,
    mapping,
    owned,
    protection_check,
    validate_compiled,
    validate_frame,
    validate_xml,
    visual_plan,
)
from epsbench.utils.canonical import canonical_json_bytes, logical_array_hash, sha256_bytes

Progress = Callable[[str, int, Any], None]


def _material_state(model: Any, renderer: Any) -> dict[str, Any]:
    model_fields = (
        "geom_rgba",
        "geom_matid",
        "cam_ipd",
        "cam_projection",
        "cam_sensorsize",
        "mat_rgba",
        "mat_emission",
        "mat_texid",
        "mat_texrepeat",
        "mat_texuniform",
        "mat_specular",
        "mat_shininess",
        "mat_reflectance",
        "tex_type",
        "tex_colorspace",
        "tex_height",
        "tex_width",
        "tex_nchannel",
        "tex_adr",
        "light_type",
        "light_active",
        "light_texid",
        "light_pos",
        "light_dir",
        "light_ambient",
        "light_diffuse",
        "light_specular",
        "light_castshadow",
    )
    state = {name: np.asarray(getattr(model, name)).tolist() for name in model_fields}
    state["tex_data_sha256"] = logical_array_hash(np.asarray(model.tex_data))
    geoms = [
        {
            **{
                name: int(getattr(g, name))
                for name in ("objid", "matid", "texid", "texuniform", "texcoord")
            },
            **{name: np.asarray(getattr(g, name)).tolist() for name in ("rgba", "texrepeat")},
            **{
                name: float(getattr(g, name))
                for name in ("emission", "specular", "shininess", "reflectance")
            },
        }
        for g in renderer.scene.geoms[: renderer.scene.ngeom]
    ]
    lights = [
        {
            **{
                name: np.asarray(getattr(light, name)).tolist()
                for name in ("pos", "dir", "ambient", "diffuse", "specular", "attenuation")
            },
            **{
                name: int(getattr(light, name))
                for name in ("id", "type", "texid", "headlight", "castshadow")
            },
        }
        for light in renderer.scene.lights[: renderer.scene.nlight]
    ]
    return {"model": state, "scene_geoms": geoms, "scene_lights": lights}


class NativeCapture:
    """One context, two ordered endpoints. Later externally bound driver must own retention.

    Constructing this object performs native work and is HELD in this source package.
    Its public progress callback preserves completed RGB/readbacks before later checks.
    A callback exception is terminal. There is no retry or automatic continuation.
    """

    def __init__(
        self,
        repository: Path,
        family: Family,
        appearance: str,
        *,
        source_head: str,
        source_tree: str,
        progress: Progress,
    ):
        import mujoco

        from epsbench.config import CorridorConfig, SingleOccluderConfig, parse_config
        from epsbench.schema import CorridorSampledGeometry
        from epsbench.sim.canonical_paired import CanonicalPairedRenderer, require_supported_runtime
        from epsbench.sim.compiled import extract_compiled_scene_contract
        from epsbench.sim.corridor import build_corridor_scene_xml
        from epsbench.sim.single_occluder import build_scene_xml

        # Existing native producer runtime guard is unchanged; no false causal task purpose.
        require_supported_runtime("osmesa")
        fixed_config((repository / "configs/development/paired_appearance_v1.json").read_bytes())
        import importlib
        import inspect
        import re

        for value in (source_head, source_tree):
            if re.fullmatch(r"[0-9a-f]{40}", value) is None:
                raise ValueError("source head/tree binding required before compilation")
        classic = importlib.import_module("mujoco.rendering.classic.renderer")
        native = importlib.import_module("mujoco._render")
        if type(classic.__file__) is not str:
            raise ValueError("SDK source file unavailable")
        renderer_file = Path(classic.__file__)
        if (
            mujoco.Renderer is not classic.Renderer
            or mujoco.mjr_render is not native.mjr_render
            or mujoco.mjr_readPixels is not native.mjr_readPixels
            or sha256_bytes(renderer_file.read_bytes()) != SDK_RENDERER_SHA256
            or Path(inspect.getsourcefile(mujoco.Renderer.render) or "").resolve()
            != renderer_file.resolve()
        ):
            raise ValueError("pinned ordinary RGB SDK implementation required")
        self.protection = protection_check(repository)
        self.family, self.appearance = family, appearance
        self.source_head, self.source_tree = source_head, source_tree
        self.progress, self.next_index, self.failed = progress, 0, False
        self.renderer: Any = None
        plan = visual_plan(family, appearance)
        # Old config is only the existing builder geometry adapter; its appearance fallback
        # is never used. The separately typed plan/record identifies the actual appearance.
        common = {
            "schema_version": "0.1.0-dev.4",
            "scene_family": family,
            "seed": FIXED["roots"][0 if family == "single_occluder" else 1],
            "render": {"width": WIDTH, "height": HEIGHT},
            "appearance": {
                "registry_version": "appearance_candidate_registry_v1",
                "profile_id": "legacy_solid_base_v1",
            },
        }
        if family == "single_occluder":
            config = parse_config(
                {
                    **common,
                    "camera": {
                        "before_lateral": -0.35,
                        "after_lateral": 0.35,
                        "forward": -3.0,
                        "height": 1.25,
                        "field_of_view_degrees": 55.0,
                    },
                    "action": {
                        "name": "lateral_right",
                        "delta_forward": 0.0,
                        "delta_lateral": 0.7,
                        "delta_yaw": 0.0,
                    },
                }
            )
            if not isinstance(config, SingleOccluderConfig):
                raise ValueError("single builder adapter")
            xml = build_scene_xml(config, plan)
        else:
            config = parse_config(
                {
                    **common,
                    "geometry": {
                        "width": {"minimum": 3.0, "maximum": 3.0},
                        "length": {"minimum": 6.0, "maximum": 6.0},
                        "wall_height": 5.0,
                    },
                    "camera": {
                        "lateral_position": 0.0,
                        "starting_forward_position": 0.5,
                        "height": 1.25,
                        "field_of_view_degrees": 55.0,
                    },
                    "action": {
                        "name": "forward",
                        "delta_forward": 0.7,
                        "delta_lateral": 0.0,
                        "delta_yaw": 0.0,
                    },
                }
            )
            if not isinstance(config, CorridorConfig):
                raise ValueError("corridor builder adapter")
            geometry = CorridorSampledGeometry(
                width=3.0,
                length=6.0,
                wall_height=5.0,
                camera_lateral_position=0.0,
                camera_before_forward_position=0.5,
                camera_after_forward_position=1.2,
                camera_height=1.25,
                field_of_view_degrees=55.0,
            )
            xml = build_corridor_scene_xml(config, geometry, plan)
        validate_xml(family, appearance, xml)
        self.xml, self.plan, self.mujoco = xml, plan, mujoco
        self.model = mujoco.MjModel.from_xml_string(xml, plan.asset_bytes)
        self.model.vis.quality.offsamples = 0
        self.data = mujoco.MjData(self.model)
        self.camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "monocular_camera"
        )
        compiled = extract_compiled_scene_contract(
            self.model, self.data, SURFACES[family], "monocular_camera"
        )
        self.compiled = json.loads(canonical_json_bytes(asdict(compiled)))
        validate_compiled(family, self.compiled)
        self.remap = mapping(family, tuple(compiled.raw_geom_ids[n] for n in SURFACES[family]))
        try:
            self.renderer = mujoco.Renderer(self.model, height=HEIGHT, width=WIDTH)
            self.paired = CanonicalPairedRenderer(
                self.renderer, progress_observer=self._paired_progress
            )
        except BaseException:
            self.close()
            raise

    def _emit(self, stage: str, value: Any) -> None:
        if stage in {
            "rgb_state_complete",
            "paired_draw_input",
            "paired_draw_output",
            "paired_read_input",
            "paired_read_output",
        }:
            if type(value) is not bytes or len(value) > 65536:
                raise ValueError("bounded immediate producer metadata")
        elif stage == "paired_read_complete":
            if (
                type(value) is not tuple
                or len(value) != 2
                or any(type(v) is not bytes for v in value)
                or tuple(map(len, value)) != (HEIGHT * WIDTH * 3, HEIGHT * WIDTH * 4)
            ):
                raise ValueError("bounded immediate native buffers")
        elif stage == "rgb_read_complete":
            if (
                type(value) is not np.ndarray
                or value.shape != (HEIGHT, WIDTH, 3)
                or value.dtype != np.uint8
            ):
                raise ValueError("bounded immediate RGB")
        self.progress(stage, self.next_index, value)

    def _paired_progress(self, stage: str, value: Any) -> None:
        # Paired engine emits immutable raw read bytes and all four snapshots unchanged.
        self._emit("paired_" + stage, value)

    def capture(self, index: int) -> Frame:
        from epsbench.sim.canonical_paired import (
            observe_canonical_paired_state,
            stable_state,
            validate_saved_state,
        )

        if (
            self.failed
            or type(index) is not int
            or index != self.next_index
            or index not in (0, 1)
            or self.renderer is None
        ):
            raise ValueError("ordered live two-endpoint context; failure/close is terminal")
        try:
            self._emit("endpoint_attempt", None)
            axis = 0 if self.family == "single_occluder" else 1
            position = FIXED["single" if axis == 0 else "corridor"]["poses"][index]
            self.model.cam_pos[self.camera_id, axis] = position
            self.mujoco.mj_forward(self.model, self.data)
            self.renderer.update_scene(self.data, camera=self.camera_id)
            for flag in (
                self.mujoco.mjtRndFlag.mjRND_SHADOW,
                self.mujoco.mjtRndFlag.mjRND_FOG,
                self.mujoco.mjtRndFlag.mjRND_HAZE,
            ):
                self.renderer.scene.flags[flag] = False
            self._emit("rgb_attempt", None)
            rgb = np.asarray(self.renderer.render(), dtype=np.uint8).copy()
            self._emit("rgb_read_complete", owned(rgb))
            rgb_state = stable_state(observe_canonical_paired_state(self.renderer))
            rgb_material = _material_state(self.model, self.renderer)
            self._emit(
                "rgb_state_complete",
                canonical_json_bytes({"stable": rgb_state, "material": rgb_material}),
            )
            pair = self.paired.capture()
            validate_saved_state(pair.stable_state, WIDTH, HEIGHT)
            # No scene update or extra draw occurs between the ordinary and owned operations.
            paired_material = _material_state(self.model, self.renderer)
            opaque = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
            for raw, label, _ in self.remap:
                opaque[pair.raw_geom_segmentation == raw] = label
            camera = {
                "world_position": self.data.cam_xpos[self.camera_id].tolist(),
                "rotation_row_major": self.data.cam_xmat[self.camera_id].tolist(),
                "fovy": float(self.model.cam_fovy[self.camera_id]),
            }
            evidence = {
                "schema": VERSION + ":endpoint",
                "source_head": self.source_head,
                "source_tree": self.source_tree,
                "config_sha256": sha256_bytes(config_bytes()),
                "appearance_record": self.plan.record,
                "scene_xml_sha256": sha256_bytes(self.xml.encode()),
                "compiled": self.compiled,
                "camera": camera,
                "action": FIXED["single" if axis == 0 else "corridor"]["delta"],
                "runtime": pair.stable_state["context_runtime"],
                "mapping": [list(v) for v in self.remap],
                "scene_map": [[v.segid_plus_one, v.objid, v.objtype] for v in pair.scene_map],
                "near": pair.near,
                "far": pair.far,
                "orientation": ORIENTATION,
                "paired_stable": dict(pair.stable_state),
                "rgb_stable": rgb_state,
                "rgb_material": rgb_material,
                "paired_material": paired_material,
                "rgb_provenance": {
                    "producer": "mujoco.Renderer.render",
                    "sdk_renderer_sha256": SDK_RENDERER_SHA256,
                    "operation": "separate_ordinary_rgb_draw_before_owned_id_depth",
                    "same_draw_as_pair": False,
                    "index": index,
                },
            }
            frame = Frame(
                self.family,
                self.appearance,
                index,
                rgb,
                pair.native_id_rgb,
                pair.native_depth_pre_metric,
                pair.raw_geom_segmentation,
                pair.depth,
                opaque,
                opaque > 0,
                opaque[:, :-1] != opaque[:, 1:],
                opaque[:-1] != opaque[1:],
                evidence,
            )
            validate_frame(frame)
            self._emit("endpoint_complete", frame)
            self.next_index += 1
            return frame
        except BaseException:
            self.failed = True
            raise

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self.failed = True


def validate_retained_native(frame: Frame) -> None:
    """Later actual review: full existing saved producer validation plus independent contracts."""
    from epsbench.sim.canonical_paired import validate_saved_state

    validate_saved_state(frame.evidence["paired_stable"], WIDTH, HEIGHT)
    validate_frame(frame)
