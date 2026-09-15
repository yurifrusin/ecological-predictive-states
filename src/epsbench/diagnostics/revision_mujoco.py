"""Fixed MuJoCo provider for the capture-contract revision study."""

from __future__ import annotations

import hashlib
import importlib.metadata
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from epsbench.diagnostics.revision_capture import (
    RevisionCaptureFailure,
    StudyCell,
    canonical_json_bytes,
    sha256_file,
)
from epsbench.diagnostics.revision_runner import PreparedEpisode
from epsbench.utils.seeding import derive_seed


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _semantic_contract(contract: Any) -> dict[str, object]:
    return {
        "raw_geom_ids": _jsonable(contract.raw_geom_ids),
        "raw_geom_world_positions": _jsonable(contract.raw_geom_world_positions),
        "raw_geom_compiled_sizes": _jsonable(contract.raw_geom_compiled_sizes),
        "raw_geom_types": _jsonable(contract.raw_geom_types),
        "raw_geom_world_rotations_row_major": _jsonable(
            contract.raw_geom_world_rotations_row_major
        ),
        "camera_field_of_view_degrees": float(contract.camera_field_of_view_degrees),
        "camera_world_position": _jsonable(contract.camera_world_position),
        "camera_world_rotation_row_major": _jsonable(contract.camera_world_rotation_row_major),
    }


def _digest_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _geometry_facts(contract: Any) -> dict[str, object]:
    planes: list[dict[str, object]] = []
    boxes: list[dict[str, object]] = []
    semantic = _semantic_contract(contract)
    for name, raw_id in sorted(contract.raw_geom_ids.items()):
        item = {
            "name": name,
            "raw_geom_id": int(raw_id),
            "center": list(contract.raw_geom_world_positions[name]),
            "rotation": np.asarray(
                contract.raw_geom_world_rotations_row_major[name], dtype=np.float64
            )
            .reshape(3, 3)
            .tolist(),
            "half_extents": list(contract.raw_geom_compiled_sizes[name]),
        }
        if contract.raw_geom_types[name] == "plane":
            planes.append(item)
        else:
            boxes.append(item)
    return {
        "finite_planes": planes,
        "oriented_boxes": boxes,
        "compiled_semantic_facts_sha256": _digest_json(semantic),
    }


def _camera_facts(
    mujoco: Any, model: Any, data: Any, camera_id: int, axis: int, position: float
) -> dict[str, object]:
    model.cam_pos[camera_id, axis] = position
    mujoco.mj_forward(model, data)
    value = {
        "camera_position": data.cam_xpos[camera_id].astype(float).tolist(),
        "camera_rotation": data.cam_xmat[camera_id].astype(float).reshape(3, 3).tolist(),
        "fovy_degrees": float(model.cam_fovy[camera_id]),
        "model_stat_extent": float(model.stat.extent),
        "model_vis_map_znear": float(model.vis.map.znear),
        "model_vis_map_zfar": float(model.vis.map.zfar),
    }
    value["state_sha256"] = _digest_json(value)
    return value


class MujocoStack:
    def __init__(
        self,
        mujoco: Any,
        model: Any,
        data: Any,
        role: str,
        backend: str,
        requested: int,
        width: int,
        height: int,
        camera_id: int,
        pose_axis: int,
        pose_positions: Mapping[str, float],
        pose_facts: Mapping[str, Mapping[str, object]],
        xml: str,
    ):
        self.mujoco = mujoco
        self.model = model
        self.data = data
        self.role = role
        self.backend = backend
        self.requested = requested
        self.width = width
        self.height = height
        self.camera_id = camera_id
        self.pose_axis = pose_axis
        self.pose_positions = pose_positions
        self.pose_facts = pose_facts
        self.xml = xml
        self.renderer = mujoco.Renderer(model, height=height, width=width)

    def make_current(self) -> None:
        context = getattr(self.renderer, "_gl_context", None)
        if context is None or not hasattr(context, "make_current"):
            raise RevisionCaptureFailure("renderer exposes no make_current operation")
        context.make_current()

    def set_pose(self, pose_name: str) -> Mapping[str, object]:
        self.model.cam_pos[self.camera_id, self.pose_axis] = self.pose_positions[pose_name]
        self.mujoco.mj_forward(self.model, self.data)
        observed = {
            "camera_position": self.data.cam_xpos[self.camera_id].astype(float).tolist(),
            "camera_rotation": self.data.cam_xmat[self.camera_id]
            .astype(float)
            .reshape(3, 3)
            .tolist(),
            "fovy_degrees": float(self.model.cam_fovy[self.camera_id]),
            "model_stat_extent": float(self.model.stat.extent),
            "model_vis_map_znear": float(self.model.vis.map.znear),
            "model_vis_map_zfar": float(self.model.vis.map.zfar),
        }
        observed["state_sha256"] = _digest_json(observed)
        return observed

    def update_scene(self) -> None:
        self.renderer.update_scene(self.data, camera=self.camera_id)

    def render_rgb(self) -> npt.NDArray[np.uint8]:
        return np.asarray(self.renderer.render()).copy()

    def enable_depth(self) -> None:
        self.renderer.enable_depth_rendering()

    def render_depth(self) -> npt.NDArray[np.float32]:
        return np.asarray(self.renderer.render()).copy()

    def disable_depth(self) -> None:
        self.renderer.disable_depth_rendering()

    def enable_segmentation(self) -> None:
        self.renderer.enable_segmentation_rendering()

    def scene_map(self) -> Mapping[int, tuple[int, int]]:
        result: dict[int, tuple[int, int]] = {}
        for geom in self.renderer.scene.geoms[: self.renderer.scene.ngeom]:
            if int(geom.segid) >= 0:
                result[int(geom.segid)] = (int(geom.objid), int(geom.objtype))
        if not result:
            raise RevisionCaptureFailure("live scene map is empty")
        return result

    def render_segmentation(self, out: npt.NDArray[np.uint8]) -> npt.NDArray[np.int32]:
        return np.asarray(self.renderer.render(out=out)).copy()

    def disable_segmentation(self) -> None:
        self.renderer.disable_segmentation_rendering()

    def provenance(self) -> Mapping[str, object]:
        from epsbench.diagnostics.gl_provenance import inspect_mujoco_offscreen_attachments
        from epsbench.diagnostics.mujoco_runner import (
            _model_sha256,
            _observed_backend,
            _runtime_hashes,
        )

        value = dict(inspect_mujoco_offscreen_attachments(self.renderer))
        value.update(_runtime_hashes(self.mujoco, self.xml))
        value.update(
            {
                "role": self.role,
                "requested_offsamples": self.requested,
                "actual_offsamples": int(self.renderer._mjr_context.offSamples),
                "actual_backend": _observed_backend(self.renderer),
                "model_mjb_sha256": _model_sha256(self.mujoco, self.model),
                "model_stat_extent": float(self.model.stat.extent),
                "model_vis_map_znear": float(self.model.vis.map.znear),
                "model_vis_map_zfar": float(self.model.vis.map.zfar),
                "python_version": __import__("platform").python_version(),
                "numpy_version": importlib.metadata.version("numpy"),
                "mujoco_version": importlib.metadata.version("mujoco"),
                "pyopengl_version": importlib.metadata.version("PyOpenGL"),
                "glfw_version": importlib.metadata.version("glfw"),
            }
        )
        return value

    def close(self) -> None:
        self.renderer.close()


def provide_fixed_episode(cell: StudyCell) -> tuple[PreparedEpisode, Any]:
    """Derive the reviewed fixed episode and precompile all models before contexts."""
    import mujoco

    if int(mujoco.mjtObj.mjOBJ_GEOM) != 5:
        raise RevisionCaptureFailure("MuJoCo geom object-type code differs from fixed schema")

    from epsbench.appearance import (
        load_appearance_registry,
        load_evaluation_seed_registry,
        resolve_appearance,
    )
    from epsbench.config import load_config
    from epsbench.config.models import CorridorConfig, SingleOccluderConfig
    from epsbench.diagnostics.mujoco_runner import _model_sha256
    from epsbench.sim.compiled import extract_compiled_scene_contract

    config_path = Path(
        "configs/corridor_v0.yaml" if cell.family == "corridor" else "configs/benchmark_v0.yaml"
    )
    config = load_config(config_path)
    if (
        config.seed != 1729
        or config.render.width != 160
        or config.render.height != 120
        or config.camera.field_of_view_degrees != 55.0
        or config.appearance.profile_id != "legacy_solid_base_v1"
    ):
        raise RevisionCaptureFailure("fixed config contract differs")
    episode_seed = derive_seed(config.seed, f"episode:{cell.episode_index}")
    registry_path = Path("configs/appearance_candidates_v0.yaml")
    seeds_path = Path("configs/evaluation_seed_candidates_v0.yaml")
    registry = load_appearance_registry(registry_path)
    seed_registry = load_evaluation_seed_registry(seeds_path)

    surface_names: tuple[str, ...]
    if cell.family == "corridor":
        if not isinstance(config, CorridorConfig):
            raise RevisionCaptureFailure("corridor config discriminator differs")
        from epsbench.sim.corridor import (
            CORRIDOR_SURFACE_NAMES,
            build_corridor_scene_xml,
            sample_corridor_geometry,
        )

        surface_names = CORRIDOR_SURFACE_NAMES
        geometry = sample_corridor_geometry(config, episode_seed)
        appearance = resolve_appearance(
            registry,
            config.appearance.profile_id,
            "corridor",
            surface_names,
            episode_seed,
            config.seed,
            seed_registry,
        )
        xml = build_corridor_scene_xml(config, geometry, appearance)
        pose_axis = 1
        pose_positions = {
            "before": geometry.camera_before_forward_position,
            "after": geometry.camera_after_forward_position,
        }
    else:
        if not isinstance(config, SingleOccluderConfig):
            raise RevisionCaptureFailure("single-occluder config discriminator differs")
        from epsbench.sim.single_occluder import (
            SINGLE_OCCLUDER_SURFACE_NAMES,
            build_scene_xml,
        )

        surface_names = SINGLE_OCCLUDER_SURFACE_NAMES
        appearance = resolve_appearance(
            registry,
            config.appearance.profile_id,
            "single_occluder",
            surface_names,
            episode_seed,
            config.seed,
            seed_registry,
        )
        xml = build_scene_xml(config, appearance)
        pose_axis = 0
        pose_positions = {
            "before": config.camera.before_lateral,
            "after": config.camera.after_lateral,
        }

    def compile_model(samples: int) -> tuple[Any, Any, int, Any]:
        model = mujoco.MjModel.from_xml_string(xml, appearance.asset_bytes)
        model.vis.quality.offsamples = samples
        data = mujoco.MjData(model)
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "monocular_camera")
        mujoco.mj_forward(model, data)
        contract = extract_compiled_scene_contract(model, data, surface_names, "monocular_camera")
        return model, data, camera_id, contract

    normalized_model, _, _, normalized_contract = compile_model(4)
    normalized_hash = _model_sha256(mujoco, normalized_model)
    semantic = _semantic_contract(normalized_contract)
    roles = {"primary": 0 if cell.policy == "joint0" else 4}
    if cell.policy == "hybrid":
        roles["segmentation"] = 0
    compiled: dict[str, tuple[Any, Any, int, Any]] = {
        role: compile_model(samples) for role, samples in roles.items()
    }
    for role, (_, _, _, contract) in compiled.items():
        if _semantic_contract(contract) != semantic:
            raise RevisionCaptureFailure(f"{role} compiled semantic facts differ")
    first_model, first_data, first_camera, _ = compiled["primary"]
    pose_facts = {
        name: _camera_facts(mujoco, first_model, first_data, first_camera, pose_axis, position)
        for name, position in pose_positions.items()
    }
    for model, data, camera_id, _contract in compiled.values():
        model.cam_pos[camera_id, pose_axis] = pose_positions["before"]
        mujoco.mj_forward(model, data)
    actual_hashes = {role: _model_sha256(mujoco, values[0]) for role, values in compiled.items()}
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], text=True).strip()
    prepared = PreparedEpisode(
        episode_seed=episode_seed,
        source={
            "head": head,
            "tree": tree,
            "xml_sha256": hashlib.sha256(xml.encode("utf-8")).hexdigest(),
            "config_sha256": sha256_file(config_path),
            "appearance_registry_sha256": sha256_file(registry_path),
            "appearance_instance": _jsonable(appearance.record),
            "asset_sha256": {
                name: hashlib.sha256(payload).hexdigest()
                for name, payload in sorted(appearance.asset_bytes.items())
            },
            "seed_registry_sha256": sha256_file(seeds_path),
            "dependency_lock_sha256": sha256_file(Path("uv.lock")),
            "episode_namespace": f"episode:{cell.episode_index}",
            "appearance_namespace_source": (
                "resolve_appearance(...episode_seed,root_seed,seed_registry)"
            ),
        },
        config=_jsonable(config),
        model={
            "normalized_mjb_sha256": normalized_hash,
            "semantic_facts_sha256": _digest_json(semantic),
            "actual_models": actual_hashes,
        },
        geometry_facts=_geometry_facts(normalized_contract),
        pose_facts=pose_facts,
    )

    def factory(samples: int, role: str) -> MujocoStack:
        if role not in compiled or roles[role] != samples:
            raise RevisionCaptureFailure("factory role/sample request differs")
        model, data, camera_id, _ = compiled.pop(role)
        return MujocoStack(
            mujoco,
            model,
            data,
            role,
            cell.backend,
            samples,
            config.render.width,
            config.render.height,
            camera_id,
            pose_axis,
            pose_positions,
            pose_facts,
            xml,
        )

    return prepared, factory
