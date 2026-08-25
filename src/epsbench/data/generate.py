"""Deterministic dataset generation for the Gate 0B single-scene slice."""

from __future__ import annotations

import json
import os
import platform
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image

from epsbench import __version__
from epsbench.annotations import derive_boundary_structure, derive_visibility
from epsbench.config import BenchmarkConfig
from epsbench.data.identity import compute_dataset_logical_hash, compute_ecological_label_hash
from epsbench.schema import (
    Action,
    ArtifactRecord,
    CameraInstrumentation,
    DatasetManifest,
    EpisodeManifest,
    FrameRecord,
    Modality,
    OcclusionRelation,
    PrivilegedInstrumentation,
    RendererProvenance,
    SurfaceReference,
    TransitionRecord,
    UnavailableAnnotation,
)
from epsbench.sim import render_transition
from epsbench.utils.canonical import (
    canonical_json_bytes,
    logical_array_hash,
    sha256_bytes,
    sha256_file,
    write_canonical_json,
)
from epsbench.utils.seeding import derive_seed, rng_for

_SURFACE_NAMES = ("support_surface", "occluding_surface", "background_surface")


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _array_artifact(
    path: Path,
    root: Path,
    array: np.ndarray[Any, Any],
    modality: Modality,
    media_type: str,
) -> ArtifactRecord:
    return ArtifactRecord(
        path=_relative(path, root),
        modality=modality,
        media_type=media_type,
        dtype=array.dtype.str,
        shape=tuple(array.shape),
        logical_sha256=logical_array_hash(array),
        file_sha256=sha256_file(path),
        byte_count=path.stat().st_size,
    )


def _json_artifact(
    path: Path,
    root: Path,
    payload: Any,
    modality: Modality,
    media_type: str,
) -> ArtifactRecord:
    logical_bytes = canonical_json_bytes(payload)
    return ArtifactRecord(
        path=_relative(path, root),
        modality=modality,
        media_type=media_type,
        dtype="json",
        shape=(len(logical_bytes),),
        logical_sha256=sha256_bytes(logical_bytes),
        file_sha256=sha256_file(path),
        byte_count=path.stat().st_size,
    )


def _surface_references(
    root_seed: int,
    episode_index: int,
) -> tuple[tuple[SurfaceReference, ...], dict[str, SurfaceReference]]:
    rng = rng_for(root_seed, f"surface-remapping:{episode_index}")
    permutation = rng.permutation(len(_SURFACE_NAMES))
    labels: set[int] = set()
    while len(labels) < len(_SURFACE_NAMES):
        labels.add(int(rng.integers(1, 2**31 - 1)))
    label_list = sorted(labels)
    by_name: dict[str, SurfaceReference] = {}
    ordered: list[SurfaceReference] = []
    for output_index, surface_index in enumerate(permutation.tolist()):
        name = _SURFACE_NAMES[surface_index]
        token = rng.bytes(8).hex()
        reference = SurfaceReference(
            surface_id=f"surface-{token}",
            segmentation_label=label_list[output_index],
        )
        by_name[name] = reference
        ordered.append(reference)
    return tuple(ordered), by_name


def _remap_segmentation(
    raw_segmentation: np.ndarray[Any, Any],
    raw_geom_ids: dict[str, int],
    references: dict[str, SurfaceReference],
) -> np.ndarray[Any, Any]:
    remapped = np.zeros(raw_segmentation.shape, dtype=np.int32)
    for name, raw_id in raw_geom_ids.items():
        remapped[raw_segmentation == raw_id] = references[name].segmentation_label
    return remapped


def _write_frame(
    root: Path,
    episode_directory: Path,
    frame_index: int,
    rgb: np.ndarray[Any, Any],
    depth: np.ndarray[Any, Any],
    segmentation: np.ndarray[Any, Any],
    camera: CameraInstrumentation,
) -> FrameRecord:
    stem = "before" if frame_index == 0 else "after"
    rgb_path = episode_directory / f"rgb_{stem}.png"
    depth_path = episode_directory / f"depth_{stem}.npy"
    segmentation_path = episode_directory / f"segmentation_{stem}.npy"
    camera_path = episode_directory / f"camera_{stem}.json"
    Image.fromarray(rgb, mode="RGB").save(rgb_path, compress_level=9, optimize=False)
    np.save(depth_path, depth, allow_pickle=False)
    np.save(segmentation_path, segmentation, allow_pickle=False)
    write_canonical_json(camera_path, camera)
    return FrameRecord(
        frame_index=frame_index,  # type: ignore[arg-type]
        width=int(rgb.shape[1]),
        height=int(rgb.shape[0]),
        rgb=_array_artifact(rgb_path, root, rgb, Modality.RGB, "image/png"),
        depth=_array_artifact(depth_path, root, depth, Modality.DEPTH, "application/x-npy"),
        segmentation=_array_artifact(
            segmentation_path,
            root,
            segmentation,
            Modality.SURFACE_REGIONS,
            "application/x-npy",
        ),
        camera_world_transform=_json_artifact(
            camera_path,
            root,
            camera,
            Modality.CAMERA_WORLD_TRANSFORM,
            "application/json",
        ),
    )


def _generate_episode(
    root: Path,
    config: BenchmarkConfig,
    episode_index: int,
) -> EpisodeManifest:
    episode_id = f"episode-{episode_index:06d}"
    episode_seed = derive_seed(config.seed, f"episode:{episode_index}")
    episode_directory = root / "episodes" / episode_id
    episode_directory.mkdir(parents=True)
    rendered = render_transition(config)
    surfaces, references = _surface_references(config.seed, episode_index)
    before_segmentation = _remap_segmentation(
        rendered.before.raw_geom_segmentation,
        rendered.raw_geom_ids,
        references,
    )
    after_segmentation = _remap_segmentation(
        rendered.after.raw_geom_segmentation,
        rendered.raw_geom_ids,
        references,
    )
    camera_before = CameraInstrumentation(
        frame_index=0,
        camera_world_position=rendered.before.camera_world_position,
        camera_world_rotation_row_major=rendered.before.camera_world_rotation_row_major,
    )
    camera_after = CameraInstrumentation(
        frame_index=1,
        camera_world_position=rendered.after.camera_world_position,
        camera_world_rotation_row_major=rendered.after.camera_world_rotation_row_major,
    )
    before = _write_frame(
        root,
        episode_directory,
        0,
        rendered.before.rgb,
        rendered.before.depth,
        before_segmentation,
        camera_before,
    )
    after = _write_frame(
        root,
        episode_directory,
        1,
        rendered.after.rgb,
        rendered.after.depth,
        after_segmentation,
        camera_after,
    )
    visibility, correspondence, events = derive_visibility(
        before_segmentation,
        after_segmentation,
        surfaces,
    )
    boundaries = (
        derive_boundary_structure(before_segmentation, surfaces, 0),
        derive_boundary_structure(after_segmentation, surfaces, 1),
    )
    transition = TransitionRecord(
        schema_version="0.1.0",
        episode_id=episode_id,
        action=Action(**config.action.model_dump()),
        surfaces=surfaces,
        before=before,
        after=after,
        visibility_states=visibility,
        region_correspondence=correspondence,
        visibility_events=events,
        occlusion_relations=(
            OcclusionRelation(
                occluder_surface_id=references["occluding_surface"].surface_id,
                occluded_surface_id=references["background_surface"].surface_id,
                frame_indices=(0, 1),
            ),
        ),
        boundary_structures=boundaries,
        dense_optical_flow=UnavailableAnnotation(
            field="dense_optical_flow",
            status="unavailable",
            reason=(
                "Dense flow is remaining Gate 0B work; this vertical slice records exact "
                "region correspondence and visibility changes without fabricating flow values."
            ),
        ),
        ecological_label_sha256="0" * 64,
    )
    transition = TransitionRecord.model_validate(
        {
            **transition.model_dump(mode="python"),
            "ecological_label_sha256": compute_ecological_label_hash(transition),
        }
    )
    transition_path = episode_directory / "transition.json"
    write_canonical_json(transition_path, transition)

    instrumentation = PrivilegedInstrumentation(
        schema_version="0.1.0",
        episode_id=episode_id,
        appearance_variant=config.appearance.variant,
        raw_geom_ids=rendered.raw_geom_ids,
        raw_to_opaque_surface_ids={
            str(raw_id): references[name].surface_id
            for name, raw_id in rendered.raw_geom_ids.items()
        },
        raw_geom_world_positions=rendered.raw_geom_positions,
    )
    instrumentation_path = episode_directory / "instrumentation.json"
    write_canonical_json(instrumentation_path, instrumentation)
    return EpisodeManifest(
        episode_id=episode_id,
        episode_index=episode_index,
        episode_seed=episode_seed,
        transition=_json_artifact(
            transition_path,
            root,
            transition,
            Modality.VISIBILITY_EVENTS,
            "application/json",
        ),
        privileged_instrumentation=_json_artifact(
            instrumentation_path,
            root,
            instrumentation,
            Modality.PRIVILEGED_GENERATION_RECORDS,
            "application/json",
        ),
        ecological_label_sha256=transition.ecological_label_sha256,
        rgb_logical_sha256=(before.rgb.logical_sha256, after.rgb.logical_sha256),
    )


def _renderer_provenance() -> RendererProvenance:
    configured_backend = os.environ.get("MUJOCO_GL")
    if configured_backend is None:
        configured_backend = "wgl-default" if platform.system() == "Windows" else "platform-default"
    return RendererProvenance(
        mujoco_version=mujoco.__version__,
        numpy_version=np.__version__,
        renderer="mujoco.Renderer",
        backend=configured_backend,
        operating_system=platform.system(),
    )


def generate_dataset(config: BenchmarkConfig, episodes: int, output: Path) -> DatasetManifest:
    """Generate a new dataset directory, refusing to overwrite existing content."""

    if episodes < 1:
        raise ValueError("episodes must be at least one")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    resolved_config_path = output / "resolved_config.json"
    write_canonical_json(resolved_config_path, config)
    resolved_config_artifact = _json_artifact(
        resolved_config_path,
        output,
        config,
        Modality.PRIVILEGED_GENERATION_RECORDS,
        "application/json",
    )
    episode_manifests = tuple(
        _generate_episode(output, config, episode_index) for episode_index in range(episodes)
    )
    manifest = DatasetManifest(
        schema_version="0.1.0",
        generator_version="0.1.0",
        root_seed=config.seed,
        config_logical_sha256=sha256_bytes(canonical_json_bytes(config)),
        appearance_variant=config.appearance.variant,
        resolved_config=resolved_config_artifact,
        renderer_provenance=_renderer_provenance(),
        episodes=episode_manifests,
        dataset_logical_sha256="0" * 64,
    )
    manifest = DatasetManifest.model_validate(
        {
            **manifest.model_dump(mode="python"),
            "dataset_logical_sha256": compute_dataset_logical_hash(manifest),
        }
    )
    write_canonical_json(output / "manifest.json", manifest)
    volatile_metadata = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "hostname": socket.gethostname(),
        "python": sys.version,
        "generator_version": __version__,
    }
    (output / "run.json").write_text(
        json.dumps(volatile_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
