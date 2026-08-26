"""Fail-closed typed access to dataset modalities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image

from epsbench.data.paths import resolve_dataset_manifest
from epsbench.schema import (
    Action,
    CameraInstrumentation,
    CorridorInstrumentation,
    CorridorSampledGeometry,
    DatasetManifest,
    EcologicalTransitionView,
    EpisodeManifest,
    FrameRecord,
    Modality,
    ModalityPermissionSet,
    PrivilegedInstrumentation,
    SceneFamily,
    TransitionRecord,
    parse_privileged_instrumentation_json,
)


class PermissionDeniedError(PermissionError):
    """Raised before any unauthorised artifact is opened."""


class DatasetLoader:
    """Dataset reader that requires a declared permission set at construction."""

    def __init__(self, root: Path, permissions: ModalityPermissionSet) -> None:
        self.root, manifest_path = resolve_dataset_manifest(root)
        self.permissions = permissions
        self._manifest = DatasetManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )

    def _require(self, *modalities: Modality) -> None:
        denied = [modality for modality in modalities if not self.permissions.permits(modality)]
        if denied:
            names = ", ".join(modality.value for modality in denied)
            raise PermissionDeniedError(f"modality permission denied: {names}")

    def _path(self, relative_path: str) -> Path:
        candidate = (self.root / relative_path).resolve()
        if not candidate.is_relative_to(self.root):
            raise ValueError("artifact path escapes the dataset root")
        return candidate

    def _episode(self, episode_index: int) -> EpisodeManifest:
        try:
            return next(
                episode
                for episode in self._manifest.episodes
                if episode.episode_index == episode_index
            )
        except StopIteration as error:
            raise IndexError(f"episode index not found: {episode_index}") from error

    def _transition(self, episode_index: int) -> TransitionRecord:
        episode = self._episode(episode_index)
        return TransitionRecord.model_validate_json(
            self._path(episode.transition.path).read_text(encoding="utf-8")
        )

    @staticmethod
    def _frame(transition: TransitionRecord, frame_index: int) -> FrameRecord:
        if frame_index == 0:
            return transition.before
        if frame_index == 1:
            return transition.after
        raise IndexError(f"frame index must be 0 or 1, got {frame_index}")

    def read_action(self, episode_index: int) -> Action:
        self._require(Modality.EXECUTED_ACTION)
        return self._transition(episode_index).action

    def read_scene_family(self) -> SceneFamily:
        self._require(Modality.SCENE_FAMILY)
        return self._manifest.scene_family

    def read_dataset_manifest(self) -> DatasetManifest:
        """Return full control metadata only to explicitly privileged callers."""

        self._require(
            Modality.TRANSITION_RECORD,
            Modality.SCENE_FAMILY,
            Modality.PRIVILEGED_GENERATION_RECORDS,
        )
        return self._manifest.model_copy(deep=True)

    def read_ecological_transition(self, episode_index: int) -> EcologicalTransitionView:
        self._require(
            Modality.EXECUTED_ACTION,
            Modality.SURFACE_REGIONS,
            Modality.VISIBILITY_FRACTIONS,
            Modality.REGION_CORRESPONDENCE,
            Modality.REGION_MASK_CHANGES,
            Modality.ECOLOGICAL_VISIBILITY_EVENTS,
            Modality.OCCLUSION_ANNOTATION,
            Modality.BOUNDARY_STRUCTURE,
        )
        transition = self._transition(episode_index)
        return EcologicalTransitionView(
            episode_id=transition.episode_id,
            action=transition.action,
            surfaces=transition.surfaces,
            visibility_states=transition.visibility_states,
            region_correspondence=transition.region_correspondence,
            region_mask_changes=transition.region_mask_changes,
            ecological_visibility_events=transition.ecological_visibility_events,
            occlusion=transition.occlusion,
            boundary_structures=transition.boundary_structures,
            dense_optical_flow=transition.dense_optical_flow,
            ecological_label_sha256=transition.ecological_label_sha256,
        )

    def read_rgb(self, episode_index: int, frame_index: int) -> npt.NDArray[np.uint8]:
        self._require(Modality.RGB)
        transition = self._transition(episode_index)
        frame = self._frame(transition, frame_index)
        with Image.open(self._path(frame.rgb.path)) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()

    def read_depth(self, episode_index: int, frame_index: int) -> npt.NDArray[np.float32]:
        self._require(Modality.DEPTH)
        transition = self._transition(episode_index)
        frame = self._frame(transition, frame_index)
        return np.asarray(
            np.load(self._path(frame.depth.path), allow_pickle=False), dtype=np.float32
        )

    def read_segmentation(self, episode_index: int, frame_index: int) -> npt.NDArray[np.int32]:
        self._require(Modality.SURFACE_REGIONS)
        transition = self._transition(episode_index)
        frame = self._frame(transition, frame_index)
        return np.asarray(
            np.load(self._path(frame.segmentation.path), allow_pickle=False), dtype=np.int32
        )

    def read_camera_world_transform(
        self, episode_index: int, frame_index: int
    ) -> CameraInstrumentation:
        self._require(Modality.CAMERA_WORLD_TRANSFORM)
        transition = self._transition(episode_index)
        frame = self._frame(transition, frame_index)
        return CameraInstrumentation.model_validate_json(
            self._path(frame.camera_world_transform.path).read_text(encoding="utf-8")
        )

    def _instrumentation(self, episode_index: int) -> PrivilegedInstrumentation:
        episode = self._episode(episode_index)
        return parse_privileged_instrumentation_json(
            self._path(episode.privileged_instrumentation.path).read_text(encoding="utf-8")
        )

    def read_raw_mujoco_geom_ids(self, episode_index: int) -> dict[str, int]:
        self._require(Modality.MUJOCO_GEOM_IDS, Modality.PRIVILEGED_GENERATION_RECORDS)
        return dict(self._instrumentation(episode_index).raw_geom_ids)

    def read_raw_world_coordinates(
        self, episode_index: int
    ) -> dict[str, tuple[float, float, float]]:
        self._require(
            Modality.RAW_SIMULATOR_COORDINATES,
            Modality.PRIVILEGED_GENERATION_RECORDS,
        )
        return dict(self._instrumentation(episode_index).raw_geom_world_positions)

    def read_sampled_corridor_geometry(self, episode_index: int) -> CorridorSampledGeometry:
        self._require(
            Modality.SAMPLED_SCENE_GEOMETRY,
            Modality.PRIVILEGED_GENERATION_RECORDS,
        )
        instrumentation = self._instrumentation(episode_index)
        if not isinstance(instrumentation, CorridorInstrumentation):
            raise ValueError("sampled corridor geometry is unavailable for this scene family")
        return instrumentation.sampled_geometry

    def read_semantic_surface_names(self, episode_index: int) -> tuple[str, ...]:
        self._require(Modality.PRIVILEGED_GENERATION_RECORDS)
        instrumentation = self._instrumentation(episode_index)
        return tuple(instrumentation.raw_geom_ids)
