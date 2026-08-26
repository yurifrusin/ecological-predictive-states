from pathlib import Path

import pytest

from epsbench.data import DatasetLoader, PermissionDeniedError
from epsbench.schema import Modality, ModalityPermissionSet


def test_ecological_permission_set_excludes_metric_and_instrumentation() -> None:
    permissions = ModalityPermissionSet.ecological_only()
    assert permissions.permits(Modality.ANALYTIC_OPTICAL_TRANSPORT)
    assert permissions.permits(Modality.REGION_MASK_CHANGES)
    assert permissions.permits(Modality.ECOLOGICAL_VISIBILITY_EVENTS)
    assert permissions.permits(Modality.OCCLUSION_ANNOTATION)
    assert permissions.permits(Modality.EXECUTED_ACTION)
    assert not permissions.permits(Modality.DEPTH)
    assert not permissions.permits(Modality.CAMERA_WORLD_TRANSFORM)
    assert not permissions.permits(Modality.MUJOCO_GEOM_IDS)
    assert not permissions.permits(Modality.RAW_SIMULATOR_COORDINATES)
    assert not permissions.permits(Modality.SAMPLED_SCENE_GEOMETRY)
    assert not permissions.permits(Modality.TRANSITION_RECORD)
    assert not permissions.permits(Modality.SCENE_FAMILY)


def test_ecological_loader_denies_depth_camera_raw_ids_and_coordinates(
    smoke_dataset: Path,
) -> None:
    loader = DatasetLoader(smoke_dataset, ModalityPermissionSet.ecological_only())
    ecological_view = loader.read_ecological_transition(0)
    ecological_payload = ecological_view.model_dump(mode="json")
    assert "depth" not in ecological_payload
    assert "camera_world_transform" not in ecological_payload
    assert "raw_geom_ids" not in ecological_payload
    assert "raw_geom_world_positions" not in ecological_payload
    assert "occlusion_oracle" not in ecological_payload
    loader.read_action(0)
    with pytest.raises(PermissionDeniedError):
        loader.read_depth(0, 0)
    with pytest.raises(PermissionDeniedError):
        loader.read_camera_world_transform(0, 0)
    with pytest.raises(PermissionDeniedError):
        loader.read_raw_mujoco_geom_ids(0)
    with pytest.raises(PermissionDeniedError):
        loader.read_raw_world_coordinates(0)
