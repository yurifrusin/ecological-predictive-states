"""Synthetic CPU checks for paired artifacts, identity domains, and typed access."""

from __future__ import annotations

import copy
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

from epsbench.data.generate import _write_frame, generate_dataset
from epsbench.data.identity import compute_canonical_paired_endpoint_hash, dataset_logical_domain
from epsbench.data.loader import DatasetLoader, PermissionDeniedError
from epsbench.data.provenance import collect_source_provenance
from epsbench.schema import (
    ArtifactRecord,
    CameraInstrumentation,
    CanonicalPairedEpisodeManifest,
    CanonicalPairedFrameRecord,
    CanonicalPairedOutputProvenance,
    DatasetManifest,
    EpisodeManifest,
    FrameRecord,
    Modality,
    ModalityPermissionSet,
    RendererProvenance,
    SceneFamily,
)
from epsbench.utils.canonical import canonical_json_bytes
from tests.unit.test_canonical_paired import capture_fake


@pytest.fixture
def written_frame(
    tmp_path: Path,
) -> tuple[Path, CanonicalPairedFrameRecord, list[dict[str, object]]]:
    _, pair = capture_fake()
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    segmentation = np.array([[456, 123], [123, 0]], dtype=np.int32)
    episode = tmp_path / "episode-000000"
    episode.mkdir()
    events: list[dict[str, object]] = []
    frame = _write_frame(
        tmp_path,
        episode,
        0,
        rgb,
        pair.depth,
        segmentation,
        CameraInstrumentation(
            frame_index=0,
            camera_world_position=(0.0, 0.0, 0.0),
            camera_world_rotation_row_major=tuple(np.eye(3).reshape(-1)),
        ),
        canonical_pair=pair,
        pair_operational_events=events,
        scene_family=SceneFamily.CORRIDOR,
        episode_seed=123,
        config_logical_sha256="1" * 64,
        scene_content_sha256="2" * 64,
        episode_id="episode-000000",
        raw_to_opaque_surface_ids={
            "7": "surface-123456789abcdef0",
            "9": "surface-23456789abcdef01",
        },
        opaque_surface_labels={"surface-123456789abcdef0": 123, "surface-23456789abcdef01": 456},
        source_provenance_sha256="3" * 64,
        renderer_execution_provenance_sha256="4" * 64,
    )
    assert isinstance(frame, CanonicalPairedFrameRecord)
    return tmp_path, frame, events


def _provenance(root: Path, frame: CanonicalPairedFrameRecord) -> CanonicalPairedOutputProvenance:
    return CanonicalPairedOutputProvenance.model_validate_json(
        (root / frame.paired_output_provenance.path).read_bytes()
    )


def test_canonical_frame_preserves_pair_fields_through_union_and_round_trip(
    written_frame: Any,
) -> None:
    root, frame, events = written_frame
    adapter = TypeAdapter(FrameRecord | CanonicalPairedFrameRecord)
    payload = adapter.dump_json(frame)
    assert b"paired_output_provenance" in payload
    parsed = adapter.validate_json(payload)
    assert isinstance(parsed, CanonicalPairedFrameRecord)
    assert canonical_json_bytes(parsed) == canonical_json_bytes(frame)
    provenance = _provenance(root, frame)
    assert compute_canonical_paired_endpoint_hash(provenance) == provenance.endpoint_logical_sha256
    assert (
        len(events) == 1
        and events[0]["endpoint_logical_sha256"] == provenance.endpoint_logical_sha256
    )
    assert provenance.rgb_provenance.association == "separate_draw_no_id_depth_correspondence"
    assert provenance.counterfactual_provenance is None
    assert "actual_current_context" not in (root / provenance.producer_state.path).read_text()
    assert events[0]["observations"]["draw_output"]["actual_current_context"] == 77


def test_legacy_frame_has_no_paired_field_or_retroactive_claim(written_frame: Any) -> None:
    _, frame, _ = written_frame
    payload = frame.model_dump(mode="python")
    del payload["paired_output_provenance"]
    old = FrameRecord.model_validate(payload)
    adapter = TypeAdapter(FrameRecord | CanonicalPairedFrameRecord)
    assert adapter.dump_json(old) == old.model_dump_json().encode()
    assert type(adapter.validate_json(adapter.dump_json(old))) is FrameRecord
    assert "paired_output_provenance" not in old.model_dump()


@pytest.mark.parametrize(
    "remove",
    ["scene_map", "native_id_rgb", "producer_state", "source_provenance_sha256", "rgb_provenance"],
)
def test_required_provenance_cannot_be_omitted(written_frame: Any, remove: str) -> None:
    root, frame, _ = written_frame
    value = _provenance(root, frame).model_dump(mode="python")
    del value[remove]
    with pytest.raises(ValidationError):
        CanonicalPairedOutputProvenance.model_validate(value)


@pytest.mark.parametrize(
    "mutation",
    [
        "raw_key",
        "raw_alias",
        "non_geom",
        "negative_label",
        "duplicate_label",
        "wrong_modality",
        "wrong_shape",
        "unbound_map",
        "rgb_role",
        "counterfactual_role",
    ],
)
def test_malformed_pair_metadata_is_rejected(written_frame: Any, mutation: str) -> None:
    root, frame, _ = written_frame
    value = _provenance(root, frame).model_dump(mode="python")
    if mutation == "raw_key":
        value["raw_to_opaque_surface_ids"]["07"] = value["raw_to_opaque_surface_ids"].pop("7")
    elif mutation == "raw_alias":
        value["raw_to_opaque_surface_ids"]["9"] = value["raw_to_opaque_surface_ids"]["7"]
    elif mutation == "non_geom":
        value["scene_map"][0]["objtype"] = 1
    elif mutation == "negative_label":
        value["opaque_surface_labels"]["surface-123456789abcdef0"] = -1
    elif mutation == "duplicate_label":
        value["opaque_surface_labels"]["surface-123456789abcdef0"] = 456
    elif mutation == "wrong_modality":
        value["native_id_rgb"]["modality"] = Modality.RGB.value
    elif mutation == "wrong_shape":
        value["native_id_rgb"]["shape"] = [2, 2, 4]
    elif mutation == "unbound_map":
        value["scene_map"][0]["objid"] = 80
    elif mutation == "rgb_role":
        value["rgb_provenance"]["producer"] = "single_occluder_counterfactual_segmentation"
    elif mutation == "counterfactual_role":
        value["counterfactual_provenance"] = copy.deepcopy(value["rgb_provenance"])
    with pytest.raises(ValidationError):
        CanonicalPairedOutputProvenance.model_validate(value)


@pytest.mark.parametrize(
    "missing",
    [None, Modality.DEPTH, Modality.MUJOCO_GEOM_IDS, Modality.PRIVILEGED_GENERATION_RECORDS],
)
def test_permissions_are_denied_before_transition_or_path_access(missing: Modality | None) -> None:
    loader = object.__new__(DatasetLoader)
    all_required = {
        Modality.DEPTH,
        Modality.MUJOCO_GEOM_IDS,
        Modality.PRIVILEGED_GENERATION_RECORDS,
    }
    loader.permissions = (
        ModalityPermissionSet(allowed=frozenset(all_required - {missing}))
        if missing
        else ModalityPermissionSet.ecological_only()
    )

    def forbidden(*_args: object) -> None:
        raise AssertionError("transition/path access occurred before permission denial")

    loader._transition = forbidden
    loader._load_npy = forbidden
    loader._load_json = forbidden
    with pytest.raises(PermissionDeniedError):
        loader.read_canonical_paired_output(999, 999)


def _loader(root: Path, frame: CanonicalPairedFrameRecord) -> DatasetLoader:
    loader = object.__new__(DatasetLoader)
    loader.root = root
    loader.permissions = ModalityPermissionSet.all_modalities()
    loader._transition = lambda _: SimpleNamespace(
        episode_id="episode-000000", before=frame, after=frame
    )
    return loader


def test_authorized_reader_uses_owned_artifacts_and_returns_retained_arrays(
    written_frame: Any,
) -> None:
    root, frame, _ = written_frame
    pair = _loader(root, frame).read_canonical_paired_output(0, 0)
    assert pair.native_id_rgb.dtype == np.uint8
    assert pair.native_depth_pre_metric.dtype == np.float32
    assert pair.provenance.association == "shared_raster_id_depth"
    assert pair.producer_state["offSamples"] == 0


def test_authorized_reader_rejects_changed_native_file(written_frame: Any) -> None:
    root, frame, _ = written_frame
    pair = _provenance(root, frame)
    path = root / pair.native_depth_pre_metric.path
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="hash mismatch"):
        _loader(root, frame).read_canonical_paired_output(0, 0)


def manifest_value(*, paired: bool) -> dict[str, Any]:
    def artifact(modality: Modality, path: str) -> ArtifactRecord:
        return ArtifactRecord(
            path=path,
            modality=modality,
            media_type="application/json",
            dtype="json",
            shape=(1,),
            logical_sha256="a" * 64,
            file_sha256="b" * 64,
            byte_count=1,
        )

    episode: dict[str, Any] = dict(
        episode_id="episode-000000",
        episode_index=0,
        episode_seed=123,
        transition=artifact(Modality.TRANSITION_RECORD, "transition.json"),
        privileged_instrumentation=artifact(
            Modality.PRIVILEGED_GENERATION_RECORDS, "instrumentation.json"
        ),
        rgb_logical_sha256=("a" * 64, "b" * 64),
    )
    for key in (
        "scene_content_sha256",
        "ecological_label_sha256",
        "analytic_transport_sha256",
        "oriented_boundary_sha256",
        "visibility_event_sha256",
        "appearance_instance_sha256",
    ):
        episode[key] = "c" * 64
    if paired:
        episode["paired_output_provenance_sha256"] = ("d" * 64, "e" * 64)
    return dict(
        schema_version="0.1.0-dev.11" if paired else "0.1.0-dev.7",
        generator_version="0.1.0",
        scene_family=SceneFamily.CORRIDOR,
        root_seed=1729,
        config_logical_sha256="a" * 64,
        appearance_registry_sha256="a" * 64,
        appearance_profile_id="legacy_solid_base_v1",
        appearance_profile_sha256="a" * 64,
        appearance_registry_snapshot=artifact(Modality.APPEARANCE_CONTROL, "appearances.json"),
        evaluation_seed_registry_sha256="a" * 64,
        evaluation_seed_registry_snapshot=artifact(Modality.APPEARANCE_CONTROL, "seeds.json"),
        appearance_assignment_schedule_source="snapshotted_evaluation_seed_registry_v1",
        resolved_config=artifact(Modality.PRIVILEGED_GENERATION_RECORDS, "config.json"),
        renderer_provenance=RendererProvenance(
            mujoco_version="3.12.0",
            numpy_version="2.4.6",
            renderer="mujoco.Renderer",
            backend="osmesa",
            operating_system="Linux",
        ),
        renderer_execution_provenance_sha256="a" * 64,
        episodes=(episode,),
        dataset_logical_sha256="a" * 64,
        source_provenance=collect_source_provenance(Path.cwd()),
        source_provenance_sha256="a" * 64,
        content_provenance_binding_sha256="a" * 64,
    )


@pytest.mark.parametrize("version", ["0.1.0-dev.7", "0.1.0-dev.8", "0.1.0-dev.9", "0.1.0-dev.10"])
def test_old_dataset_schemas_preserve_serialization_and_identity_without_pairs(
    version: str,
) -> None:
    value = manifest_value(paired=False)
    value["schema_version"] = version
    if version in {"0.1.0-dev.8", "0.1.0-dev.10"}:
        value["appearance_assignment_schedule_source"] = (
            "snapshotted_revision_partition_seed_registry_v1"
        )
    manifest = DatasetManifest.model_validate(value)
    assert type(manifest.episodes[0]) is EpisodeManifest
    assert "paired_output_provenance_sha256" not in dataset_logical_domain(manifest)["episodes"][0]
    assert b"paired_output" not in canonical_json_bytes(manifest)
    assert canonical_json_bytes(
        DatasetManifest.model_validate_json(canonical_json_bytes(manifest))
    ) == canonical_json_bytes(manifest)


def test_new_dataset_requires_pair_identity_and_roundtrips_child_episode() -> None:
    value = manifest_value(paired=True)
    manifest = DatasetManifest.model_validate(value)
    assert isinstance(manifest.episodes[0], CanonicalPairedEpisodeManifest)
    assert dataset_logical_domain(manifest)["episodes"][0]["paired_output_provenance_sha256"] == [
        "d" * 64,
        "e" * 64,
    ]
    parsed = DatasetManifest.model_validate_json(canonical_json_bytes(manifest))
    assert isinstance(parsed.episodes[0], CanonicalPairedEpisodeManifest)
    assert canonical_json_bytes(parsed) == canonical_json_bytes(manifest)
    del value["episodes"][0]["paired_output_provenance_sha256"]
    with pytest.raises(ValidationError, match="paired provenance"):
        DatasetManifest.model_validate(value)


def test_old_dataset_cannot_gain_pair_provenance_by_relabeling_episode() -> None:
    value = manifest_value(paired=True)
    value["schema_version"] = "0.1.0-dev.7"
    with pytest.raises(ValidationError, match="paired provenance"):
        DatasetManifest.model_validate(value)


def test_default_capture_remains_legacy() -> None:
    assert inspect.signature(generate_dataset).parameters["capture_mode"].default == "legacy"
