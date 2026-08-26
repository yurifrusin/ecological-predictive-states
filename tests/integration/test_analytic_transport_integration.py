from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from epsbench.config import CorridorConfig, SingleOccluderConfig
from epsbench.data import (
    DatasetLoader,
    DatasetValidationError,
    PermissionDeniedError,
    generate_dataset,
    validate_dataset,
)
from epsbench.data.identity import compute_analytic_transport_hash
from epsbench.schema import (
    AvailableDenseOpticalTransport,
    AvailableOcclusionAnnotation,
    Modality,
    ModalityPermissionSet,
    TransitionRecord,
    UnavailableOcclusionAnnotation,
)
from tests.dataset_mutations import (
    commit_episode_payloads,
    commit_raw_transition_payload,
    load_episode_payloads,
    load_manifest,
    rewrite_array_artifact,
)


def _transition(root: Path, episode_index: int = 0) -> TransitionRecord:
    manifest = load_manifest(root)
    return TransitionRecord.model_validate_json(
        (root / manifest.episodes[episode_index].transition.path).read_text(encoding="utf-8")
    )


def _available_transport(root: Path, episode_index: int = 0) -> AvailableDenseOpticalTransport:
    transport = _transition(root, episode_index).analytic_optical_transport
    assert isinstance(transport, AvailableDenseOpticalTransport)
    return transport


@pytest.mark.parametrize(
    "fixture_name",
    ["deterministic_datasets", "deterministic_corridor_datasets"],
)
def test_transport_artifacts_and_identities_are_byte_deterministic(
    fixture_name: str,
    request: pytest.FixtureRequest,
) -> None:
    first, second = request.getfixturevalue(fixture_name)
    assert isinstance(first, Path)
    assert isinstance(second, Path)
    for episode_index in (0, 1):
        first_transport = _available_transport(first, episode_index)
        second_transport = _available_transport(second, episode_index)
        assert first_transport.analytic_transport_sha256 == (
            second_transport.analytic_transport_sha256
        )
        for first_direction, second_direction in (
            (first_transport.forward, second_transport.forward),
            (first_transport.backward, second_transport.backward),
        ):
            for first_record, second_record in (
                (first_direction.vectors_fixed, second_direction.vectors_fixed),
                (first_direction.validity, second_direction.validity),
                (first_direction.reasons, second_direction.reasons),
            ):
                assert (first / first_record.path).read_bytes() == (
                    second / second_record.path
                ).read_bytes()


@pytest.mark.parametrize(
    "fixture_name",
    ["appearance_datasets", "corridor_appearance_datasets"],
)
def test_appearance_change_preserves_transport_artifacts_and_identity(
    fixture_name: str,
    request: pytest.FixtureRequest,
) -> None:
    base, alternate = request.getfixturevalue(fixture_name)
    assert isinstance(base, Path)
    assert isinstance(alternate, Path)
    base_transport = _available_transport(base)
    alternate_transport = _available_transport(alternate)
    assert base_transport.analytic_transport_sha256 == (
        alternate_transport.analytic_transport_sha256
    )
    for first, second in (
        (base_transport.forward, alternate_transport.forward),
        (base_transport.backward, alternate_transport.backward),
    ):
        assert first.vectors_fixed.logical_sha256 == second.vectors_fixed.logical_sha256
        assert first.validity.logical_sha256 == second.validity.logical_sha256
        assert first.reasons.logical_sha256 == second.reasons.logical_sha256


@pytest.mark.parametrize("fixture_name", ["smoke_dataset", "corridor_dataset"])
def test_opaque_surface_remapping_is_outside_transport_identity(
    fixture_name: str,
    request: pytest.FixtureRequest,
) -> None:
    root = request.getfixturevalue(fixture_name)
    assert isinstance(root, Path)
    transition = _transition(root)
    transport = _available_transport(root)
    changed_surfaces = tuple(
        surface.model_copy(update={"surface_id": f"surface-{index + 1:016x}"})
        for index, surface in enumerate(transition.surfaces)
    )
    changed = transition.model_copy(update={"surfaces": changed_surfaces})
    changed_transport = changed.analytic_optical_transport
    assert isinstance(changed_transport, AvailableDenseOpticalTransport)
    assert compute_analytic_transport_hash(changed_transport) == (
        transport.analytic_transport_sha256
    )


def test_seed_change_with_fixed_single_occluder_content_preserves_transport(
    benchmark_config: SingleOccluderConfig,
    smoke_dataset: Path,
    tmp_path: Path,
) -> None:
    changed_seed = benchmark_config.model_copy(update={"seed": benchmark_config.seed + 1})
    changed_path = tmp_path / "changed-seed"
    generate_dataset(changed_seed, 1, changed_path)
    assert _available_transport(smoke_dataset).analytic_transport_sha256 == (
        _available_transport(changed_path).analytic_transport_sha256
    )


def test_camera_action_or_geometry_change_changes_transport_identity(
    benchmark_config: SingleOccluderConfig,
    corridor_config: CorridorConfig,
    smoke_dataset: Path,
    corridor_dataset: Path,
    tmp_path: Path,
) -> None:
    changed_single = SingleOccluderConfig.model_validate(
        {
            **benchmark_config.model_dump(mode="python"),
            "camera": {
                **benchmark_config.camera.model_dump(mode="python"),
                "after_lateral": 0.45,
            },
            "action": {
                **benchmark_config.action.model_dump(mode="python"),
                "delta_lateral": 0.8,
            },
        }
    )
    changed_corridor = CorridorConfig.model_validate(
        {
            **corridor_config.model_dump(mode="python"),
            "action": {
                **corridor_config.action.model_dump(mode="python"),
                "delta_forward": 0.8,
            },
        }
    )
    changed_corridor_geometry = CorridorConfig.model_validate(
        {
            **corridor_config.model_dump(mode="python"),
            "geometry": {
                **corridor_config.geometry.model_dump(mode="python"),
                "width": {"minimum": 3.3, "maximum": 3.7},
            },
        }
    )
    changed_single_path = tmp_path / "changed-single"
    changed_corridor_path = tmp_path / "changed-corridor"
    changed_corridor_geometry_path = tmp_path / "changed-corridor-geometry"
    generate_dataset(changed_single, 1, changed_single_path)
    generate_dataset(changed_corridor, 1, changed_corridor_path)
    generate_dataset(changed_corridor_geometry, 1, changed_corridor_geometry_path)
    assert _available_transport(smoke_dataset).analytic_transport_sha256 != (
        _available_transport(changed_single_path).analytic_transport_sha256
    )
    assert _available_transport(corridor_dataset).analytic_transport_sha256 != (
        _available_transport(changed_corridor_path).analytic_transport_sha256
    )
    assert _available_transport(corridor_dataset).analytic_transport_sha256 != (
        _available_transport(changed_corridor_geometry_path).analytic_transport_sha256
    )


@pytest.mark.parametrize("fixture_name", ["smoke_dataset", "corridor_dataset"])
def test_loader_returns_complete_typed_transport_bundle(
    fixture_name: str,
    request: pytest.FixtureRequest,
) -> None:
    root = request.getfixturevalue(fixture_name)
    assert isinstance(root, Path)
    loader = DatasetLoader(root, ModalityPermissionSet.ecological_only())
    bundle = loader.read_analytic_optical_transport(0)
    rgb_shape = loader.read_segmentation(0, 0).shape
    assert bundle.forward_vectors_fixed.dtype == np.int32
    assert bundle.backward_vectors_fixed.dtype == np.int32
    assert bundle.forward_vectors_fixed.shape == (*rgb_shape, 2)
    assert bundle.backward_vectors_fixed.shape == (*rgb_shape, 2)
    assert bundle.forward_validity.dtype == bundle.forward_reasons.dtype == np.uint8
    assert bundle.backward_validity.dtype == bundle.backward_reasons.dtype == np.uint8
    assert bundle.fixed_point_scale == 1024
    assert np.array_equal(bundle.forward_validity == 1, bundle.forward_reasons == 0)
    assert np.array_equal(bundle.backward_validity == 1, bundle.backward_reasons == 0)
    assert np.all(bundle.forward_vectors_fixed[bundle.forward_validity == 0] == 0)
    assert np.all(bundle.backward_vectors_fixed[bundle.backward_validity == 0] == 0)


@pytest.mark.parametrize("fixture_name", ["smoke_dataset", "corridor_dataset"])
def test_transport_permission_is_denied_before_transition_or_artifact_access(
    fixture_name: str,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = request.getfixturevalue(fixture_name)
    assert isinstance(root, Path)
    permissions = ModalityPermissionSet(allowed=frozenset({Modality.EXECUTED_ACTION}))
    loader = DatasetLoader(root, permissions)

    def fail_if_transition_opened(episode_index: int) -> None:
        raise AssertionError(f"transition opened for episode {episode_index}")

    monkeypatch.setattr(loader, "_transition", fail_if_transition_opened)
    with pytest.raises(PermissionDeniedError, match="analytic_optical_transport"):
        loader.read_analytic_optical_transport(0)


@pytest.mark.parametrize("fixture_name", ["smoke_dataset", "corridor_dataset"])
def test_ecological_transport_view_has_no_metric_semantic_or_boundary_ownership_leakage(
    fixture_name: str,
    request: pytest.FixtureRequest,
) -> None:
    root = request.getfixturevalue(fixture_name)
    assert isinstance(root, Path)
    payload = (
        DatasetLoader(
            root,
            ModalityPermissionSet.ecological_only(),
        )
        .read_ecological_transition(0)
        .model_dump_json()
    )
    for forbidden in (
        "camera_world",
        "raw_geom",
        "world_position",
        "sampled_geometry",
        "support_surface",
        "corridor_floor",
        "boundary_ownership_map",
        "owned_boundary",
    ):
        assert forbidden not in payload


@pytest.mark.parametrize("fixture_name", ["smoke_dataset", "corridor_dataset"])
@pytest.mark.parametrize(
    "corruption",
    [
        "valid_vector",
        "invalid_nonzero_vector",
        "validity",
        "reason",
        "unknown_reason",
        "scale",
        "swapped_directions",
        "foreign_episode_artifact",
    ],
)
def test_hash_rebuilt_transport_corruption_is_rejected(
    fixture_name: str,
    corruption: str,
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    source = request.getfixturevalue(fixture_name)
    assert isinstance(source, Path)
    broken = tmp_path / f"{fixture_name}-{corruption}"
    shutil.copytree(source, broken)
    transition, instrumentation = load_episode_payloads(broken, 0)
    transport = transition["analytic_optical_transport"]
    forward = transport["forward"]

    if corruption in {"valid_vector", "invalid_nonzero_vector"}:
        record = _artifact(forward["vectors_fixed"])
        vectors = np.load(broken / record.path, allow_pickle=False)
        validity = np.load(broken / forward["validity"]["path"], allow_pickle=False)
        desired = 1 if corruption == "valid_vector" else 0
        row, column = np.argwhere(validity == desired)[0]
        vectors[row, column, 0] += 1
        forward["vectors_fixed"] = rewrite_array_artifact(
            broken,
            record,
            vectors,
        ).model_dump(mode="json")
    elif corruption == "validity":
        record = _artifact(forward["validity"])
        validity = np.load(broken / record.path, allow_pickle=False)
        row, column = np.argwhere(validity == 1)[0]
        validity[row, column] = 0
        forward["validity"] = rewrite_array_artifact(
            broken,
            record,
            validity,
        ).model_dump(mode="json")
    elif corruption in {"reason", "unknown_reason"}:
        record = _artifact(forward["reasons"])
        reasons = np.load(broken / record.path, allow_pickle=False)
        row, column = np.argwhere(reasons == 0)[0]
        reasons[row, column] = 3 if corruption == "reason" else 255
        forward["reasons"] = rewrite_array_artifact(
            broken,
            record,
            reasons,
        ).model_dump(mode="json")
    elif corruption == "scale":
        transport["quantisation"]["fixed_point_scale"] = 2048
        commit_raw_transition_payload(broken, 0, transition)
        with pytest.raises(DatasetValidationError, match="transition failed schema validation"):
            validate_dataset(broken)
        return
    elif corruption == "swapped_directions":
        backward = transport["backward"]
        for field in ("vectors_fixed", "validity", "reasons"):
            forward[field], backward[field] = backward[field], forward[field]
    else:
        second_transition = _transition(broken, 1).model_dump(mode="json")
        forward["vectors_fixed"] = second_transition["analytic_optical_transport"]["forward"][
            "vectors_fixed"
        ]

    commit_episode_payloads(broken, 0, transition, instrumentation)
    with pytest.raises(DatasetValidationError):
        validate_dataset(broken)


def _artifact(payload: dict[str, Any]) -> Any:
    from epsbench.schema import ArtifactRecord

    return ArtifactRecord.model_validate(payload, strict=False)


@pytest.mark.parametrize("fixture_name", ["smoke_dataset", "corridor_dataset"])
def test_transport_slice_preserves_unavailable_claims_and_existing_occlusion_posture(
    fixture_name: str,
    request: pytest.FixtureRequest,
) -> None:
    root = request.getfixturevalue(fixture_name)
    assert isinstance(root, Path)
    transition = _transition(root)
    assert transition.ecological_visibility_events.status == "unavailable"
    assert transition.ecological_visibility_events.reason_category == (
        "oriented_boundary_ownership_unavailable"
    )
    if fixture_name == "smoke_dataset":
        assert isinstance(transition.occlusion, AvailableOcclusionAnnotation)
    else:
        assert isinstance(transition.occlusion, UnavailableOcclusionAnnotation)
    for change in transition.region_mask_changes:
        assert set(change.model_dump()) == {
            "surface_id",
            "change",
            "affected_image_pixels",
        }
    payload = transition.model_dump_json()
    assert "boundary_ownership_map" not in payload
    assert "owned_boundary" not in payload
