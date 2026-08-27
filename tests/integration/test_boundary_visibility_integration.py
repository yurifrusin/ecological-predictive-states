from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from epsbench.config import SingleOccluderConfig
from epsbench.data import (
    DatasetLoader,
    DatasetValidationError,
    PermissionDeniedError,
    generate_dataset,
    validate_dataset,
)
from epsbench.schema import (
    AvailableOcclusionAnnotation,
    BoundaryKind,
    Modality,
    ModalityPermissionSet,
    TransitionRecord,
)
from epsbench.sim import compute_single_occluder_boundary_visibility
from tests.dataset_mutations import (
    commit_declared_boundary_identity_corruption,
    commit_declared_event_identity_corruption,
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


def _artifact(payload: dict[str, Any]) -> Any:
    from epsbench.schema import ArtifactRecord

    return ArtifactRecord.model_validate(payload, strict=False)


def test_single_occluder_boundary_owner_and_events_match_counterfactual_relation(
    smoke_dataset: Path,
) -> None:
    transition = _transition(smoke_dataset)
    boundary = transition.oriented_boundary_ownership
    assert boundary.status == "available"
    assert all(item.kind != BoundaryKind.UNRESOLVED_BOUNDARY for item in boundary.elements)
    assert any(item.kind == BoundaryKind.OCCLUDING_CONTOUR for item in boundary.elements)
    assert any(item.kind == BoundaryKind.ATTACHED_JUNCTION for item in boundary.elements)
    assert isinstance(transition.occlusion, AvailableOcclusionAnnotation)
    relation = transition.occlusion.relations[0]
    owned_pairs = {
        (
            item.owner_surface_id,
            item.positive_surface_id
            if item.owner_surface_id == item.negative_surface_id
            else item.negative_surface_id,
            item.frame_index,
        )
        for item in boundary.elements
        if item.kind == BoundaryKind.OCCLUDING_CONTOUR
    }
    assert all(
        (
            relation.occluder_surface_id,
            relation.occluded_surface_id,
            frame_index,
        )
        in owned_pairs
        for frame_index in relation.frame_indices
    )
    summaries = transition.ecological_visibility_events.occluding_event_summaries
    assert {item.kind for item in summaries} == {"accretion", "deletion"}
    assert all(item.owner_surface_id == relation.occluder_surface_id for item in summaries)
    assert all(item.affected_surface_id == relation.occluded_surface_id for item in summaries)
    assert all(item.pixel_count > 0 for item in summaries)


def test_reversed_lateral_action_reverses_accretion_and_deletion_sides(
    benchmark_config: SingleOccluderConfig,
) -> None:
    reversed_config = SingleOccluderConfig.model_validate(
        {
            **benchmark_config.model_dump(mode="python"),
            "camera": {
                **benchmark_config.camera.model_dump(mode="python"),
                "before_lateral": benchmark_config.camera.after_lateral,
                "after_lateral": benchmark_config.camera.before_lateral,
            },
            "action": {
                "name": "lateral_left",
                "delta_forward": 0.0,
                "delta_lateral": -benchmark_config.action.delta_lateral,
                "delta_yaw": 0.0,
            },
        }
    )
    forward = compute_single_occluder_boundary_visibility(benchmark_config)
    reversed_analysis = compute_single_occluder_boundary_visibility(reversed_config)
    forward_deletion_column = np.argwhere(forward.before_fate_codes == 1)[:, 1].mean()
    forward_accretion_column = np.argwhere(forward.after_origin_codes == 1)[:, 1].mean()
    reverse_deletion_column = np.argwhere(reversed_analysis.before_fate_codes == 1)[:, 1].mean()
    reverse_accretion_column = np.argwhere(reversed_analysis.after_origin_codes == 1)[:, 1].mean()
    assert forward_deletion_column < forward_accretion_column
    assert reverse_deletion_column > reverse_accretion_column
    assert forward_deletion_column == pytest.approx(reverse_accretion_column)
    assert forward_accretion_column == pytest.approx(reverse_deletion_column)


def test_corridor_attached_seams_have_no_occluding_contour_or_causal_events(
    corridor_dataset: Path,
) -> None:
    transition = _transition(corridor_dataset)
    assert all(
        item.kind != BoundaryKind.OCCLUDING_CONTOUR
        for item in transition.oriented_boundary_ownership.elements
    )
    assert any(
        item.kind == BoundaryKind.ATTACHED_JUNCTION
        for item in transition.oriented_boundary_ownership.elements
    )
    assert transition.ecological_visibility_events.occluding_event_summaries == ()
    assert isinstance(transition.occlusion, AvailableOcclusionAnnotation)
    assert transition.occlusion.oracle_rule == "oriented_boundary_ownership_v1"
    assert transition.occlusion.relations == ()


@pytest.mark.parametrize(
    "fixture_name",
    ["appearance_datasets", "corridor_appearance_datasets"],
)
def test_appearance_changes_preserve_boundary_and_event_identities(
    fixture_name: str,
    request: pytest.FixtureRequest,
) -> None:
    base, alternate = request.getfixturevalue(fixture_name)
    assert isinstance(base, Path)
    assert isinstance(alternate, Path)
    base_manifest = validate_dataset(base)
    alternate_manifest = validate_dataset(alternate)
    assert base_manifest.episodes[0].oriented_boundary_sha256 == (
        alternate_manifest.episodes[0].oriented_boundary_sha256
    )
    assert base_manifest.episodes[0].visibility_event_sha256 == (
        alternate_manifest.episodes[0].visibility_event_sha256
    )


def test_opaque_remapping_changes_public_boundary_and_event_identity_but_not_transport(
    benchmark_config: SingleOccluderConfig,
    smoke_dataset: Path,
    tmp_path: Path,
) -> None:
    changed = benchmark_config.model_copy(update={"seed": benchmark_config.seed + 1})
    changed_root = tmp_path / "changed-remapping"
    generate_dataset(changed, 1, changed_root)
    baseline = load_manifest(smoke_dataset).episodes[0]
    remapped = load_manifest(changed_root).episodes[0]
    assert baseline.analytic_transport_sha256 == remapped.analytic_transport_sha256
    assert baseline.oriented_boundary_sha256 != remapped.oriented_boundary_sha256
    assert baseline.visibility_event_sha256 != remapped.visibility_event_sha256


@pytest.mark.parametrize(
    ("method", "modality"),
    [
        ("read_oriented_boundaries", Modality.ORIENTED_BOUNDARY_OWNERSHIP),
        ("read_ecological_visibility_events", Modality.ECOLOGICAL_VISIBILITY_EVENTS),
    ],
)
def test_boundary_and_event_permissions_fail_before_transition_access(
    smoke_dataset: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    modality: Modality,
) -> None:
    loader = DatasetLoader(
        smoke_dataset,
        ModalityPermissionSet(allowed=frozenset({Modality.EXECUTED_ACTION})),
    )

    def fail_if_opened(episode_index: int) -> None:
        raise AssertionError(f"transition opened for episode {episode_index}")

    monkeypatch.setattr(loader, "_transition", fail_if_opened)
    with pytest.raises(PermissionDeniedError, match=modality.value):
        getattr(loader, method)(0)


def test_ecological_boundary_and_event_bundle_has_no_semantic_metric_or_raw_leakage(
    smoke_dataset: Path,
) -> None:
    payload = (
        DatasetLoader(smoke_dataset, ModalityPermissionSet.ecological_only())
        .read_ecological_transition(0)
        .model_dump_json()
    )
    for forbidden in (
        "support_surface",
        "occluding_surface",
        "background_surface",
        "raw_geom",
        "world_position",
        "axis_interval_gaps",
        "contact_tolerance",
        "camera_world",
        "depth_",
    ):
        assert forbidden not in payload


@pytest.mark.parametrize(
    "corruption",
    [
        "boundary_owner",
        "boundary_kind",
        "event_code",
        "event_unknown_code",
        "event_owner",
        "event_summary",
    ],
)
def test_fully_rehashed_boundary_and_event_corruption_is_rejected(
    smoke_dataset: Path,
    tmp_path: Path,
    corruption: str,
) -> None:
    broken = tmp_path / corruption
    shutil.copytree(smoke_dataset, broken)
    transition, instrumentation = load_episode_payloads(broken, 0)
    if corruption in {"boundary_owner", "boundary_kind"}:
        element = next(
            item
            for item in transition["oriented_boundary_ownership"]["elements"]
            if item["kind"] == "occluding_contour"
        )
        if corruption == "boundary_owner":
            if element["owner_side"] == "negative_axis_side":
                element["owner_side"] = "positive_axis_side"
                element["owner_surface_id"] = element["positive_surface_id"]
            else:
                element["owner_side"] = "negative_axis_side"
                element["owner_surface_id"] = element["negative_surface_id"]
        else:
            element["kind"] = "attached_junction"
            element["owner_side"] = "none"
            element["owner_surface_id"] = None
    elif corruption in {"event_code", "event_unknown_code", "event_owner"}:
        direction = transition["ecological_visibility_events"]["before_fate"]
        field = (
            "event_codes"
            if corruption in {"event_code", "event_unknown_code"}
            else "owner_surface_labels"
        )
        record = _artifact(direction[field])
        array = np.load(broken / record.path, allow_pickle=False)
        codes = np.load(broken / direction["event_codes"]["path"], allow_pickle=False)
        row, column = np.argwhere(codes == 1)[0]
        if corruption in {"event_code", "event_unknown_code"}:
            array[row, column] = 0 if corruption == "event_code" else 255
        else:
            array[row, column] = transition["surfaces"][0]["segmentation_label"]
        direction[field] = rewrite_array_artifact(
            broken,
            record,
            array,
        ).model_dump(mode="json")
    else:
        transition["ecological_visibility_events"]["occluding_event_summaries"][0][
            "pixel_count"
        ] += 1
    commit_episode_payloads(broken, 0, transition, instrumentation)
    with pytest.raises(DatasetValidationError):
        validate_dataset(broken)


@pytest.mark.parametrize("field", ["oriented_boundary_ownership", "ecological_visibility_events"])
def test_fully_rehashed_boundary_or_event_method_corruption_is_rejected(
    smoke_dataset: Path,
    tmp_path: Path,
    field: str,
) -> None:
    broken = tmp_path / f"{field}-method"
    shutil.copytree(smoke_dataset, broken)
    transition, _ = load_episode_payloads(broken, 0)
    transition[field]["method"] = "fabricated_method_v999"
    commit_raw_transition_payload(broken, 0, transition)
    with pytest.raises(DatasetValidationError):
        validate_dataset(broken)


@pytest.mark.parametrize("identity", ["boundary", "event"])
def test_fully_rehashed_declared_boundary_or_event_identity_corruption_is_rejected(
    smoke_dataset: Path,
    tmp_path: Path,
    identity: str,
) -> None:
    broken = tmp_path / f"{identity}-identity"
    shutil.copytree(smoke_dataset, broken)
    if identity == "boundary":
        commit_declared_boundary_identity_corruption(broken, 0, "1" * 64)
    else:
        commit_declared_event_identity_corruption(broken, 0, "2" * 64)
    with pytest.raises(DatasetValidationError, match="identity"):
        validate_dataset(broken)


def test_fully_rehashed_attachment_geometry_evidence_corruption_is_rejected(
    smoke_dataset: Path,
    tmp_path: Path,
) -> None:
    broken = tmp_path / "attachment-evidence"
    shutil.copytree(smoke_dataset, broken)
    transition, instrumentation = load_episode_payloads(broken, 0)
    instrumentation["attachment_contract"]["pair_evidence"][0]["axis_interval_gaps"][0] += 0.01
    commit_episode_payloads(broken, 0, transition, instrumentation)
    with pytest.raises(DatasetValidationError, match="attachment contract"):
        validate_dataset(broken)
