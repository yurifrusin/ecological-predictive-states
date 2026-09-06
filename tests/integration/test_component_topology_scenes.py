"""Ordinary development seeds only; complete reseals still require public reconstruction."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from epsbench.annotations.component_topology import domain_hash, seal_annotation
from epsbench.config import AppearanceConfig, load_config
from epsbench.data import DatasetLoader, create_inspection_image, generate_dataset, validate_dataset
from epsbench.data.identity import compute_ecological_label_hash, compute_visibility_event_hash
from epsbench.data.loader import PermissionDeniedError
from epsbench.schema import (
    ArtifactRecord,
    ComponentTopologyAnnotation,
    Modality,
    ModalityPermissionSet,
)
from epsbench.utils.canonical import canonical_json_bytes, sha256_file
from tests.dataset_mutations import (
    _write_manifest,
    commit_episode_payloads,
    load_episode_payloads,
    load_manifest,
    rewrite_array_artifact,
    rewrite_json_artifact,
)


@pytest.fixture(scope="module", params=["benchmark", "corridor"])
def topology_dataset(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    output = tmp_path_factory.mktemp("component-topology") / str(request.param)
    generate_dataset(
        load_config(Path(f"configs/{request.param}_v0.yaml")), 2, output, component_topology=True
    )
    return output


def test_complete_scene_reconstruction_and_identity_chain(topology_dataset: Path) -> None:
    manifest = validate_dataset(topology_dataset)
    loader = DatasetLoader(topology_dataset, ModalityPermissionSet.ecological_only())
    for episode in manifest.episodes:
        bundle = loader.read_component_topology(episode.episode_index)
        topology = bundle.annotation
        assert topology == seal_annotation(topology)
        assert sum(c.pixel_count for c in topology.frames[0].components) == int(
            np.count_nonzero(loader.read_segmentation(episode.episode_index, 0))
        )
        assert len(topology.events) > 0
        assert sorted(
            i for e in topology.events for i in (*e.before_component_ids, *e.after_component_ids)
        ) == sorted(c.component_id for f in topology.frames for c in f.components)
        assert len(
            np.unique(bundle.before_component_labels[bundle.before_component_labels > 0])
        ) == len(topology.frames[0].components)
    assert (
        loader.read_component_topology(0).annotation != loader.read_component_topology(1).annotation
    )


def test_appearance_and_repeat_invariance(topology_dataset: Path, tmp_path: Path) -> None:
    manifest = load_manifest(topology_dataset)
    config = load_config(
        Path(
            "configs/benchmark_v0.yaml"
            if manifest.scene_family.value == "single_occluder"
            else "configs/corridor_v0.yaml"
        )
    )
    alternate = config.model_copy(
        update={
            "appearance": AppearanceConfig(
                registry_version="appearance_candidate_registry_v1",
                profile_id="legacy_solid_alternate_v1",
            )
        }
    )
    generate_dataset(alternate, 2, tmp_path / "alternate", component_topology=True)
    generate_dataset(config, 2, tmp_path / "repeat", component_topology=True)
    baseline = DatasetLoader(topology_dataset, ModalityPermissionSet.ecological_only())
    for root in (tmp_path / "alternate", tmp_path / "repeat"):
        validate_dataset(root)
        loader = DatasetLoader(root, ModalityPermissionSet.ecological_only())
        for index in (0, 1):
            assert (
                loader.read_component_topology(index).annotation
                == baseline.read_component_topology(index).annotation
            )


@pytest.mark.parametrize(
    "method",
    ["read_component_topology", "read_ecological_transition", "read_ecological_visibility_events"],
)
def test_topology_permission_is_checked_before_transition_open(
    topology_dataset: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    permissions = ModalityPermissionSet(
        allowed=ModalityPermissionSet.all_modalities().allowed - {Modality.COMPONENT_TOPOLOGY}
    )
    loader = DatasetLoader(topology_dataset, permissions)

    def unexpected_open(*args: Any) -> None:
        pytest.fail("unauthorised access opened a transition")

    monkeypatch.setattr(loader, "_transition", unexpected_open)
    with pytest.raises(PermissionDeniedError, match="component_topology"):
        getattr(loader, method)(0)


def test_dedicated_loader_exposes_no_control_metadata(topology_dataset: Path) -> None:
    loader = DatasetLoader(
        topology_dataset, ModalityPermissionSet(allowed=frozenset({Modality.COMPONENT_TOPOLOGY}))
    )
    payload = canonical_json_bytes(loader.read_component_topology(0).annotation).decode()
    for field in (
        "raw_geom",
        "world_position",
        "profile_id",
        "benchmark_role",
        "episode_seed",
        "camera",
        "sampled_geometry",
        "texture",
        "generation_record",
        "registry",
    ):
        assert field not in payload


@pytest.mark.parametrize(
    "mutation",
    [
        "pixel_count",
        "bounds",
        "mask_hash",
        "map",
        "support_count",
        "target_pair",
        "event_kind",
        "event_id",
        "component_id",
        "whole_substitution",
    ],
)
def test_complete_joint_reseals_rejected(
    topology_dataset: Path, tmp_path: Path, mutation: str
) -> None:
    root = tmp_path / "false"
    shutil.copytree(topology_dataset, root)
    transition, instrumentation = load_episode_payloads(root, 0)
    payload = transition["ecological_visibility_events"]["capabilities"]["component_topology"]
    component = next(c for c in payload["frames"][0]["components"] if c["pixel_count"] > 2)
    if mutation == "pixel_count":
        component["pixel_count"] -= 1
    elif mutation == "bounds":
        component["bounds_top_left_bottom_right_exclusive"][2] += 1
    elif mutation == "mask_hash":
        component["mask_sha256"] = "1" * 64
    elif mutation == "map":
        record = ArtifactRecord.model_validate_json(
            json.dumps(payload["frames"][0]["component_labels"])
        )
        array = np.load(root / record.path, allow_pickle=False)
        array.flat[int(np.flatnonzero(array > 0)[0])] = 0
        payload["frames"][0]["component_labels"] = rewrite_array_artifact(
            root, record, array
        ).model_dump(mode="json")
    elif mutation in {"support_count", "target_pair"}:
        support = next(s for s in payload["supports"] if s["edge"])
        support["forward_count"] += 1
        if mutation == "target_pair":
            support["backward_count"] += 1
    elif mutation == "event_kind":
        event = next(e for e in payload["events"] if e["kind"] == "one_to_one_continuation")
        event["kind"] = "many_to_one_merge"
        event["event_id"] = domain_hash(
            "event_id", {k: v for k, v in event.items() if k != "event_id"}
        )
    elif mutation == "event_id":
        payload["events"][0]["event_id"] = "1" * 64
    elif mutation == "component_id":
        old_id, new_id = component["component_id"], "component-" + "1" * 64
        payload = json.loads(json.dumps(payload).replace(old_id, new_id))
        payload["supports"].sort(
            key=lambda s: (s["surface_id"], s["before_component_id"], s["after_component_id"])
        )
        for event in payload["events"]:
            event["before_component_ids"].sort()
            event["after_component_ids"].sort()
        payload["events"].sort(
            key=lambda e: (e["surface_id"], e["before_component_ids"], e["after_component_ids"])
        )
    else:
        # Jointly change every existing support and all raster-local component mask roots.
        for frame in payload["frames"]:
            for item in frame["components"]:
                item["mask_sha256"] = domain_hash("component_mask", {"false": item["component_id"]})
        for support in payload["supports"]:
            if support["edge"]:
                support["forward_count"] += 7
    false = seal_annotation(
        ComponentTopologyAnnotation.model_validate_json(canonical_json_bytes(payload))
    )
    transition["ecological_visibility_events"]["capabilities"]["component_topology"] = (
        false.model_dump(mode="json")
    )
    commit_episode_payloads(root, 0, transition, instrumentation)
    # All outer claims are genuinely resealed; reconstruction must be the rejecting check.
    with pytest.raises(ValueError, match=r"component.*independent reconstruction"):
        validate_dataset(root)


def test_unupdated_ecological_identity_rejected(topology_dataset: Path, tmp_path: Path) -> None:
    root = tmp_path / "false"
    shutil.copytree(topology_dataset, root)
    loader = DatasetLoader(root, ModalityPermissionSet.all_modalities())
    transition = loader._transition(0)
    topology = loader.read_component_topology(0).annotation
    changed = topology.model_copy(update={"component_topology_sha256": "1" * 64})
    events = transition.ecological_visibility_events.model_copy(
        update={
            "capabilities": transition.ecological_visibility_events.capabilities.model_copy(
                update={"component_topology": changed}
            )
        }
    )
    events = events.model_copy(
        update={"visibility_event_sha256": compute_visibility_event_hash(events)}
    )
    false = transition.model_copy(update={"ecological_visibility_events": events})
    assert compute_ecological_label_hash(false) != transition.ecological_label_sha256
    manifest = load_manifest(root)
    episode = manifest.episodes[0].model_copy(
        update={
            "transition": rewrite_json_artifact(
                root, manifest.episodes[0].transition, false.model_dump(mode="json")
            ),
            "visibility_event_sha256": events.visibility_event_sha256,
        }
    )
    _write_manifest(root, manifest, [episode, *manifest.episodes[1:]])
    with pytest.raises(ValueError, match="ecological"):
        validate_dataset(root)


@pytest.mark.parametrize(
    "mutation", ["traversal", "alias", "dtype", "shape", "content_hash", "hardlink"]
)
def test_component_artifact_safety(topology_dataset: Path, tmp_path: Path, mutation: str) -> None:
    root = tmp_path / "false"
    shutil.copytree(topology_dataset, root)
    transition, instrumentation = load_episode_payloads(root, 0)
    frames = transition["ecological_visibility_events"]["capabilities"]["component_topology"][
        "frames"
    ]
    record = frames[0]["component_labels"]
    if mutation == "hardlink":
        path = root / record["path"]
        alias = tmp_path / "linked.npy"
        os.link(path, alias)
    elif mutation == "traversal":
        record["path"] = "../outside.npy"
    elif mutation == "alias":
        frames[1]["component_labels"] = dict(record)
    elif mutation == "dtype":
        record["dtype"] = "uint8"
    elif mutation == "shape":
        record["shape"] = [1, 1]
    else:
        record["logical_sha256"] = "1" * 64
    if mutation in {"alias", "content_hash"}:
        payload = transition["ecological_visibility_events"]["capabilities"]["component_topology"]
        annotation = ComponentTopologyAnnotation.model_validate_json(canonical_json_bytes(payload))
        transition["ecological_visibility_events"]["capabilities"]["component_topology"] = (
            seal_annotation(annotation).model_dump(mode="json")
        )
        commit_episode_payloads(root, 0, transition, instrumentation)
    elif mutation != "hardlink":
        manifest = load_manifest(root)
        episode = manifest.episodes[0].model_copy(
            update={
                "transition": rewrite_json_artifact(
                    root, manifest.episodes[0].transition, transition
                )
            }
        )
        _write_manifest(root, manifest, [episode, *manifest.episodes[1:]])
    match = "reused|alias|duplicate|logical hash" if mutation in {"alias", "content_hash"} else None
    with pytest.raises(ValueError, match=match):
        validate_dataset(root)


def test_inspection_validates_and_preserves_complete_dataset(
    topology_dataset: Path, tmp_path: Path
) -> None:
    before = {
        p.relative_to(topology_dataset): sha256_file(p)
        for p in topology_dataset.rglob("*")
        if p.is_file()
    }
    create_inspection_image(topology_dataset, 0, tmp_path / "topology.png")
    assert before == {
        p.relative_to(topology_dataset): sha256_file(p)
        for p in topology_dataset.rglob("*")
        if p.is_file()
    }
    with pytest.raises(ValueError, match="outside"):
        create_inspection_image(topology_dataset, 0, topology_dataset / "bad.png")
    with pytest.raises(FileExistsError):
        create_inspection_image(topology_dataset, 0, tmp_path / "topology.png")
