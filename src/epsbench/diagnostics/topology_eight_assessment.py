"""Saved-array assessments for the fixed topology development study; no capture."""

from __future__ import annotations

import io
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, model_validator

from epsbench.annotations.component_topology import (
    ByteMap,
    FrameIndex,
    IntMap,
    derive_component_topology,
    validate_component_topology,
)
from epsbench.data.loader import DatasetLoader, PermissionDeniedError
from epsbench.schema import (
    Action,
    ArtifactRecord,
    CameraInstrumentation,
    ComponentTopologyAnnotation,
    Modality,
    ModalityPermissionSet,
    Sha256,
    SurfaceId,
    SurfaceReference,
)
from epsbench.utils.canonical import canonical_json_bytes, logical_array_hash, sha256_bytes

PROTOCOL = "epsbench.topology_development.eight_transition.v1"
CELL_IDS = (
    "00-forward-base-r0",
    "01-forward-base-r1",
    "02-forward-alternate-r0",
    "03-forward-alternate-r1",
    "04-reverse-base-r0",
    "05-reverse-base-r1",
    "06-reverse-alternate-r0",
    "07-reverse-alternate-r1",
)
INPUT_NAMES = (
    "segmentation_before",
    "segmentation_after",
    "forward_vectors",
    "backward_vectors",
    "forward_validity",
    "backward_validity",
    "forward_reasons",
    "backward_reasons",
    "before_fate_codes",
    "after_origin_codes",
)


class PrivilegedAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    protocol: Literal["epsbench.topology_development.eight_transition.v1"]
    background_surface_id: SurfaceId
    semantic_to_opaque: dict[str, SurfaceId]
    rgb_hashes: tuple[Sha256, Sha256]
    native_hashes: tuple[tuple[Sha256, Sha256], tuple[Sha256, Sha256]]
    producer_states: tuple[dict[str, Any], dict[str, Any]]
    camera_states: tuple[CameraInstrumentation, CameraInstrumentation]
    action: Action
    raw_world_coordinates: dict[str, tuple[float, float, float]]
    raw_geom_ids: dict[str, int]
    endpoint_reconstruction: Literal["passed"]

    @model_validator(mode="after")
    def exact_correspondence(self) -> PrivilegedAssessment:
        if (
            set(self.semantic_to_opaque)
            != {"background_surface", "support_surface", "occluding_surface"}
            or len(set(self.semantic_to_opaque.values())) != 3
        ):
            raise ValueError("privileged surface correspondence must be exact and bijective")
        if self.background_surface_id != self.semantic_to_opaque["background_surface"]:
            raise ValueError("background identification differs from surface correspondence")
        if tuple(camera.frame_index for camera in self.camera_states) != (0, 1):
            raise ValueError("camera endpoints must be ordered")
        return self


@dataclass(frozen=True)
class PublicTopologyInputs:
    """Only public image-plane arrays and opaque declarations enter derivation."""

    segmentations: tuple[IntMap, IntMap]
    surfaces: tuple[SurfaceReference, ...]
    vectors: tuple[IntMap, IntMap]
    validity: tuple[ByteMap, ByteMap]
    reasons: tuple[ByteMap, ByteMap]
    event_codes: tuple[ByteMap, ByteMap]
    analytic_transport_sha256: str

    def arrays(self) -> tuple[np.ndarray[Any, Any], ...]:
        return (
            *self.segmentations,
            *self.vectors,
            *self.validity,
            *self.reasons,
            *self.event_codes,
        )


def load_public_inputs(loader: DatasetLoader) -> PublicTopologyInputs:
    if loader.permissions != ModalityPermissionSet.ecological_only():
        raise PermissionDeniedError(
            "public assessment requires exactly ecological-only permissions"
        )
    view = loader.read_ecological_transition(0)
    transport = loader.read_analytic_optical_transport(0)
    events = loader.read_ecological_visibility_events(0)
    return PublicTopologyInputs(
        (loader.read_segmentation(0, 0), loader.read_segmentation(0, 1)),
        view.surfaces,
        (transport.forward_vectors_fixed, transport.backward_vectors_fixed),
        (transport.forward_validity, transport.backward_validity),
        (transport.forward_reasons, transport.backward_reasons),
        (events.before_fate_codes, events.after_origin_codes),
        transport.analytic_transport_sha256,
    )


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        import os

        os.fsync(stream.fileno())
    if path.read_bytes() != payload:
        raise ValueError("saved assessment readback differs")


def _json(path: Path, value: Any) -> None:
    _publish(path, canonical_json_bytes(value) + b"\n")


def _array_bytes(array: np.ndarray[Any, Any]) -> bytes:
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return stream.getvalue()


def derive_public_artifacts(
    inputs: PublicTopologyInputs, output: Path
) -> ComponentTopologyAnnotation:
    """Publish complete inputs/maps, then validate persisted arrays independently."""
    if output.exists():
        raise FileExistsError("standalone public assessment directory already exists")
    output.mkdir(parents=True)
    for name, array in zip(INPUT_NAMES, inputs.arrays(), strict=True):
        _publish(output / f"{name}.npy", _array_bytes(array))
    metadata = {
        "protocol": PROTOCOL,
        "surfaces": [surface.model_dump(mode="json") for surface in inputs.surfaces],
        "analytic_transport_sha256": inputs.analytic_transport_sha256,
        "input_logical_hashes": dict(
            zip(INPUT_NAMES, map(logical_array_hash, inputs.arrays()), strict=True)
        ),
    }
    _json(output / "inputs.json", metadata)

    def save_map(frame: FrameIndex, array: IntMap) -> ArtifactRecord:
        payload = _array_bytes(array)
        name = f"components_{frame}.npy"
        _publish(output / name, payload)
        return ArtifactRecord(
            path=name,
            modality=Modality.COMPONENT_TOPOLOGY,
            media_type="application/x-npy",
            dtype="int32",
            shape=array.shape,
            logical_sha256=logical_array_hash(array),
            file_sha256=sha256_bytes(payload),
            byte_count=len(payload),
        )

    annotation = derive_component_topology(
        inputs.segmentations,
        inputs.surfaces,
        inputs.vectors,
        inputs.validity,
        inputs.reasons,
        inputs.event_codes,
        inputs.analytic_transport_sha256,
        save_map,
    )
    _json(output / "topology.json", annotation)
    validate_public_artifacts(output)
    return annotation


def validate_public_artifacts(output: Path) -> ComponentTopologyAnnotation:
    metadata = json.loads((output / "inputs.json").read_bytes())
    if (
        set(metadata)
        != {"protocol", "surfaces", "analytic_transport_sha256", "input_logical_hashes"}
        or metadata["protocol"] != PROTOCOL
    ):
        raise ValueError("invalid standalone public input envelope")
    arrays = [np.load(output / f"{name}.npy", allow_pickle=False) for name in INPUT_NAMES]
    if metadata["input_logical_hashes"] != dict(
        zip(INPUT_NAMES, map(logical_array_hash, arrays), strict=True)
    ):
        raise ValueError("saved public input hashes differ")
    annotation = ComponentTopologyAnnotation.model_validate_json(
        (output / "topology.json").read_bytes()
    )
    maps = (
        np.load(output / "components_0.npy", allow_pickle=False),
        np.load(output / "components_1.npy", allow_pickle=False),
    )
    for frame in annotation.frames:
        payload = (output / frame.component_labels.path).read_bytes()
        if (
            len(payload) != frame.component_labels.byte_count
            or sha256_bytes(payload) != frame.component_labels.file_sha256
        ):
            raise ValueError("saved component map bytes differ")
    validate_component_topology(
        annotation,
        (arrays[0], arrays[1]),
        tuple(SurfaceReference.model_validate(s) for s in metadata["surfaces"]),
        (arrays[2], arrays[3]),
        (arrays[4], arrays[5]),
        (arrays[6], arrays[7]),
        (arrays[8], arrays[9]),
        metadata["analytic_transport_sha256"],
        maps,
    )
    return annotation


def _privileged_assessment(loader: DatasetLoader, public: PublicTopologyInputs) -> dict[str, Any]:
    """Permission check precedes all privileged reads, including instrumentation."""
    loader._require(*tuple(Modality))
    from epsbench.sim.canonical_paired import SceneMapEntry, convert_native_depth, decode_id_colors

    manifest = loader.read_dataset_manifest()
    if manifest.schema_version != "0.1.0-dev.11" or len(manifest.episodes) != 1:
        raise ValueError("assessment requires one canonical paired episode")
    raw_ids = loader.read_raw_mujoco_geom_ids(0)
    endpoints = [loader.read_canonical_paired_output(0, frame) for frame in (0, 1)]
    correspondence = {
        name: endpoints[0].provenance.raw_to_opaque_surface_ids[str(raw_id)]
        for name, raw_id in sorted(raw_ids.items())
    }
    if set(correspondence) != {"background_surface", "support_surface", "occluding_surface"}:
        raise ValueError("intended background declaration is absent")
    for frame, endpoint in enumerate(endpoints):
        provenance = endpoint.provenance
        raw = decode_id_colors(
            endpoint.native_id_rgb,
            tuple(
                SceneMapEntry(item.segid_plus_one, item.objid, item.objtype)
                for item in provenance.scene_map
            ),
        )
        segmentation = np.zeros(raw.shape, dtype=np.int32)
        for raw_id, opaque in provenance.raw_to_opaque_surface_ids.items():
            segmentation[raw == int(raw_id)] = provenance.opaque_surface_labels[opaque]
        if not np.array_equal(segmentation, public.segmentations[frame]) or not np.array_equal(
            convert_native_depth(endpoint.native_depth_pre_metric, provenance.near, provenance.far),
            loader.read_depth(0, frame),
        ):
            raise ValueError("canonical endpoint independent reconstruction differs")
        if {
            name: provenance.raw_to_opaque_surface_ids[str(raw_id)]
            for name, raw_id in raw_ids.items()
        } != correspondence:
            raise ValueError("endpoint surface correspondence differs")
    transition = loader._transition(0)
    return {
        "protocol": PROTOCOL,
        "background_surface_id": correspondence["background_surface"],
        "semantic_to_opaque": correspondence,
        "rgb_hashes": [logical_array_hash(loader.read_rgb(0, frame)) for frame in (0, 1)],
        "native_hashes": [
            [logical_array_hash(e.native_id_rgb), logical_array_hash(e.native_depth_pre_metric)]
            for e in endpoints
        ],
        "producer_states": [e.producer_state for e in endpoints],
        "camera_states": [
            loader.read_camera_world_transform(0, frame).model_dump(mode="json") for frame in (0, 1)
        ],
        "action": transition.action.model_dump(mode="json"),
        "raw_world_coordinates": loader.read_raw_world_coordinates(0),
        "raw_geom_ids": raw_ids,
        "endpoint_reconstruction": "passed",
    }


def assess_cell(
    dataset: Path, exploratory: Path, privileged_output: Path, inspection_output: Path
) -> dict[str, Any]:
    public = load_public_inputs(DatasetLoader(dataset, ModalityPermissionSet.ecological_only()))
    annotation = derive_public_artifacts(public, exploratory)
    privileged = _privileged_assessment(
        DatasetLoader(dataset, ModalityPermissionSet.all_modalities()), public
    )
    privileged = PrivilegedAssessment.model_validate_json(
        canonical_json_bytes(privileged)
    ).model_dump(mode="json")
    _json(privileged_output, privileged)
    # All panels originate from saved arrays; PIL has no renderer context.
    maps = [
        np.load(exploratory / f"components_{frame}.npy", allow_pickle=False) for frame in (0, 1)
    ]
    panels = []
    for array in (*public.segmentations, *maps):
        rgb = np.zeros((*array.shape, 3), dtype=np.uint8)
        for label in np.unique(array):
            if label:
                rgb[array == label] = np.frombuffer(
                    bytes.fromhex(sha256_bytes(str(int(label)).encode()))[:3], dtype=np.uint8
                )
        panels.append(rgb)
    stream = io.BytesIO()
    Image.fromarray(np.concatenate(panels, axis=1)).save(stream, format="PNG")
    _publish(inspection_output, stream.getvalue())
    return {
        "protocol": PROTOCOL,
        "public_reconstruction": "passed",
        "endpoint_reconstruction": "passed",
        "topology_sha256": annotation.component_topology_sha256,
        "capability": annotation.status,
        "event_kinds": [event.kind.value for event in annotation.events],
    }


def _inventory(root: Path, *, exclude_run: bool = False) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("assessment inventory refuses symlinks")
        if path.is_file():
            name = path.relative_to(root).as_posix()
            if exclude_run and name == "run.json":
                continue
            result[name] = sha256_bytes(path.read_bytes())
    if not result:
        raise ValueError("empty assessment inventory")
    return result


def _direction_relation(
    a: ComponentTopologyAnnotation,
    b: ComponentTopologyAnnotation,
    pa: dict[str, Any],
    pb: dict[str, Any],
) -> dict[str, Any]:
    identities: list[dict[str, tuple[str, str]]] = []
    for annotation, private in ((a, pa), (b, pb)):
        names = {opaque: name for name, opaque in private["semantic_to_opaque"].items()}
        if len(names) != len(private["semantic_to_opaque"]):
            raise ValueError("surface correspondence must be bijective")
        identities.append(
            {
                c.component_id: (names[c.surface_id], c.mask_sha256)
                for f in annotation.frames
                for c in f.components
            }
        )
    ia, ib = identities
    components_equal = all(
        sorted(ia[c.component_id] for c in a.frames[frame].components)
        == sorted(ib[c.component_id] for c in b.frames[1 - frame].components)
        for frame in (0, 1)
    )
    exchange = {
        "one_to_many_split": "many_to_one_merge",
        "many_to_one_merge": "one_to_many_split",
        "component_appearance": "component_disappearance",
        "component_disappearance": "component_appearance",
    }

    def events(
        annotation: ComponentTopologyAnnotation, ids: dict[str, tuple[str, str]], reverse: bool
    ) -> list[Any]:
        return sorted(
            (
                exchange.get(e.kind.value, e.kind.value) if reverse else e.kind.value,
                sorted(
                    ids[i] for i in (e.after_component_ids if reverse else e.before_component_ids)
                ),
                sorted(
                    ids[i] for i in (e.before_component_ids if reverse else e.after_component_ids)
                ),
            )
            for e in annotation.events
        )

    supports_a = sorted(
        (
            ia[s.before_component_id],
            ia[s.after_component_id],
            s.edge,
            s.forward_count,
            s.backward_count,
        )
        for s in a.supports
    )
    supports_b = sorted(
        (
            ib[s.after_component_id],
            ib[s.before_component_id],
            s.edge,
            s.backward_count,
            s.forward_count,
        )
        for s in b.supports
    )
    cameras_exchange = all(
        {k: v for k, v in pa["camera_states"][frame].items() if k != "frame_index"}
        == {k: v for k, v in pb["camera_states"][1 - frame].items() if k != "frame_index"}
        for frame in (0, 1)
    )
    actions_reverse = (
        pa["action"]["name"] == "lateral_right"
        and pb["action"]["name"] == "lateral_left"
        and all(
            pa["action"][key] == -pb["action"][key]
            for key in ("delta_forward", "delta_lateral", "delta_yaw")
        )
    )
    return {
        "components_exchange_equal": components_equal,
        "events_exchange_equal": events(a, ia, False) == events(b, ib, True),
        "support_edges_exchange_equal": [s[:3] for s in supports_a] == [s[:3] for s in supports_b],
        "camera_endpoints_exchange_equal": cameras_exchange,
        "producer_endpoints_exchange_equal": pa["producer_states"] == pb["producer_states"][::-1],
        "actions_reverse_equal": actions_reverse,
        "geometry_equal": pa["raw_world_coordinates"] == pb["raw_world_coordinates"]
        and pa["raw_geom_ids"] == pb["raw_geom_ids"],
        "directional_counts_exchange_equal_descriptive": supports_a == supports_b,
        "forward_supports": supports_a,
        "reverse_exchanged_supports": supports_b,
    }


def compare_study(root: Path, cells: Sequence[str] = CELL_IDS) -> dict[str, Any]:
    if tuple(cells) != CELL_IDS:
        raise ValueError("comparison requires exact ordered eight-cell membership")
    annotations = [validate_public_artifacts(root / "exploratory" / cell) for cell in cells]
    private = [
        PrivilegedAssessment.model_validate_json(
            (root / "privileged" / f"{cell}.json").read_bytes()
        ).model_dump(mode="json")
        for cell in cells
    ]
    if any(p["protocol"] != PROTOCOL or p["endpoint_reconstruction"] != "passed" for p in private):
        raise ValueError("invalid privileged assessment identity or endpoint status")
    repeats = [
        {
            "ordinals": [a, b],
            "dataset_equal_except_run_json": _inventory(
                root / "datasets" / cells[a], exclude_run=True
            )
            == _inventory(root / "datasets" / cells[b], exclude_run=True),
            "standalone_equal": _inventory(root / "exploratory" / cells[a])
            == _inventory(root / "exploratory" / cells[b]),
        }
        for a, b in ((0, 1), (2, 3), (4, 5), (6, 7))
    ]
    appearances = []
    for a, b in ((0, 2), (1, 3), (4, 6), (5, 7)):
        controls = (
            "native_hashes",
            "producer_states",
            "camera_states",
            "action",
            "semantic_to_opaque",
            "raw_world_coordinates",
            "raw_geom_ids",
        )
        appearances.append(
            {
                "ordinals": [a, b],
                "rgb_changed_both_endpoints": all(
                    x != y
                    for x, y in zip(private[a]["rgb_hashes"], private[b]["rgb_hashes"], strict=True)
                ),
                "controlled_terms_equal": all(
                    private[a][key] == private[b][key] for key in controls
                ),
                "standalone_equal": _inventory(root / "exploratory" / cells[a])
                == _inventory(root / "exploratory" / cells[b]),
            }
        )
    directions = [
        {
            "ordinals": [a, b],
            **_direction_relation(annotations[a], annotations[b], private[a], private[b]),
        }
        for a, b in ((0, 4), (1, 5), (2, 6), (3, 7))
    ]
    targets = []
    for index, (annotation, privileged) in enumerate(zip(annotations, private, strict=True)):
        kinds = [
            e.kind.value
            for e in annotation.events
            if e.surface_id == privileged["background_surface_id"]
        ]
        expected = "one_to_many_split" if index < 4 else "many_to_one_merge"
        targets.append(
            {
                "ordinal": index,
                "background_event_kinds": kinds,
                "expected": expected,
                "matches": kinds == [expected],
                "overall_capability": annotation.status,
                "all_events": [e.model_dump(mode="json") for e in annotation.events],
            }
        )
    positive = (
        all(t["matches"] for t in targets)
        and all(r["dataset_equal_except_run_json"] and r["standalone_equal"] for r in repeats)
        and all(
            a["rgb_changed_both_endpoints"]
            and a["controlled_terms_equal"]
            and a["standalone_equal"]
            for a in appearances
        )
        and all(
            d["components_exchange_equal"]
            and d["events_exchange_equal"]
            and d["support_edges_exchange_equal"]
            and d["camera_endpoints_exchange_equal"]
            and d["producer_endpoints_exchange_equal"]
            and d["actions_reverse_equal"]
            and d["geometry_equal"]
            for d in directions
        )
    )
    return {
        "protocol": PROTOCOL,
        "repeat_pairs": repeats,
        "appearance_pairs": appearances,
        "direction_pairs": directions,
        "target_cells": targets,
        "hypothesis_positive_bounded": positive,
        "phase_gate_effect": "NONE",
    }
