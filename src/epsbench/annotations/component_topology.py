"""Slice 6 public image-plane oracle. No apparatus or appearance inputs are accepted."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from epsbench.schema import (
    ArtifactRecord,
    ComponentEventKind,
    ComponentFrame,
    ComponentRecord,
    ComponentSupport,
    ComponentTopologyAnnotation,
    ComponentTopologyEvent,
    SurfaceReference,
)
from epsbench.utils.canonical import canonical_json_bytes, logical_array_hash, sha256_bytes

IntMap = npt.NDArray[np.int32]
ByteMap = npt.NDArray[np.uint8]
FrameIndex = Literal[0, 1]
FRAME_INDICES: tuple[FrameIndex, FrameIndex] = (0, 1)
DOMAIN = "epsbench.component_topology.v1."


def domain_hash(kind: str, payload: Any) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "domain": DOMAIN + kind,
                "envelope_version": "logical_domain_envelope_v1",
                "payload": payload,
            }
        )
    )


def component_identity(frame: int, surface: str, index: int) -> str:
    # A frame-local rank identity, not a persistence claim or raster-mask identity.
    return "component-" + domain_hash(
        "component_id",
        {
            "frame_index": frame,
            "surface_id": surface,
            "surface_component_index": index,
        },
    )


def _flood_components(segmentation: IntMap, label: int) -> list[list[tuple[int, int]]]:
    height, width = segmentation.shape
    unseen = segmentation == label
    result: list[list[tuple[int, int]]] = []
    for row, column in np.argwhere(unseen):
        r, c = int(row), int(column)
        if not unseen[r, c]:
            continue
        queue = deque([(r, c)])
        unseen[r, c] = False
        pixels: list[tuple[int, int]] = []
        while queue:
            y, x = queue.popleft()
            pixels.append((y, x))
            for yy, xx in ((y - 1, x), (y, x - 1), (y, x + 1), (y + 1, x)):
                if 0 <= yy < height and 0 <= xx < width and unseen[yy, xx]:
                    unseen[yy, xx] = False
                    queue.append((yy, xx))
        result.append(sorted(pixels))
    return result


def _union_components(segmentation: IntMap, label: int) -> list[list[tuple[int, int]]]:
    """Independent validation algorithm: raster scan with north/west union-find."""
    width = segmentation.shape[1]
    parents: dict[int, int] = {}

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for row, column in np.argwhere(segmentation == label):
        r, c = int(row), int(column)
        index = r * width + c
        parents[index] = index
        for neighbour in (index - width if r else -1, index - 1 if c else -1):
            if neighbour in parents:
                first, second = find(index), find(neighbour)
                parents[max(first, second)] = min(first, second)
    groups: dict[int, list[tuple[int, int]]] = {}
    for index in parents:
        groups.setdefault(find(index), []).append(divmod(index, width))
    return sorted((sorted(pixels) for pixels in groups.values()), key=lambda pixels: pixels[0])


def extract_components(
    segmentation: IntMap,
    surfaces: tuple[SurfaceReference, ...],
    frame: FrameIndex,
    *,
    independent: bool = False,
) -> tuple[IntMap, tuple[ComponentRecord, ...]]:
    if segmentation.dtype != np.int32 or segmentation.ndim != 2:
        raise ValueError("opaque segmentation must be a two-dimensional int32 array")
    labels = [s.segmentation_label for s in surfaces]
    ids = [s.surface_id for s in surfaces]
    if len(set(labels)) != len(labels) or len(set(ids)) != len(ids):
        raise ValueError("opaque surface declarations must be unique")
    if set(np.unique(segmentation)) - {0, *labels}:
        raise ValueError("opaque segmentation contains undeclared labels")
    label_map = np.zeros(segmentation.shape, dtype=np.int32)
    records: list[ComponentRecord] = []
    algorithm = _union_components if independent else _flood_components
    for surface in sorted(surfaces, key=lambda s: s.surface_id):
        groups = algorithm(segmentation, surface.segmentation_label)
        minima = [pixels[0] for pixels in groups]
        if minima != sorted(set(minima)):
            raise ValueError("component coordinate ties are forbidden")
        for index, pixels in enumerate(groups):
            map_label = len(records) + 1
            rows, columns = np.asarray(pixels, dtype=np.int64).T
            label_map[rows, columns] = map_label
            mask = np.zeros(segmentation.shape, dtype=np.uint8)
            mask[rows, columns] = 1
            records.append(
                ComponentRecord(
                    frame_index=frame,
                    component_id=component_identity(frame, surface.surface_id, index),
                    surface_id=surface.surface_id,
                    surface_component_index=index,
                    map_label=map_label,
                    pixel_count=len(pixels),
                    minimum_row_column=pixels[0],
                    bounds_top_left_bottom_right_exclusive=(
                        int(rows.min()),
                        int(columns.min()),
                        int(rows.max()) + 1,
                        int(columns.max()) + 1,
                    ),
                    mask_sha256=domain_hash(
                        "component_mask",
                        {"shape": list(mask.shape), "logical_sha256": logical_array_hash(mask)},
                    ),
                )
            )
    return label_map, tuple(records)


def _counts(
    source: IntMap,
    target: IntMap,
    vectors: IntMap,
    validity: ByteMap,
    reasons: ByteMap,
    source_records: tuple[ComponentRecord, ...],
    target_records: tuple[ComponentRecord, ...],
    *,
    independent: bool,
) -> dict[tuple[str, str], int]:
    if (
        vectors.dtype != np.int32
        or vectors.shape != (*source.shape, 2)
        or validity.dtype != np.uint8
        or reasons.dtype != np.uint8
        or validity.shape != source.shape
        or reasons.shape != source.shape
    ):
        raise ValueError("transport arrays must have the canonical aligned dtypes and dimensions")
    if not set(np.unique(validity)).issubset({0, 1}) or not set(np.unique(reasons)).issubset(
        set(range(5))
    ):
        raise ValueError("transport contains unknown validity or reason codes")
    if not np.array_equal(validity == 1, reasons == 0) or np.any(vectors[validity == 0] != 0):
        raise ValueError("transport validity/reasons/invalid vectors are not canonical")
    by_source = {c.map_label: c for c in source_records}
    by_target = {c.map_label: c for c in target_records}
    height, width = target.shape
    counts: dict[tuple[str, str], int] = {}

    def add(r: int, c: int, y: int, x: int) -> None:
        if not (0 <= y < height and 0 <= x < width):
            return
        a = by_source.get(int(source[r, c]))
        b = by_target.get(int(target[y, x]))
        if a is not None and b is not None and a.surface_id == b.surface_id:
            key = a.component_id, b.component_id
            counts[key] = counts.get(key, 0) + 1

    eligible = np.argwhere((validity == 1) & (reasons == 0) & (source != 0))
    if independent:
        # Python integer arithmetic provides a second implementation of target-cell assignment.
        for row, column in eligible:
            r, c = int(row), int(column)
            x = (c * 1024 + 512 + int(vectors[r, c, 0])) // 1024
            y = (r * 1024 + 512 + int(vectors[r, c, 1])) // 1024
            add(r, c, y, x)
    elif len(eligible):
        target_yx = (
            eligible.astype(np.int64) * 1024
            + 512
            + vectors[eligible[:, 0], eligible[:, 1]][:, ::-1].astype(np.int64)
        ) // 1024
        for (r, c), (y, x) in zip(eligible, target_yx, strict=True):
            add(int(r), int(c), int(y), int(x))
    return counts


def _events(
    frames: tuple[tuple[ComponentRecord, ...], tuple[ComponentRecord, ...]],
    maps: tuple[IntMap, IntMap],
    codes: tuple[ByteMap, ByteMap],
    supports: tuple[ComponentSupport, ...],
    *,
    independent: bool,
) -> tuple[ComponentTopologyEvent, ...]:
    nodes = {c.component_id: c for frame in frames for c in frame}
    adjacency: dict[str, set[str]] = {identity: set() for identity in nodes}
    for support in supports:
        if support.edge:
            adjacency[support.before_component_id].add(support.after_component_id)
            adjacency[support.after_component_id].add(support.before_component_id)
    groups: list[set[str]] = []
    if independent:
        # Equivalence closure by set merging, separate from the generation traversal.
        groups = [{identity} for identity in sorted(nodes)]
        for support in supports:
            if support.edge:
                touching = [
                    g
                    for g in groups
                    if support.before_component_id in g or support.after_component_id in g
                ]
                merged = set().union(*touching)
                groups = [g for g in groups if g not in touching] + [merged]
    else:
        remaining = set(nodes)
        while remaining:
            todo = [min(remaining)]
            group: set[str] = set()
            while todo:
                identity = todo.pop()
                if identity in group:
                    continue
                group.add(identity)
                todo.extend(sorted(adjacency[identity] - group))
            remaining -= group
            groups.append(group)
    events: list[ComponentTopologyEvent] = []
    for group in groups:
        before = tuple(sorted(i for i in group if nodes[i].frame_index == 0))
        after = tuple(sorted(i for i in group if nodes[i].frame_index == 1))
        first = nodes[min(group)]
        if before and after:
            kind = (
                ComponentEventKind.CONTINUATION
                if len(before) == len(after) == 1
                else ComponentEventKind.SPLIT
                if len(before) == 1
                else ComponentEventKind.MERGE
                if len(after) == 1
                else ComponentEventKind.COMPLEX
            )
        else:
            opposite = frames[1 - first.frame_index]
            surface_absent = not any(c.surface_id == first.surface_id for c in opposite)
            selected_codes = codes[first.frame_index][maps[first.frame_index] == first.map_label]
            proven_lifecycle = bool(np.all(np.isin(selected_codes, (1, 2))))
            kind = (
                (ComponentEventKind.DISAPPEARANCE if before else ComponentEventKind.APPEARANCE)
                if surface_absent or proven_lifecycle
                else ComponentEventKind.INDETERMINATE
            )
        payload = {
            "surface_id": first.surface_id,
            "kind": kind.value,
            "before_component_ids": list(before),
            "after_component_ids": list(after),
        }
        events.append(
            ComponentTopologyEvent(
                event_id=domain_hash("event_id", payload),
                surface_id=first.surface_id,
                kind=kind,
                before_component_ids=before,
                after_component_ids=after,
            )
        )
    return tuple(
        sorted(events, key=lambda e: (e.surface_id, e.before_component_ids, e.after_component_ids))
    )


def portable_domain(annotation: ComponentTopologyAnnotation) -> dict[str, Any]:
    return {
        "status": annotation.status,
        "nodes": [
            {
                "frame_index": c.frame_index,
                "component_id": c.component_id,
                "surface_id": c.surface_id,
                "surface_component_index": c.surface_component_index,
            }
            for f in annotation.frames
            for c in f.components
        ],
        "edges": [
            {
                "surface_id": s.surface_id,
                "before_component_id": s.before_component_id,
                "after_component_id": s.after_component_id,
            }
            for s in annotation.supports
            if s.edge
        ],
        "events": [e.model_dump(mode="json") for e in annotation.events],
    }


def annotation_domain(annotation: ComponentTopologyAnnotation) -> dict[str, Any]:
    payload = annotation.model_dump(mode="json", exclude={"component_topology_sha256"})
    for frame in payload["frames"]:
        artifact = frame.pop("component_labels")
        frame["component_labels"] = {
            k: artifact[k] for k in ("modality", "dtype", "shape", "logical_sha256")
        }
    return payload


def seal_annotation(annotation: ComponentTopologyAnnotation) -> ComponentTopologyAnnotation:
    """Seal declared claims; whole-dataset validation separately reconstructs truth."""
    frames = tuple(
        f.model_copy(
            update={
                "component_map_sha256": domain_hash(
                    "component_map",
                    {
                        "frame_index": f.frame_index,
                        "shape": list(f.component_labels.shape),
                        "logical_sha256": f.component_labels.logical_sha256,
                    },
                ),
                "component_set_sha256": domain_hash(
                    "component_set", [c.model_dump(mode="json") for c in f.components]
                ),
            }
        )
        for f in annotation.frames
    )
    sealed = annotation.model_copy(
        update={
            "frames": frames,
            "supports_sha256": domain_hash(
                "supports", [s.model_dump(mode="json") for s in annotation.supports]
            ),
            "events_sha256": domain_hash(
                "events", [e.model_dump(mode="json") for e in annotation.events]
            ),
            "portable_graph_sha256": domain_hash("portable_graph", portable_domain(annotation)),
        }
    )
    return sealed.model_copy(
        update={"component_topology_sha256": domain_hash("annotation", annotation_domain(sealed))}
    )


def derive_component_topology(
    segmentations: tuple[IntMap, IntMap],
    surfaces: tuple[SurfaceReference, ...],
    vectors: tuple[IntMap, IntMap],
    validity: tuple[ByteMap, ByteMap],
    reasons: tuple[ByteMap, ByteMap],
    event_codes: tuple[ByteMap, ByteMap],
    analytic_transport_sha256: str,
    map_artifact: Callable[[FrameIndex, IntMap], ArtifactRecord],
    *,
    independent: bool = False,
) -> ComponentTopologyAnnotation:
    """Reconstruct using only opaque segmentation and retained public transport/event arrays."""
    if segmentations[0].shape != segmentations[1].shape:
        raise ValueError("before/after segmentation dimensions must match")
    for codes in event_codes:
        if codes.dtype != np.uint8 or codes.shape != segmentations[0].shape:
            raise ValueError("public event codes must be aligned uint8 arrays")
        if not set(np.unique(codes)).issubset(set(range(6))):
            raise ValueError("unknown public pixel-event code")
    extracted = tuple(
        extract_components(seg, surfaces, frame, independent=independent)
        for frame, seg in zip(FRAME_INDICES, segmentations, strict=True)
    )
    maps = extracted[0][0], extracted[1][0]
    records = extracted[0][1], extracted[1][1]
    forward = _counts(
        maps[0],
        maps[1],
        vectors[0],
        validity[0],
        reasons[0],
        records[0],
        records[1],
        independent=independent,
    )
    backward = _counts(
        maps[1],
        maps[0],
        vectors[1],
        validity[1],
        reasons[1],
        records[1],
        records[0],
        independent=independent,
    )
    supports = tuple(
        sorted(
            (
                ComponentSupport(
                    surface_id=a.surface_id,
                    before_component_id=a.component_id,
                    after_component_id=b.component_id,
                    forward_count=forward.get((a.component_id, b.component_id), 0),
                    backward_count=backward.get((b.component_id, a.component_id), 0),
                    edge=bool(
                        forward.get((a.component_id, b.component_id), 0)
                        + backward.get((b.component_id, a.component_id), 0)
                    ),
                )
                for a in records[0]
                for b in records[1]
                if a.surface_id == b.surface_id
            ),
            key=lambda s: (s.surface_id, s.before_component_id, s.after_component_id),
        )
    )
    events = _events(records, maps, event_codes, supports, independent=independent)
    frames = tuple(
        ComponentFrame(
            frame_index=frame,
            components=records[frame],
            component_labels=map_artifact(frame, maps[frame]),
            component_map_sha256="0" * 64,
            component_set_sha256="0" * 64,
        )
        for frame in FRAME_INDICES
    )
    annotation = ComponentTopologyAnnotation(
        status="indeterminate"
        if any(e.kind == ComponentEventKind.INDETERMINATE for e in events)
        else "available",
        schema_version="component_topology_annotation_v1",
        connectivity="four_neighbour_equal_opaque_label_v1",
        coordinates="row_column_zero_based_half_open_extents_v1",
        component_id_domain="epsbench.component_topology.v1.component_id",
        map_domain="epsbench.component_topology.v1.component_map",
        set_domain="epsbench.component_topology.v1.component_set",
        support_domain="epsbench.component_topology.v1.supports",
        event_domain="epsbench.component_topology.v1.events",
        annotation_domain="epsbench.component_topology.v1.annotation",
        portable_domain="epsbench.component_topology.v1.portable_graph",
        target_cell_rule="floor_source_center_plus_fixed1024_transport_v1",
        support_rule="valid_reason_zero_either_direction_same_surface_v1",
        classification_rule="connected_bipartite_graph_or_proven_isolate_v1",
        acceptance_rule="complex_retained_indeterminate_capability_fail_closed_v1",
        frames=(frames[0], frames[1]),
        supports=supports,
        events=events,
        segmentation_logical_sha256=(
            logical_array_hash(segmentations[0]),
            logical_array_hash(segmentations[1]),
        ),
        event_codes_logical_sha256=(
            logical_array_hash(event_codes[0]),
            logical_array_hash(event_codes[1]),
        ),
        analytic_transport_sha256=analytic_transport_sha256,
        supports_sha256="0" * 64,
        events_sha256="0" * 64,
        portable_graph_sha256="0" * 64,
        component_topology_sha256="0" * 64,
    )
    return seal_annotation(annotation)


def validate_component_topology(
    annotation: ComponentTopologyAnnotation,
    segmentations: tuple[IntMap, IntMap],
    surfaces: tuple[SurfaceReference, ...],
    vectors: tuple[IntMap, IntMap],
    validity: tuple[ByteMap, ByteMap],
    reasons: tuple[ByteMap, ByteMap],
    event_codes: tuple[ByteMap, ByteMap],
    analytic_transport_sha256: str,
    component_maps: tuple[IntMap, IntMap],
) -> None:
    """Reject complete reseals using only the retained public ecological evidence."""

    def compare_map(frame: FrameIndex, expected: IntMap) -> ArtifactRecord:
        actual = component_maps[frame]
        if actual.dtype != np.int32 or not np.array_equal(actual, expected):
            raise ValueError("component label map differs from independent reconstruction")
        artifact = annotation.frames[frame].component_labels
        if (
            artifact.logical_sha256 != logical_array_hash(expected)
            or artifact.shape != expected.shape
        ):
            raise ValueError("component map identity differs from independent reconstruction")
        return artifact

    expected = derive_component_topology(
        segmentations,
        surfaces,
        vectors,
        validity,
        reasons,
        event_codes,
        analytic_transport_sha256,
        compare_map,
        independent=True,
    )
    if annotation != expected:
        raise ValueError("component topology differs from independent reconstruction")
