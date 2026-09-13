"""Closed-form public-only topology cases, independent algorithms and strict domains."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from epsbench.annotations.component_topology import (
    FrameIndex,
    IntMap,
    derive_component_topology,
    domain_hash,
    extract_components,
    seal_annotation,
    validate_component_topology,
)
from epsbench.schema import (
    ArtifactRecord,
    ComponentTopologyAnnotation,
    Modality,
    SurfaceReference,
)
from epsbench.schema import (
    ComponentEventKind as Kind,
)
from epsbench.utils.canonical import canonical_json_bytes, logical_array_hash

SURFACES = (
    SurfaceReference(surface_id="surface-1111111111111111", segmentation_label=1),
    SurfaceReference(surface_id="surface-2222222222222222", segmentation_label=2),
)


def artifact(frame: FrameIndex, array: IntMap) -> ArtifactRecord:
    return ArtifactRecord(
        path=f"component_{frame}.npy",
        modality=Modality.COMPONENT_TOPOLOGY,
        media_type="application/x-npy",
        dtype="int32",
        shape=array.shape,
        logical_sha256=logical_array_hash(array),
        file_sha256="0" * 64,
        byte_count=1,
    )


def fixture_topology(
    before: list[list[int]],
    after: list[list[int]],
    forward: tuple[tuple[int, int], ...] = (),
    backward: tuple[tuple[int, int], ...] = (),
    *,
    codes: tuple[int, int] = (0, 0),
    independent: bool = False,
    substitution: ComponentTopologyAnnotation | None = None,
) -> ComponentTopologyAnnotation:
    """Mapping endpoints are flattened row-major pixel indices, never component labels."""
    arrays = np.asarray(before, dtype=np.int32), np.asarray(after, dtype=np.int32)
    shape = arrays[0].shape
    vectors = tuple(np.zeros((*shape, 2), dtype=np.int32) for _ in range(2))
    validity = tuple(np.zeros(shape, dtype=np.uint8) for _ in range(2))
    reasons = tuple(np.full(shape, 4, dtype=np.uint8) for _ in range(2))
    for direction, pairs in enumerate((forward, backward)):
        for source, target in pairs:
            r, c = divmod(source, shape[1])
            y, x = divmod(target, shape[1])
            vectors[direction][r, c] = (1024 * (x - c), 1024 * (y - r))
            validity[direction][r, c] = 1
            reasons[direction][r, c] = 0
    events = tuple(np.full(shape, code, dtype=np.uint8) for code in codes)
    annotation = derive_component_topology(
        arrays,
        SURFACES,
        (vectors[0], vectors[1]),
        (validity[0], validity[1]),
        (reasons[0], reasons[1]),
        (events[0], events[1]),
        "0" * 64,
        artifact,
        independent=independent,
    )
    if substitution is not None:
        maps = (
            extract_components(arrays[0], SURFACES, 0)[0],
            extract_components(arrays[1], SURFACES, 1)[0],
        )
        validate_component_topology(
            substitution,
            arrays,
            SURFACES,
            (vectors[0], vectors[1]),
            (validity[0], validity[1]),
            (reasons[0], reasons[1]),
            (events[0], events[1]),
            "0" * 64,
            maps,
        )
    return annotation


@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize(
    ("before", "after", "forward", "expected"),
    [
        ([[1, 1, 1]], [[1, 1, 1]], ((0, 0), (1, 1), (2, 2)), [Kind.CONTINUATION]),
        ([[1, 1, 1]], [[1, 0, 1]], ((0, 0), (2, 2)), [Kind.SPLIT]),
        ([[1, 0, 1]], [[1, 1, 1]], ((0, 0), (2, 2)), [Kind.MERGE]),
        ([[0, 0, 0]], [[1, 0, 1]], (), [Kind.APPEARANCE, Kind.APPEARANCE]),
        ([[1, 0, 1]], [[0, 0, 0]], (), [Kind.DISAPPEARANCE, Kind.DISAPPEARANCE]),
        ([[1, 1, 0, 1, 1]], [[1, 1, 0, 1, 1]], ((0, 0), (1, 3), (3, 0), (4, 3)), [Kind.COMPLEX]),
        ([[1, 1, 1]], [[1, 0, 1]], (), [Kind.INDETERMINATE] * 3),
        ([[1]], [[1]], ((0, 0),), [Kind.CONTINUATION]),
    ],
)
def test_closed_form_events(
    before: list[list[int]],
    after: list[list[int]],
    forward: tuple[tuple[int, int], ...],
    expected: list[Kind],
    independent: bool,
) -> None:
    annotation = fixture_topology(before, after, forward, independent=independent)
    assert sorted(e.kind for e in annotation.events) == sorted(expected)
    assert annotation.status == ("indeterminate" if Kind.INDETERMINATE in expected else "available")
    # The complete Cartesian table retains one-direction-only facts as zero backward counts.
    assert all(s.backward_count == 0 for s in annotation.supports)


def test_whole_surface_lifecycle_is_distinct_from_multiple_component_events() -> None:
    annotation = fixture_topology([[0, 0, 0]], [[1, 0, 1]])
    assert [e.kind for e in annotation.events] == [Kind.APPEARANCE] * 2
    assert len({e.surface_id for e in annotation.events}) == 1


@pytest.mark.parametrize("code", [1, 2])
def test_frame_exit_or_deletion_is_not_split_or_merge(code: int) -> None:
    annotation = fixture_topology([[1, 0, 1]], [[1, 0, 0]], ((0, 0),), codes=(code, 0))
    assert sorted(e.kind for e in annotation.events) == sorted(
        [Kind.CONTINUATION, Kind.DISAPPEARANCE]
    )
    reverse = fixture_topology([[1, 0, 0]], [[1, 0, 1]], ((0, 0),), codes=(0, code))
    assert sorted(e.kind for e in reverse.events) == sorted([Kind.CONTINUATION, Kind.APPEARANCE])


@pytest.mark.parametrize("code", [1, 2])
def test_causal_pixels_do_not_force_component_lifecycle_with_supported_continuation(
    code: int,
) -> None:
    annotation = fixture_topology([[1, 1]], [[1, 1]], ((0, 0),), codes=(code, code))
    assert [e.kind for e in annotation.events] == [Kind.CONTINUATION]


def test_four_connectivity_one_pixels_background_and_surface_local_rank() -> None:
    array = np.asarray([[1, 0, 2], [0, 1, 0]], dtype=np.int32)
    mapping, components = extract_components(array, SURFACES, 0)
    independent_map, independent_components = extract_components(
        array, SURFACES, 0, independent=True
    )
    np.testing.assert_array_equal(mapping, [[1, 0, 3], [0, 2, 0]])
    np.testing.assert_array_equal(mapping, independent_map)
    assert components == independent_components
    assert [c.pixel_count for c in components] == [1, 1, 1]
    assert [c.surface_component_index for c in components] == [0, 1, 0]
    assert len({c.component_id for c in components}) == 3


def test_reversal_has_split_merge_duality_and_separate_direction_counts() -> None:
    split = fixture_topology([[1, 1, 1]], [[1, 0, 1]], ((0, 0), (2, 2)), ((0, 0),))
    merge = fixture_topology([[1, 0, 1]], [[1, 1, 1]], ((0, 0),), ((0, 0), (2, 2)))
    assert [e.kind for e in split.events] == [Kind.SPLIT]
    assert [e.kind for e in merge.events] == [Kind.MERGE]
    assert sorted((s.forward_count, s.backward_count) for s in split.supports) == [(1, 0), (1, 1)]
    assert sorted((s.backward_count, s.forward_count) for s in merge.supports) == [(1, 0), (1, 1)]


@pytest.mark.parametrize(
    ("dx", "supported"), [(511, False), (512, True), (1535, True), (1536, False), (-513, False)]
)
@pytest.mark.parametrize("independent", [False, True])
def test_exact_half_open_target_cells_without_clamping(
    dx: int, supported: bool, independent: bool
) -> None:
    before = np.asarray([[1, 0]], dtype=np.int32)
    after = np.asarray([[0, 1]], dtype=np.int32)
    flow = np.asarray([[[dx, 0], [0, 0]]], dtype=np.int32)
    zero_flow = np.zeros_like(flow)
    annotation = derive_component_topology(
        (before, after),
        SURFACES,
        (flow, zero_flow),
        (np.asarray([[1, 0]], dtype=np.uint8), np.zeros((1, 2), dtype=np.uint8)),
        (np.asarray([[0, 4]], dtype=np.uint8), np.full((1, 2), 4, dtype=np.uint8)),
        (np.zeros((1, 2), dtype=np.uint8), np.zeros((1, 2), dtype=np.uint8)),
        "0" * 64,
        artifact,
        independent=independent,
    )
    assert annotation.supports[0].edge is supported
    assert annotation.supports[0].forward_count == int(supported)


def test_cross_surface_targets_and_zero_pairs_are_retained_without_edges() -> None:
    annotation = fixture_topology([[1, 0, 2]], [[2, 0, 1]], ((0, 0), (2, 2)))
    assert len(annotation.supports) == 2
    assert all(not s.edge and s.forward_count == s.backward_count == 0 for s in annotation.supports)
    assert all(e.kind == Kind.INDETERMINATE for e in annotation.events)


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "connectivity",
        "coordinates",
        "component_id_domain",
        "map_domain",
        "set_domain",
        "support_domain",
        "event_domain",
        "annotation_domain",
        "portable_domain",
        "target_cell_rule",
        "support_rule",
        "classification_rule",
        "acceptance_rule",
    ],
)
def test_unknown_versions_and_domains_rejected(field: str) -> None:
    payload = fixture_topology([[1]], [[1]], ((0, 0),)).model_dump(mode="json")
    payload[field] = "unknown_v99"
    with pytest.raises(ValidationError):
        ComponentTopologyAnnotation.model_validate_json(canonical_json_bytes(payload))


@pytest.mark.parametrize(
    "field",
    [
        "rgb",
        "depth",
        "camera_world_position",
        "raw_geom_id",
        "semantic_name",
        "profile_id",
        "benchmark_role",
        "final_root_registry",
        "generation_record",
    ],
)
def test_public_record_rejects_privileged_fields_at_each_level(field: str) -> None:
    for level in ("annotation", "component", "event", "support"):
        payload = fixture_topology([[1]], [[1]], ((0, 0),)).model_dump(mode="json")
        target: dict[str, Any] = (
            payload
            if level == "annotation"
            else payload["frames"][0]["components"][0]
            if level == "component"
            else payload["events"][0]
            if level == "event"
            else payload["supports"][0]
        )
        target[field] = "forbidden"
        with pytest.raises(ValidationError):
            ComponentTopologyAnnotation.model_validate_json(canonical_json_bytes(payload))


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "missing",
        "extra",
        "order",
        "surface",
        "event",
        "omit_indeterminate",
        "cross_edge",
    ],
)
def test_noncanonical_complete_graphs_fail_closed(mutation: str) -> None:
    payload = fixture_topology([[1, 0, 1, 2]], [[1, 0, 1, 2]]).model_dump(mode="json")
    components = payload["frames"][0]["components"]
    if mutation == "duplicate":
        components[1]["component_id"] = components[0]["component_id"]
    elif mutation == "missing":
        components.pop()
    elif mutation == "extra":
        components.append(components[0])
    elif mutation == "order":
        components.reverse()
    elif mutation == "surface":
        components[0]["surface_id"] = SURFACES[1].surface_id
    elif mutation == "event":
        payload["events"][0]["kind"] = "unknown_event"
    elif mutation == "omit_indeterminate":
        payload["events"].pop()
    else:
        payload["supports"][0]["surface_id"] = SURFACES[1].surface_id
    with pytest.raises(ValidationError):
        ComponentTopologyAnnotation.model_validate_json(canonical_json_bytes(payload))


def test_unique_logical_domains_and_independent_equality() -> None:
    assert (
        len(
            {
                domain_hash(k, {})
                for k in (
                    "component_id",
                    "component_mask",
                    "component_map",
                    "component_set",
                    "supports",
                    "events",
                    "event_id",
                    "annotation",
                    "portable_graph",
                    "definition",
                    "lock",
                )
            }
        )
        == 11
    )
    kwargs = {
        "before": [[1, 1, 0, 1, 1]],
        "after": [[1, 1, 0, 1, 1]],
        "forward": ((0, 0), (1, 3), (3, 0), (4, 3)),
    }
    assert fixture_topology(**kwargs) == fixture_topology(**kwargs, independent=True)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("before", "after", "pairs", "wrong_kind"),
    [
        ([[1, 1, 1]], [[1, 0, 1]], ((0, 0), (2, 2)), Kind.MERGE),
        ([[1, 0, 1]], [[1, 1, 1]], ((0, 0), (2, 2)), Kind.SPLIT),
        ([[1, 1, 0, 1, 1]], [[1, 1, 0, 1, 1]], ((0, 0), (1, 3), (3, 0), (4, 3)), Kind.SPLIT),
    ],
)
def test_resealed_false_split_merge_and_hidden_complex_are_rejected(
    before: list[list[int]],
    after: list[list[int]],
    pairs: tuple[tuple[int, int], ...],
    wrong_kind: Kind,
) -> None:
    annotation = fixture_topology(before, after, pairs)
    event = annotation.events[0].model_copy(update={"kind": wrong_kind})
    event = event.model_copy(
        update={
            "event_id": domain_hash("event_id", event.model_dump(mode="json", exclude={"event_id"}))
        }
    )
    false = seal_annotation(annotation.model_copy(update={"events": (event,)}))
    with pytest.raises(ValueError, match="independent reconstruction"):
        fixture_topology(before, after, pairs, substitution=false)
