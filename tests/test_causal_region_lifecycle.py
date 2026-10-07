"""Public software fixtures and independent raster-set reference, not physical evidence."""

from __future__ import annotations

import base64
import json
from dataclasses import replace
from itertools import pairwise
from typing import Any

import numpy as np
import pytest

from epsbench.diagnostics.boundary_observation import VisibleRaster
from epsbench.diagnostics.causal_region_lifecycle import (
    REQUIRED,
    State,
    TrustedObservation,
    action_bytes,
    advance,
    decode,
    reconstruct,
)
from epsbench.schema import Action, Modality, ModalityPermissionSet
from epsbench.utils.canonical import canonical_json_bytes

A, B = "surface-000000000000000a", "surface-000000000000000b"
PERMISSION = ModalityPermissionSet(allowed=REQUIRED)
ACTION = Action(name="forward", delta_forward=0.5, delta_lateral=0.0, delta_yaw=0.0)


def raster(i: int, pixels: list[list[int]], mapping: tuple[tuple[int, str], ...]) -> VisibleRaster:
    return VisibleRaster(i, np.asarray(pixels, dtype=np.int32), mapping)


class Fake:
    def __init__(self, observation: TrustedObservation) -> None:
        self.value = observation
        self.calls: list[int] = []

    def observation(self, i: int) -> TrustedObservation:
        self.calls.append(i)
        return self.value


def receipt(r: VisibleRaster) -> TrustedObservation:
    return TrustedObservation(r, True, True, True)


def step(state: State | None, r: VisibleRaster) -> State:
    result = advance(
        state,
        Fake(receipt(r)),
        PERMISSION,
        r.sequence_index,
        r.sequence_index,
        None if state is None else ACTION,
    )
    assert result.unresolved is None and result.state is not None
    return result.state


def independent(
    before: VisibleRaster, after: VisibleRaster, inventory: tuple[str, ...]
) -> dict[str, Any]:
    """Direct original-raster cell sets; never converter masks or its classification."""

    def support(r: VisibleRaster, token: str) -> set[tuple[int, int]]:
        labels = {label for label, k in r.identities if k == token}
        return {
            (y, x)
            for y in range(r.segmentation.shape[0])
            for x in range(r.segmentation.shape[1])
            if int(r.segmentation[y, x]) in labels
        }

    result = {}
    area = before.segmentation.size
    for k in inventory:
        old, new = support(before, k), support(after, k)
        b, a, o, g, lost = len(old), len(new), len(old & new), len(new - old), len(old - new)
        if not old and new:
            changes = [("region_appeared", g)]
        elif old and not new:
            changes = [("region_disappeared", lost)]
        else:
            changes = []
            if new - old:
                changes.append(("gained_image_pixels", g))
            if old - new:
                changes.append(("lost_image_pixels", lost))
            if not changes:
                changes = [("mask_unchanged", 0)]
        result[k] = (b, a, o, b / area, a / area, changes)
    return result


def check_reference(state: State, before: VisibleRaster, after: VisibleRaster) -> None:
    labels = reconstruct(state)
    expected = independent(before, after, state.inventory)
    assert labels.inventory == tuple(sorted(expected))
    assert len(labels.visibility) == len(labels.correspondence) == len(expected)
    for v, c in zip(labels.visibility, labels.correspondence, strict=True):
        actual_changes = [
            (e.change.value, e.affected_image_pixels)
            for e in labels.changes
            if e.surface_id == v.surface_id
        ]
        assert (
            v.before_visible_pixels,
            v.after_visible_pixels,
            c.same_image_coordinate_overlap_pixels,
            v.before_projected_image_fraction,
            v.after_projected_image_fraction,
            actual_changes,
        ) == expected[v.surface_id]


def test_lifecycle_absence_new_reappearance_and_reference() -> None:
    frames = (
        raster(0, [[1, 0, 0]], ((1, A),)),
        raster(1, [[0, 2, 0]], ((2, B),)),
        raster(2, [[0, 0, 0]], ()),
        raster(3, [[0, 0, 3]], ((3, A),)),
    )
    s = step(None, frames[0])
    assert s.inventory == (A,) and s.nodes[0].previous_mask is None
    with pytest.raises(ValueError):
        reconstruct(s)
    for before, after in pairwise(frames):
        prior_bytes = s.canonical_bytes()
        s = step(s, after)
        check_reference(s, before, after)
        assert decode(prior_bytes).canonical_bytes() == prior_bytes
        assert decode(s.canonical_bytes()) == s
    assert s.inventory == (A, B)
    assert [(n.first_seen, n.last_seen, n.status) for n in s.nodes] == [
        (0, 3, "VISIBLE"),
        (1, 1, "REMEMBERED_ABSENT"),
    ]
    assert len(s.executed_commands) == 3
    assert reconstruct(s).coverage()["inventory_denominator"] == 2


@pytest.mark.parametrize(
    "old,new",
    [
        ([1, 0, 0], [1, 0, 0]),
        ([1, 0, 0], [1, 1, 0]),
        ([1, 1, 0], [1, 0, 0]),
        ([1, 0, 0], [0, 1, 0]),
        ([1, 1, 0], [0, 0, 0]),
    ],
)
def test_exact_classification_precedence(old: list[int], new: list[int]) -> None:
    r0 = raster(0, [old], ((1, A),))
    r1 = raster(1, [new], ((1, A),) if any(new) else ())
    check_reference(step(step(None, r0), r1), r0, r1)


@pytest.mark.parametrize("field", ["complete_image", "complete_association", "stable_identity"])
def test_incomplete_atomic_and_no_gap_resume(field: str) -> None:
    s = step(None, raster(0, [[1, 0]], ((1, A),)))
    original = s.canonical_bytes()
    r = receipt(raster(1, [[0, 0]], ()))
    fields: dict[str, Any] = {field: False}
    bad = replace(r, **fields, unresolved_labels=1, unresolved_pixels=1)
    provider = Fake(bad)
    result = advance(s, provider, PERMISSION, 1, 1, ACTION)
    assert result.state is s and result.unresolved is not None
    assert result.state.canonical_bytes() == original
    assert result.unresolved.known_inventory == result.unresolved.excluded_inventory == 1
    assert result.unresolved.unresolved_pixels == 1
    assert s.nodes[0].status == "VISIBLE"
    provider.calls.clear()
    result = advance(s, provider, PERMISSION, 2, 2, ACTION)
    assert result.state is s and result.unresolved is not None and not provider.calls


def test_missing_association_is_unresolved_not_new_or_empty() -> None:
    s = step(None, raster(0, [[1]], ((1, A),)))
    provider = Fake(TrustedObservation(None, True, False, False, None, None))
    result = advance(s, provider, PERMISSION, 1, 1, ACTION)
    assert result.state is s and result.unresolved is not None
    assert result.unresolved.unresolved_labels is None
    assert result.state.inventory == (A,) and result.state.nodes[0].status == "VISIBLE"


@pytest.mark.parametrize("allowed", [frozenset(), REQUIRED | {Modality.DEPTH}])
def test_permission_denied_before_fetch(allowed: frozenset[Modality]) -> None:
    p = Fake(receipt(raster(0, [[0]], ())))
    with pytest.raises(PermissionError):
        advance(None, p, ModalityPermissionSet(allowed=allowed), 0, 0)
    assert not p.calls


def test_future_repeated_skipped_and_action_denied_before_fetch() -> None:
    p = Fake(receipt(raster(0, [[0]], ())))
    with pytest.raises(PermissionError):
        advance(None, p, PERMISSION, 1, 0)
    for i in (1, 2):
        assert advance(None, p, PERMISSION, i, i).unresolved is not None
    s = step(None, raster(0, [[0]], ()))
    assert advance(s, p, PERMISSION, 0, 0, ACTION).unresolved is not None
    assert advance(s, p, PERMISSION, 1, 1).unresolved is not None
    invalid = Action.model_construct(
        name="forward", delta_forward=float("nan"), delta_lateral=0.0, delta_yaw=0.0
    )
    assert advance(s, p, PERMISSION, 1, 1, invalid).unresolved is not None
    assert not p.calls


def test_shape_and_receipt_chronology_rejection() -> None:
    s = step(None, raster(0, [[1]], ((1, A),)))
    for r in (raster(1, [[1, 0]], ((1, A),)), raster(2, [[1]], ((1, A),))):
        result = advance(s, Fake(receipt(r)), PERMISSION, 1, 1, ACTION)
        assert result.state is s and result.unresolved is not None


def test_empty_inventory_initial_and_completed_not_applicable() -> None:
    s = step(step(None, raster(0, [[0]], ())), raster(1, [[0]], ()))
    assert decode(s.canonical_bytes()) == s
    assert reconstruct(s).coverage()["status"] == "NOT_APPLICABLE"
    assert reconstruct(s).coverage()["never_observed_units"] == "UNKNOWN_OMITTED"


def test_token_renaming_and_input_snapshots() -> None:
    raw = np.array([[1, 0, 0]], dtype=np.int32)
    r0 = VisibleRaster(0, raw, ((1, A),))
    raw[:] = 0
    assert int(r0.segmentation[0, 0]) == 1
    with pytest.raises(ValueError):
        r0.segmentation[0, 0] = 0
    left = step(step(None, r0), raster(1, [[0, 0, 0]], ()))
    right = step(step(None, raster(0, [[1, 0, 0]], ((1, B),))), raster(1, [[0, 0, 0]], ()))
    assert left.nodes[0].current_mask == right.nodes[0].current_mask
    assert left.nodes[0].previous_mask == right.nodes[0].previous_mask
    assert [e.change for e in reconstruct(left).changes] == [
        e.change for e in reconstruct(right).changes
    ]
    assert (
        action_bytes(ACTION)
        == b'{"delta_forward":0.5,"delta_lateral":0.0,"delta_yaw":0.0,"name":"forward"}'
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "extra",
        "bool_index",
        "wrong_identity",
        "wrong_version",
        "unsorted",
        "duplicate_token",
        "nonzero_padding",
        "short_mask",
        "extra_base64",
        "invalid_base64",
        "status",
        "indices",
        "missing_command",
        "bad_action",
        "null_predecessor",
        "new_with_previous",
        "overlap",
    ],
)
def test_strict_decode(mutation: str) -> None:
    s = step(step(None, raster(0, [[1, 2, 0]], ((1, A), (2, B)))), raster(1, [[0, 0, 0]], ()))
    v: Any = json.loads(s.canonical_bytes())
    n = v["nodes"][0]
    if mutation == "extra":
        v["hidden_depth"] = 0
    elif mutation == "bool_index":
        v["decision_index"] = True
    elif mutation == "wrong_identity":
        v["identity_kind"] = "SAME_FACE"
    elif mutation == "wrong_version":
        v["version"] = "other"
    elif mutation == "unsorted":
        v["nodes"].reverse()
    elif mutation == "duplicate_token":
        v["nodes"][1]["token"] = n["token"]
    elif mutation == "nonzero_padding":
        n["current_mask"] = base64.b64encode(b"\x01").decode()
    elif mutation == "short_mask":
        n["current_mask"] = ""
    elif mutation == "extra_base64":
        n["current_mask"] += "="
    elif mutation == "invalid_base64":
        n["current_mask"] = "!!!!"
    elif mutation == "status":
        n["status"] = "VISIBLE"
    elif mutation == "indices":
        n["last_seen"] = 1
    elif mutation == "missing_command":
        v["executed_commands"] = []
    elif mutation == "bad_action":
        v["executed_commands"][0]["delta_forward"] = -1.0
    elif mutation == "null_predecessor":
        n["previous_mask"] = None
    elif mutation == "new_with_previous":
        n["first_seen"] = 1
    elif mutation == "overlap":
        v["nodes"][1]["previous_mask"] = n["previous_mask"]
    with pytest.raises((ValueError, TypeError)):
        decode(canonical_json_bytes(v))


@pytest.mark.parametrize("spelling", ["newline", "whitespace", "duplicate_key", "action_integer"])
def test_noncanonical_json(spelling: str) -> None:
    s = step(step(None, raster(0, [[0]], ())), raster(1, [[0]], ()))
    encoded = s.canonical_bytes()
    if spelling == "newline":
        encoded += b"\n"
    elif spelling == "whitespace":
        encoded = encoded.replace(b":", b": ", 1)
    elif spelling == "duplicate_key":
        encoded = encoded.replace(b"{", b'{"decision_index":1,', 1)
    elif spelling == "action_integer":
        encoded = encoded.replace(b'"delta_lateral":0.0', b'"delta_lateral":0')
    with pytest.raises(ValueError):
        decode(encoded)


def test_smoke_serialization_validation_inspection() -> None:
    r0, r1 = raster(0, [[1, 0, 0]], ((1, A),)), raster(1, [[0, 0, 0]], ())
    s = step(step(None, r0), r1)
    restored = decode(s.canonical_bytes())
    check_reference(restored, r0, r1)
    assert restored.storage()["mask_payload_bytes"] == 2
    print(
        "Synthetic lifecycle smoke:",
        json.dumps(
            {
                "digest": restored.digest(),
                "storage": restored.storage(),
                "node_status": restored.nodes[0].status,
                "labels": [
                    (e.change.value, e.affected_image_pixels) for e in reconstruct(restored).changes
                ],
                "coverage": reconstruct(restored).coverage(),
            },
            sort_keys=True,
        ),
    )


@pytest.mark.parametrize("value", [-1, True, "unknown", 1])
def test_unresolved_counts_cannot_certify_absence(value: Any) -> None:
    s = step(None, raster(0, [[1]], ((1, A),)))
    observation = TrustedObservation(raster(1, [[0]], ()), True, True, True, value, None)
    result = advance(s, Fake(observation), PERMISSION, 1, 1, ACTION)
    assert result.state is s and result.unresolved is not None
    assert result.state.nodes[0].status == "VISIBLE"


def test_action_snapshot_and_forged_mutable_permission() -> None:
    s = step(None, raster(0, [[0]], ()))
    action = Action(name="forward", delta_forward=0.5, delta_lateral=0.0, delta_yaw=0.0)
    p = Fake(receipt(raster(1, [[0]], ())))
    result = advance(s, p, PERMISSION, 1, 1, action)
    assert result.state is not None and result.unresolved is None
    saved = result.state.canonical_bytes()
    object.__setattr__(action, "delta_forward", 9.0)
    assert result.state.canonical_bytes() == saved
    p.calls.clear()
    mutable: Any = set(REQUIRED)
    forged = ModalityPermissionSet.model_construct(allowed=mutable)
    with pytest.raises(PermissionError):
        advance(s, p, forged, 1, 1, ACTION)
    assert not p.calls


def test_provider_failure_leaves_prior_state_unchanged() -> None:
    class Broken:
        def observation(self, i: int) -> TrustedObservation:
            raise RuntimeError("public provider failure")

    s = step(None, raster(0, [[1]], ((1, A),)))
    saved = s.canonical_bytes()
    with pytest.raises(RuntimeError, match="public provider failure"):
        advance(s, Broken(), PERMISSION, 1, 1, ACTION)
    assert s.canonical_bytes() == saved


def test_impossible_first_seen_predecessor_rejected() -> None:
    s = step(step(None, raster(0, [[1]], ((1, A),))), raster(1, [[1]], ((1, A),)))
    value = json.loads(s.canonical_bytes())
    value["nodes"][0]["previous_mask"] = "AA=="
    with pytest.raises(ValueError, match="first-seen predecessor"):
        decode(canonical_json_bytes(value))
