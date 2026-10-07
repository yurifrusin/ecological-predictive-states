"""Independent public endpoint cases; no annotation producer or physical qualification."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from epsbench.diagnostics.boundary_observation import VisibleRaster
from epsbench.diagnostics.causal_region_lifecycle import (
    REQUIRED as LIFECYCLE_PERMISSIONS,
)
from epsbench.diagnostics.causal_region_lifecycle import (
    State,
    TrustedObservation,
)
from epsbench.diagnostics.causal_region_lifecycle import (
    advance as observe,
)
from epsbench.diagnostics.observed_contour_relation import (
    REQUIRED,
    RULES,
    EndpointEvidence,
    Pair,
    Reference,
    advance,
    bind,
    decode,
    initialize,
)
from epsbench.schema import (
    Action,
    BoundaryAxis,
    BoundaryKind,
    BoundaryOwnerSide,
    Modality,
    ModalityPermissionSet,
    OcclusionRelation,
    OrientedBoundaryElement,
    UnavailableOcclusionAnnotation,
)
from epsbench.utils.canonical import canonical_json_bytes, sha256_bytes

A, B, C = ("surface-" + f"{v:016x}" for v in (1, 2, 3))
EPISODE = "a" * 32
PERMISSIONS = ModalityPermissionSet(allowed=REQUIRED)
ACTION = Action(name="forward", delta_forward=0.5, delta_lateral=0.0, delta_yaw=0.0)


class RasterProvider:
    def __init__(self, raster: VisibleRaster) -> None:
        self.raster = raster

    def observation(self, i: int) -> TrustedObservation:
        return TrustedObservation(self.raster, True, True, True)


def lifecycle(previous: State | None = None, visible: bool = True) -> State:
    i = 0 if previous is None else previous.decision_index + 1
    pixels = [[1, 2, 1]] if visible else [[1, 0, 1]]
    raster = VisibleRaster(
        i, np.array(pixels, dtype=np.int32), ((1, A), (2, B)) if visible else ((1, A),)
    )
    result = observe(
        previous,
        RasterProvider(raster),
        ModalityPermissionSet(allowed=LIFECYCLE_PERMISSIONS),
        i,
        i,
        None if previous is None else ACTION,
    )
    assert result.state is not None and result.unresolved is None
    return result.state


def contour(owner: str = A, affected: str = B, col: int = 0) -> OrientedBoundaryElement:
    return OrientedBoundaryElement(
        frame_index=1,
        axis=BoundaryAxis.HORIZONTAL,
        row=0,
        column=col,
        negative_surface_id=owner,
        positive_surface_id=affected,
        kind=BoundaryKind.OCCLUDING_CONTOUR,
        owner_side=BoundaryOwnerSide.NEGATIVE_AXIS_SIDE,
        owner_surface_id=owner,
    )


def evidence(
    state: State,
    elements: tuple[OrientedBoundaryElement, ...] | None = None,
    rule: str = "oriented_boundary_ownership_complete_v2",
) -> EndpointEvidence:
    elements = (contour(),) if elements is None else elements
    # Handwritten reported pair, not production aggregation.
    reports = (
        (OcclusionRelation(occluder_surface_id=A, occluded_surface_id=B, frame_indices=(1,)),)
        if elements
        else ()
    )
    return EndpointEvidence(bind(EPISODE, state), True, True, True, elements, reports, rule)


class Fake:
    def __init__(self, value: EndpointEvidence) -> None:
        self.value = value
        self.calls: list[int] = []

    def endpoint_one(self, i: int) -> EndpointEvidence:
        self.calls.append(i)
        return self.value


def project(ref: Reference, state: State, value: EndpointEvidence, predecessor: State) -> Reference:
    update = advance(
        ref, state, EPISODE, Fake(value), PERMISSIONS, state.decision_index, predecessor=predecessor
    )
    assert update.unresolved is None
    return update.reference


def independent(elements: tuple[OrientedBoundaryElement, ...]) -> set[tuple[str, str]]:
    """Original endpoint records; independent owner-side selection, no module aggregation."""
    expected = set()
    for e in elements:
        if e.frame_index == 1 and e.kind == BoundaryKind.OCCLUDING_CONTOUR:
            if e.owner_side == BoundaryOwnerSide.NEGATIVE_AXIS_SIDE:
                assert e.negative_surface_id is not None and e.positive_surface_id is not None
                expected.add((e.negative_surface_id, e.positive_surface_id))
            else:
                assert e.positive_surface_id is not None and e.negative_surface_id is not None
                expected.add((e.positive_surface_id, e.negative_surface_id))
    return expected


@pytest.mark.parametrize("rule", sorted(RULES))
def test_positive_empty_stale_unavailable_roundtrip(rule: str) -> None:
    s0 = lifecycle()
    initial = initialize(s0, EPISODE)
    assert not initial.pairs and initial.availability == "INITIAL"
    s1 = lifecycle(s0)
    ev = evidence(s1, rule=rule)
    current = project(initial, s1, ev, s0)
    assert {(p.owner, p.affected) for p in current.current()} == independent(ev.elements)
    assert current.oracle_rule == rule
    s2 = lifecycle(s1)
    empty = project(current, s2, evidence(s2, (), rule), s1)
    assert empty.availability == "AVAILABLE" and not empty.current()
    assert empty.pairs[0].latest_support == 1
    s3 = lifecycle(s2, False)
    unavailable = EndpointEvidence(
        bind(EPISODE, s3),
        False,
        False,
        True,
        (),
        (),
        None,
        UnavailableOcclusionAnnotation(
            status="unavailable",
            reason_category="oriented_corridor_occlusion_oracle_unavailable",
            reason="public fixture unavailable",
        ),
    )
    stale = project(empty, s3, unavailable, s2)
    assert (
        stale.availability == "UNAVAILABLE" and stale.pairs == current.pairs and not stale.current()
    )
    assert decode(stale.canonical_bytes()) == stale
    assert "lifecycle_root" not in json.loads(stale.feature_bytes())
    assert "episode_key" not in json.loads(stale.feature_bytes())


def test_opposite_pairs_and_ambiguous_coexistence() -> None:
    s0, s1 = lifecycle(), lifecycle(lifecycle())
    elements = (contour(), contour(B, A, 1))
    reports = tuple(
        OcclusionRelation(occluder_surface_id=o, occluded_surface_id=a, frame_indices=(1,))
        for o, a in ((A, B), (B, A))
    )
    value = replace(evidence(s1, elements), reported_pairs=reports)
    result = project(initialize(s0, EPISODE), s1, value, s0)
    assert {(p.owner, p.affected) for p in result.current()} == independent(elements)
    ambiguous = OrientedBoundaryElement(
        frame_index=1,
        axis=BoundaryAxis.HORIZONTAL,
        row=0,
        column=1,
        negative_surface_id=B,
        positive_surface_id=A,
        kind=BoundaryKind.MULTI_SURFACE_JUNCTION_AMBIGUOUS,
        owner_side=BoundaryOwnerSide.NONE,
        owner_surface_id=None,
    )
    value = evidence(s1, (contour(), ambiguous))
    assert len(project(initialize(s0, EPISODE), s1, value, s0).current()) == 1


@pytest.mark.parametrize(
    "case",
    [
        "missing_report",
        "extra_unknown_report",
        "unknown_positive",
        "absent_side",
        "visible_binding",
        "episode",
        "digest",
        "shape",
        "index",
        "endpoint_zero",
        "report_both_endpoints",
        "mixed_availability",
        "unavailable_with_positive",
        "missing_unavailable",
        "third_rule",
        "duplicate_location",
    ],
)
def test_contradictions_atomic(case: str) -> None:
    s0 = lifecycle()
    s1 = lifecycle(s0, case != "absent_side")
    initial = initialize(s0, EPISODE)
    value = evidence(s1)
    if case == "missing_report":
        value = replace(value, reported_pairs=())
    elif case == "extra_unknown_report":
        value = replace(
            value,
            reported_pairs=(
                *value.reported_pairs,
                OcclusionRelation(occluder_surface_id=A, occluded_surface_id=C, frame_indices=(1,)),
            ),
        )
    elif case == "unknown_positive":
        value = replace(
            value,
            elements=(contour(A, C),),
            reported_pairs=(
                OcclusionRelation(occluder_surface_id=A, occluded_surface_id=C, frame_indices=(1,)),
            ),
        )
    elif case == "visible_binding":
        value = replace(value, exact_visible_binding=False)
    elif case == "episode":
        value = replace(value, binding=replace(value.binding, episode_key="b" * 32))
    elif case == "digest":
        value = replace(value, binding=replace(value.binding, lifecycle_root="0" * 64))
    elif case == "shape":
        value = replace(value, binding=replace(value.binding, shape=(1, 4)))
    elif case == "index":
        value = replace(value, binding=replace(value.binding, observation_index=2))
    elif case == "endpoint_zero":
        value = replace(value, elements=(value.elements[0].model_copy(update={"frame_index": 0}),))
    elif case == "report_both_endpoints":
        value = replace(
            value,
            reported_pairs=(value.reported_pairs[0].model_copy(update={"frame_indices": (0, 1)}),),
        )
    elif case == "mixed_availability":
        value = replace(value, boundary_available=False)
    elif case == "unavailable_with_positive":
        value = replace(value, boundary_available=False, occlusion_available=False)
    elif case == "missing_unavailable":
        value = replace(
            value,
            boundary_available=False,
            occlusion_available=False,
            elements=(),
            reported_pairs=(),
            oracle_rule=None,
        )
    elif case == "third_rule":
        value = replace(value, oracle_rule="invented_rule")
    elif case == "duplicate_location":
        value = replace(value, elements=(*value.elements, *value.elements))
    before = initial.canonical_bytes()
    result = advance(initial, s1, EPISODE, Fake(value), PERMISSIONS, 1, predecessor=s0)
    assert result.unresolved is not None and result.reference is initial
    assert result.reference.canonical_bytes() == before
    if case == "extra_unknown_report":
        assert "contradict" in result.unresolved  # Equality checked before unknown-token treatment.


def test_prefetch_permissions_future_chronology_episode() -> None:
    s0 = lifecycle()
    s1 = lifecycle(s0)
    r0 = initialize(s0, EPISODE)
    p = Fake(evidence(s1))
    for allowed in (frozenset(), REQUIRED | {Modality.DEPTH}, LIFECYCLE_PERMISSIONS):
        with pytest.raises(PermissionError):
            advance(r0, s1, EPISODE, p, ModalityPermissionSet(allowed=allowed), 1, predecessor=s0)
    with pytest.raises(PermissionError):
        advance(r0, s1, EPISODE, p, PERMISSIONS, 0, predecessor=s0)
    assert advance(r0, s0, EPISODE, p, PERMISSIONS, 0, predecessor=s0).unresolved is not None
    assert (
        advance(r0, lifecycle(s1), EPISODE, p, PERMISSIONS, 2, predecessor=s0).unresolved
        is not None
    )
    assert advance(r0, s1, "b" * 32, p, PERMISSIONS, 1, predecessor=s0).unresolved is not None
    assert not p.calls


@pytest.mark.parametrize(
    "mutation",
    [
        "extra",
        "duplicate_key",
        "newline",
        "bool_index",
        "initial_pairs",
        "unsupported_rule",
        "future_support",
        "reflexive",
        "duplicate_pair",
        "wrong_identity",
    ],
)
def test_decode_strict(mutation: str) -> None:
    s0 = lifecycle()
    s1 = lifecycle(s0)
    result = project(initialize(s0, EPISODE), s1, evidence(s1), s0)
    v: Any = json.loads(result.canonical_bytes())
    if mutation == "extra":
        v["raw_id"] = 9
    elif mutation == "bool_index":
        v["binding"]["observation_index"] = True
    elif mutation == "initial_pairs":
        v["binding"]["observation_index"] = 0
    elif mutation == "unsupported_rule":
        v["oracle_rule"] = "other"
    elif mutation == "future_support":
        v["pairs"][0]["latest_support"] = 2
    elif mutation == "reflexive":
        v["pairs"][0]["affected"] = A
    elif mutation == "duplicate_pair":
        v["pairs"].append(v["pairs"][0])
    elif mutation == "wrong_identity":
        v["binding"]["identity_kind"] = "FACE"
    encoded = canonical_json_bytes(v)
    if mutation == "duplicate_key":
        encoded = encoded.replace(b"{", b'{"version":"observed-contour-relation-v1",', 1)
    elif mutation == "newline":
        encoded += b"\n"
    with pytest.raises(ValueError):
        decode(encoded)


def test_renaming_and_smoke_inspection() -> None:
    s0 = lifecycle()
    s1 = lifecycle(s0)
    result = project(initialize(s0, EPISODE), s1, evidence(s1), s0)
    restored = decode(result.canonical_bytes())
    assert restored == result and len(restored.current()) == 1
    v = json.loads(result.canonical_bytes())
    v["pairs"][0]["owner"], v["pairs"][0]["affected"] = B, A
    renamed = decode(canonical_json_bytes(v))
    assert renamed.pairs[0].latest_support == result.pairs[0].latest_support
    assert len(renamed.current()) == len(result.current())
    print(
        "Synthetic contour smoke:",
        json.dumps(
            {
                "digest": sha256_bytes(restored.canonical_bytes()),
                "storage": restored.storage(),
                "availability": restored.availability,
                "latest_support": restored.pairs[0].latest_support,
                "physical_truth": "UNQUALIFIED",
                "prediction": "ABSENT",
            },
            sort_keys=True,
        ),
    )


def test_positive_side_owner_refresh_and_full_projection_renaming() -> None:
    s0 = lifecycle()
    s1 = lifecycle(s0)
    element = contour().model_copy(
        update={"owner_side": BoundaryOwnerSide.POSITIVE_AXIS_SIDE, "owner_surface_id": B}
    )
    ev = replace(
        evidence(s1, (element,)),
        reported_pairs=(
            OcclusionRelation(occluder_surface_id=B, occluded_surface_id=A, frame_indices=(1,)),
        ),
    )
    first = project(initialize(s0, EPISODE), s1, ev, s0)
    assert {(p.owner, p.affected) for p in first.current()} == independent(ev.elements)
    s2 = lifecycle(s1)
    refreshed = project(first, s2, replace(ev, binding=bind(EPISODE, s2)), s1)
    assert refreshed.pairs[0].latest_support == 2 and len(refreshed.current()) == 1
    rename = {A: B, B: A}
    states = []
    previous = None
    for i in (0, 1):
        r = VisibleRaster(i, np.array([[1, 2, 1]], dtype=np.int32), ((1, B), (2, A)))
        u = observe(
            previous,
            RasterProvider(r),
            ModalityPermissionSet(allowed=LIFECYCLE_PERMISSIONS),
            i,
            i,
            None if previous is None else ACTION,
        )
        assert u.state is not None and u.unresolved is None
        previous = u.state
        states.append(previous)
    e = element.model_copy(
        update={"negative_surface_id": B, "positive_surface_id": A, "owner_surface_id": A}
    )
    renamed_ev = replace(
        ev,
        binding=bind(EPISODE, states[1]),
        elements=(e,),
        reported_pairs=(
            OcclusionRelation(occluder_surface_id=A, occluded_surface_id=B, frame_indices=(1,)),
        ),
    )
    renamed = project(initialize(states[0], EPISODE), states[1], renamed_ev, states[0])
    assert {(p.owner, p.affected, p.latest_support) for p in renamed.pairs} == {
        (rename[p.owner], rename[p.affected], p.latest_support) for p in first.pairs
    }


def test_provider_failure_and_malformed_owned_record_retention() -> None:
    s0 = lifecycle()
    s1 = lifecycle(s0)
    initial = initialize(s0, EPISODE)
    saved = initial.canonical_bytes()

    class Broken:
        def endpoint_one(self, observation_index: int) -> EndpointEvidence:
            raise RuntimeError("public endpoint failure")

    with pytest.raises(RuntimeError, match="public endpoint failure"):
        advance(initial, s1, EPISODE, Broken(), PERMISSIONS, 1, predecessor=s0)
    assert initial.canonical_bytes() == saved
    ev = evidence(s1)
    bad = ev.elements[0].model_copy(update={"owner_surface_id": C})
    result = advance(
        initial, s1, EPISODE, Fake(replace(ev, elements=(bad,))), PERMISSIONS, 1, predecessor=s0
    )
    assert result.unresolved is not None and result.reference is initial
    assert result.reference.canonical_bytes() == saved


def test_initialization_requires_zero_and_current_inventory_cannot_drop_history() -> None:
    s0 = lifecycle()
    s1 = lifecycle(s0)
    with pytest.raises(ValueError):
        initialize(s1, EPISODE)
    supported = project(initialize(s0, EPISODE), s1, evidence(s1), s0)
    s2 = lifecycle(s1)
    contradictory_lifecycle = replace(s2, nodes=())
    p = Fake(evidence(contradictory_lifecycle, ()))
    rejected = advance(
        supported, contradictory_lifecycle, EPISODE, p, PERMISSIONS, 2, predecessor=s1
    )
    assert rejected.unresolved is not None and rejected.reference is supported and not p.calls


@pytest.mark.parametrize(
    "case", ["root", "branch", "commands", "first_seen", "last_seen", "new_history"]
)
def test_predecessor_binding_and_continuation_before_fetch(case: str) -> None:
    s0 = lifecycle()
    s1 = lifecycle(s0)
    s2 = lifecycle(s1, False)
    s3 = lifecycle(s2, False)
    prior = project(initialize(s0, EPISODE), s1, evidence(s1), s0)
    predecessor, current = s1, s2
    if case == "root":
        prior = replace(prior, binding=replace(prior.binding, lifecycle_root="0" * 64))
    elif case == "branch":
        predecessor = lifecycle(s0, False)
        prior = replace(prior, binding=bind(EPISODE, predecessor))
    elif case == "commands":
        changed = canonical_json_bytes(ACTION.model_copy(update={"delta_forward": 0.25}))
        current = replace(s2, executed_commands=(changed, *s2.executed_commands[1:]))
    else:
        prior = project(prior, s2, evidence(s2, ()), s1)
        predecessor, current = s2, s3
        nodes = list(current.nodes)
        if case == "first_seen":
            nodes[0] = replace(nodes[0], first_seen=1)
        elif case == "last_seen":
            nodes[1] = replace(nodes[1], last_seen=0)
        else:
            nodes.append(replace(nodes[1], token=C))
        current = replace(current, nodes=tuple(nodes))
    provider = Fake(evidence(current, ()))
    before = prior.canonical_bytes()
    result = advance(
        prior,
        current,
        EPISODE,
        provider,
        PERMISSIONS,
        current.decision_index,
        predecessor=predecessor,
    )
    assert result.unresolved is not None and result.reference is prior
    assert result.reference.canonical_bytes() == before and not provider.calls


@pytest.mark.parametrize("case", ["duplicate", "noncanonical"])
def test_report_list_canonical_before_pair_equality(case: str) -> None:
    s0 = lifecycle()
    s1 = lifecycle(s0)
    prior = initialize(s0, EPISODE)
    ev = evidence(s1, (contour(), contour(B, A, 1)))
    ab = ev.reported_pairs[0]
    ba = OcclusionRelation(occluder_surface_id=B, occluded_surface_id=A, frame_indices=(1,))
    reports = (ab, ab, ba) if case == "duplicate" else (ba, ab)
    provider = Fake(replace(ev, reported_pairs=reports))
    before = prior.canonical_bytes()
    result = advance(prior, s1, EPISODE, provider, PERMISSIONS, 1, predecessor=s0)
    assert result.unresolved is not None and "canonical" in result.unresolved
    assert result.reference is prior and prior.canonical_bytes() == before
    assert provider.calls == [1]


def test_accepted_continuation_can_add_new_observed_region() -> None:
    s0 = lifecycle()
    s1 = lifecycle(s0)
    prior = project(initialize(s0, EPISODE), s1, evidence(s1), s0)
    raster = VisibleRaster(2, np.array([[1, 2, 1]], dtype=np.int32), ((1, A), (2, C)))
    result = observe(
        s1,
        RasterProvider(raster),
        ModalityPermissionSet(allowed=LIFECYCLE_PERMISSIONS),
        2,
        2,
        ACTION,
    )
    assert result.state is not None and result.unresolved is None
    s2 = result.state
    ev = replace(
        evidence(s2, (contour(A, C),)),
        reported_pairs=(
            OcclusionRelation(occluder_surface_id=A, occluded_surface_id=C, frame_indices=(1,)),
        ),
    )
    accepted = project(prior, s2, ev, s1)
    assert {(p.owner, p.affected) for p in accepted.current()} == {(A, C)}
    assert accepted.pairs[0].latest_support == 1


def test_incoming_history_cannot_claim_new_current_token() -> None:
    s0 = lifecycle()
    s1 = lifecycle(s0)
    prior = project(initialize(s0, EPISODE), s1, evidence(s1), s0)
    raster = VisibleRaster(2, np.array([[1, 2, 1]], dtype=np.int32), ((1, A), (2, C)))
    result = observe(
        s1,
        RasterProvider(raster),
        ModalityPermissionSet(allowed=LIFECYCLE_PERMISSIONS),
        2,
        2,
        ACTION,
    )
    assert result.state is not None
    s2 = result.state
    forged = replace(prior, pairs=(Pair(A, C, 1),))
    provider = Fake(evidence(s2, ()))
    before = forged.canonical_bytes()
    rejected = advance(forged, s2, EPISODE, provider, PERMISSIONS, 2, predecessor=s1)
    assert rejected.unresolved is not None and rejected.reference is forged
    assert rejected.reference.canonical_bytes() == before and not provider.calls
