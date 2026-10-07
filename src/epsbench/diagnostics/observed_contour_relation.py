"""Endpoint-one sampled contour positives; separate from the frozen lifecycle format."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Literal, Protocol

from epsbench.diagnostics.causal_region_lifecycle import IDENTITY, State, integer, keys
from epsbench.schema import (
    BoundaryKind,
    Modality,
    ModalityPermissionSet,
    OcclusionRelation,
    OrientedBoundaryElement,
    UnavailableOcclusionAnnotation,
)
from epsbench.utils.canonical import canonical_json_bytes

VERSION = "observed-contour-relation-v1"
REQUIRED = frozenset({Modality.ORIENTED_BOUNDARY_OWNERSHIP, Modality.OCCLUSION_ANNOTATION})
RULES = frozenset(
    {
        "oriented_boundary_ownership_complete_v2",
        "oriented_boundary_ownership_with_counterfactual_crosscheck_v1",
    }
)


@dataclass(frozen=True)
class Binding:
    """Trusted episode/lifecycle provenance, excluded from relation feature bytes."""

    episode_key: str
    lifecycle_root: str
    observation_index: int
    shape: tuple[int, int]
    identity_kind: str = IDENTITY

    def __post_init__(self) -> None:
        if (
            type(self.episode_key) is not str
            or re.fullmatch(r"[0-9a-f]{32}", self.episode_key) is None
        ):
            raise ValueError("opaque same-episode binding required")
        if (
            type(self.lifecycle_root) is not str
            or re.fullmatch(r"[0-9a-f]{64}", self.lifecycle_root) is None
        ):
            raise ValueError("exact lifecycle digest required")
        integer(self.observation_index)
        if (
            type(self.shape) is not tuple
            or len(self.shape) != 2
            or any(integer(v) == 0 for v in self.shape)
        ):
            raise ValueError("positive exact lifecycle shape required")
        if self.identity_kind != IDENTITY:
            raise ValueError("oracle REGION unit required")


def bind(episode_key: str, lifecycle: State) -> Binding:
    if type(lifecycle) is not State:
        raise ValueError("accepted exact lifecycle state required")
    return Binding(episode_key, lifecycle.digest(), lifecycle.decision_index, lifecycle.shape)


@dataclass(frozen=True)
class Pair:
    owner: str
    affected: str
    latest_support: int


@dataclass(frozen=True)
class Reference:
    binding: Binding
    availability: Literal["INITIAL", "AVAILABLE", "UNAVAILABLE"]
    oracle_rule: str | None
    pairs: tuple[Pair, ...]

    def __post_init__(self) -> None:
        if type(self.binding) is not Binding or type(self.pairs) is not tuple:
            raise ValueError("exact immutable reference binding/pairs required")
        t = self.binding.observation_index
        if t == 0:
            if self.availability != "INITIAL" or self.oracle_rule is not None or self.pairs:
                raise ValueError("initialize without completed evidence or pairs")
        elif self.availability == "AVAILABLE":
            if self.oracle_rule not in RULES:
                raise ValueError("one of the two existing oracle rules required")
        elif self.availability != "UNAVAILABLE" or self.oracle_rule is not None:
            raise ValueError("valid unavailable reference required")
        pair_keys = []
        for p in self.pairs:
            if type(p) is not Pair:
                raise ValueError("exact pair required")
            for token in (p.owner, p.affected):
                if type(token) is not str or re.fullmatch(r"surface-[0-9a-f]{16}", token) is None:
                    raise ValueError("opaque observed region token required")
            if p.owner == p.affected or not 1 <= integer(p.latest_support) <= t:
                raise ValueError("nonreflexive positive support time required")
            if self.availability == "UNAVAILABLE" and p.latest_support == t:
                raise ValueError("unavailable evidence cannot assert current support")
            pair_keys.append((p.owner, p.affected))
        if pair_keys != sorted(set(pair_keys)):
            raise ValueError("canonical unique directed pairs required")

    def current(self) -> tuple[Pair, ...]:
        return tuple(p for p in self.pairs if p.latest_support == self.binding.observation_index)

    def feature_bytes(self) -> bytes:
        """No episode/digest/shape/rule metadata or boundary coordinates in features."""
        return canonical_json_bytes(
            {
                "version": VERSION,
                "observation_index": self.binding.observation_index,
                "availability": self.availability,
                "pairs": [asdict(p) for p in self.pairs],
            }
        )

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(
            {
                "version": VERSION,
                "binding": asdict(self.binding),
                "availability": self.availability,
                "oracle_rule": self.oracle_rule,
                "pairs": [asdict(p) for p in self.pairs],
            }
        )

    def storage(self) -> dict[str, int]:
        return {
            "stored_pairs": len(self.pairs),
            "current_pairs": len(self.current()),
            "feature_bytes": len(self.feature_bytes()),
            "serialized_bytes": len(self.canonical_bytes()),
        }


def decode(payload: bytes) -> Reference:
    if type(payload) is not bytes:
        raise ValueError("exact canonical bytes required")
    root = keys(json.loads(payload), {"version", "binding", "availability", "oracle_rule", "pairs"})
    if root["version"] != VERSION or type(root["pairs"]) is not list:
        raise ValueError("version and pair array required")
    b = keys(
        root["binding"],
        {"episode_key", "lifecycle_root", "observation_index", "shape", "identity_kind"},
    )
    if type(b["shape"]) is not list or len(b["shape"]) != 2:
        raise ValueError("two shape dimensions required")
    binding = Binding(
        b["episode_key"],
        b["lifecycle_root"],
        b["observation_index"],
        tuple(b["shape"]),
        b["identity_kind"],
    )
    pairs = []
    for value in root["pairs"]:
        p = keys(value, {"owner", "affected", "latest_support"})
        pairs.append(Pair(p["owner"], p["affected"], p["latest_support"]))
    reference = Reference(binding, root["availability"], root["oracle_rule"], tuple(pairs))
    if reference.canonical_bytes() != payload:
        raise ValueError("canonical JSON decode/re-encode equality required")
    return reference


@dataclass(frozen=True)
class EndpointEvidence:
    """Trusted completed endpoint1-only projection; never a full transition object."""

    binding: Binding
    boundary_available: bool
    occlusion_available: bool
    exact_visible_binding: bool
    elements: tuple[OrientedBoundaryElement, ...]
    reported_pairs: tuple[OcclusionRelation, ...]
    oracle_rule: str | None
    unavailable: UnavailableOcclusionAnnotation | None = None


class Provider(Protocol):
    def endpoint_one(self, observation_index: int) -> EndpointEvidence: ...


@dataclass(frozen=True)
class Update:
    reference: Reference
    unresolved: str | None


def initialize(lifecycle: State, episode_key: str) -> Reference:
    if type(lifecycle) is not State or lifecycle.decision_index != 0:
        raise ValueError("initialize only at lifecycle zero")
    return Reference(bind(episode_key, lifecycle), "INITIAL", None, ())


def advance(
    reference: Reference,
    lifecycle: State,
    episode_key: str,
    provider: Provider,
    permissions: ModalityPermissionSet,
    decision_index: int,
) -> Update:
    if (
        type(permissions) is not ModalityPermissionSet
        or type(permissions.allowed) is not frozenset
        or any(type(m) is not Modality for m in permissions.allowed)
        or permissions.allowed != REQUIRED
    ):
        raise PermissionError(
            "exact separate ownership/occlusion permissions required before access"
        )
    if type(reference) is not Reference or type(lifecycle) is not State:
        raise ValueError("exact prior relation reference and accepted lifecycle required")
    t, cutoff = integer(lifecycle.decision_index), integer(decision_index)
    if t > cutoff:
        raise PermissionError("future endpoint denied before access")

    def reject(reason: str) -> Update:
        return Update(reference, reason)

    expected = bind(episode_key, lifecycle)
    if (
        t != reference.binding.observation_index + 1
        or episode_key != reference.binding.episode_key
        or lifecycle.shape != reference.binding.shape
    ):
        return reject("nonconsecutive or incompatible lifecycle/episode binding")
    known = set(lifecycle.inventory)
    if any(p.owner not in known or p.affected not in known for p in reference.pairs):
        return reject("remembered relation does not belong to accepted lifecycle inventory")
    evidence = provider.endpoint_one(t)
    if (
        type(evidence) is not EndpointEvidence
        or type(evidence.binding) is not Binding
        or evidence.binding != expected
    ):
        return reject("endpoint does not bind exact accepted lifecycle/episode")
    if evidence.exact_visible_binding is not True:
        return reject("current raster/association binding unresolved")
    if type(evidence.elements) is not tuple or type(evidence.reported_pairs) is not tuple:
        return reject("immutable endpoint-only records required")
    if (
        type(evidence.boundary_available) is not bool
        or type(evidence.occlusion_available) is not bool
    ):
        return reject("exact availability assertions required")
    if evidence.boundary_available != evidence.occlusion_available:
        return reject("available/unavailable records contradict")
    if not evidence.boundary_available:
        if evidence.elements or evidence.reported_pairs or evidence.oracle_rule is not None:
            return reject("unavailable evidence contains positive records or rule")
        if type(evidence.unavailable) is not UnavailableOcclusionAnnotation:
            return reject("existing unavailable annotation required")
        try:
            UnavailableOcclusionAnnotation.model_validate_json(
                canonical_json_bytes(evidence.unavailable)
            )
        except ValueError:
            return reject("invalid unavailable annotation")
        return Update(Reference(expected, "UNAVAILABLE", None, reference.pairs), None)
    if (
        type(evidence.oracle_rule) is not str
        or evidence.oracle_rule not in RULES
        or evidence.unavailable is not None
    ):
        return reject("available evidence lacks existing oracle rule or contradicts unavailability")
    try:
        elements = tuple(
            OrientedBoundaryElement.model_validate_json(canonical_json_bytes(e))
            for e in evidence.elements
            if type(e) is OrientedBoundaryElement
        )
        reports = tuple(
            OcclusionRelation.model_validate_json(canonical_json_bytes(p))
            for p in evidence.reported_pairs
            if type(p) is OcclusionRelation
        )
        if len(elements) != len(evidence.elements) or len(reports) != len(evidence.reported_pairs):
            return reject("wrong endpoint record types")
        h, w = lifecycle.shape
        locations = []
        positives = set()
        for e in elements:
            if e.frame_index != 1:
                return reject("endpoint0 features denied")
            bound = e.column < w - 1 if e.axis.value == "horizontal" else e.row < h - 1
            if e.row >= h or e.column >= w or not bound:
                return reject("boundary coordinate outside current raster")
            locations.append((e.axis.value, e.row, e.column))
            if e.kind == BoundaryKind.OCCLUDING_CONTOUR:
                owner = e.owner_surface_id
                other = (
                    e.positive_surface_id
                    if owner == e.negative_surface_id
                    else e.negative_surface_id
                )
                if owner is None or other is None:
                    return reject("supported contour lacks two controlled sides")
                positives.add((owner, other))
        if len(locations) != len(set(locations)):
            return reject("duplicate endpoint boundary locations")
        reported = set()
        for p in reports:
            if p.frame_indices != (1,):
                return reject("reported relation is not endpoint1-only")
            reported.add((p.occluder_surface_id, p.occluded_surface_id))
        # Compare BEFORE visibility/inventory; filtering must never hide contradictions.
        if positives != reported:
            return reject("reported pairs contradict endpoint contour positives")
        visible = {n.token for n in lifecycle.nodes if n.status == "VISIBLE"}
        if any(owner not in visible or affected not in visible for owner, affected in positives):
            return reject("supported contour side unknown or currently absent")
        latest = {(p.owner, p.affected): p.latest_support for p in reference.pairs}
        latest.update(dict.fromkeys(positives, t))
        pairs = tuple(
            Pair(owner, affected, time) for (owner, affected), time in sorted(latest.items())
        )
        return Update(Reference(expected, "AVAILABLE", evidence.oracle_rule, pairs), None)
    except (ValueError, TypeError):
        return reject("malformed endpoint ownership/occlusion records")
