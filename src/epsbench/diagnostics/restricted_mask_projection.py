"""Shared before-only mask projection and neutral-target binding; no model or producer."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np

from epsbench.diagnostics.boundary_observation import BoundaryObservationView, VisibleRaster
from epsbench.diagnostics.causal_region_lifecycle import (
    State,
    TrustedObservation,
    action_bytes,
    advance,
    decode,
)
from epsbench.diagnostics.neutral_observation_target import Candidate, Evaluation, Masks
from epsbench.diagnostics.neutral_observation_target import evaluate as neutral_evaluate
from epsbench.diagnostics.visible_forecast_contract import (
    Array,
    CausalInput,
    Command,
    Limits,
    RasterProvider,
    TokenFrame,
    _json,
    _keys,
    integer,
    permissions,
)
from epsbench.schema import Action, ModalityPermissionSet
from epsbench.utils.canonical import canonical_json_bytes, sha256_bytes

VERSION = "restricted-mask-projection-v1"


@dataclass(frozen=True)
class Trust:
    """External adapter assertions, not inferred physical qualification."""

    complete_history: bool
    complete_image: bool
    complete_association: bool
    stable_identity: bool

    def check(self) -> None:
        if type(self) is not Trust or any(
            v is not True
            for v in (
                self.complete_history,
                self.complete_image,
                self.complete_association,
                self.stable_identity,
            )
        ):
            raise ValueError("complete trusted prefix/association required")


def numeric(action: Action) -> Command:
    checked = Action.model_validate_json(action_bytes(action))
    return (
        Fraction(checked.delta_forward),
        Fraction(checked.delta_lateral),
        Fraction(checked.delta_yaw),
    )


def raster(frame: TokenFrame) -> VisibleRaster:
    """Lossless relabeling of supplied observed masks, never a geometry computation."""
    segmentation = np.zeros(frame.shape, dtype=np.int32)
    pairs = []
    for label, (name, mask) in enumerate(frame.masks, 1):
        segmentation[mask] = label
        pairs.append((label, name))
    return VisibleRaster(frame.index, segmentation, tuple(pairs))


@dataclass(frozen=True)
class PrefixProvider:
    frames: tuple[TokenFrame, ...]
    trust: Trust

    def raster(self, index: int) -> VisibleRaster:
        integer(index)
        if index >= len(self.frames):
            raise PermissionError("future prefix access denied")
        return raster(self.frames[index])

    def observation(self, index: int) -> TrustedObservation:
        return TrustedObservation(
            self.raster(index),
            self.trust.complete_image,
            self.trust.complete_association,
            self.trust.stable_identity,
        )


def lifecycle(source: CausalInput, trust: Trust, executed: tuple[bytes, ...]) -> State:
    """Rebuild through the accepted ledger from genuine complete supplied history."""
    permissions(source.permissions)
    trust.check()
    if len(executed) != len(source.executed):
        raise ValueError("genuine complete Action log required")
    commands = tuple(Action.model_validate_json(c) for c in executed)
    if tuple(numeric(c) for c in commands) != source.executed:
        raise ValueError("stored Action/Command numeric equivalence must be exact")
    provider = PrefixProvider(source.frames, trust)
    state = None
    for i in range(len(source.frames)):
        update = advance(
            state,
            provider,
            source.permissions,
            i,
            len(source.frames) - 1,
            None if i == 0 else commands[i - 1],
        )
        if update.unresolved is not None or update.state is None:
            raise ValueError("trusted prefix cannot yield accepted lifecycle")
        state = update.state
    assert state is not None
    return state


@dataclass(frozen=True, init=False)
class Projection:
    """Immutable shared numeric P_t; alignment and full prefix stay outside features."""

    _source: bytes
    _state: bytes
    _features: bytes
    _binding: bytes
    alignment: tuple[str, ...]
    access: ModalityPermissionSet
    limits: Limits
    trust: Trust
    availability: tuple[bool, bool]
    episode: str
    source_head: str
    cutoff: int
    announced: bytes

    def __init__(
        self,
        source: CausalInput,
        state: State,
        access: ModalityPermissionSet,
        trust: Trust,
        availability: tuple[bool, bool],
        episode: str,
        source_head: str,
        cutoff: int,
        announced: Action,
    ) -> None:
        permissions(access)  # Before caller serialization/parsing or observation access.
        if type(source) is not CausalInput or type(state) is not State:
            raise ValueError("real complete causal prefix and accepted lifecycle required")
        permissions(source.permissions)
        integer(cutoff)
        t = len(source.frames) - 1
        if t > cutoff or state.decision_index > cutoff:
            raise PermissionError("future before-state denied")
        if t < 1:
            raise ValueError("two completed observations required")
        if type(trust) is not Trust:
            raise ValueError("typed external assertions required")
        trust.check()
        if (
            type(availability) is not tuple
            or len(availability) != 2
            or any(type(v) is not bool for v in availability)
        ):
            raise ValueError("two exact endpoint availability flags required")
        for value, pattern in ((episode, r"[0-9a-f]{32}"), (source_head, r"[0-9a-f]{40}")):
            if type(value) is not str or re.fullmatch(pattern, value) is None:
                raise ValueError("opaque episode and exact source identity required")
        source = CausalInput.from_bytes(source.canonical_bytes(), access, source.limits)
        state = decode(state.canonical_bytes())
        announced_bytes = action_bytes(announced)
        if numeric(announced) != source.announced:
            raise ValueError("announced Action/Command equivalence must be exact")
        if (
            lifecycle(source, trust, state.executed_commands).canonical_bytes()
            != state.canonical_bytes()
        ):
            raise ValueError(
                "complete prefix/lifecycle chronology, masks, actions or inventory differ"
            )
        names = source.inventory
        lookup = {name: i for i, name in enumerate(names)}
        previous = dict(source.frames[-2].masks)
        current = dict(source.frames[-1].masks)
        empty = np.zeros(source.shape, dtype=np.bool_)
        nodes = []
        for node in state.nodes:
            a, b = current.get(node.token, empty), previous.get(node.token, empty)
            nodes.append(
                {
                    "current": a.tolist(),
                    "previous": b.tolist(),
                    "difference": (a.astype(np.int8) - b.astype(np.int8)).tolist(),
                    "status": node.status,
                    "first_seen_age": t - node.first_seen,
                    "last_seen_age": t - node.last_seen,
                }
            )
        endpoints = []
        provider = PrefixProvider(source.frames, trust)
        view = BoundaryObservationView(provider, access, t)
        for index, available in zip((t - 1, t), availability, strict=True):
            pairs = None
            if available:
                edges = view.observe(index).edges
                pairs = sorted(
                    {
                        tuple(
                            sorted((lookup[e.negative_surface_id], lookup[e.positive_surface_id]))
                        )
                        for e in edges
                        if e.negative_surface_id is not None and e.positive_surface_id is not None
                    }
                )
            endpoints.append({"age": t - index, "available": available, "pairs": pairs})
        features: dict[str, Any] = {
            "nodes": nodes,
            "endpoints": endpoints,
            "executed_action": source.executed[-1],
            "announced_action": source.announced,
            "current_union": np.logical_or.reduce(
                [empty, *[a for _, a in source.frames[-1].masks]]
            ).tolist(),
            "previous_union": np.logical_or.reduce(
                [empty, *[a for _, a in source.frames[-2].masks]]
            ).tolist(),
        }
        # Commands keep exact reduced rational strings; no float conversion in features.
        features["executed_action"] = [str(q) for q in source.executed[-1]]
        features["announced_action"] = [str(q) for q in source.announced]
        feature_bytes = canonical_json_bytes(features)
        binding = canonical_json_bytes(
            {
                "version": VERSION,
                "episode": episode,
                "source_head": source_head,
                "input_sha256": source.digest,
                "state_sha256": state.digest(),
                "announced_action": announced.model_dump(mode="json"),
                "feature_sha256": sha256_bytes(feature_bytes),
                "alignment": names,
                "index": t,
                "target_index": t + 1,
                "shape": source.shape,
            }
        )
        for field_name, field_value in {
            "_source": source.canonical_bytes(),
            "_state": state.canonical_bytes(),
            "_features": feature_bytes,
            "_binding": binding,
            "alignment": names,
            "access": access.model_copy(deep=True),
            "limits": source.limits,
            "trust": trust,
            "availability": availability,
            "episode": episode,
            "source_head": source_head,
            "cutoff": cutoff,
            "announced": announced_bytes,
        }.items():
            object.__setattr__(self, field_name, field_value)

    def feature_bytes(self, previous_mask_ablation: bool = False) -> bytes:
        if type(previous_mask_ablation) is not bool:
            raise ValueError("exact ablation flag required")
        if not previous_mask_ablation:
            return self._features
        p = json.loads(self._features)
        del p["previous_union"]
        for node in p["nodes"]:
            del node["previous"]
            del node["difference"]
        return canonical_json_bytes(p)

    def binding_bytes(self) -> bytes:
        return self._binding

    def revalidate(self) -> CausalInput:
        permissions(self.access)
        source = CausalInput.from_bytes(self._source, self.access, self.limits)
        checked = Projection(
            source,
            decode(self._state),
            self.access,
            self.trust,
            self.availability,
            self.episode,
            self.source_head,
            self.cutoff,
            Action.model_validate_json(self.announced),
        )
        if (
            checked._features != self._features
            or checked._binding != self._binding
            or checked.alignment != self.alignment
        ):
            raise ValueError("projection snapshot/binding differs")
        return source

    def save(self, known: Masks, new: Array | None) -> bytes:
        source = self.revalidate()
        candidate = Candidate(source, known, new).canonical_bytes()
        return canonical_json_bytes(
            {
                "version": VERSION + ":candidate",
                "binding_sha256": sha256_bytes(self._binding),
                "candidate": json.loads(candidate),
            }
        )

    def persistence(self) -> bytes:
        source = self.revalidate()
        current = dict(source.frames[-1].masks)
        empty = np.zeros(source.shape, dtype=np.bool_)
        return self.save(tuple((name, current.get(name, empty)) for name in self.alignment), empty)

    def evaluate(self, saved: bytes, provider: RasterProvider) -> Evaluation:
        source = self.revalidate()  # Permission/source context before candidate parsing/fetch.
        p = _json(saved)
        _keys(p, {"version", "binding_sha256", "candidate"})
        if p["version"] != VERSION + ":candidate" or p["binding_sha256"] != sha256_bytes(
            self._binding
        ):
            raise ValueError("exact projection/context binding required")
        candidate = canonical_json_bytes(p["candidate"])
        checked = Candidate.from_bytes(candidate, source)
        if candidate != checked.canonical_bytes() or saved != canonical_json_bytes(p):
            raise ValueError("canonical saved candidate required")
        return neutral_evaluate(source, candidate, provider)

    def storage(self) -> dict[str, int]:
        return {
            "feature_bytes": len(self._features),
            "alignment_bytes": len(canonical_json_bytes(list(self.alignment))),
            "binding_bytes": len(self._binding),
            "full_prefix_context_bytes": len(self._source),
            "lifecycle_context_bytes": len(self._state),
        }
