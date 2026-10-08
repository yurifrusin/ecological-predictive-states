"""Three-image lawful memory reference; no predictor, loader or physical qualification."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from epsbench.diagnostics.boundary_observation import BoundaryObservationView
from epsbench.diagnostics.restricted_mask_projection import PrefixProvider, Trust
from epsbench.diagnostics.visible_forecast_contract import (
    CausalInput,
    Limits,
    TokenFrame,
    permissions,
)
from epsbench.schema import Modality, ModalityPermissionSet
from epsbench.utils.canonical import canonical_json_bytes, sha256_bytes

VERSION = "bounded-prefix-memory-v1"
LIMITS = Limits(3, 1024, 16)


def access_check(access: ModalityPermissionSet) -> None:
    permissions(access)  # Before caller serialization, prefix inspection or endpoint extraction.
    if type(access.allowed) is not frozenset or any(
        type(m) is not Modality for m in access.allowed
    ):
        raise PermissionError("exact frozen typed ecological modalities required")


def bounded(source: CausalInput) -> None:
    if type(source) is not CausalInput:
        raise ValueError("validated exact CausalInput required")
    access_check(source.permissions)
    if type(source.frames) is not tuple or len(source.frames) != 3:
        raise ValueError("exactly three complete consecutive images required")
    if any(type(f) is not TokenFrame for f in source.frames):
        raise ValueError("exact observed TokenFrames required; missing images are not admitted")
    LIMITS.check(3, source.shape, len(source.inventory))
    if any(f.shape != source.shape for f in source.frames):
        raise ValueError("fixed complete image shape required")


@dataclass(frozen=True, init=False)
class BoundedPrefixMemory:
    """Immutable numeric prefix; opaque alignment and scientific bindings stay outside features."""

    _source: bytes
    _features: bytes
    _binding: bytes
    alignment: tuple[str, ...]
    access: ModalityPermissionSet
    limits: Limits

    def __init__(
        self,
        source: CausalInput,
        access: ModalityPermissionSet,
        trust: Trust,
        availability: tuple[bool, bool, bool],
        episode: str,
        source_head: str,
    ) -> None:
        access_check(access)
        bounded(source)
        if type(trust) is not Trust:
            raise ValueError("exact external Trust required")
        trust.check()
        if (
            type(availability) is not tuple
            or len(availability) != 3
            or any(type(v) is not bool for v in availability)
        ):
            raise ValueError("three exact relation-endpoint availability flags required")
        for value, pattern in ((episode, r"[0-9a-f]{32}"), (source_head, r"[0-9a-f]{40}")):
            if type(value) is not str or re.fullmatch(pattern, value) is None:
                raise ValueError("opaque episode and exact source identity required")
        # Reuse the established complete-prefix validator, then own its immutable bytes.
        source = CausalInput.from_bytes(source.canonical_bytes(), access, source.limits)
        bounded(source)
        names = source.inventory
        lookup = {name: i for i, name in enumerate(names)}
        view = BoundaryObservationView(PrefixProvider(source.frames, trust), access, 2)
        empty = np.zeros(source.shape, dtype=np.bool_)
        frames = []
        for frame, available in zip(source.frames, availability, strict=True):
            masks = dict(frame.masks)
            pairs = None
            if available:
                pairs = sorted(
                    {
                        tuple(
                            sorted((lookup[e.negative_surface_id], lookup[e.positive_surface_id]))
                        )
                        for e in view.observe(frame.index).edges
                        if e.negative_surface_id is not None and e.positive_surface_id is not None
                    }
                )
            frames.append(
                {
                    "index": frame.index,
                    "masks": [masks.get(name, empty).tolist() for name in names],
                    "observed": [name in masks for name in names],
                    "contacts": {"available": available, "pairs": pairs},
                }
            )
        features: dict[str, Any] = {
            "shape": source.shape,
            "frames": frames,
            "executed": [[str(q) for q in command] for command in source.executed],
            "announced": [str(q) for q in source.announced],
        }
        feature_bytes = canonical_json_bytes(features)
        binding = canonical_json_bytes(
            {
                "version": VERSION,
                "episode": episode,
                "source_head": source_head,
                "input_sha256": source.digest,
                "feature_sha256": sha256_bytes(feature_bytes),
                "alignment": names,
                "index": 2,
                "target_index": 3,
                "announced": [str(q) for q in source.announced],
            }
        )
        for field_name, field_value in {
            "_source": source.canonical_bytes(),
            "_features": feature_bytes,
            "_binding": binding,
            "alignment": names,
            "access": access.model_copy(deep=True),
            "limits": source.limits,
        }.items():
            object.__setattr__(self, field_name, field_value)

    def feature_bytes(self) -> bytes:
        access_check(self.access)
        return self._features

    def binding_bytes(self) -> bytes:
        access_check(self.access)
        return self._binding

    def revalidate(self) -> CausalInput:
        access_check(self.access)
        source = CausalInput.from_bytes(self._source, self.access, self.limits)
        bounded(source)
        return source

    def storage(self) -> dict[str, int]:
        access_check(self.access)
        return {
            "numeric_bytes": len(self._features),
            "prefix_bytes": len(self._source),
            "binding_bytes": len(self._binding),
        }
