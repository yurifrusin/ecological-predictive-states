"""The four fixed fictional readiness sets; no sampler or physical producer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from epsbench.diagnostics.boundary_observation import VisibleRaster
from epsbench.diagnostics.causal_region_lifecycle import TrustedObservation, advance
from epsbench.diagnostics.restricted_mask_projection import Projection, Trust, numeric
from epsbench.diagnostics.visible_forecast_contract import REQUIRED, CausalView, Limits
from epsbench.schema import Action, ModalityPermissionSet
from epsbench.utils.canonical import canonical_json_bytes, sha256_bytes

ACCESS = ModalityPermissionSet(allowed=REQUIRED)


@dataclass(frozen=True)
class Fixture:
    number: int
    projection: Projection
    target: VisibleRaster

    _binding: bytes = field(init=False, repr=False)
    _target: bytes = field(init=False, repr=False)
    _fetched: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_binding", self.projection.binding_bytes())
        object.__setattr__(self, "_target", self.target_bytes())

    def target_bytes(self) -> bytes:
        return canonical_json_bytes(
            {
                "index": self.target.sequence_index,
                "labels": self.target.segmentation.tolist(),
                "identities": self.target.identities,
            }
        )

    def fetch(self, projection: Projection) -> VisibleRaster:
        projection.revalidate()
        if projection.binding_bytes() != self._binding or self._fetched:
            raise PermissionError("exact single-fetch fictional episode/action/source required")
        self.check()
        object.__setattr__(self, "_fetched", True)
        return self.target

    def check(self) -> None:
        """Independently enumerate every cell and adjacency from literal sets."""
        i = self.number
        if type(i) is not int or i not in (1, 2, 3, 4):
            raise ValueError("fixed fictional case required")
        r0 = c0 = 8 if i <= 2 else 18
        n = 1 if i <= 2 else 2
        p = self.projection
        source = p.revalidate()
        names = tuple(f"surface-{16 * i + j:016x}" for j in range(1, n + 1))
        new = f"surface-{16 * i + n + 1:016x}"
        expected_action = -0.5 if i % 2 else 0.5
        if (
            p.episode != f"{i:032x}"
            or p.alignment != names
            or float(source.announced[1]) != expected_action
        ):
            raise ValueError("literal episode/action/inventory differs")
        target = self.target
        if p.binding_bytes() != self._binding or self.target_bytes() != self._target:
            raise ValueError("frozen fixture binding/target changed")
        lookup = dict(target.identities)
        observed_pairs: set[tuple[int, int]] = set()
        for row in range(32):
            for col in range(32):
                known = next(
                    (
                        j
                        for j in range(1, n + 1)
                        if r0 <= row <= r0 + 3 and c0 + 4 * (j - 1) <= col <= c0 + 4 * j - 1
                    ),
                    0,
                )
                new_here = r0 <= row <= r0 + 3 and col == (c0 - 1 if i % 2 else c0 + 4 * n)
                expected = names[known - 1] if known else new if new_here else None
                if lookup.get(int(target.segmentation[row, col])) != expected:
                    raise ValueError("literal target set differs")
                for frame in source.frames:
                    for j, (_, mask) in enumerate(frame.masks, 1):
                        if bool(mask[row, col]) != (known == j):
                            raise ValueError("literal prefix set differs")
                if col < 31:
                    right = next(
                        (
                            j
                            for j in range(1, n + 1)
                            if r0 <= row <= r0 + 3 and c0 + 4 * (j - 1) <= col + 1 <= c0 + 4 * j - 1
                        ),
                        0,
                    )
                    if known and right and known != right:
                        observed_pairs.add((min(known - 1, right - 1), max(known - 1, right - 1)))
        f = json.loads(p.feature_bytes())
        if any(
            e["pairs"] != [list(pair) for pair in sorted(observed_pairs)]
            or e["available"] is not True
            for e in f["endpoints"]
        ):
            raise ValueError("independently enumerated adjacency differs")


class _Prefix:
    def __init__(self, raster: VisibleRaster) -> None:
        self.frame = raster

    def raster(self, index: int) -> VisibleRaster:
        if index not in (0, 1):
            raise PermissionError("fictional prefix only")
        return VisibleRaster(index, self.frame.segmentation, self.frame.identities)

    def observation(self, index: int) -> TrustedObservation:
        return TrustedObservation(self.raster(index), True, True, True)


def fixtures(source_head: str) -> tuple[Fixture, ...]:
    """Materialize only the declared public software sets, never sampled geometry."""
    result = []
    executed = Action(name="lateral_right", delta_forward=0.0, delta_lateral=0.75, delta_yaw=0.0)
    for i in (1, 2, 3, 4):
        r0 = c0 = 8 if i <= 2 else 18
        n = 1 if i <= 2 else 2
        names = tuple(f"surface-{16 * i + j:016x}" for j in range(1, n + 1))
        labels = np.zeros((32, 32), dtype=np.int32)
        for j in range(1, n + 1):
            labels[r0 : r0 + 4, c0 + 4 * (j - 1) : c0 + 4 * j] = j
        provider = _Prefix(VisibleRaster(0, labels, tuple(enumerate(names, 1))))
        announced = Action(
            name="lateral_left" if i % 2 else "lateral_right",
            delta_forward=0.0,
            delta_lateral=-0.5 if i % 2 else 0.5,
            delta_yaw=0.0,
        )
        source = CausalView(provider, ACCESS, 1, Limits(2, 1024, 3)).materialize(
            (numeric(executed),), numeric(announced)
        )
        state = advance(None, provider, ACCESS, 0, 1, None).state
        assert state is not None
        state = advance(state, provider, ACCESS, 1, 1, executed).state
        assert state is not None
        projection = Projection(
            source,
            state,
            ACCESS,
            Trust(True, True, True, True),
            (True, True),
            f"{i:032x}",
            source_head,
            1,
            announced,
        )
        target_labels = labels.copy()
        target_labels[r0 : r0 + 4, c0 - 1 if i % 2 else c0 + 4 * n] = n + 1
        target = VisibleRaster(
            2,
            target_labels,
            (*tuple(enumerate(names, 1)), (n + 1, f"surface-{16 * i + n + 1:016x}")),
        )
        item = Fixture(i, projection, target)
        item.check()
        result.append(item)
    return tuple(result)


def public_key() -> bytes:
    return bytes.fromhex(sha256_bytes(b"restricted-mask-model-v2-public-readiness"))
