"""Versioned privileged point-sample qualification; no collection or legacy override."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from epsbench.diagnostics import disclosure_producer as producer
from epsbench.diagnostics import disclosure_reference as reference
from epsbench.diagnostics.restricted_exact_raster import Box, Raster
from epsbench.schema import ModalityPermissionSet
from epsbench.utils.canonical import canonical_json_bytes, sha256_bytes

VERSION = "oracle-region-unit-point-sample-v1"


@dataclass(frozen=True)
class Qualification:
    """Instrument evidence, not an authenticated external approval credential."""

    policy: str
    input_sha256: str
    produced: Raster
    reference: reference.Audit

    @property
    def contradiction(self) -> bool:
        return self.produced.labels != self.reference.labels

    @property
    def qualified(self) -> bool:
        return not (self.contradiction or self.produced.ties or self.reference.ties)


def qualify(
    access: ModalityPermissionSet,
    policy: str,
    lateral: Q,
    boxes: tuple[Box, ...],
    extent: tuple[Q, Q],
    raw: Raster,
    independent: reference.Audit,
) -> Qualification:
    """Qualify separately obtained instrument records with explicit opt-in.

    Invalid inputs raise; background is only a valid, measured no-hit label.
    Boundary/cause diagnostics remain privileged and unchanged.
    """
    producer.access_check(access)  # Before policy, geometry or callback access.
    if type(policy) is not str or policy != VERSION:
        raise ValueError("explicit supported point-sample policy required")
    producer.domain(lateral, boxes, extent)
    binding = sha256_bytes(
        canonical_json_bytes(
            {
                "policy": policy,
                "lateral": str(lateral),
                "boxes": [[str(v) for v in (*b.lower, *b.upper)] for b in boxes],
                "floor": [str(v) for v in extent],
                "calibration": "fixed-32x32-origin-lateral-minus3-1",
            }
        )
    )
    if type(raw) is not Raster or type(independent) is not reference.Audit:
        raise ValueError("exact retained instrument records required")
    for grid in (raw.labels, independent.labels):
        if (
            type(grid) is not tuple
            or len(grid) != 32
            or any(type(row) is not tuple or len(row) != 32 for row in grid)
            or any(type(v) is not int or not 0 <= v <= len(boxes) + 1 for row in grid for v in row)
        ):
            raise ValueError("complete exact instrument grids required")
    for diagnostics, raw_ties in (
        (raw.ties, True),
        (independent.ties, False),
        (independent.boundaries, False),
    ):
        if type(diagnostics) is not tuple:
            raise ValueError("exact pixel diagnostics required")
        seen = set()
        for item in diagnostics:
            if type(item) is not tuple or len(item) != (3 if raw_ties else 2):
                raise ValueError("exact pixel diagnostic entries required")
            row, column = item[:2]
            if (
                any(type(v) is not int or not 0 <= v < 32 for v in (row, column))
                or (row, column) in seen
            ):
                raise ValueError("unique in-bounds pixel diagnostics required")
            seen.add((row, column))
            if raw_ties:
                winners = item[-1]
                if (
                    type(winners) is not tuple
                    or len(winners) < 2
                    or any(type(v) is not int or not 1 <= v <= len(boxes) + 1 for v in winners)
                    or len(set(winners)) != len(winners)
                ):
                    raise ValueError("distinct exact tie units required")
    return Qualification(policy, binding, raw, independent)
