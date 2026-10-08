"""Public-only oracle organization inputs. No simulator or private-data adapter."""

from __future__ import annotations

import struct
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class Permission(Enum):
    PUBLIC_ORACLE_SOFTWARE = "public_oracle_software"


@dataclass(frozen=True)
class OracleInput:
    """Full lawful history; slots express association equality, never ID embeddings."""

    masks: NDArray[np.bool_]
    present: NDArray[np.bool_]
    observed: NDArray[np.bool_]
    contacts: NDArray[np.bool_]
    available: NDArray[np.bool_]
    executed: tuple[int, int]
    announced: tuple[int, int]

    @property
    def region_count(self) -> int:
        return int(self.masks.shape[1])

    def checked(self) -> OracleInput:
        if type(self.masks) is not np.ndarray:
            raise ValueError("exact Boolean lawful array required")
        n = self.region_count if self.masks.ndim == 4 else 0
        if n not in (2, 3, 4):
            raise ValueError("public software permits exactly 2, 3 or 4 before-known regions")
        shapes = ((3, n, 32, 32), (3, n), (3, n), (3, n, n), (3, n, n))
        arrays = (self.masks, self.present, self.observed, self.contacts, self.available)
        for array, shape in zip(arrays, shapes, strict=True):
            if type(array) is not np.ndarray or array.dtype != np.bool_ or array.shape != shape:
                raise ValueError("exact Boolean lawful tensor shape required")
        nonempty = self.masks.any(axis=(2, 3))
        if np.any(nonempty != self.observed) or np.any(self.observed & ~self.present):
            raise ValueError("observed flags must describe present nonempty masks")
        if not np.all(self.observed.any(axis=0)):
            raise ValueError("unseen inventory denied")
        if np.any(self.masks.sum(axis=1) > 1):
            raise ValueError("region masks must not overlap")
        if not np.array_equal(self.available, self.available.transpose(0, 2, 1)):
            raise ValueError("contact availability must be symmetric")
        record_pairs = self.present[:, :, None] & self.present[:, None, :]
        if np.any(self.available & ~record_pairs):
            raise ValueError("missing records cannot claim contact availability")
        factual = np.zeros((3, n, n), dtype=np.bool_)
        for t in range(3):
            for r in range(n):
                for q in range(r + 1, n):
                    a, b = self.masks[t, r], self.masks[t, q]
                    touching = bool(
                        (a[1:] & b[:-1]).any()
                        or (a[:-1] & b[1:]).any()
                        or (a[:, 1:] & b[:, :-1]).any()
                        or (a[:, :-1] & b[:, 1:]).any()
                    )
                    factual[t, r, q] = factual[t, q, r] = touching
        if np.any((self.contacts != factual) & self.available):
            raise ValueError("contact facts must match lawful four-neighbour masks")
        if np.any(self.contacts & ~self.available):
            raise ValueError("unavailable contact cannot be asserted")
        if not np.array_equal(self.contacts, self.contacts.transpose(0, 2, 1)):
            raise ValueError("four-neighbour contacts must be symmetric")
        if np.any(np.diagonal(self.contacts, axis1=1, axis2=2)):
            raise ValueError("self contacts denied")
        if (
            type(self.executed) is not tuple
            or type(self.announced) is not tuple
            or len(self.executed) != 2
            or len(self.announced) != 2
        ):
            raise ValueError("exact two executed and two announced commands required")
        if any(type(v) is not int or abs(v) > 3 for v in (*self.executed, *self.announced)):
            raise ValueError("bounded scalar commands required")
        # Own immutable copies: no mutable reader aliases remain.
        copied = tuple(a.copy() for a in arrays)
        for array in copied:
            array.flags.writeable = False
        return OracleInput(
            copied[0], copied[1], copied[2], copied[3], copied[4], self.executed, self.announced
        )

    def permuted(self, order: tuple[int, ...] | None = None) -> OracleInput:
        self.checked()
        n = self.region_count
        order = tuple(reversed(range(n))) if order is None else order
        if (
            type(order) is not tuple
            or len(order) != n
            or any(type(i) is not int for i in order)
            or set(order) != set(range(n))
        ):
            raise ValueError("complete region-slot bijection required")
        p = list(order)
        return OracleInput(
            self.masks[:, p],
            self.present[:, p],
            self.observed[:, p],
            self.contacts[:, p][:, :, p],
            self.available[:, p][:, :, p],
            self.executed,
            self.announced,
        ).checked()

    def serialized(self) -> bytes:
        x = self.checked()
        return (
            (
                b"EPS-PUBLIC-ORACLE-V3\0"
                if x.region_count == 2
                else b"EPS-PUBLIC-ORACLE-N34-V1\0" + bytes([x.region_count])
            )
            + b"".join(
                a.tobytes() for a in (x.masks, x.present, x.observed, x.contacts, x.available)
            )
            + struct.pack("<4q", *x.executed, *x.announced)
        )

    def storage_bytes(self) -> int:
        return (
            sum(
                a.nbytes
                for a in (self.masks, self.present, self.observed, self.contacts, self.available)
            )
            + 4 * 8
        )


def read_public(permission: Permission, supplier: Callable[[], OracleInput]) -> OracleInput:
    """Permission is checked before calling a supplier or reading any arrays."""
    if type(permission) is not Permission or permission is not Permission.PUBLIC_ORACLE_SOFTWARE:
        raise PermissionError("public oracle software permission required")
    value = supplier()
    if type(value) is not OracleInput:
        raise TypeError("typed lawful input required; arbitrary records denied")
    return value.checked()


@dataclass(frozen=True)
class PublicFixture:
    inputs: OracleInput
    success: tuple[tuple[int, int], ...]
    support: tuple[tuple[int, int], ...]

    def permuted(self, order: tuple[int, ...]) -> PublicFixture:
        inputs = self.inputs.permuted(order)
        return PublicFixture(
            inputs, tuple(self.success[i] for i in order), tuple(self.support[i] for i in order)
        )

    def __post_init__(self) -> None:
        self.inputs.checked()
        for targets in (self.success, self.support):
            if (
                type(targets) is not tuple
                or len(targets) != self.inputs.region_count
                or any(type(row) is not tuple or len(row) != 2 for row in targets)
            ):
                raise ValueError("all-region/two-arm target shape required")
        for r in range(self.inputs.region_count):
            for a in range(2):
                value = self.success[r][a]
                if type(value) is not int or value not in (0, 1):
                    raise ValueError("public binary target required")
                if type(self.support[r][a]) is not int or self.support[r][a] != 16 * value:
                    raise ValueError("literal public support teacher required")


def public_fixtures() -> tuple[PublicFixture, ...]:
    """Literal v3 tensors, software teachers only; no scenes or generation seeds."""
    patterns = ((0, 0), (1, 0), (0, 1), (1, 1), (0, 0), (1, 0), (0, 1), (1, 1))
    result = []
    for i, target in enumerate(patterns):
        masks = np.zeros((3, 2, 32, 32), dtype=np.bool_)
        masks[0, 0, 2:6, 2 + i : 6 + i] = True
        masks[1, 0, 3:7, 3 + i : 7 + i] = True
        masks[:, 1, 24:28, 24:28] = True
        inputs = OracleInput(
            masks,
            np.ones((3, 2), dtype=np.bool_),
            np.asarray(masks.any(axis=(2, 3)), dtype=np.bool_),
            np.zeros((3, 2, 2), dtype=np.bool_),
            np.ones((3, 2, 2), dtype=np.bool_),
            (1, 3),
            (-2, 2),
        ).checked()
        result.append(
            PublicFixture(inputs, (target, (1, 1)), ((16 * target[0], 16 * target[1]), (16, 16)))
        )
    return tuple(result)


def public_region_fixtures(n: int) -> tuple[PublicFixture, ...]:
    """Literal reviewed N=3/4 extension of the unchanged eight v3 software cases."""
    if type(n) is not int or n not in (3, 4):
        raise ValueError("new public fixtures require exactly N=3 or N=4")
    result = []
    for base in public_fixtures():
        masks = np.zeros((3, n, 32, 32), dtype=np.bool_)
        masks[:, :2] = base.inputs.masks
        masks[:, 2, 16:20, 4:8] = True
        if n == 4:
            masks[:, 3, 16:20, 8:12] = True
        contacts = np.zeros((3, n, n), dtype=np.bool_)
        if n == 4:
            contacts[:, 2, 3] = contacts[:, 3, 2] = True
        inputs = OracleInput(
            masks,
            np.ones((3, n), dtype=np.bool_),
            np.asarray(masks.any(axis=(2, 3)), dtype=np.bool_),
            contacts,
            np.ones((3, n, n), dtype=np.bool_),
            base.inputs.executed,
            base.inputs.announced,
        ).checked()
        result.append(
            PublicFixture(
                inputs, base.success + ((1, 1),) * (n - 2), base.support + ((16, 16),) * (n - 2)
            )
        )
    return tuple(result)


def decode_public(permission: Permission, supplier: Callable[[], bytes]) -> OracleInput:
    if type(permission) is not Permission or permission is not Permission.PUBLIC_ORACLE_SOFTWARE:
        raise PermissionError("public oracle software permission required before byte read")
    payload = supplier()
    old_prefix = b"EPS-PUBLIC-ORACLE-V3\0"
    new_prefix = b"EPS-PUBLIC-ORACLE-N34-V1\0"
    if type(payload) is not bytes:
        raise ValueError("exact versioned lawful payload required")
    if payload.startswith(old_prefix):
        n, offset = 2, len(old_prefix)
    elif payload.startswith(new_prefix) and len(payload) > len(new_prefix):
        n, offset = payload[len(new_prefix)], len(new_prefix) + 1
        if n not in (3, 4):
            raise ValueError("new bounded format permits only N=3/4")
    else:
        raise ValueError("exact versioned lawful payload required")
    if len(payload) != offset + 3072 * n + 6 * n + 6 * n * n + 32:
        raise ValueError("exact cardinality payload length required")
    arrays = []
    for shape in ((3, n, 32, 32), (3, n), (3, n), (3, n, n), (3, n, n)):
        length = int(np.prod(shape))
        raw = payload[offset : offset + length]
        if any(v not in (0, 1) for v in raw):
            raise ValueError("canonical Boolean bytes required")
        arrays.append(np.frombuffer(raw, dtype=np.bool_).reshape(shape))
        offset += length
    commands = struct.unpack("<4q", payload[offset:])
    return OracleInput(
        arrays[0],
        arrays[1],
        arrays[2],
        arrays[3],
        arrays[4],
        (commands[0], commands[1]),
        (commands[2], commands[3]),
    ).checked()
