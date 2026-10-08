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

    def checked(self) -> OracleInput:
        n = self.masks.shape[1] if self.masks.ndim == 4 else 0
        if n != 2:
            raise ValueError("public readiness requires exactly two before-known regions")
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
            a, b = self.masks[t]
            touching = bool(
                (a[1:] & b[:-1]).any()
                or (a[:-1] & b[1:]).any()
                or (a[:, 1:] & b[:, :-1]).any()
                or (a[:, :-1] & b[:, 1:]).any()
            )
            factual[t, 0, 1] = factual[t, 1, 0] = touching
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

    def permuted(self) -> OracleInput:
        p = [1, 0]
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
            b"EPS-PUBLIC-ORACLE-V3\0"
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
    success: tuple[tuple[int, int], tuple[int, int]]
    support: tuple[tuple[int, int], tuple[int, int]]

    def __post_init__(self) -> None:
        self.inputs.checked()
        for targets in (self.success, self.support):
            if (
                type(targets) is not tuple
                or len(targets) != 2
                or any(type(row) is not tuple or len(row) != 2 for row in targets)
            ):
                raise ValueError("two-region/two-arm target shape required")
        for r in range(2):
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


def decode_public(permission: Permission, supplier: Callable[[], bytes]) -> OracleInput:
    if type(permission) is not Permission or permission is not Permission.PUBLIC_ORACLE_SOFTWARE:
        raise PermissionError("public oracle software permission required before byte read")
    payload = supplier()
    prefix = b"EPS-PUBLIC-ORACLE-V3\0"
    if (
        type(payload) is not bytes
        or len(payload) != len(prefix) + 6212
        or not payload.startswith(prefix)
    ):
        raise ValueError("exact versioned lawful payload required")
    offset = len(prefix)
    arrays = []
    for shape in ((3, 2, 32, 32), (3, 2), (3, 2), (3, 2, 2), (3, 2, 2)):
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
