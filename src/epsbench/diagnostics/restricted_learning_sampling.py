"""Unbiased deterministic sampling primitive; no study domain or seed allocation."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def commitment(domain: str, seed: bytes) -> str:
    return hashlib.sha256(
        len(domain.encode()).to_bytes(8, "little") + domain.encode() + seed
    ).hexdigest()


class Entropy:
    def __init__(self, domain: str, seed: bytes) -> None:
        if type(domain) is not str or not domain or type(seed) is not bytes or not seed:
            raise ValueError("explicit domain/private seed bytes required")
        self.domain, self.seed, self.counter = domain.encode(), seed, 0

    def below(self, n: int) -> int:
        if type(n) is not int or not 1 <= n <= 2**64:
            raise ValueError("bounded positive integer range")
        limit = 2**64 - (2**64 % n)
        while True:
            data = (
                len(self.domain).to_bytes(8, "little")
                + self.domain
                + len(self.seed).to_bytes(8, "little")
                + self.seed
                + self.counter.to_bytes(8, "little")
            )
            value = int.from_bytes(hashlib.sha256(data).digest()[:8], "little")
            self.counter += 1
            if value < limit:
                return value % n


def shuffled(values: Sequence[T], domain: str, seed: bytes) -> tuple[T, ...]:
    output = list(values)
    entropy = Entropy(domain, seed)
    for i in range(len(output) - 1, 0, -1):
        j = entropy.below(i + 1)
        output[i], output[j] = output[j], output[i]
    return tuple(output)
