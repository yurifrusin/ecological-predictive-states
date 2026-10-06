"""Privileged membership descriptors and precollection commitments; no allocation."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from typing import Any

from epsbench.diagnostics.restricted_learning_contract import (
    ARCHITECTURE,
    INITIALIZATIONS,
    PROPOSAL,
    ROSTER,
    VERSION,
    digest,
    sha,
)
from epsbench.diagnostics.visible_forecast_contract import _json, _keys
from epsbench.utils.canonical import canonical_json_bytes

BOOTSTRAP_DOMAIN = "M0-occupancy-development-v1/bootstrap-v1"
INIT_DOMAIN = "M0-occupancy-development-v1/init-v1"
ANNOUNCED = (Q(-5, 4), Q(-3, 4), Q(3, 4), Q(5, 4))
PREFIXES = ((Q(2), Q(1), Q(0)), (Q(-2), Q(-1), Q(0)))


@dataclass(frozen=True)
class Geometry:
    foreground_width: Q
    background_width: Q
    foreground_y: Q
    background_y: Q
    background_x: Q

    def __post_init__(self) -> None:
        values = (
            self.foreground_width,
            self.background_width,
            self.foreground_y,
            self.background_y,
            self.background_x,
        )
        domains = (
            (Q(3, 5), Q(7, 10), Q(4, 5)),
            (Q(7, 20), Q(9, 20), Q(11, 20)),
            (Q(-1, 4), Q(0), Q(1, 4)),
            (Q(5, 2), Q(3), Q(7, 2)),
            (Q(0), Q(1, 20), Q(1, 10)),
        )
        if any(type(v) is not Q or v not in d for v, d in zip(values, domains, strict=True)):
            raise ValueError("descriptor outside frozen 243-member rational domain")
        # Historical width1/2 is outside this domain; never extend it to admit old items.

    def payload(self) -> list[str]:
        return [
            str(v)
            for v in (
                self.foreground_width,
                self.background_width,
                self.foreground_y,
                self.background_y,
                self.background_x,
            )
        ]

    @property
    def identity(self) -> str:
        return digest(canonical_json_bytes({"domain": VERSION, "geometry": self.payload()}))

    @classmethod
    def parse(cls, p: Any) -> Geometry:
        if type(p) is not list or len(p) != 5 or any(type(v) is not str for v in p):
            raise ValueError("strict rational descriptor required")
        values = tuple(Q(v) for v in p)
        if [str(v) for v in values] != p:
            raise ValueError("canonical rational descriptor required")
        return cls(*values)


def decisions(geometry: str) -> tuple[str, ...]:
    sha(geometry)
    return tuple(f"{geometry}/prefix-{p}/action-{a}" for p in range(2) for a in range(4))


@dataclass(frozen=True)
class MembershipLock:
    train: tuple[Geometry, ...]
    development: tuple[Geometry, ...]
    evaluation: tuple[Geometry, ...]
    nested_train: tuple[str, ...]
    split_commitment: str
    identity_commitment: str
    initialization_commitments: tuple[tuple[str, str], ...]
    schedule_commitments: tuple[tuple[int, str], ...]
    bootstrap_commitment: str
    source_head: str
    source_tree: str
    source_files: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        collections = (
            self.train,
            self.development,
            self.evaluation,
            self.nested_train,
            self.initialization_commitments,
            self.schedule_commitments,
            self.source_files,
        )
        if any(type(v) is not tuple for v in collections) or any(
            type(pair) is not tuple or len(pair) != 2
            for v in (self.initialization_commitments, self.schedule_commitments, self.source_files)
            for pair in v
        ):
            raise ValueError("owned exact immutable membership containers required")
        if (len(self.train), len(self.development), len(self.evaluation)) != (64, 16, 32):
            raise ValueError("complete 112 geometry split required")
        all_items = (*self.train, *self.development, *self.evaluation)
        if (
            any(type(g) is not Geometry for g in all_items)
            or len({g.identity for g in all_items}) != 112
        ):
            raise ValueError("unique geometry units; no dependent-view splitting")
        if (
            len(self.nested_train) != 16
            or len(set(self.nested_train)) != 16
            or (not set(self.nested_train) <= {g.identity for g in self.train})
        ):
            raise ValueError("committed nested 16-training subset required")
        if tuple(k for k, _ in self.initialization_commitments) != INITIALIZATIONS or (
            tuple(k for k, _ in self.schedule_commitments) != (16, 64)
        ):
            raise ValueError("paired initialization/budget schedule commitments required")
        for h in (
            self.split_commitment,
            self.identity_commitment,
            self.bootstrap_commitment,
            *(v for _, v in self.initialization_commitments),
            *(v for _, v in self.schedule_commitments),
            *(v for _, v in self.source_files),
        ):
            sha(h)
        for h in (self.source_head, self.source_tree):
            if type(h) is not str or len(h) != 40 or any(c not in "0123456789abcdef" for c in h):
                raise ValueError("exact source head/tree required")
        if not self.source_files or len({n for n, _ in self.source_files}) != len(
            self.source_files
        ):
            raise ValueError("unique logical source file hashes required")

    @property
    def units(self) -> tuple[Geometry, ...]:
        return (*self.train, *self.development, *self.evaluation)

    @property
    def decision_roster(self) -> tuple[str, ...]:
        return tuple(d for g in self.units for d in decisions(g.identity))

    @property
    def forecast_roster(self) -> tuple[str, ...]:
        return tuple(
            f"{f.key}/{d}" for f in ROSTER for g in self.evaluation for d in decisions(g.identity)
        )

    def split(self, identity: str) -> str:
        for name, values in (
            ("train", self.train),
            ("development", self.development),
            ("evaluation", self.evaluation),
        ):
            if any(g.identity == identity for g in values):
                return name
        raise ValueError("geometry not committed")

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(
            {
                "version": VERSION,
                "architecture": ARCHITECTURE,
                "proposal": PROPOSAL,
                "splits": {
                    "train": [g.payload() for g in self.train],
                    "development": [g.payload() for g in self.development],
                    "evaluation": [g.payload() for g in self.evaluation],
                },
                "nested_train": list(self.nested_train),
                "split_commitment": self.split_commitment,
                "identity_commitment": self.identity_commitment,
                "initialization_domain": INIT_DOMAIN,
                "initialization_commitments": dict(self.initialization_commitments),
                "schedule_commitments": {str(k): v for k, v in self.schedule_commitments},
                "bootstrap_domain": BOOTSTRAP_DOMAIN,
                "bootstrap_commitment": self.bootstrap_commitment,
                "source_head": self.source_head,
                "source_tree": self.source_tree,
                "source_files": dict(self.source_files),
                "decisions": list(self.decision_roster),
                "forecast_roster": list(self.forecast_roster),
            }
        )

    @property
    def digest(self) -> str:
        return digest(self.canonical_bytes())

    @classmethod
    def from_bytes(cls, data: bytes) -> MembershipLock:
        if type(data) is not bytes or len(data) > 4 * 1024 * 1024:
            raise ValueError("bounded membership bytes required")
        p = _json(data)
        _keys(
            p,
            {
                "version",
                "architecture",
                "proposal",
                "splits",
                "nested_train",
                "split_commitment",
                "identity_commitment",
                "initialization_domain",
                "initialization_commitments",
                "schedule_commitments",
                "bootstrap_domain",
                "bootstrap_commitment",
                "source_head",
                "source_tree",
                "source_files",
                "decisions",
                "forecast_roster",
            },
        )
        _keys(p["splits"], {"train", "development", "evaluation"})
        _keys(p["initialization_commitments"], set(INITIALIZATIONS))
        _keys(p["schedule_commitments"], {"16", "64"})
        if p["initialization_domain"] != INIT_DOMAIN or p["bootstrap_domain"] != BOOTSTRAP_DOMAIN:
            raise ValueError("commitment domains differ")
        result = cls(
            tuple(Geometry.parse(g) for g in p["splits"]["train"]),
            tuple(Geometry.parse(g) for g in p["splits"]["development"]),
            tuple(Geometry.parse(g) for g in p["splits"]["evaluation"]),
            tuple(p["nested_train"]),
            p["split_commitment"],
            p["identity_commitment"],
            tuple((i, p["initialization_commitments"][i]) for i in INITIALIZATIONS),
            tuple((b, p["schedule_commitments"][str(b)]) for b in (16, 64)),
            p["bootstrap_commitment"],
            p["source_head"],
            p["source_tree"],
            tuple(sorted(p["source_files"].items())),
        )
        if result.canonical_bytes() != data:
            raise ValueError("canonical complete membership/roster/specification required")
        return result
