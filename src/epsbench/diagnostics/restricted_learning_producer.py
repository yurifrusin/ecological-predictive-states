"""Isolated privileged descriptor adapter; no allocation or launch entrypoint.

Import denied by source checks. A later exact-source collection decision must
supply the private MembershipLock/nonces and evaluator archive. Source acceptance
alone is not that decision; collect() remains disabled in this package.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction as Q
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from epsbench.diagnostics.occupancy_reference import Solid, audit, status
from epsbench.diagnostics.restricted_exact_raster import VERSION as RASTER_VERSION
from epsbench.diagnostics.restricted_exact_raster import Box, raster
from epsbench.diagnostics.restricted_learning_contract import (
    LIMITS,
    REQUIRED,
    VERSION,
    InputEvidence,
    digest,
)
from epsbench.diagnostics.restricted_learning_membership import (
    ANNOUNCED,
    BOOTSTRAP_DOMAIN,
    INIT_DOMAIN,
    PREFIXES,
    Geometry,
    MembershipLock,
)
from epsbench.diagnostics.restricted_learning_retention import Archive, QualifiedTarget
from epsbench.diagnostics.restricted_learning_sampling import commitment, shuffled
from epsbench.diagnostics.visible_forecast_contract import CausalInput, TokenFrame
from epsbench.schema import ModalityPermissionSet


def exact_bytes(value: Any) -> bytes:
    import json

    def exact(v: Any) -> str:
        if type(v) is not Q:
            raise TypeError("unsupported privileged annotation")
        return str(v)

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=exact
    ).encode()


SOURCE_FILES = (
    "src/epsbench/diagnostics/restricted_learning_collection.py",
    "src/epsbench/diagnostics/restricted_learning_contract.py",
    "src/epsbench/diagnostics/restricted_learning_membership.py",
    "src/epsbench/diagnostics/restricted_learning_sampling.py",
    "src/epsbench/diagnostics/restricted_learning_scoring.py",
    "src/epsbench/diagnostics/restricted_learning_retention.py",
    "src/epsbench/diagnostics/restricted_learning_producer.py",
    "src/epsbench/diagnostics/restricted_exact_raster.py",
    "src/epsbench/diagnostics/visible_forecast_contract.py",
    "src/epsbench/diagnostics/boundary_observation.py",
    "src/epsbench/diagnostics/occupancy_reference.py",
    "src/epsbench/diagnostics/causal_history_fixture.py",
    "src/epsbench/schema.py",
    "src/epsbench/utils/canonical.py",
    "docs/RESTRICTED_OCCUPANCY_LEARNING_V1.md",
    "docs/RESTRICTED_OCCUPANCY_ARCHITECTURE_V1.md",
)
SPLIT_DOMAIN = "M0-occupancy-development-v1/split-v1"
IDENTITY_DOMAIN = "M0-occupancy-development-v1/identity-v1"
SCHEDULE_DOMAIN = "M0-occupancy-development-v1/schedule-v1"


@dataclass(frozen=True)
class PrivateSeeds:
    split: bytes
    identity: bytes
    schedule: bytes
    bootstrap: bytes
    initializations: tuple[int, int, int]

    def __post_init__(self) -> None:
        values = (self.split, self.identity, self.schedule, self.bootstrap)
        if any(type(v) is not bytes or len(v) != 32 for v in values) or len(set(values)) != 4:
            raise ValueError("four distinct privately allocated 32-byte domain seeds required")
        if (
            type(self.initializations) is not tuple
            or len(self.initializations) != 3
            or any(type(v) is not int or not 0 <= v < 2**64 for v in self.initializations)
            or len(set(self.initializations)) != 3
        ):
            raise ValueError("three distinct paired private uint64 initialization seeds required")


def derived_splits(seeds: PrivateSeeds) -> tuple[tuple[Geometry, ...], ...]:
    # Actual domain is lazy and forbidden to every source-check process.
    domain = tuple(
        Geometry(*v)
        for v in product(
            (Q(3, 5), Q(7, 10), Q(4, 5)),
            (Q(7, 20), Q(9, 20), Q(11, 20)),
            (Q(-1, 4), Q(0), Q(1, 4)),
            (Q(5, 2), Q(3), Q(7, 2)),
            (Q(0), Q(1, 20), Q(1, 10)),
        )
    )
    ordered = shuffled(domain, SPLIT_DOMAIN, seeds.split)
    return ordered[:64], ordered[64:80], ordered[80:112]


def schedule(geometries: tuple[Geometry, ...], seed: bytes, budget: int) -> tuple[str, ...]:
    counters = {g.identity: 0 for g in geometries}
    decision_order = {
        g.identity: shuffled(tuple(range(8)), SCHEDULE_DOMAIN + "/decisions/" + g.identity, seed)
        for g in geometries
    }
    result: list[str] = []
    cycle = 0
    while len(result) < 16000:
        for g in shuffled(geometries, SCHEDULE_DOMAIN + f"/budget-{budget}/cycle-{cycle}", seed):
            k = decision_order[g.identity][counters[g.identity] % 8]
            result.append(f"{g.identity}/prefix-{k // 4}/action-{k % 4}")
            counters[g.identity] += 1
            if len(result) == 16000:
                break
        cycle += 1
    return tuple(result)


def identities(lock: MembershipLock, seed: bytes) -> dict[str, bytes]:
    return {
        f"{g.identity}/prefix-{p}": hashlib.sha256(
            IDENTITY_DOMAIN.encode() + b"\0" + seed + bytes.fromhex(g.identity) + bytes([p])
        ).digest()
        for g in lock.units
        for p in range(2)
    }


def verify_private_lock(lock: MembershipLock, seeds: PrivateSeeds, root: Path) -> dict[str, bytes]:
    from epsbench.diagnostics.restricted_learning_contract import INITIALIZATIONS

    derived = derived_splits(seeds)
    if (
        derived != (lock.train, lock.development, lock.evaluation)
        or lock.nested_train != tuple(g.identity for g in derived[0][:16])
        or lock.split_commitment != commitment(SPLIT_DOMAIN, seeds.split)
    ):
        raise ValueError("unbiased private split/nested subset derivation differs")
    init = tuple(
        (i, commitment(INIT_DOMAIN, seed.to_bytes(8, "little")))
        for i, seed in zip(INITIALIZATIONS, seeds.initializations, strict=True)
    )
    orders = tuple(
        (
            b,
            digest(
                exact_bytes(schedule(lock.train[:16] if b == 16 else lock.train, seeds.schedule, b))
            ),
        )
        for b in (16, 64)
    )
    nonces = identities(lock, seeds.identity)
    if (
        lock.initialization_commitments != init
        or lock.schedule_commitments != orders
        or (
            lock.bootstrap_commitment != commitment(BOOTSTRAP_DOMAIN, seeds.bootstrap)
            or lock.identity_commitment
            != digest(exact_bytes({k: v.hex() for k, v in sorted(nonces.items())}))
        )
    ):
        raise ValueError("private seed/domain/complete schedule commitments differ")

    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    if (
        git("rev-parse", "HEAD") != lock.source_head
        or git("rev-parse", "HEAD^{tree}") != lock.source_tree
        or (git("status", "--porcelain", "--untracked-files=no"))
    ):
        raise ValueError("clean exact accepted source HEAD/tree required before collection")
    recipe = tuple(
        (name, digest((root / name).read_bytes().replace(b"\r\n", b"\n"))) for name in SOURCE_FILES
    )
    if dict(lock.source_files) != dict(recipe) or len(lock.source_files) != len(recipe):
        raise ValueError("complete fixed logical source recipe differs")
    return nonces


class DescriptorAdapter:
    """Evaluator-only callable adapter, never supplied to an author/learner.

    Construction requires an already retained complete membership lock and a
    separate exact-source collection receipt. The receipt is evidence, not code
    manufacturing owner authority; independent review verifies its authorization.
    """

    def __init__(
        self,
        lock: MembershipLock,
        archive: Archive,
        nonces: dict[str, bytes],
        collection_receipt: bytes,
        seeds: PrivateSeeds,
        source_root: Path,
    ) -> None:
        from epsbench.diagnostics.visible_forecast_contract import _json, _keys

        if (
            type(lock) is not MembershipLock
            or archive.events[0]["kind"] != "MEMBERSHIP_LOCK"
            or (archive.read("membership-lock.json") != lock.canonical_bytes())
        ):
            raise PermissionError("actual precollection membership lock required")
        receipt = _json(collection_receipt)
        _keys(receipt, {"version", "source_head", "source_tree", "membership", "owner_decision"})
        if (
            receipt["version"] != VERSION
            or receipt["source_head"] != lock.source_head
            or (
                receipt["source_tree"] != lock.source_tree
                or receipt["membership"] != lock.digest
                or receipt["owner_decision"] != "EXACT_SOURCE_COLLECTION_AUTHORIZED"
            )
        ):
            raise PermissionError("separate exact-source collection decision required")
        if nonces != verify_private_lock(lock, seeds, source_root):
            raise ValueError("privately derived perprefix identities differ")
        if (
            set(nonces) != {f"{g.identity}/prefix-{p}" for g in lock.units for p in range(2)}
            or any(type(n) is not bytes or len(n) != 32 for n in nonces.values())
            or len(set(nonces.values())) != 224
            or digest(exact_bytes({k: v.hex() for k, v in sorted(nonces.items())}))
            != lock.identity_commitment
        ):
            raise ValueError("precommitted distinct private pergeometry identity nonces required")
        self.lock, self.archive = lock, archive
        self.geometries = {g.identity: g for g in lock.units}
        self.mapping = {
            k: {
                i: "surface-"
                + hashlib.sha256(VERSION.encode() + b"\0" + n + bytes([i])).hexdigest()[:16]
                for i in (1, 2, 3)
            }
            for k, n in nonces.items()
        }
        if len({v for m in self.mapping.values() for v in m.values()}) != 672:
            raise ValueError("opaque collision")
        self.prefixes: dict[str, InputEvidence] = {}
        self.calls: set[tuple[str, int]] = set()
        self._prefix_cache: dict[
            str, tuple[tuple[TokenFrame, ...], tuple[dict[str, Any], ...]]
        ] = {}
        archive.event("COLLECTION_AUTHORITY", "collection-authority.json", collection_receipt)
        archive.event(
            "PRIVATE_IDENTITIES",
            "identity-private.json",
            exact_bytes(
                {"nonces": {k: v.hex() for k, v in nonces.items()}, "raw_to_opaque": self.mapping}
            ),
        )

    def _frame(
        self, g: Geometry, lateral: Q, key: str, observed: set[int]
    ) -> tuple[TokenFrame, dict[str, Any]]:
        self.archive.check()
        fg = Solid(
            (-g.foreground_width, g.foreground_y, Q(2, 5)),
            (g.foreground_width, g.foreground_y + Q(1, 5), Q(8, 5)),
        )
        bg = Solid(
            (g.background_x - g.background_width, g.background_y, Q(1, 2)),
            (g.background_x + g.background_width, g.background_y + Q(1, 5), Q(3, 2)),
        )

        produced = raster(
            lateral,
            (Box(fg.lower, fg.upper), Box(bg.lower, bg.upper)),
            (Q(4), Q(7)),
            self.archive.check,
        )
        raw = np.array(produced.labels, dtype=np.int64)
        if raw.dtype != np.int64 or raw.shape != (32, 32) or np.any((raw < 0) | (raw > 3)):
            raise ValueError("bounded supported raw raster required")
        raw_data = raw.astype("<i8").tobytes()
        index = len(self.archive.events)
        self.archive.event("RAW_PRIVATE", f"raw-{index}.bin", raw_data)
        self.archive.event(
            "AUDIT_PRIVATE",
            f"audit-{index}.json",
            exact_bytes({"version": RASTER_VERSION, "ties": produced.ties}),
        )
        result = audit((lateral, Q(-3), Q(1)), (fg, bg), (Q(4), Q(7)), 32, self.archive.check)
        ref_data = np.array(result.labels, dtype="<i8").tobytes()
        self.archive.event(
            "REFERENCE_PRIVATE", f"reference-{index}.json", exact_bytes(asdict(result))
        )
        mapping = self.mapping["/".join(key.split("/")[:2])]
        all_observed = observed | {int(v) for v in np.unique(raw) if v}
        records = {
            str(label): asdict(
                status(result, label, label in all_observed, (lateral, Q(-3), Q(1)), fg)
            )
            for label in (1, 2, 3)
        }
        annotation = {
            "raw": digest(raw_data),
            "reference": digest(ref_data),
            "ties": result.ties,
            "boundaries": result.boundaries,
            "statuses": records,
        }
        self.archive.event("AUDIT_PRIVATE", f"audit-{index + 1}.json", exact_bytes(annotation))
        if raw_data != ref_data or result.ties or produced.ties:
            raise ValueError("unique independent reference disagreement; no replacement")
        masks = tuple((mapping[i], raw == i) for i in (1, 2, 3) if np.any(raw == i))
        return TokenFrame(0, (32, 32), masks), annotation

    def prefix(self, decision: str) -> tuple[InputEvidence, bytes]:
        if decision not in self.lock.decision_roster or decision in self.prefixes:
            raise PermissionError("fresh exact committed decision; no prefix replay")
        geometry, prefix_name, _ = decision.split("/")
        prefix = int(prefix_name.removeprefix("prefix-"))
        poses = PREFIXES[prefix]
        g = self.geometries[geometry]
        observed: set[int] = set()
        frames, annotations = [], []
        prefix_key = geometry + "/" + prefix_name
        cached = self._prefix_cache.get(prefix_key)
        for i, lateral in enumerate(poses):
            if (prefix_key, i) in self.calls and cached is None:
                raise PermissionError("no producer repeat")
            if cached is None:
                self.calls.add((prefix_key, i))
                frame, annotation = self._frame(g, lateral, decision, observed)
            else:
                frame, annotation = cached[0][i], cached[1][i]
            frames.append(TokenFrame(i, frame.shape, frame.masks))
            annotations.append(annotation)
            observed.update(
                label
                for label, h in self.mapping[geometry + "/" + prefix_name].items()
                if h in {t for t, _ in frame.masks}
            )
        if cached is None:
            self._prefix_cache[prefix_key] = (tuple(frames), tuple(annotations))
        if 3 not in {
            label
            for label, h in self.mapping[geometry + "/" + prefix_name].items()
            if h in {t for f in frames[:2] for t, _ in f.masks}
        }:
            raise ValueError("background must previously be observed")
        hidden = annotations[2]["statuses"]["3"]["cause"]
        if hidden != "COMPLETE_IN_FRUSTUM_OCCLUSION":
            raise ValueError("pure full in-frame decision hiding required")
        action_index = int(decision.split("/")[-1].removeprefix("action-"))
        action = ANNOUNCED[action_index]
        executed = tuple((Q(0), poses[i + 1] - poses[i], Q(0)) for i in range(2))
        source = InputEvidence(
            CausalInput(
                tuple(frames),
                executed,
                (Q(0), action, Q(0)),
                ModalityPermissionSet(allowed=REQUIRED),
                LIMITS,
            )
        )
        self.prefixes[decision] = source
        return source, exact_bytes(
            {"version": VERSION, "audits": annotations, "decision_occlusion": hidden}
        )

    def target(self, decision: str) -> QualifiedTarget:
        if decision not in self.prefixes or (decision, 3) in self.calls:
            raise PermissionError("frozen input before single target access")
        self.calls.add((decision, 3))
        geometry, prefix_name, action_name = decision.split("/")
        source = self.prefixes[decision]
        observed = {
            i
            for i, h in self.mapping[geometry + "/" + prefix_name].items()
            if h in source.prefix.inventory
        }
        frame, annotation = self._frame(
            self.geometries[geometry],
            ANNOUNCED[int(action_name.removeprefix("action-"))],
            decision,
            observed,
        )
        visible = {h for h, _ in frame.masks}
        annotation.update({"version": VERSION, "descriptor": geometry})
        # New future tokens remain private annotation content, outside observed truth.
        return QualifiedTarget(
            tuple((h, h in visible) for h in source.prefix.inventory), exact_bytes(annotation)
        )


def collect() -> None:
    raise PermissionError(
        "collection entrypoint disabled; separately authorized exact-source collection required"
    )
