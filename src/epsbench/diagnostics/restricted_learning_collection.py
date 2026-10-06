"""Explicit private preparation and single-use qualification; no default launch.

Receipts are external evidence, not owner authority. Actual use additionally requires
private archive ACLs and isolated evaluator/author processes. Source smoke injects
symbolic backends and never imports the actual descriptor producer.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from epsbench.diagnostics.restricted_learning_contract import VERSION, digest, sha
from epsbench.diagnostics.restricted_learning_retention import Archive, reject_reparse
from epsbench.diagnostics.visible_forecast_contract import _json, _keys
from epsbench.utils.canonical import canonical_json_bytes

INCLUSIVE_BYTES = 1024**3
PACKAGING_RESERVE = INCLUSIVE_BYTES // 2


@dataclass(frozen=True)
class Decision:
    """Externally supplied identity, separately checked against exact source/lock."""

    owner_decision: str
    source_head: str
    source_tree: str
    access_staging: str
    membership: str | None = None

    def __post_init__(self) -> None:
        sha(self.owner_decision)
        sha(self.access_staging)
        if self.membership is not None:
            sha(self.membership)
        for value in (self.source_head, self.source_tree):
            if (
                type(value) is not str
                or len(value) != 40
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError("external exact source identity required")

    def receipt(self) -> bytes:
        return canonical_json_bytes(
            {
                "version": VERSION,
                "source_head": self.source_head,
                "source_tree": self.source_tree,
                "membership": self.membership,
                "owner_decision": self.owner_decision,
            }
        )


class _Backend(Protocol):
    def source(self, root: Path, decision: Decision) -> tuple[tuple[str, bytes], ...]: ...
    def prepare(
        self, seeds: tuple[bytes, ...], decision: Decision, files: tuple[tuple[str, bytes], ...]
    ) -> bytes: ...
    def validate(self, lock: bytes, seeds: tuple[bytes, ...], root: Path) -> Any: ...
    def qualify(
        self, lock: Any, seeds: tuple[bytes, ...], root: Path, archive: Archive, receipt: bytes
    ) -> None: ...
    def inspect(self, root: Path) -> dict[str, Any]: ...


def _runtime() -> bytes:
    # Volatile provenance is retained separately from scientific source/membership hashes.
    return canonical_json_bytes(
        {
            "python": sys.version,
            "platform": platform.platform(),
            "implementation": platform.python_implementation(),
            "pid": os.getpid(),
            "utc_ns": time.time_ns(),
            "planning_cpu_seconds": 3600,
            "planning_process_ram_bytes": 4 * 1024**3,
            "resource_enforcement": "cooperative deadline/payload; no hard host guarantee",
        }
    )


def _total(root: Path) -> int:
    total = 0
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in (*dirs, *names):
            path = Path(directory) / name
            reject_reparse(path)
            if path.is_file():
                total += path.stat().st_size
            elif not path.is_dir():
                raise ValueError("regular retained files/directories required")
    if total > INCLUSIVE_BYTES:
        raise RuntimeError("inclusive original evidence bound exceeded")
    return total


def _read(root: Path, name: str) -> bytes:
    path = root / name
    reject_reparse(path)
    if not path.is_file() or path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("bounded private precommit artifact required")
    return path.read_bytes()


def _evidence_root(root: Path) -> str:
    _total(root)
    files = []
    for path in sorted(root.rglob("*")):
        reject_reparse(path)
        if path.is_file():
            hashed = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    hashed.update(chunk)
            files.append((path.relative_to(root).as_posix(), hashed.hexdigest()))
    return digest(canonical_json_bytes(files))


def _private(seeds: tuple[bytes, ...]) -> Any:
    from epsbench.diagnostics.restricted_learning_producer import PrivateSeeds

    if len(seeds) != 7 or any(
        type(v) is not bytes or len(v) != size
        for v, size in zip(seeds, (32, 32, 32, 32, 8, 8, 8), strict=True)
    ):
        raise ValueError("retained exact private seed preimages required")
    return PrivateSeeds(
        seeds[0],
        seeds[1],
        seeds[2],
        seeds[3],
        (
            int.from_bytes(seeds[4], "little"),
            int.from_bytes(seeds[5], "little"),
            int.from_bytes(seeds[6], "little"),
        ),
    )


def _failure(archive: Archive, error: Exception) -> None:
    # A pre-dataset failure has no membership journal. Preserve its stage privately.
    try:
        archive.write(
            "operator-failure.json",
            canonical_json_bytes(
                {
                    "disposition": "INCONCLUSIVE",
                    "status": "FAILED_CLOSED",
                    "failure_kind": type(error).__name__,
                    "phase_gate_effect": "NONE",
                    "prior_evidence_root": _evidence_root(archive.root),
                }
            ),
            failure=True,
        )
        error.add_note("aggregate retained failure root: " + _evidence_root(archive.root))
    except Exception as preservation_error:
        error.add_note("operator preservation error: " + type(preservation_error).__name__)


def _prepare(
    attempt: Path,
    source_root: Path,
    decision: Decision,
    backend: _Backend,
    entropy: Callable[[int], bytes],
) -> dict[str, Any]:
    if decision.membership is not None:
        raise PermissionError("preparation decision must precede membership")
    # Exclusive directory creation precedes any entropy draw; never replace an attempt.
    archive = Archive(attempt, max_bytes=INCLUSIVE_BYTES - PACKAGING_RESERVE)
    try:
        archive.write("preparation-decision.json", decision.receipt())
        archive.write(
            "access-staging.json",
            canonical_json_bytes(
                {
                    "declaration": decision.access_staging,
                    "assumption": "private ACLs and separate evaluator/author processes",
                }
            ),
        )
        archive.write("prepare-runtime.json", _runtime())
        files = backend.source(source_root, decision)
        recipe = []
        for i, (name, content) in enumerate(files):
            archive.write(f"source-{i}.bin", content)
            recipe.append(
                {"logical_path": name, "file": f"source-{i}.bin", "sha256": digest(content)}
            )
        archive.write("source-recipe.json", canonical_json_bytes({"files": recipe}))
        seeds = []
        for i, size in enumerate((32, 32, 32, 32, 8, 8, 8)):
            archive.check()
            value = entropy(size)
            # Retain even colliding/malformed preimages before validation. No redraw.
            archive.write(f"seed-{i}.bin", value, failure=True)
            seeds.append(value)
        archive.check()
        lock = backend.prepare(tuple(seeds), decision, files)
        archive.write("membership-lock.json", lock)
        backend.validate(lock, tuple(seeds), source_root)
        binding = archive.write(
            "precommit.json",
            canonical_json_bytes(
                {
                    "version": VERSION,
                    "membership": digest(lock),
                    "source_head": decision.source_head,
                    "source_tree": decision.source_tree,
                    "preparation_decision": digest(decision.receipt()),
                    "access_staging": decision.access_staging,
                    "source_recipe": digest(archive.read("source-recipe.json")),
                    "seed_preimages": [digest(archive.read(f"seed-{i}.bin")) for i in range(7)],
                    "inclusive_bytes": INCLUSIVE_BYTES,
                    "packaging_reserve": PACKAGING_RESERVE,
                }
            ),
        )
        used = _total(attempt)
        return {
            "disposition": "PRECOMMITTED",
            "membership": digest(lock),
            "precommit": binding,
            "private_attempt_root": _evidence_root(attempt),
            "retained_bytes": used,
            "packaging_reserve": PACKAGING_RESERVE,
            "phase_gate_effect": "NONE",
        }
    except Exception as error:
        _failure(archive, error)
        raise


def _qualify(
    attempt: Path,
    source_root: Path,
    decision: Decision,
    backend: _Backend,
    collection_receipt: bytes,
) -> dict[str, Any]:
    reject_reparse(attempt)
    if not attempt.is_dir() or decision.membership is None:
        raise PermissionError(
            "existing private precommit and separate membership decision required"
        )
    # This exclusive claim is permanent, including failures before dataset creation.
    claim = attempt / "qualification-claim.json"
    reject_reparse(claim)
    with claim.open("xb") as stream:
        stream.write(decision.receipt())
        stream.flush()
        os.fsync(stream.fileno())
    journal: Archive | None = None
    lock: Any = None
    try:
        supplied = _json(collection_receipt)
        _keys(supplied, {"version", "source_head", "source_tree", "membership", "owner_decision"})
        if (
            supplied["version"] != VERSION
            or supplied["source_head"] != decision.source_head
            or supplied["source_tree"] != decision.source_tree
            or supplied["membership"] != decision.membership
            or supplied["owner_decision"] != "EXACT_SOURCE_COLLECTION_AUTHORIZED"
            or canonical_json_bytes(supplied) != collection_receipt
        ):
            raise PermissionError("unchanged external exact-source collection receipt required")
        receipt_path = attempt / "qualification-receipt.json"
        with receipt_path.open("xb") as stream:
            stream.write(collection_receipt)
            stream.flush()
            os.fsync(stream.fileno())
        if (attempt / "operator-failure.json").exists():
            raise PermissionError("terminal preparation cannot qualify")
        lock_bytes = _read(attempt, "membership-lock.json")
        precommit_bytes = _read(attempt, "precommit.json")
        precommit = _json(precommit_bytes)
        _keys(
            precommit,
            {
                "version",
                "membership",
                "source_head",
                "source_tree",
                "preparation_decision",
                "access_staging",
                "source_recipe",
                "seed_preimages",
                "inclusive_bytes",
                "packaging_reserve",
            },
        )
        if (
            precommit["version"] != VERSION
            or precommit["membership"] != digest(lock_bytes)
            or precommit["membership"] != decision.membership
            or precommit["source_head"] != decision.source_head
            or precommit["source_tree"] != decision.source_tree
            or precommit["access_staging"] != decision.access_staging
            or precommit["inclusive_bytes"] != INCLUSIVE_BYTES
            or precommit["packaging_reserve"] != PACKAGING_RESERVE
        ):
            raise ValueError("unchanged exact private precommit required")
        seeds = tuple(_read(attempt, f"seed-{i}.bin") for i in range(7))
        if [digest(s) for s in seeds] != precommit["seed_preimages"] or digest(
            _read(attempt, "preparation-decision.json")
        ) != precommit["preparation_decision"]:
            raise ValueError("retained seed/decision preimages changed")
        files = backend.source(source_root, decision)
        recipe_bytes = _read(attempt, "source-recipe.json")
        recipe = _json(recipe_bytes)
        expected = [
            {"logical_path": name, "file": f"source-{i}.bin", "sha256": digest(content)}
            for i, (name, content) in enumerate(files)
        ]
        if (
            recipe != {"files": expected}
            or digest(recipe_bytes) != precommit["source_recipe"]
            or any(
                _read(attempt, r["file"]) != content
                for r, (_, content) in zip(expected, files, strict=True)
            )
        ):
            raise ValueError("retained complete logical source preimages changed")
        lock = backend.validate(lock_bytes, seeds, source_root)
        used = _total(attempt)
        remaining = INCLUSIVE_BYTES - PACKAGING_RESERVE - used
        journal = Archive(attempt / "dataset", max_bytes=remaining)
        journal.write("qualification-runtime.json", _runtime())
        backend.qualify(lock, seeds, source_root, journal, collection_receipt)
        inspected = backend.inspect(journal.root)
        if inspected["failed"] or inspected["seal_a"] is None or inspected["seal_b"] is not None:
            raise ValueError("qualification-only complete immutable SealA required")
        total = _total(attempt)
        if total + PACKAGING_RESERVE > INCLUSIVE_BYTES:
            raise RuntimeError("review packaging reservation exceeded")
        return {
            "disposition": "QUALIFIED",
            "membership": decision.membership,
            "precommit": digest(precommit_bytes),
            "seal_a": inspected["seal_a"],
            "history_root": inspected["history_root"],
            "private_attempt_root": _evidence_root(attempt),
            "events": inspected["events"],
            "retained_bytes": total,
            "packaging_reserve": PACKAGING_RESERVE,
            "phase_gate_effect": "NONE",
        }
    except Exception as error:
        # Lifecycle preserves per-member pending coverage; retain any earlier failure too.
        if journal is not None:
            if not journal.failed and journal.events and hasattr(lock, "decision_roster"):
                try:
                    journal.fail(lock.decision_roster, error)
                except Exception as preservation_error:
                    error.add_note(
                        "pending-coverage preservation error: " + type(preservation_error).__name__
                    )
            _failure(journal, error)
        else:
            path = attempt / "qualification-failure.json"
            try:
                with path.open("xb") as stream:
                    stream.write(
                        canonical_json_bytes(
                            {
                                "disposition": "INCONCLUSIVE",
                                "status": "FAILED_CLOSED",
                                "failure_kind": type(error).__name__,
                            }
                        )
                    )
                    stream.flush()
                    os.fsync(stream.fileno())
                error.add_note("aggregate retained failure root: " + _evidence_root(attempt))
            except Exception as preservation_error:
                error.add_note(
                    "qualification preservation error: " + type(preservation_error).__name__
                )
        raise


class _ActualBackend:
    """Every producer import and membership derivation is lazy, actual-use only."""

    def source(self, root: Path, decision: Decision) -> tuple[tuple[str, bytes], ...]:
        from epsbench.diagnostics.restricted_learning_producer import SOURCE_FILES

        def git(*args: str) -> str:
            return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

        if (
            git("rev-parse", "HEAD") != decision.source_head
            or git("rev-parse", "HEAD^{tree}") != decision.source_tree
            or git("status", "--porcelain")
        ):
            raise ValueError("clean exact externally accepted source required")
        return tuple(
            (name, (root / name).read_bytes().replace(b"\r\n", b"\n")) for name in SOURCE_FILES
        )

    def prepare(
        self, seeds: tuple[bytes, ...], decision: Decision, files: tuple[tuple[str, bytes], ...]
    ) -> bytes:
        from epsbench.diagnostics.restricted_learning_contract import INITIALIZATIONS
        from epsbench.diagnostics.restricted_learning_membership import (
            BOOTSTRAP_DOMAIN,
            INIT_DOMAIN,
            MembershipLock,
        )
        from epsbench.diagnostics.restricted_learning_producer import (
            IDENTITY_DOMAIN,
            SPLIT_DOMAIN,
            derived_splits,
            exact_bytes,
            schedule,
        )
        from epsbench.diagnostics.restricted_learning_sampling import commitment

        private = _private(seeds)
        train, development, evaluation = derived_splits(private)
        nonces = {
            f"{g.identity}/prefix-{p}": hashlib.sha256(
                IDENTITY_DOMAIN.encode()
                + b"\0"
                + private.identity
                + bytes.fromhex(g.identity)
                + bytes([p])
            )
            .digest()
            .hex()
            for g in (*train, *development, *evaluation)
            for p in range(2)
        }
        return MembershipLock(
            train,
            development,
            evaluation,
            tuple(g.identity for g in train[:16]),
            commitment(SPLIT_DOMAIN, private.split),
            digest(exact_bytes(dict(sorted(nonces.items())))),
            tuple(
                (i, commitment(INIT_DOMAIN, seed.to_bytes(8, "little")))
                for i, seed in zip(INITIALIZATIONS, private.initializations, strict=True)
            ),
            tuple(
                (
                    b,
                    digest(
                        exact_bytes(schedule(train[:16] if b == 16 else train, private.schedule, b))
                    ),
                )
                for b in (16, 64)
            ),
            commitment(BOOTSTRAP_DOMAIN, private.bootstrap),
            decision.source_head,
            decision.source_tree,
            tuple((name, digest(content)) for name, content in files),
        ).canonical_bytes()

    def validate(self, lock: bytes, seeds: tuple[bytes, ...], root: Path) -> Any:
        from epsbench.diagnostics.restricted_learning_membership import MembershipLock
        from epsbench.diagnostics.restricted_learning_producer import verify_private_lock

        private = _private(seeds)
        result = MembershipLock.from_bytes(lock)
        verify_private_lock(result, private, root)
        return result

    def qualify(
        self, lock: Any, seeds: tuple[bytes, ...], root: Path, archive: Archive, receipt: bytes
    ) -> None:
        from epsbench.diagnostics.restricted_learning_producer import (
            DescriptorAdapter,
            verify_private_lock,
        )
        from epsbench.diagnostics.restricted_learning_retention import Lifecycle

        private = _private(seeds)
        lifecycle = Lifecycle(lock, archive)
        provider = DescriptorAdapter(
            lock, archive, verify_private_lock(lock, private, root), receipt, private, root
        )
        lifecycle.qualify(provider)

    def inspect(self, root: Path) -> dict[str, Any]:
        from epsbench.diagnostics.restricted_learning_retention import inspect_archive

        return inspect_archive(root)


def prepare_private_precommit(
    attempt: Path, source_root: Path, preparation_decision: Decision
) -> dict[str, Any]:
    """Actual allocation ONLY after separate external preparation authorization."""
    import secrets

    return _prepare(
        attempt, source_root, preparation_decision, _ActualBackend(), secrets.token_bytes
    )


def qualify_once(
    attempt: Path,
    source_root: Path,
    qualification_decision: Decision,
    collection_receipt: bytes,
) -> dict[str, Any]:
    """Actual one-shot qualification ONLY after exact membership/source authorization."""
    return _qualify(
        attempt, source_root, qualification_decision, _ActualBackend(), collection_receipt
    )
