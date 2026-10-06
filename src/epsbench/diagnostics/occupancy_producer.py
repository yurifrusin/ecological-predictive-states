"""Actual pinned producer/collection entrypoint. Source checks must deny this module.

No launch CLI, seed allocator, historical controller, transport or native renderer.
Calling collect requires a separately reviewed exact-source launch, not source approval.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict
from fractions import Fraction as Q
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from epsbench.diagnostics import causal_history_fixture as producer
from epsbench.diagnostics.boundary_observation import VisibleRaster
from epsbench.diagnostics.occupancy_collection import (
    MEMBERS,
    VERSION,
    Qualification,
    Retention,
    digest,
    encode,
)
from epsbench.diagnostics.occupancy_reference import Solid, audit, status
from epsbench.diagnostics.visible_forecast_contract import Command

SOURCE_HEAD = "bf9d2f80caa92184232f70c101d835263433b323"
SOURCE_TREE = "cf3711631fdb498d467465ab4fceb478b0f84f07"
PROPOSAL_SHA256 = "753f461c597ea5e58c29c63cfec0bf92403d6a39ac53a4c1905537fc67571494"
# Source logical LF bytes, independent of checkout newline conversion.
PRODUCER_SHA256 = "f924d14cbb96fb18981b54288058629596b6ebe7dca40f6a3c9cf2d458e1d600"
OCCLUDER = Solid((Q(-3, 5), Q(0), Q(2, 5)), (Q(3, 5), Q(1, 5), Q(8, 5)))
BACKGROUND = Solid((Q(-1, 2), Q(3), Q(1, 2)), (Q(1, 2), Q(16, 5), Q(3, 2)))
TABLE: dict[str, tuple[str, Q, Q, Q, Q]] = {
    "R": ("RH", Q(2), Q(0), Q(-2), Q(-2)),
    "H": ("RH", Q(2), Q(0), Q(1, 4), Q(1, 4)),
    "F": ("F", Q(2), Q(57, 8), Q(1), Q(65, 8)),
    "P": ("P", Q(2), Q(63, 10), Q(1, 10), Q(32, 5)),
}


def logical_source(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def floats(p: tuple[Q, Q, Q]) -> tuple[float, float, float]:
    return float(p[0]), float(p[1]), float(p[2])


def action(lateral: Q) -> Command:
    return (Q(0), lateral, Q(0))


def source_plan(nonces: dict[str, bytes]) -> dict[str, Any]:
    here = Path(__file__).parent
    return {
        "version": VERSION,
        "proposal_sha256": PROPOSAL_SHA256,
        "baseline_head": SOURCE_HEAD,
        "baseline_tree": SOURCE_TREE,
        "code": {
            name: digest(logical_source(here / name))
            for name in (
                "occupancy_producer.py",
                "occupancy_reference.py",
                "occupancy_collection.py",
                "visible_forecast_contract.py",
                "causal_history_fixture.py",
            )
        },
        "table": {
            m: [unit, *map(str, (a0, a1, announced, a2))]
            for m, (unit, a0, a1, announced, a2) in TABLE.items()
        },
        "geometry": {
            "camera": [32, 32, "-3", "1", "0", "90"],
            "occluder": asdict(OCCLUDER),
            "background": asdict(BACKGROUND),
            "support": ["4", "7"],
        },
        "limits": {
            "frames": 2,
            "pixels": 1024,
            "tokens": 3,
            "calls": 10,
            "total_bytes": 16777216,
            "failure_reserve": 262144,
            "cooperative_seconds": 60,
        },
        "identity_nonce_hashes": {u: digest(n) for u, n in nonces.items()},
        "membership": list(MEMBERS),
        "forecasts": 8,
        "physical_units": 3,
        "finite_near_far": "NOT_APPLICABLE",
    }


class SceneAdapter:
    def __init__(self, retained: Retention, nonces: dict[str, bytes]) -> None:
        if (
            set(nonces) != {"RH", "F", "P"}
            or any(type(n) is not bytes or len(n) != 32 for n in nonces.values())
            or len(set(nonces.values())) != 3
        ):
            raise ValueError("three distinct private preallocated 32-byte identity nonces required")
        if digest(logical_source(Path(producer.__file__))) != PRODUCER_SHA256:
            raise ValueError("pinned producer source differs")
        self.retained = retained
        self.target_gate: Callable[[], None] | None = None
        self.scene = producer.Scene(
            producer.Camera(32, 32, -3.0, 1.0, 0.0, 90.0),
            producer.Box(floats(OCCLUDER.lower), floats(OCCLUDER.upper)),
            producer.Box(floats(BACKGROUND.lower), floats(BACKGROUND.upper)),
            (4.0, 7.0),
        )
        self.mapping = {
            u: {
                label: "surface-"
                + hashlib.sha256(VERSION.encode() + b"\0" + n + bytes([label])).hexdigest()[:16]
                for label in (1, 2, 3)
            }
            for u, n in nonces.items()
        }
        if len({t for mapping in self.mapping.values() for t in mapping.values()}) != 9:
            raise ValueError("opaque identity collision")
        self.observed: dict[str, set[int]] = {u: set() for u in nonces}
        self.statuses: dict[str, dict[int, str]] = {}
        self.visible_counts: dict[str, int] = {}
        self.calls: set[tuple[str, int | str]] = set()
        retained.write(
            "identity-private.json",
            encode(
                {"nonces": {u: n.hex() for u, n in nonces.items()}, "raw_to_opaque": self.mapping}
            ),
        )

    def _frame(self, unit: str, key: int | str, index: int, lateral: Q) -> VisibleRaster:
        if (unit, key) in self.calls:
            raise RuntimeError("producer call cannot be repeated")
        self.calls.add((unit, key))
        self.retained.producer_call()
        raw, points = self.scene.frame(float(lateral))
        if not np.all(np.isfinite(points)):
            raise ValueError("nonfinite producer instrumentation failed closed")
        del points  # Never materialized into causal state; no depth/world projection.
        self.retained.check()
        if raw.dtype != np.int64 or raw.shape != (32, 32) or np.any((raw < 0) | (raw > 3)):
            raise ValueError("unsupported producer labels")
        raw_sha = self.retained.write(f"raw-{unit}-{key}.bin", raw.astype("<i8").tobytes())
        origin = (lateral, Q(-3), Q(1))
        result = audit(origin, (OCCLUDER, BACKGROUND), (Q(4), Q(7)), 32, self.retained.check)
        # Preserve independent pixels/ties and raw provenance even when agreement fails.
        ref_sha = self.retained.write(f"reference-{unit}-{key}.json", encode(asdict(result)))
        self.retained.write(
            f"audit-{unit}-{key}.json",
            encode(
                {
                    "raw": raw_sha,
                    "reference": ref_sha,
                    "agreement": bool(np.array_equal(raw, np.array(result.labels))),
                    "unique": not result.ties,
                }
            ),
        )
        if not np.array_equal(raw, np.array(result.labels)) or result.ties:
            raise ValueError("exact unique rational/producer occupancy qualification failed")
        if index < 2:
            self.observed[unit].update(int(v) for v in np.unique(raw) if v != 0)
        self.visible_counts[f"{unit}-{key}"] = int(np.count_nonzero(raw == 3))
        records = {
            label: asdict(status(result, label, label in self.observed[unit], origin, OCCLUDER))
            for label in sorted(self.observed[unit])
        }
        self.statuses[f"{unit}-{key}"] = {
            label: record["cause"] for label, record in records.items()
        }
        self.retained.write(
            f"status-{unit}-{key}.json",
            encode(
                {
                    "records": records,
                    "raw_to_opaque": {label: self.mapping[unit][label] for label in records},
                    "finite_near_far": "NOT_APPLICABLE",
                }
            ),
        )
        identities = tuple(
            (label, self.mapping[unit][label]) for label in (1, 2, 3) if np.any(raw == label)
        )
        return VisibleRaster(index, raw.astype(np.int32), identities)

    def prefix(self, unit: str, index: int) -> VisibleRaster:
        if unit not in ("RH", "F", "P") or type(index) is not int or index not in (0, 1):
            raise PermissionError("fixed prefix membership required")
        member = "R" if unit == "RH" else unit
        return self._frame(
            unit, index, index, (TABLE[member][1] if index == 0 else TABLE[member][2])
        )

    def target(self, member: str) -> VisibleRaster:
        if member not in MEMBERS or self.target_gate is None:
            raise PermissionError("complete retained seal required before producer target")
        self.target_gate()
        unit, _, _, _, a2 = TABLE[member]
        return self._frame(unit, member, 2, a2)

    def relations(self) -> None:
        required = {
            "RH-0": "VISIBLE",
            "RH-1": "COMPLETE_IN_FRUSTUM_OCCLUSION",
            "RH-R": "VISIBLE",
            "RH-H": "COMPLETE_IN_FRUSTUM_OCCLUSION",
            "F-0": "VISIBLE",
            "F-1": "COMPLETE_FRAME_LOSS",
            "F-F": "COMPLETE_FRAME_LOSS",
            "P-0": "VISIBLE",
            "P-1": "PARTIAL_FRAME_LOSS",
            "P-P": "PARTIAL_FRAME_LOSS",
        }
        if any(self.visible_counts.get(k, 0) == 0 for k in ("P-1", "P-P")):
            raise ValueError("P requires nonempty surviving visible background support")
        if self.retained.calls != 10 or any(
            self.statuses.get(k, {}).get(3) != v for k, v in required.items()
        ):
            raise ValueError("fixed required relations unqualified; no replacement")


def collect(root: Path, nonces: dict[str, bytes], expected_head: str) -> dict[str, Any]:
    """Actual launch entrypoint; no source check may import or invoke this module."""
    checkout = Path(__file__).resolve().parents[3]
    if len(expected_head) != 40 or any(c not in "0123456789abcdef" for c in expected_head):
        raise ValueError("exact independently reviewed source head required")
    command = ["git", "-c", "safe.directory=" + checkout.as_posix(), "-C", str(checkout)]
    head = subprocess.check_output([*command, "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        [*command, "status", "--porcelain", "--untracked-files=normal"], text=True
    ).strip()
    if head != expected_head or dirty:
        raise ValueError("reviewed source identity/clean checkout differs before collection")
    retained = Retention(root)
    try:
        adapter = SceneAdapter(retained, nonces)
        commands: dict[str, tuple[tuple[Command, ...], Command]] = {
            m: ((action(a1 - a0),), action(announced))
            for m, (_, a0, a1, announced, _) in TABLE.items()
        }
        controller = Qualification(
            adapter, retained, {**source_plan(nonces), "reviewed_head": head}, commands
        )
        adapter.target_gate = controller.verify_seal
        result = controller.run()
        retained.stage = "RELATIONS"
        adapter.relations()
        retained.write(
            "qualification.json",
            encode(
                {
                    "version": VERSION,
                    "status": "QUALIFIED",
                    "producer_calls": retained.calls,
                    "phase_gate_effect": "NONE",
                }
            ),
        )
        retained.write(
            "runtime-metadata.json",
            encode(
                {
                    "elapsed_seconds": time.monotonic() - retained.start,
                    "retained_bytes_before_metadata": retained.used,
                    "producer_calls": retained.calls,
                    "python": sys.version,
                    "numpy": version("numpy"),
                    "epsbench": version("epsbench"),
                }
            ),
        )
        return result
    except Exception as error:
        try:
            if not retained.failed:
                retained.fail(list(MEMBERS), error)
            if not (retained.root / "runtime-metadata.json").exists():
                retained.write(
                    "runtime-metadata.json",
                    encode(
                        {
                            "elapsed_seconds": time.monotonic() - retained.start,
                            "retained_bytes_before_metadata": retained.used,
                            "producer_calls": retained.calls,
                            "python": sys.version,
                            "numpy": version("numpy"),
                            "epsbench": version("epsbench"),
                        }
                    ),
                    True,
                )
        except Exception as receipt_error:
            raise error from receipt_error
        raise
