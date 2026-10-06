"""One-attempt local analytic lifecycle. Launch authorization is separate from source review."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from epsbench.diagnostics.return_view_analytic import (
    ANNOUNCED,
    CONFIG_BYTES,
    CONFIG_SHA256,
    EXECUTED,
    FIXED,
    Frame,
    Producer,
)
from epsbench.diagnostics.return_view_core import (
    REQUIRED,
    HistoryView,
    collision_bijection,
    persistence,
    return_cache,
    score,
    tokens,
)
from epsbench.schema import ModalityPermissionSet
from epsbench.utils.canonical import canonical_json_bytes

# Conservative uncompressed allocation, including file headers and incomplete writes.
# 8 * (labels int64 + worldpoints 3*float64 + segmentation int32) <= 5.53 MiB;
# 8 * (all internal edges * 5 int32) <= 5.86 MiB;
# 4 all-token forecasts <= 0.23 MiB binary; controls reserve <= 1 MiB.
# All artifacts stay below 13 MiB; 16 MiB admitted total is an application byte bound.
LIMIT = 16 * 1024 * 1024
FRAME_LIMIT = 1500 * 1024
FORECAST_LIMIT = 128 * 1024
CONTROL_LIMIT = 64 * 1024
PATHS = {
    **{f"frame-{m}-{i}.npz": FRAME_LIMIT for m in range(2) for i in range(4)},
    **{f"forecast-{m}-{r}.npz": FORECAST_LIMIT for m in range(2) for r in range(2)},
    **{
        p: CONTROL_LIMIT
        for p in (
            "attempt.json",
            "config.json",
            "authorization.json",
            "seal.json",
            "exposure.json",
            "result.json",
            "inspection.json",
            "failure.json",
        )
    },
}
assert sum(PATHS.values()) <= LIMIT


class Store:
    def __init__(self, root: Path) -> None:
        if any(p.is_symlink() for p in (root, *root.parents)):
            raise ValueError("linked namespace denied")
        root.mkdir(parents=False, exist_ok=False)
        self.root = root
        self.used = 0
        self.hashes: dict[str, str] = {}
        self.write("attempt.json", canonical_json_bytes({"schema": "return-view-attempt-v1"}))

    def write(self, name: str, data: bytes) -> None:
        if name not in PATHS or name in self.hashes or len(data) > PATHS[name]:
            raise ValueError("path, version, count or per-file allocation denied")
        # Reserve before opening; partial failures never release the reservation or namespace.
        self.used += len(data)
        if self.used > LIMIT:
            raise ValueError("16 MiB admitted total exceeded")
        with (self.root / name).open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        self.hashes[name] = hashlib.sha256(data).hexdigest()


def pack(**arrays: Any) -> bytes:
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    return stream.getvalue()


def retain_frame(store: Store, member: int, index: int, frame: Frame) -> None:
    o = frame.observation
    if (
        o.raster.sequence_index != index
        or frame.raw_labels.dtype != np.int64
        or frame.world_points.dtype != np.float64
        or frame.raw_labels.shape != o.raster.segmentation.shape
        or frame.world_points.shape != (*frame.raw_labels.shape, 3)
        or frame.raw_labels.shape[0] > 120
        or frame.raw_labels.shape[1] > 160
        or not np.all(np.isfinite(frame.world_points))
    ):
        raise ValueError("bounded typed frame chronology and provenance required")
    # Privileged evaluation bundle; never returned by a rule view. No pickle/object arrays.
    labels = dict(o.raster.identities)
    inverse: dict[str | None, int] = {t: label for label, t in labels.items()}
    edges = np.array(
        [
            [
                0 if e.axis.value == "horizontal" else 1,
                e.row,
                e.column,
                inverse.get(e.negative_surface_id, 0),
                inverse.get(e.positive_surface_id, 0),
            ]
            for e in o.boundary.edges
        ],
        dtype=np.int32,
    ).reshape(-1, 5)
    store.write(
        f"frame-{member}-{index}.npz",
        pack(
            raw_labels=frame.raw_labels,
            world_points=frame.world_points,
            segmentation=o.raster.segmentation,
            edges=edges,
            identities=np.array(o.raster.identities, dtype="U32"),
            schema=np.array("return-view-frame-v1"),
        ),
    )


def attempt(
    root: Path,
    frame: Callable[[int, int], Frame],
    authorization: bytes,
    config: bytes,
    executed: tuple[Fraction, Fraction],
    announced: Fraction,
) -> dict[str, Any]:
    """Lifecycle seam for distinct tiny tests; public launch uses fixed config."""
    store = Store(root)
    try:
        store.write("authorization.json", authorization)
        store.write("config.json", config)
        past = []
        views = []
        forecasts = []
        for member in range(2):
            history = []
            for index in range(3):
                item = frame(member, index)
                retain_frame(store, member, index, item)
                history.append(item.observation)
            past.append(history)
            full = HistoryView(
                tuple(history), executed, announced, ModalityPermissionSet(allowed=REQUIRED), False
            )
            recent = HistoryView(
                tuple(history[1:]),
                executed,
                announced,
                ModalityPermissionSet(allowed=REQUIRED),
                True,
            )
            views.append((full, recent))
            rules = (persistence(recent), return_cache(full))
            forecasts.append(rules)
            for rule, forecast in enumerate(rules):
                store.write(
                    f"forecast-{member}-{rule}.npz",
                    pack(
                        schema=np.array("return-view-forecast-v1"),
                        rule=np.array(forecast.rule),
                        tokens=np.array([t for t, _ in forecast.masks], dtype="U32"),
                        known=np.array([a is not None for _, a in forecast.masks]),
                        masks=np.array(
                            [
                                np.zeros(history[0].raster.segmentation.shape, dtype=bool)
                                if a is None
                                else a
                                for _, a in forecast.masks
                            ]
                        ),
                    ),
                )
        # The seal commits every member's input, provenance and both forecasts in one fsync.
        store.write(
            "seal.json",
            canonical_json_bytes(
                {"domain": "return-view-seal-v1", "members": [0, 1], "hashes": dict(store.hashes)}
            ),
        )
        # Whole membership is durably exposed before either target generation or read.
        store.write(
            "exposure.json",
            canonical_json_bytes(
                {
                    "domain": "return-view-exposure-v1",
                    "members": [0, 1],
                    "target": 3,
                    "seal": store.hashes["seal.json"],
                }
            ),
        )
        targets = []
        for member in range(2):
            target = frame(member, 3)
            retain_frame(store, member, 3, target)
            targets.append(target)
        mapping = collision_bijection(views[0][1], views[1][1])
        target_tokens = []
        for target in targets:
            associated = target.observation.raster.segmentation[target.raw_labels == 3]
            target_tokens.append(
                dict(target.observation.raster.identities)[int(associated[0])]
                if len(associated)
                else None
            )
        relations = {
            "recent_equal": mapping is not None,
            "bijection_covers_complete_histories": mapping is not None
            and set(tokens(views[1][0])) <= set(mapping)
            and set(tokens(views[0][0])) <= set(mapping.values()),
            "complete_histories_distinct": mapping is not None
            and views[0][0].canonical_bytes() != views[1][0].canonical_bytes(mapping),
            "independent_static_return": all(
                np.array_equal(
                    targets[m].raw_labels,
                    # Earlier retained raw labels are independently read, not copied into target.
                    np.load(root / f"frame-{m}-0.npz", allow_pickle=False)["raw_labels"],
                )
                for m in range(2)
            ),
            "target_visible_in_both_recent_frames": all(
                np.any(np.load(root / f"frame-{m}-{i}.npz", allow_pickle=False)["raw_labels"] == 3)
                for m in range(2)
                for i in (1, 2)
            ),
            "target_correspondence_same_bijection": mapping is not None
            and target_tokens[0] is not None
            and target_tokens[1] is not None
            and mapping.get(target_tokens[1]) == target_tokens[0],
            "target_nonempty": all(np.any(t.raw_labels == 3) for t in targets),
            "target_different": not np.array_equal(
                targets[0].raw_labels == 3, targets[1].raw_labels == 3
            ),
            "cache_exact": all(
                score(forecasts[m][1], targets[m].observation)["exact"] for m in range(2)
            ),
        }
        result = {
            "schema": "return-view-result-v1",
            "status": "ADMITTED_WITNESS" if all(relations.values()) else "INCONCLUSIVE",
            "relations": relations,
            "scores": [[score(f, targets[m].observation) for f in forecasts[m]] for m in range(2)],
            "pose_count": 8,
            "native_rgb_flow": "UNPROVEN",
            "phase_gate_effect": "NONE",
        }
        store.write("result.json", canonical_json_bytes(result))
        store.write(
            "inspection.json",
            canonical_json_bytes(
                {
                    "hashes": dict(store.hashes),
                    "bytes_before_inspection": store.used,
                    "limit": LIMIT,
                }
            ),
        )
        validate_and_inspect(root)
        return result
    except Exception as exc:
        store.write(
            "failure.json",
            canonical_json_bytes(
                {"status": "INCONCLUSIVE", "exception_type": type(exc).__name__, "no_retry": True}
            ),
        )
        raise


def launch(root: Path, decision_path: Path) -> dict[str, Any]:
    raw = decision_path.read_bytes()
    if len(raw) > CONTROL_LIMIT:
        raise ValueError("launch decision too large")
    decision = json.loads(raw)
    source_root = Path(__file__).resolve().parents[3]
    subprocess.check_output(
        [
            "git",
            "ls-files",
            "--error-unmatch",
            str(Path(__file__).resolve().relative_to(source_root)),
        ],
        cwd=source_root,
        text=True,
    )
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_root, text=True).strip()
    clean = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=source_root, text=True
    )
    expected = {
        "schema": "return-view-launch-decision-v1",
        "source_head": head,
        "config_sha256": CONFIG_SHA256,
        "authorization_path": str(decision_path.resolve()),
        "output_path": str(root.resolve()),
        "authorize_one_analytic_attempt": True,
        "independent_exact_head_reviews_complete": True,
    }
    if decision != expected or clean:
        raise PermissionError(
            "separate exact-source/config/path launch decision and clean source required"
        )
    producer = Producer(FIXED)
    return attempt(root, producer.frame, raw, CONFIG_BYTES, EXECUTED, ANNOUNCED)


def validate_and_inspect(root: Path) -> dict[str, int]:
    """Bounded retained-file integrity inspection; never generates or reevaluates poses."""
    sizes = {}
    hashes = {}
    for path in root.iterdir():
        if path.name not in PATHS or path.is_symlink() or not path.is_file():
            raise ValueError("unadmitted retained path")
        size = path.stat().st_size
        if size > PATHS[path.name]:
            raise ValueError("retained allocation exceeded")
        sizes[path.name] = size
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if sum(sizes.values()) > LIMIT:
        raise ValueError("retained total exceeds capacity")
    inspection = json.loads((root / "inspection.json").read_bytes())
    before = {k: v for k, v in hashes.items() if k != "inspection.json"}
    if inspection["hashes"] != before:
        raise ValueError("retained files differ from final inspection binding")
    seal = json.loads((root / "seal.json").read_bytes())
    exposure = json.loads((root / "exposure.json").read_bytes())
    expected_preseal = {
        k: v
        for k, v in before.items()
        if k in {"attempt.json", "config.json", "authorization.json"}
        or k.startswith("forecast-")
        or (k.startswith("frame-") and not k.endswith("-3.npz"))
    }
    if (
        seal["hashes"] != expected_preseal
        or seal["members"] != [0, 1]
        or exposure
        != {
            "domain": "return-view-exposure-v1",
            "members": [0, 1],
            "target": 3,
            "seal": hashes["seal.json"],
        }
    ):
        raise ValueError("whole-membership seal/exposure binding differs")
    return {"file_count": len(sizes), "total_bytes": sum(sizes.values()), "limit": LIMIT}
