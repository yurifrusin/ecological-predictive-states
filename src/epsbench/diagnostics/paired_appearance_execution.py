"""Small fixed apparatus driver and durable component retention; no native work on import."""

from __future__ import annotations

import os
import re
import stat
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from epsbench.diagnostics import paired_appearance as p
from epsbench.diagnostics.appearance_study import PAIRED_V1, StudySpec, require_spec
from epsbench.diagnostics.paired_appearance_runtime import (
    AppearanceExecutionBinding,
    LaunchDecision,
    require_decision,
    require_native_binding,
)
from epsbench.utils.canonical import canonical_json_bytes, sha256_bytes

MIB = 1024**2
HOST_ACTUAL = 768 * 1024
DUMMY_TOTAL = 256 * 1024
DUMMY_CASES = ("normal", "interrupted", "deadline", "overbudget")
STAGES = (
    "endpoint_attempt",
    "rgb_attempt",
    "rgb_read_complete",
    "rgb_state_complete",
    "paired_draw_input",
    "paired_draw_attempt",
    "paired_draw_complete",
    "paired_draw_output",
    "paired_read_input",
    "paired_read_attempt",
    "paired_read_complete",
    "paired_read_output",
    "endpoint_complete",
)
METADATA = {
    "rgb_state_complete",
    "paired_draw_input",
    "paired_draw_output",
    "paired_read_input",
    "paired_read_output",
}
SPECS = {
    "rgb": ("uint8", (120, 160, 3)),
    "native_id": ("uint8", (120, 160, 3)),
    "native_depth": ("float32", (120, 160)),
    "raw": ("int32", (120, 160)),
    "depth": ("float32", (120, 160)),
    "opaque": ("int32", (120, 160)),
    "controlled": ("bool", (120, 160)),
    "horizontal": ("bool", (120, 159)),
    "vertical": ("bool", (119, 160)),
}


def safe_path(path: Path) -> None:
    """Cooperative owned namespace; reject links/reparse points and file aliases."""
    if not path.is_absolute():
        raise ValueError("absolute owned path required")
    for item in (path, *path.parents):
        if item.exists() or item.is_symlink():
            facts = item.lstat()
            if (
                stat.S_ISLNK(facts.st_mode)
                or getattr(facts, "st_file_attributes", 0) & 0x400
                or (stat.S_ISREG(facts.st_mode) and facts.st_nlink != 1)
            ):
                raise ValueError("linked/aliased path denied")


def flush_directory(path: Path) -> None:
    # Portable Windows directory-fsync is unavailable; no power-loss attestation.
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def exclusive_file(path: Path, payload: bytes) -> None:
    safe_path(path)
    with path.open("xb") as stream:
        if stream.write(payload) != len(payload):
            raise OSError("short component write")
        stream.flush()
        os.fsync(stream.fileno())
    flush_directory(path.parent)


def consume_attempt(
    root: Path, binding: AppearanceExecutionBinding, decision: LaunchDecision
) -> None:
    require_decision(binding, decision)
    safe_path(root)
    if root.name != binding.output_id:
        raise ValueError("owned output basename differs")
    root.mkdir()  # Exclusive namespace is itself the nonrefundable overlap/relaunch lock.
    flush_directory(root.parent)
    exclusive_file(
        root / "consumed.json",
        canonical_json_bytes(
            {
                "binding": binding.model_dump(mode="json"),
                "decision": decision.model_dump(mode="json"),
                "binding_root": binding.root,
            }
        ),
    )
    # No reset if either create or flush fails; the directory remains consumed.


def consumed(root: Path, binding: AppearanceExecutionBinding) -> None:
    safe_path(root)
    path = root / "consumed.json"
    safe_path(path)
    if path.stat().st_size > 16384:
        raise ValueError("attempt anchor bound")
    value = p.strict_json(path.read_bytes())
    if set(value) != {"binding", "decision", "binding_root"}:
        raise ValueError("closed attempt anchor")
    if value["binding"] != binding.model_dump(mode="json") or value["binding_root"] != binding.root:
        raise ValueError("consumed binding differs")
    require_decision(binding, LaunchDecision.model_validate(value["decision"]))


class ByteBoundedSink:
    """Only fixed component paths and small callback receipts, never an archive/journal service."""

    def __init__(self, root: Path, binding: AppearanceExecutionBinding):
        consumed(root, binding)
        self.study = binding.preparation.study
        self.root, self.binding = root, binding
        self.poisoned = False
        self.context = -1
        self.endpoint = 0
        self.stage = 0
        self.sequence = 0
        self.used = [0] * 16
        self.shared = (root / "consumed.json").stat().st_size
        self.counts = {
            k: 0
            for k in (
                "endpoint_attempt",
                "rgb_attempt",
                "rgb_read_complete",
                "paired_draw_attempt",
                "paired_draw_complete",
                "paired_read_attempt",
                "paired_read_complete",
            )
        }
        if set(v.name for v in root.iterdir()) != {"consumed.json"}:
            raise ValueError("existing sink files deny resume")
        for name in ("endpoints", "events"):
            (root / name).mkdir()
        self.put("binding.json", canonical_json_bytes(binding.model_dump(mode="json")))

    def put(self, name: str, data: bytes) -> dict[str, Any]:
        if self.poisoned:
            raise OSError("poisoned sink denies further writes")
        try:
            if type(data) is not bytes or (
                name not in {"binding.json", "terminal.json"}
                and re.fullmatch(r"events/[0-9]{4}\.json", name) is None
                and not (
                    re.fullmatch(r"endpoints/e[0-9]{2}/[a-z0-9_-]+\.(bin|json)", name)
                    and name.split("/")[-1]
                    in {
                        *(n + ".bin" for n in (*p.ARRAY_NAMES, "read_id", "read_depth")),
                        *(n + ".json" for n in (*METADATA, "frame")),
                    }
                )
            ):
                raise ValueError("fixed bounded component path")
            if name.startswith("endpoints/"):
                slot = int(name.split("/")[1][1:])
                if slot not in range(16) or self.used[slot] + len(data) > MIB:
                    raise ValueError("endpoint byte cap")
                self.used[slot] += len(data)
            else:
                # Reserve 16KiB for a terminal result before admitting ordinary writes.
                cap = 2 * MIB if name == "terminal.json" else 2 * MIB - 16384
                if self.shared + len(data) > cap:
                    raise ValueError("shared byte cap")
                self.shared += len(data)
            path = self.root / name
            safe_path(path)
            if not path.parent.exists():
                path.parent.mkdir()
                flush_directory(path.parent.parent)
            exclusive_file(path, data)
            return {"path": name, "bytes": len(data), "sha256": sha256_bytes(data)}
        except BaseException:
            self.poisoned = True
            raise

    def begin_context(self, ordinal: int) -> None:
        if (
            self.poisoned
            or type(ordinal) is not int
            or ordinal != self.context + 1
            or ordinal not in range(8)
            or (self.context >= 0 and self.endpoint != 2)
        ):
            raise ValueError("fixed context order; no resume")
        self.context, self.endpoint, self.stage = ordinal, 0, 0

    def progress(self, stage: str, index: int, value: Any) -> None:
        if self.poisoned:
            raise OSError("poisoned sink")
        try:
            if (
                self.context not in range(8)
                or type(index) is not int
                or index != self.endpoint
                or index not in (0, 1)
                or stage != STAGES[self.stage]
            ):
                raise ValueError("closed callback order")
            slot = self.context * 2 + index
            prefix = f"endpoints/e{slot:02d}/"
            refs: list[dict[str, Any]] = []
            if stage in self.counts:
                self.counts[stage] += 1  # Observed callback count, not inferred native success.
            if stage in METADATA:
                if type(value) is not bytes or len(value) > 65536:
                    raise ValueError("metadata envelope cap")
                if type(p.strict_json(value)) is not dict:
                    raise ValueError("producer metadata mapping required")
                refs.append(self.put(prefix + stage + ".json", value))
            elif stage == "rgb_read_complete":
                if (
                    type(value) is not np.ndarray
                    or value.dtype != np.uint8
                    or value.shape != (120, 160, 3)
                ):
                    raise ValueError("completed RGB shape/dtype")
                refs.append(self.put(prefix + "rgb.bin", value.tobytes()))
            elif stage == "paired_read_complete":
                if (
                    type(value) is not tuple
                    or len(value) != 2
                    or any(type(v) is not bytes for v in value)
                    or tuple(map(len, value)) != (57600, 76800)
                ):
                    raise ValueError("completed paired raw lengths")
                for name, data in zip(("read_id", "read_depth"), value, strict=True):
                    refs.append(self.put(prefix + name + ".bin", data))
            elif stage == "endpoint_complete":
                if type(value) is not p.Frame:
                    raise ValueError("completed frame type")
                family, appearance, _ = p.contexts(study=self.study)[self.context]
                if (value.family, value.appearance, value.index) != (family, appearance, index):
                    raise ValueError("frame membership")
                p.validate_frame(value, study=self.study)
                if (value.evidence["source_head"], value.evidence["source_tree"]) != (
                    self.binding.preparation.source_head,
                    self.binding.preparation.source_tree,
                ):
                    raise ValueError("frame source binding")
                arrays = {}
                for name in p.ARRAY_NAMES:
                    data = getattr(value, name).tobytes()
                    arrays[name] = (
                        {
                            "path": prefix + "rgb.bin",
                            "bytes": len(data),
                            "sha256": sha256_bytes(data),
                        }
                        if name == "rgb"
                        else self.put(prefix + name + ".bin", data)
                    )
                meta = canonical_json_bytes(
                    {
                        "family": family,
                        "appearance": appearance,
                        "index": index,
                        "evidence": value.evidence,
                        "arrays": arrays,
                    }
                )
                if len(meta) > 65536:
                    raise ValueError("final evidence envelope cap")
                refs.append(self.put(prefix + "frame.json", meta))
            elif value is not None:
                raise ValueError("attempt/complete callbacks have no payload")
            event = canonical_json_bytes(
                {
                    "sequence": self.sequence,
                    "binding_root": self.binding.root,
                    "context": self.context,
                    "index": index,
                    "stage": stage,
                    "refs": refs,
                }
            )
            if len(event) > 4096:
                raise ValueError("callback receipt cap")
            self.put(f"events/{self.sequence:04d}.json", event)
            self.sequence += 1
            self.stage += 1
            if stage == "endpoint_complete":
                self.endpoint += 1
                self.stage = 0
        except BaseException:
            self.poisoned = True
            raise

    def terminal(self, result: dict[str, Any]) -> None:
        data = canonical_json_bytes(result)
        if len(data) > 16384:
            raise ValueError("terminal reserve")
        self.put("terminal.json", data)


def read_ref(root: Path, value: dict[str, Any], cap: int) -> bytes:
    if (
        set(value) != {"path", "bytes", "sha256"}
        or type(value["bytes"]) is not int
        or not 0 <= value["bytes"] <= cap
        or type(value["path"]) is not str
        or re.fullmatch(r"endpoints/e[0-9]{2}/[a-z0-9_-]+\.(bin|json)", value["path"]) is None
    ):
        raise ValueError("closed component reference")
    path = root / value["path"]
    safe_path(path)
    if not path.is_file() or path.stat().st_size != value["bytes"]:
        raise ValueError("component missing/length differs")
    data = path.read_bytes()
    if sha256_bytes(data) != value["sha256"]:
        raise ValueError("component hash differs")
    return data


def replay(
    root: Path, binding: AppearanceExecutionBinding, *, native: bool = False
) -> dict[str, Any]:
    """Validate complete prefix. Incomplete suffix is reported; never resumes capture."""
    study = binding.preparation.study
    frames: dict[tuple[int, int], p.Frame] = {}
    counts = {
        k: 0
        for k in (
            "endpoint_attempt",
            "rgb_attempt",
            "rgb_read_complete",
            "paired_draw_attempt",
            "paired_draw_complete",
            "paired_read_attempt",
            "paired_read_complete",
        )
    }
    result: dict[str, Any] = {"status": "INCONCLUSIVE", "frames": frames, "counts": counts}
    try:
        consumed(root, binding)
        binding_file = root / "binding.json"
        safe_path(binding_file)
        if binding_file.stat().st_size > 16384 or p.strict_json(
            binding_file.read_bytes()
        ) != binding.model_dump(mode="json"):
            raise ValueError("retained binding differs")
        total = 0
        for path in root.rglob("*"):
            safe_path(path)
            if path.is_file():
                size = path.stat().st_size
                total += size
                if path.relative_to(root).parts[0] == "endpoints":
                    if size > MIB:
                        raise ValueError("oversized endpoint component")
                elif size > 2 * MIB:
                    raise ValueError("oversized shared component")
        if total > 18 * MIB:
            raise ValueError("original tree cap")
        for slot in range(16):
            directory = root / f"endpoints/e{slot:02d}"
            if directory.exists() and sum(v.stat().st_size for v in directory.iterdir()) > MIB:
                raise ValueError("endpoint total cap")
        if (
            sum(
                v.stat().st_size
                for v in root.rglob("*")
                if v.is_file() and "endpoints" not in v.relative_to(root).parts
            )
            > 2 * MIB
        ):
            raise ValueError("shared total cap")
        paths = sorted((root / "events").iterdir())
        payloads: dict[str, bytes] = {}
        for sequence, path in enumerate(paths):
            if path.name != f"{sequence:04d}.json" or path.stat().st_size > 4096:
                raise ValueError("event gap/name/size")
            event = p.strict_json(path.read_bytes())
            context, index, stage = (
                sequence // (2 * len(STAGES)),
                (sequence // len(STAGES)) % 2,
                STAGES[sequence % len(STAGES)],
            )
            if (
                set(event) != {"sequence", "binding_root", "context", "index", "stage", "refs"}
                or type(event["sequence"]) is not int
                or type(event["context"]) is not int
                or type(event["index"]) is not int
                or event["sequence"] != sequence
                or event["binding_root"] != binding.root
                or event["context"] != context
                or event["index"] != index
                or event["stage"] != stage
                or context >= 8
                or type(event["refs"]) is not list
                or any(type(v) is not dict for v in event["refs"])
            ):
                raise ValueError("callback membership/order/binding")
            prefix = f"endpoints/e{context * 2 + index:02d}/"
            expected = (
                [prefix + stage + ".json"]
                if stage in METADATA
                else [prefix + "rgb.bin"]
                if stage == "rgb_read_complete"
                else [prefix + "read_id.bin", prefix + "read_depth.bin"]
                if stage == "paired_read_complete"
                else [prefix + "frame.json"]
                if stage == "endpoint_complete"
                else []
            )
            if [v.get("path") for v in event["refs"]] != expected:
                raise ValueError("event component inventory")
            if stage in counts:
                counts[stage] += 1
            for ref in event["refs"]:
                payloads[ref["path"]] = read_ref(
                    root, ref, 65536 if ref["path"].endswith(".json") else 76800
                )
            if stage == "endpoint_complete":
                meta = p.strict_json(payloads[prefix + "frame.json"])
                family, appearance, _ = p.contexts(study=study)[context]
                if (
                    set(meta) != {"family", "appearance", "index", "evidence", "arrays"}
                    or type(meta["index"]) is not int
                    or (meta["family"], meta["appearance"], meta["index"])
                    != (family, appearance, index)
                    or set(meta["arrays"]) != set(SPECS)
                ):
                    raise ValueError("frame closed membership/arrays")
                arrays = {}
                for name, (dtype, shape) in SPECS.items():
                    if meta["arrays"][name]["path"] != prefix + name + ".bin":
                        raise ValueError("array component path")
                    raw = read_ref(root, meta["arrays"][name], 76800)
                    arrays[name] = np.frombuffer(raw, dtype=dtype).reshape(shape)
                frame = p.Frame(
                    family, appearance, index, **arrays, evidence=meta["evidence"], study=study
                )
                if (frame.evidence["source_head"], frame.evidence["source_tree"]) != (
                    binding.preparation.source_head,
                    binding.preparation.source_tree,
                ):
                    raise ValueError("retained source differs")
                if frame.rgb.tobytes() != payloads[prefix + "rgb.bin"]:
                    raise ValueError("RGB callback differs")
                for name, read_name in (("native_id", "read_id"), ("native_depth", "read_depth")):
                    if (
                        np.flipud(getattr(frame, name)).tobytes()
                        != payloads[prefix + read_name + ".bin"]
                    ):
                        raise ValueError("completed raw read differs from final orientation")
                rgb = p.strict_json(payloads[prefix + "rgb_state_complete.json"])
                if rgb != {
                    "stable": frame.evidence["rgb_stable"],
                    "material": frame.evidence["rgb_material"],
                }:
                    raise ValueError("RGB saved callback differs")
                snapshots = [
                    p.strict_json(payloads[prefix + k + ".json"])
                    for k in (
                        "paired_draw_input",
                        "paired_draw_output",
                        "paired_read_input",
                        "paired_read_output",
                    )
                ]
                excluded = {
                    "projection_matrix_float32",
                    "modelview_matrix_float32",
                    "clip_origin",
                    "clip_depth_mode",
                }
                if (
                    {k: v for k, v in snapshots[0].items() if k not in excluded}
                    != {k: v for k, v in snapshots[1].items() if k not in excluded}
                    or snapshots[1] != snapshots[2]
                    or snapshots[2] != snapshots[3]
                ):
                    raise ValueError("saved producer snapshot drift")
                stable = {k: snapshots[1][k] for k in frame.evidence["paired_stable"]}
                # Omit only existing volatile framebuffer/object handles.
                attachments = p.strict_json(canonical_json_bytes(stable["offscreen_attachments"]))
                attachments["offFBO"].pop("framebuffer", None)
                for role in ("color0", "depth"):
                    attachments["offFBO"][role].pop("object_name", None)
                stable["offscreen_attachments"] = attachments
                if stable != frame.evidence["paired_stable"]:
                    raise ValueError("retained paired state differs from callback")
                p.validate_frame(frame, study=study)
                if native:
                    from epsbench.diagnostics.paired_appearance_native import (
                        validate_retained_native,
                    )

                    validate_retained_native(frame)
                frames[(context, index)] = frame
        if len(frames) == 16:
            expected_files = {"consumed.json", "binding.json", "terminal.json"}
            expected_files |= {f"events/{i:04d}.json" for i in range(16 * len(STAGES))}
            for slot in range(16):
                prefix = f"endpoints/e{slot:02d}/"
                expected_files |= {
                    prefix + n + ".bin" for n in (*p.ARRAY_NAMES, "read_id", "read_depth")
                }
                expected_files |= {prefix + n + ".json" for n in (*METADATA, "frame")}
            actual_files = {v.relative_to(root).as_posix() for v in root.rglob("*") if v.is_file()}
            if actual_files - expected_files:
                raise ValueError("unsupported complete-tree components")
        result["status"] = (
            "COMPLETE" if len(frames) == 16 and len(paths) == 16 * len(STAGES) else "PREFIX"
        )
        result["events"] = len(paths)
    except (ValueError, TypeError, KeyError, OSError, IndexError) as error:
        result["reason"] = str(error)
    return result


class Capture(Protocol):
    def capture(self, index: int) -> p.Frame: ...
    def close(self) -> None: ...


Factory = Callable[[int, Callable[[str, int, Any], None]], Capture]


def ready(
    frames: dict[tuple[int, int], p.Frame], context: int, index: int, study: StudySpec = PAIRED_V1
) -> dict[str, Any]:
    require_spec(study)
    family, appearance, repeat = p.contexts(study=study)[context]
    frame = frames[(context, index)]
    if (frame.family, frame.appearance, frame.index) != (family, appearance, index):
        raise ValueError("ready frame membership differs")
    p.validate_frame(frame, study)
    if appearance == study.appearances[1]:
        # Existing textured-only per-surface criteria become decidable at this endpoint.
        # comparison uses the fixed matched solid; it also checks every existing texture condition.
        solid = p.contexts(study=study).index((family, study.appearances[0], repeat))
        comparison = p.comparison(frames[(solid, index)], frame, study=study)
        if comparison["status"] == "FAIL":
            return comparison
    if repeat == 1:
        original = p.contexts(study=study).index((family, appearance, 0))
        if not p.repeat_equal(frames[(original, index)], frame, study=study):
            return {"status": "FAIL", "reason": "exact regeneration mismatch"}
    return {"status": "PASS"}


def run_matrix(
    sink: ByteBoundedSink, factory: Factory, deadline: float, *, native_replay: bool = False
) -> dict[str, Any]:
    """Pure control seam: source tests inject arbitrary fixtures; native entry binds its factory."""
    study = sink.study
    frames: dict[tuple[int, int], p.Frame] = {}
    reason: dict[str, Any] = {"status": "INCONCLUSIVE", "reason": "incomplete fixed matrix"}
    operational_error: str | None = None
    try:
        for context, _ in enumerate(p.contexts(study=study)):
            if time.monotonic() >= deadline:
                raise TimeoutError("work watchdog exhausted")
            sink.begin_context(context)
            capture: Capture | None = None
            try:
                capture = factory(context, sink.progress)
                for index in range(2):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("work watchdog exhausted")
                    capture.capture(index)
                    retained = replay(sink.root, sink.binding, native=native_replay)
                    if (
                        retained["status"] == "INCONCLUSIVE"
                        or (context, index) not in retained["frames"]
                    ):
                        raise ValueError(retained.get("reason", "retained endpoint unavailable"))
                    frames = retained["frames"]
                    check = ready(frames, context, index, study)
                    if check["status"] == "FAIL":
                        reason = check
                        break
                else:
                    continue
                break
            finally:
                if capture is not None:
                    capture.close()
        else:
            complete = {
                key: (frames[(ordinal, 0)], frames[(ordinal, 1)])
                for ordinal, key in enumerate(p.contexts(study=study))
            }
            reason = p.assess(complete, study=study)
            if reason["status"] == "PASS":
                reason = {**reason, "status": "CAPTURE_COMPLETE"}
        if time.monotonic() >= deadline:
            raise TimeoutError("work watchdog exhausted before result")
    except Exception as error:
        operational_error = str(error)[:1024]
        if reason["status"] != "FAIL":
            reason = {"status": "INCONCLUSIVE", "reason": operational_error}
    result = {
        **reason,
        "binding_root": sink.binding.root,
        "phase_gate_effect": "NONE",
        "completed_endpoints": len(frames),
        "unattempted_slots": 16 - sink.counts["endpoint_attempt"],
        "counts": sink.counts.copy(),
        "cleanup_pending": True,
        "apparatus_status": "FAIL" if reason["status"] == "FAIL" else "INCONCLUSIVE",
        "scientific_result": reason.copy(),
        "operational_status": "INCONCLUSIVE" if operational_error else "COMPLETE",
    }
    if operational_error:
        result.update(status="INCONCLUSIVE", operational_error=operational_error)
    try:
        sink.terminal(result)
    except Exception as error:
        result = {
            **result,
            "status": "INCONCLUSIVE",
            "operational_status": "INCONCLUSIVE",
            "retention_error": str(error)[:1024],
        }
    return result


def run_native(
    repository: Path,
    root: Path,
    binding: AppearanceExecutionBinding,
    decision: LaunchDecision,
    deadline: float,
) -> dict[str, Any]:
    require_decision(binding, decision)
    require_native_binding(repository, binding)
    study = binding.preparation.study
    if binding.purpose != study.native_purpose:
        raise PermissionError("dummy cannot become a native matrix")
    from epsbench.diagnostics.paired_appearance_native import NativeCapture

    def factory(context: int, progress: Callable[[str, int, Any], None]) -> Capture:
        family, appearance, _ = p.contexts(study=study)[context]
        return NativeCapture(
            repository,
            family,
            appearance,
            source_head=binding.preparation.source_head,
            source_tree=binding.preparation.source_tree,
            progress=progress,
            binding=binding,
        )

    return run_matrix(ByteBoundedSink(root, binding), factory, deadline, native_replay=True)


def run_dummy(root: Path, case: str, deadline: float) -> None:
    """Four held tiny qualification bodies; no SDK/full-shape matrix or source-test invocation."""
    if case not in DUMMY_CASES:
        raise ValueError("one of exactly four dummy cases required")
    # Each joint-budget slot is exclusive, including partial files/terminal receipts.
    path = root / ("dummy-" + case)
    path.mkdir()
    used = 0
    poisoned = False

    def put(name: str, data: bytes) -> None:
        nonlocal used, poisoned
        if poisoned:
            raise ValueError("tiny dummy budget poisoned")
        try:
            if used + len(data) > 32 * 1024:
                raise ValueError("tiny dummy budget poisoned")
            used += len(data)
            exclusive_file(path / name, data)
        except BaseException:
            poisoned = True
            raise

    put("rgb.bin", b"tiny-completed-rgb")
    put("pair.bin", b"tiny-completed-id-depth")
    if case == "interrupted":
        raise RuntimeError("intentional interruption after completed paired read")
    if case == "deadline":
        while time.monotonic() < deadline:
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))
        raise TimeoutError("intentional dummy watchdog")
    if case == "overbudget":
        put("overflow.bin", b"x" * (32 * 1024 - used + 1))
        raise AssertionError("overflow was not denied")
    if (path / "rgb.bin").read_bytes() != b"tiny-completed-rgb" or (
        path / "pair.bin"
    ).read_bytes() != b"tiny-completed-id-depth":
        raise ValueError("tiny callback replay")
    put("complete.json", b'{"native":false,"status":"COMPLETE"}')
