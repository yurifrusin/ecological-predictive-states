"""Immutable ledger and artifacts for the bounded capture-contract revision study."""

from __future__ import annotations

import hashlib
import io
import json
import ntpath
import os
import platform
import posixpath
import stat
import subprocess
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

SCHEMA = "revision_capture_cell/v1"
LEDGER_SCHEMA = "revision_capture_ledger/v1"
ARCHIVE_SHA256 = "072edf97d7cd44a7a95c6b189e358bc34ddfd50be4890167dacfd12ccf0d59a2"
MANIFEST_SHA256 = "ba8018ef1a50d5ce3997d86aac36e7509096053866745fd3b73006b7752b305a"
ANCHOR_MEMBERS = {
    "capture/wgl-offsamples-4/before_raw_geom_ids.npy": "b5974ac705cd42a42e25abbbe8ed57cd3fec4218eb72b4d623c64fe81459b4d8",  # noqa: E501
    "capture/wgl-offsamples-4/after_raw_geom_ids.npy": "8308f8028b9b0b32ad294baf1ff4bcb6315cbd77ed5bbddc27bb94b35a58af9a",  # noqa: E501
    "capture/osmesa-offsamples-4/before_raw_geom_ids.npy": "6bcb47a40bd671cb8ace39753d2243d75727605dfafdf5d77172663bee956039",  # noqa: E501
    "capture/osmesa-offsamples-4/after_raw_geom_ids.npy": "859ce504f08fc4bcb0ec493dedc13d81620c682201943866dee13a406caa4415",  # noqa: E501
}
FAMILIES = ("corridor", "single_occluder")
BACKENDS = ("wgl", "osmesa")
POLICIES = ("joint4", "hybrid", "joint0")
POSES = ("before", "after")
STUDY_HOST = "DESKTOP-TPUQMNG"


class RevisionCaptureFailure(RuntimeError):
    """An integrity or one-attempt failure that permanently stops the study."""


@dataclass(frozen=True)
class StudyCell:
    ordinal: int
    family: str
    episode_index: int
    backend: str
    policy: str

    @property
    def name(self) -> str:
        return (
            f"{self.ordinal:02d}-{self.family}-episode-{self.episode_index:06d}-"
            f"{self.backend}-{self.policy}"
        )


def fixed_cells() -> tuple[StudyCell, ...]:
    cells: list[StudyCell] = []
    ordinal = 0
    for family in FAMILIES:
        for episode_index in range(4):
            for policy, backend in (
                ("joint4", "wgl"),
                ("joint4", "osmesa"),
                ("hybrid", "wgl"),
                ("hybrid", "osmesa"),
                ("joint0", "wgl"),
                ("joint0", "osmesa"),
            ):
                cells.append(StudyCell(ordinal, family, episode_index, backend, policy))
                ordinal += 1
    return tuple(cells)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _safe_member(name: str) -> None:
    path = Path(name.replace("\\", "/"))
    if not name or "\\" in name or path.is_absolute() or ".." in path.parts:
        raise RevisionCaptureFailure(f"unsafe archive member path: {name!r}")
    if any(part in ("", ".") for part in path.parts):
        raise RevisionCaptureFailure(f"non-canonical archive member path: {name!r}")


def verify_input_archive(path: Path) -> dict[str, object]:
    if not path.is_file() or sha256_file(path) != ARCHIVE_SHA256:
        raise RevisionCaptureFailure("fixed input archive SHA-256 mismatch")
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [item.filename for item in infos]
            if len(names) != len(set(names)):
                raise RevisionCaptureFailure("duplicate archive member")
            for name in names:
                _safe_member(name)
            manifest_bytes = archive.read("evidence-manifest.json")
            if sha256_bytes(manifest_bytes) != MANIFEST_SHA256:
                raise RevisionCaptureFailure("fixed evidence manifest SHA-256 mismatch")
            manifest = json.loads(manifest_bytes)
            records = manifest.get("files")
            if not isinstance(records, list):
                raise RevisionCaptureFailure("evidence manifest files is not a list")
            declared: dict[str, str] = {}
            for item in records:
                if not isinstance(item, dict):
                    raise RevisionCaptureFailure("invalid evidence manifest member record")
                raw_name, raw_digest = item.get("path"), item.get("sha256")
                if not isinstance(raw_name, str) or not isinstance(raw_digest, str):
                    raise RevisionCaptureFailure("invalid manifest path/hash")
                name, digest = raw_name, raw_digest
                _safe_member(name)
                if name in declared:
                    raise RevisionCaptureFailure("duplicate manifest path")
                payload = archive.read(name)
                if len(payload) != item.get("bytes") or sha256_bytes(payload) != digest:
                    raise RevisionCaptureFailure(f"manifest member mismatch: {name}")
                declared[name] = digest
            if set(names) != set(declared) | {"evidence-manifest.json"}:
                raise RevisionCaptureFailure("archive membership differs from manifest")
            for name, digest in ANCHOR_MEMBERS.items():
                if declared.get(name) != digest:
                    raise RevisionCaptureFailure(f"fixed anchor binding differs: {name}")
    except (OSError, zipfile.BadZipFile, KeyError, json.JSONDecodeError) as exc:
        raise RevisionCaptureFailure("cannot verify fixed input archive") from exc
    return {
        "archive_sha256": ARCHIVE_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "members": declared,
    }


def load_anchor_array(path: Path, backend: str, pose_name: str) -> npt.NDArray[np.int32]:
    """Load one fixed historical control only after validating the entire packet."""
    verify_input_archive(path)
    if backend not in BACKENDS or pose_name not in POSES:
        raise RevisionCaptureFailure("invalid fixed anchor selector")
    name = f"capture/{backend}-offsamples-4/{pose_name}_raw_geom_ids.npy"
    try:
        with zipfile.ZipFile(path) as archive:
            payload = archive.read(name)
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise RevisionCaptureFailure("cannot load fixed anchor member") from exc
    if sha256_bytes(payload) != ANCHOR_MEMBERS[name]:
        raise RevisionCaptureFailure("fixed anchor member changed after verification")
    value = np.load(io.BytesIO(payload), allow_pickle=False)
    if value.dtype != np.int32 or value.shape != (120, 160):
        raise RevisionCaptureFailure("fixed anchor array schema differs")
    return cast(npt.NDArray[np.int32], value)


def _reject_linked_path(path: Path) -> None:
    current = path.absolute()
    while True:
        if current.exists():
            info = current.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise RevisionCaptureFailure(f"symlink path component rejected: {current}")
            attrs = getattr(info, "st_file_attributes", 0)
            reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            if attrs & reparse:
                raise RevisionCaptureFailure(f"reparse path component rejected: {current}")
        if current.parent == current:
            return
        current = current.parent


def _directory_fsync(path: Path) -> bool:
    if os.name == "nt":
        return False
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return True


def publish_bytes(path: Path, payload: bytes) -> dict[str, object]:
    """Publish exactly once by fsynced temporary file and hard link."""
    _reject_linked_path(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    token = hashlib.sha256(os.urandom(32)).hexdigest()[:16]
    temporary = path.with_name(f".{path.name}.{token}.tmp")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    fd = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fsync = _directory_fsync(path.parent)
    except Exception:
        raise
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    if path.read_bytes() != payload:
        raise RevisionCaptureFailure(f"published artifact readback differs: {path}")
    return {
        "path": path.name,
        "sha256": sha256_bytes(payload),
        "bytes": len(payload),
        "directory_fsync": directory_fsync,
    }


def array_bytes(value: npt.NDArray[Any]) -> tuple[bytes, dict[str, object]]:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise RevisionCaptureFailure("object arrays are forbidden")
    dtype = array.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(array.astype(dtype, copy=False))
    if np.issubdtype(canonical.dtype, np.floating) and not np.isfinite(canonical).all():
        raise RevisionCaptureFailure("non-finite array")
    stream = io.BytesIO()
    np.save(stream, canonical, allow_pickle=False)
    metadata: dict[str, object] = {
        "dtype": canonical.dtype.str,
        "shape": list(canonical.shape),
        "byte_order": "little",
        "order": "C",
    }
    return stream.getvalue(), metadata


class ArtifactWriter:
    def __init__(self, root: Path):
        self.root = root

    def array(
        self,
        relative: str,
        value: npt.NDArray[Any],
        *,
        complete: bool = True,
        validated: bool = False,
        allow_nonfinite: bool = False,
    ) -> dict[str, object]:
        array = np.asarray(value)
        if allow_nonfinite:
            if array.dtype.hasobject or not np.issubdtype(array.dtype, np.number):
                raise RevisionCaptureFailure("retained malformed array must be non-object numeric")
            dtype = array.dtype.newbyteorder("<")
            canonical = np.ascontiguousarray(array.astype(dtype, copy=False))
            stream = io.BytesIO()
            np.save(stream, canonical, allow_pickle=False)
            payload = stream.getvalue()
            metadata: dict[str, object] = {
                "dtype": canonical.dtype.str,
                "shape": list(canonical.shape),
                "byte_order": "little",
                "order": "C",
            }
        else:
            payload, metadata = array_bytes(array)
        record = publish_bytes(self.root / relative, payload)
        return {**record, **metadata, "complete": complete, "validated": validated}

    def json(
        self, relative: str, value: object, *, complete: bool = True, validated: bool = False
    ) -> dict[str, object]:
        payload = canonical_json_bytes(value)
        record = publish_bytes(self.root / relative, payload)
        return {
            **record,
            "media_type": "application/json",
            "complete": complete,
            "validated": validated,
        }


def publish_analysis_result(root: Path, value: object) -> dict[str, object]:
    """Publish nested analysis arrays separately so JSON remains finite."""
    writer = ArtifactWriter(root)
    counter = 0

    def materialize(item: Any) -> Any:
        nonlocal counter
        if isinstance(item, np.ndarray):
            name = f"array-{counter:06d}.npy"
            counter += 1
            # Analytic no-hit maps intentionally contain NaN, unlike captured depth.
            array = np.asarray(item)
            if array.dtype.hasobject:
                raise RevisionCaptureFailure("analysis object arrays are forbidden")
            dtype = array.dtype.newbyteorder("<")
            canonical = np.ascontiguousarray(array.astype(dtype, copy=False))
            stream = io.BytesIO()
            np.save(stream, canonical, allow_pickle=False)
            record = publish_bytes(root / name, stream.getvalue())
            return {
                **record,
                "dtype": canonical.dtype.str,
                "shape": list(canonical.shape),
                "byte_order": "little",
                "order": "C",
                "complete": True,
                "validated": True,
            }
        if isinstance(item, Mapping):
            return {str(key): materialize(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [materialize(child) for child in item]
        if isinstance(item, np.generic):
            return item.item()
        return item

    report = materialize(value)
    return writer.json("capture-revision-analysis.json", report, validated=True)


class AttemptLock:
    """O_EXCL lock. Failure intentionally retains the file forever."""

    def __init__(self, root: Path):
        self.root = root
        self.path = root / "capture-attempt.lock"
        self.fd: int | None = None
        self.identity: tuple[int, int] | None = None

    def acquire(self) -> None:
        _reject_linked_path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.write(self.fd, canonical_json_bytes({"pid": os.getpid()}))
            os.fsync(self.fd)
            acquired = os.fstat(self.fd)
            self.identity = (acquired.st_dev, acquired.st_ino)
            _directory_fsync(self.root)
        except FileExistsError as exc:
            raise RevisionCaptureFailure("attempt lock already exists; study stopped") from exc

    def release_after_success(self, terminal_revision: Path) -> None:
        if self.fd is None or self.identity is None or not terminal_revision.is_file():
            raise RevisionCaptureFailure("cannot release lock before terminal revision")
        os.fsync(self.fd)
        descriptor = os.fstat(self.fd)
        descriptor_identity = (descriptor.st_dev, descriptor.st_ino)
        if descriptor_identity != self.identity:
            os.close(self.fd)
            self.fd = None
            raise RevisionCaptureFailure("attempt lock descriptor identity changed")
        # Native Windows does not permit unlinking this file while its os.open
        # descriptor remains open. Close first, then independently re-check the
        # pathname before unlinking; a mismatch retains whichever path now exists.
        os.close(self.fd)
        self.fd = None
        try:
            path_info = self.path.stat(follow_symlinks=False)
        except FileNotFoundError as exc:
            raise RevisionCaptureFailure("acquired attempt lock path disappeared") from exc
        path_identity = (path_info.st_dev, path_info.st_ino)
        if (
            path_identity != self.identity
            or stat.S_ISLNK(path_info.st_mode)
            or path_info.st_nlink != 1
        ):
            raise RevisionCaptureFailure("attempt lock ownership/file identity changed")
        self.path.unlink()
        self.identity = None
        _directory_fsync(self.root)


def detect_study_runtime(
    *, system: str | None = None, environ: Mapping[str, str] | None = None
) -> str:
    actual_system = platform.system() if system is None else system
    actual_environ = os.environ if environ is None else environ
    if actual_system == "Windows":
        return "windows"
    if actual_system == "Linux" and (
        "WSL_INTEROP" in actual_environ or "WSL_DISTRO_NAME" in actual_environ
    ):
        return "wsl"
    raise RevisionCaptureFailure("study handoff requires native Windows or WSL")


def _canonical_native_root(value: object, runtime: str) -> str:
    if not isinstance(value, str) or not value:
        raise RevisionCaptureFailure("handoff root is not a nonempty native path")
    if runtime == "windows":
        drive, tail = ntpath.splitdrive(value)
        if drive.upper() != "C:" or not tail.startswith("\\") or value.startswith("\\\\"):
            raise RevisionCaptureFailure("Windows handoff root must be a native C-drive path")
        if "/" in value or value.startswith("\\\\?\\") or ntpath.normpath(value) != value:
            raise RevisionCaptureFailure("Windows handoff root is aliased or non-canonical")
        return value
    if runtime == "wsl":
        if not value.startswith("/mnt/c/") or posixpath.normpath(value) != value:
            raise RevisionCaptureFailure("WSL handoff root must be canonical under /mnt/c")
        return value
    raise RevisionCaptureFailure("invalid handoff runtime")


def translate_study_root(
    value: str, source_runtime: str, *, executing_runtime: str | None = None
) -> str:
    execution = detect_study_runtime() if executing_runtime is None else executing_runtime
    if source_runtime not in ("windows", "wsl") or execution not in ("windows", "wsl"):
        raise RevisionCaptureFailure("invalid wslpath translation runtime")
    direction = "-u" if source_runtime == "windows" else "-w"
    command = (
        ["wsl.exe", "--exec", "wslpath", "-a", direction, value]
        if execution == "windows"
        else ["wslpath", "-a", direction, value]
    )
    try:
        translated = subprocess.check_output(
            command,
            text=True,
            encoding="utf-8",
            stderr=subprocess.PIPE,
            timeout=10,
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RevisionCaptureFailure("independent wslpath translation failed") from exc
    return _canonical_native_root(translated, "wsl" if source_runtime == "windows" else "windows")


def verify_handoff_records(
    root: Path,
    *,
    current_runtime: str | None = None,
    current_host: str | None = None,
    current_root: str | None = None,
    translator: Callable[[str, str], str] = translate_study_root,
) -> dict[str, object]:
    """Require the completed two-way Windows/WSL token exchange."""
    directory = root / "handoff"
    _reject_linked_path(root)
    _reject_linked_path(directory)
    expected = {
        "windows.json",
        "wsl.json",
        "verified-windows.json",
        "verified-wsl.json",
    }
    if not directory.is_dir() or {item.name for item in directory.iterdir()} != expected:
        raise RevisionCaptureFailure("complete two-way handoff records are absent")
    runtime_now = detect_study_runtime() if current_runtime is None else current_runtime
    host_now = platform.node() if current_host is None else current_host
    root_now = str(root.resolve()) if current_root is None else current_root
    if host_now.upper() != STUDY_HOST or runtime_now not in ("windows", "wsl"):
        raise RevisionCaptureFailure("handoff invocation host/runtime differs from study binding")
    records: dict[str, dict[str, Any]] = {}
    for runtime in ("windows", "wsl"):
        source_path = directory / f"{runtime}.json"
        verified_path = directory / f"verified-{runtime}.json"
        _reject_linked_path(source_path)
        _reject_linked_path(verified_path)
        source = json.loads(source_path.read_text(encoding="utf-8"))
        verified = json.loads(verified_path.read_text(encoding="utf-8"))
        if (
            source.get("schema") != "revision_capture_handoff/v1"
            or source.get("runtime") != runtime
            or str(source.get("host", "")).upper() != STUDY_HOST
            or verified.get("schema") != "revision_capture_handoff_verification/v1"
            or verified.get("runtime") != runtime
            or str(verified.get("host", "")).upper() != STUDY_HOST
            or verified.get("own_token") != source.get("token")
            or not verified.get("expected_peer_root")
            or verified.get("observed_root") != source.get("observed_root")
            or not isinstance(source.get("token"), str)
            or not source.get("token")
        ):
            raise RevisionCaptureFailure("invalid handoff record")
        records[runtime] = source
        records[f"verified-{runtime}"] = verified
    windows_root = _canonical_native_root(records["windows"].get("observed_root"), "windows")
    wsl_root = _canonical_native_root(records["wsl"].get("observed_root"), "wsl")
    if _canonical_native_root(root_now, runtime_now) != records[runtime_now].get("observed_root"):
        raise RevisionCaptureFailure("current canonical output root differs from handoff binding")
    if records["windows"]["token"] == records["wsl"]["token"]:
        raise RevisionCaptureFailure("handoff tokens are not independent")
    if (
        records["verified-windows"].get("peer_token") != records["wsl"]["token"]
        or records["verified-wsl"].get("peer_token") != records["windows"]["token"]
        or records["verified-windows"].get("expected_peer_root")
        != records["wsl"].get("observed_root")
        or records["verified-wsl"].get("expected_peer_root")
        != records["windows"].get("observed_root")
    ):
        raise RevisionCaptureFailure("handoff peer token/root mismatch")
    if (
        translator(windows_root, "windows") != wsl_root
        or translator(wsl_root, "wsl") != windows_root
    ):
        raise RevisionCaptureFailure("independent native/WSL root translation mismatch")
    return {
        "windows_root": windows_root,
        "wsl_root": wsl_root,
        "windows_token_sha256": sha256_bytes(str(records["windows"]["token"]).encode("utf-8")),
        "wsl_token_sha256": sha256_bytes(str(records["wsl"]["token"]).encode("utf-8")),
    }


def _revision_path(root: Path, index: int) -> Path:
    return root / "ledger" / f"revision-{index:04d}.json"


def validate_ledger(root: Path, *, allow_stopped_tail: bool = False) -> list[dict[str, Any]]:
    directory = root / "ledger"
    if not directory.is_dir():
        raise RevisionCaptureFailure("ledger directory absent")
    entries = list(directory.iterdir())
    if any(path.name.endswith(".tmp") or path.name.startswith(".") for path in entries):
        raise RevisionCaptureFailure("unexpected ledger temporary file")
    paths = sorted(directory.glob("revision-*.json"))
    if set(entries) != set(paths):
        raise RevisionCaptureFailure("foreign ledger entry")
    expected = [_revision_path(root, i) for i in range(len(paths))]
    if paths != expected or not paths:
        raise RevisionCaptureFailure("ledger revision gap or empty chain")
    records: list[dict[str, Any]] = []
    predecessor: str | None = None
    for index, path in enumerate(paths):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RevisionCaptureFailure("invalid ledger JSON") from exc
        if (
            record.get("schema") != LEDGER_SCHEMA
            or record.get("revision") != index
            or record.get("predecessor_sha256") != predecessor
        ):
            raise RevisionCaptureFailure("invalid ledger chain")
        predecessor = sha256_file(path)
        records.append(cast(dict[str, Any], record))
    initial = records[0]
    expected_cells = [cell.__dict__ | {"name": cell.name} for cell in fixed_cells()]
    if initial.get("event") != "initialised" or initial.get("cells") != expected_cells:
        raise RevisionCaptureFailure("initial fixed cell plan differs")
    next_ordinal = 0
    expecting_terminal = False
    for position, record in enumerate(records[1:], start=1):
        event = record.get("event")
        if not expecting_terminal:
            if (
                next_ordinal >= len(fixed_cells())
                or event != "reserved"
                or record.get("cell_ordinal") != next_ordinal
                or record.get("cell_name") != fixed_cells()[next_ordinal].name
            ):
                raise RevisionCaptureFailure("ledger reservation history is invalid")
            expecting_terminal = True
        else:
            if event not in {"complete", "failed"} or record.get("cell_ordinal") != next_ordinal:
                raise RevisionCaptureFailure("ledger terminal history is invalid")
            if event == "failed" and position != len(records) - 1:
                raise RevisionCaptureFailure("failed ledger event must be terminal")
            expecting_terminal = False
            next_ordinal += 1
    tail = records[-1]
    if not allow_stopped_tail and tail.get("event") in {"reserved", "failed"}:
        raise RevisionCaptureFailure("reserved/failed ledger tail permanently stops study")
    return records


def initialise_ledger(root: Path, plan_binding: Mapping[str, object]) -> Path:
    if root.exists():
        raise RevisionCaptureFailure("output root already exists")
    root.mkdir(parents=True)
    cells = [cell.__dict__ | {"name": cell.name} for cell in fixed_cells()]
    record = {
        "schema": LEDGER_SCHEMA,
        "revision": 0,
        "predecessor_sha256": None,
        "event": "initialised",
        "plan_binding": dict(plan_binding),
        "cells": cells,
    }
    publish_bytes(_revision_path(root, 0), canonical_json_bytes(record))
    return _revision_path(root, 0)


def append_revision(
    root: Path, event: str, cell: StudyCell, details: Mapping[str, object] | None = None
) -> Path:
    records = validate_ledger(root, allow_stopped_tail=True)
    tail_event = records[-1]["event"]
    if event == "reserved" and tail_event in {"reserved", "failed"}:
        raise RevisionCaptureFailure("stopped ledger cannot reserve another cell")
    if event in {"complete", "failed"} and tail_event != "reserved":
        raise RevisionCaptureFailure("terminal event requires matching reserved tail")
    if event in {"complete", "failed"} and records[-1].get("cell_ordinal") != cell.ordinal:
        raise RevisionCaptureFailure("terminal event cell differs from reserved tail")
    completed = [item["cell_ordinal"] for item in records if item["event"] == "complete"]
    if event == "reserved" and completed != list(range(cell.ordinal)):
        raise RevisionCaptureFailure("wrong next cell or corrupt completion history")
    index = len(records)
    record = {
        "schema": LEDGER_SCHEMA,
        "revision": index,
        "predecessor_sha256": sha256_file(_revision_path(root, index - 1)),
        "event": event,
        "cell_ordinal": cell.ordinal,
        "cell_name": cell.name,
        "details": dict(details or {}),
    }
    publish_bytes(_revision_path(root, index), canonical_json_bytes(record))
    return _revision_path(root, index)


def load_cell_result(receipt: Path) -> dict[str, Any]:
    """Resolve a receipt's artifact references for in-memory analysis."""
    if receipt.is_dir():
        receipt = receipt / "receipt.json"
    root = receipt.parent
    value = json.loads(receipt.read_text(encoding="utf-8"))

    def resolve(item: Any) -> Any:
        if isinstance(item, dict) and "path" in item and "sha256" in item:
            path = root / str(item["path"])
            if sha256_file(path) != item["sha256"]:
                raise RevisionCaptureFailure(f"artifact hash mismatch: {path}")
            if path.suffix == ".npy":
                return np.load(path, allow_pickle=False)
            if path.suffix == ".json":
                return json.loads(path.read_text(encoding="utf-8"))
        if isinstance(item, dict):
            return {key: resolve(child) for key, child in item.items()}
        if isinstance(item, list):
            return [resolve(child) for child in item]
        return item

    return cast(dict[str, Any], resolve(value))
