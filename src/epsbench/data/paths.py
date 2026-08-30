"""Fail-closed, handle-bound access to files owned by one artifact root."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from collections.abc import Iterator, Set
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

_WINDOWS_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:")


class UnsafeOwnedFileError(ValueError):
    """Raised when a declared file is not uniquely owned below its root."""


@dataclass(frozen=True)
class OwnedRegularFile:
    """One verified identity, open handle, and immutable byte snapshot."""

    path: Path
    handle: BinaryIO
    payload: bytes
    device: int
    inode: int
    byte_count: int


def _resolve_artifact_root(root: Path) -> Path:
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        raise UnsafeOwnedFileError("artifact root does not exist") from error
    if not resolved_root.is_dir():
        raise UnsafeOwnedFileError("artifact root is not a directory")
    return resolved_root


def _canonical_logical_path(relative_path: str) -> PurePosixPath:
    if (
        not isinstance(relative_path, str)
        or "\\" in relative_path
        or _WINDOWS_DRIVE_PREFIX.match(relative_path)
        or "\x00" in relative_path
    ):
        raise UnsafeOwnedFileError(f"artifact path is not canonical: {relative_path}")
    logical_path = PurePosixPath(relative_path)
    if (
        logical_path.is_absolute()
        or relative_path in {"", "."}
        or ".." in logical_path.parts
        or logical_path.as_posix() != relative_path
    ):
        raise UnsafeOwnedFileError(f"artifact path is not canonical: {relative_path}")
    return logical_path


def _verify_current_path(
    resolved_root: Path,
    logical_path: PurePosixPath,
    relative_path: str,
    expected_identity: tuple[int, int] | None = None,
) -> tuple[Path, os.stat_result]:
    cursor = resolved_root
    for part in logical_path.parts[:-1]:
        cursor /= part
        try:
            component_stat = cursor.lstat()
        except OSError as error:
            raise UnsafeOwnedFileError(f"artifact parent is missing: {relative_path}") from error
        if stat.S_ISLNK(component_stat.st_mode):
            raise UnsafeOwnedFileError(
                f"artifact parent must not be a symbolic-link alias: {relative_path}"
            )
        if not stat.S_ISDIR(component_stat.st_mode):
            raise UnsafeOwnedFileError(f"artifact parent is not a directory: {relative_path}")

    candidate = resolved_root.joinpath(*logical_path.parts)
    try:
        candidate_stat = candidate.lstat()
    except OSError as error:
        raise UnsafeOwnedFileError(f"missing artifact: {relative_path}") from error
    if stat.S_ISLNK(candidate_stat.st_mode):
        raise UnsafeOwnedFileError(f"artifact must not be a symbolic-link alias: {relative_path}")
    if not stat.S_ISREG(candidate_stat.st_mode):
        raise UnsafeOwnedFileError(f"artifact must be a regular file: {relative_path}")
    if candidate_stat.st_nlink != 1:
        raise UnsafeOwnedFileError(f"artifact must not be a hard-link alias: {relative_path}")
    try:
        resolved = candidate.resolve(strict=True)
        resolved_stat = resolved.stat()
    except OSError as error:
        raise UnsafeOwnedFileError(f"artifact cannot be resolved: {relative_path}") from error
    if not resolved.is_relative_to(resolved_root):
        raise UnsafeOwnedFileError(f"artifact resolves outside its root: {relative_path}")
    candidate_identity = (candidate_stat.st_dev, candidate_stat.st_ino)
    resolved_identity = (resolved_stat.st_dev, resolved_stat.st_ino)
    if (
        candidate_identity != resolved_identity
        or resolved_stat.st_nlink != 1
        or (expected_identity is not None and resolved_identity != expected_identity)
    ):
        raise UnsafeOwnedFileError(f"artifact changed while being validated: {relative_path}")
    return resolved, resolved_stat


def _verify_open_handle(
    owned: OwnedRegularFile,
    resolved_root: Path,
    logical_path: PurePosixPath,
    relative_path: str,
) -> None:
    try:
        handle_stat = os.fstat(owned.handle.fileno())
    except (OSError, ValueError) as error:
        raise UnsafeOwnedFileError(f"artifact handle became invalid: {relative_path}") from error
    identity = (handle_stat.st_dev, handle_stat.st_ino)
    if (
        not stat.S_ISREG(handle_stat.st_mode)
        or handle_stat.st_nlink != 1
        or identity != (owned.device, owned.inode)
        or handle_stat.st_size != owned.byte_count
    ):
        raise UnsafeOwnedFileError(f"artifact changed while being consumed: {relative_path}")
    _, current_stat = _verify_current_path(
        resolved_root,
        logical_path,
        relative_path,
        expected_identity=identity,
    )
    if current_stat.st_size != owned.byte_count:
        raise UnsafeOwnedFileError(f"artifact changed while being consumed: {relative_path}")
    try:
        owned.handle.seek(0)
        current_payload = owned.handle.read()
    except (OSError, ValueError) as error:
        raise UnsafeOwnedFileError(f"artifact cannot be re-read: {relative_path}") from error
    if current_payload != owned.payload:
        raise UnsafeOwnedFileError(f"artifact bytes changed while being consumed: {relative_path}")


@contextmanager
def open_owned_regular_file(root: Path, relative_path: str) -> Iterator[OwnedRegularFile]:
    """Open one owned file and keep its verified identity bound through consumption."""

    resolved_root = _resolve_artifact_root(root)
    logical_path = _canonical_logical_path(relative_path)
    resolved, before_stat = _verify_current_path(resolved_root, logical_path, relative_path)
    try:
        handle = resolved.open("rb")
    except OSError as error:
        raise UnsafeOwnedFileError(f"artifact cannot be opened: {relative_path}") from error
    try:
        try:
            opened_stat = os.fstat(handle.fileno())
        except OSError as error:
            raise UnsafeOwnedFileError(f"artifact cannot be inspected: {relative_path}") from error
        identity = (opened_stat.st_dev, opened_stat.st_ino)
        if (
            not stat.S_ISREG(opened_stat.st_mode)
            or opened_stat.st_nlink != 1
            or identity != (before_stat.st_dev, before_stat.st_ino)
            or opened_stat.st_size != before_stat.st_size
        ):
            raise UnsafeOwnedFileError(f"artifact changed while being opened: {relative_path}")
        current_path, current_stat = _verify_current_path(
            resolved_root,
            logical_path,
            relative_path,
            expected_identity=identity,
        )
        if current_stat.st_size != opened_stat.st_size:
            raise UnsafeOwnedFileError(f"artifact changed while being opened: {relative_path}")
        try:
            handle.seek(0)
            payload = handle.read()
        except OSError as error:
            raise UnsafeOwnedFileError(
                f"artifact cannot be snapshotted: {relative_path}"
            ) from error
        if len(payload) != opened_stat.st_size:
            raise UnsafeOwnedFileError(f"artifact changed while being snapshotted: {relative_path}")
        owned = OwnedRegularFile(
            path=current_path,
            handle=handle,
            payload=payload,
            device=opened_stat.st_dev,
            inode=opened_stat.st_ino,
            byte_count=opened_stat.st_size,
        )
        try:
            yield owned
        finally:
            _verify_open_handle(owned, resolved_root, logical_path, relative_path)
    finally:
        handle.close()


def sha256_open_file(owned: OwnedRegularFile) -> str:
    """Hash the immutable snapshot captured from an owned file handle."""

    return hashlib.sha256(owned.payload).hexdigest()


def validate_exact_owned_file_tree(root: Path, expected_files: Set[str]) -> None:
    """Require an exact, non-link tree of uniquely owned regular files."""

    try:
        root_stat = root.lstat()
    except OSError as error:
        raise UnsafeOwnedFileError("artifact root does not exist") from error
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)

    def is_link_or_reparse(value: os.stat_result) -> bool:
        return stat.S_ISLNK(value.st_mode) or bool(
            getattr(value, "st_file_attributes", 0) & reparse_flag
        )

    if is_link_or_reparse(root_stat) or not stat.S_ISDIR(root_stat.st_mode):
        raise UnsafeOwnedFileError("artifact root must be a non-link directory")
    resolved_root = root.resolve(strict=True)
    canonical_files: set[str] = set()
    expected_directories: set[str] = set()
    for relative_path in expected_files:
        logical_path = _canonical_logical_path(relative_path)
        canonical_files.add(logical_path.as_posix())
        parts = logical_path.parts
        expected_directories.update(
            PurePosixPath(*parts[:index]).as_posix() for index in range(1, len(parts))
        )
    if canonical_files != set(expected_files):
        raise UnsafeOwnedFileError("artifact tree expectation is not canonical")

    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    file_identities: set[tuple[int, int]] = set()

    def visit(directory: Path, prefix: PurePosixPath | None = None) -> None:
        try:
            entries = list(os.scandir(directory))
        except OSError as error:
            raise UnsafeOwnedFileError("artifact tree cannot be enumerated") from error
        for entry in entries:
            relative = PurePosixPath(entry.name) if prefix is None else prefix / entry.name
            relative_path = relative.as_posix()
            try:
                entry_stat = Path(entry.path).lstat()
            except OSError as error:
                raise UnsafeOwnedFileError(
                    f"artifact tree entry cannot be inspected: {relative_path}"
                ) from error
            if is_link_or_reparse(entry_stat):
                raise UnsafeOwnedFileError(
                    f"artifact tree entry must not be a link alias: {relative_path}"
                )
            if stat.S_ISDIR(entry_stat.st_mode):
                actual_directories.add(relative_path)
                visit(Path(entry.path), relative)
                continue
            if not stat.S_ISREG(entry_stat.st_mode) or entry_stat.st_nlink != 1:
                raise UnsafeOwnedFileError(
                    f"artifact tree entry must be one owned regular file: {relative_path}"
                )
            identity = (entry_stat.st_dev, entry_stat.st_ino)
            if identity in file_identities:
                raise UnsafeOwnedFileError(
                    f"artifact tree entries must have unique identities: {relative_path}"
                )
            file_identities.add(identity)
            actual_files.add(relative_path)

    visit(resolved_root)
    if actual_files != canonical_files or actual_directories != expected_directories:
        raise UnsafeOwnedFileError("artifact tree contains missing or additional entries")


class UnsafeDatasetManifestError(ValueError):
    """Raised when the root manifest is missing, aliased, or not a safe regular file."""


@contextmanager
def open_dataset_manifest(root: Path) -> Iterator[tuple[Path, OwnedRegularFile]]:
    """Open the root manifest and keep its verified handle live through schema decoding."""

    try:
        resolved_root = _resolve_artifact_root(root)
        with open_owned_regular_file(resolved_root, "manifest.json") as manifest:
            yield resolved_root, manifest
    except UnsafeOwnedFileError as error:
        message = str(error).replace("artifact", "manifest.json")
        raise UnsafeDatasetManifestError(message) from error
