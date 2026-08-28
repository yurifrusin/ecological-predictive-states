"""Fail-closed resolution for the dataset root manifest."""

from __future__ import annotations

import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

_WINDOWS_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:")


class UnsafeOwnedFileError(ValueError):
    """Raised when a declared file is not uniquely owned below its root."""


@dataclass(frozen=True)
class OwnedRegularFile:
    """Resolved identity for a non-linked regular file owned by one artifact root."""

    path: Path
    device: int
    inode: int
    byte_count: int


def resolve_owned_regular_file(root: Path, relative_path: str) -> OwnedRegularFile:
    """Resolve one canonical relative path without following aliases or links."""

    try:
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        raise UnsafeOwnedFileError("artifact root does not exist") from error
    if not resolved_root.is_dir():
        raise UnsafeOwnedFileError("artifact root is not a directory")
    if (
        "\\" in relative_path
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
    unresolved_identity = (candidate_stat.st_dev, candidate_stat.st_ino)
    resolved_identity = (resolved_stat.st_dev, resolved_stat.st_ino)
    if unresolved_identity != resolved_identity or resolved_stat.st_nlink != 1:
        raise UnsafeOwnedFileError(f"artifact changed while being validated: {relative_path}")
    return OwnedRegularFile(
        path=resolved,
        device=resolved_stat.st_dev,
        inode=resolved_stat.st_ino,
        byte_count=resolved_stat.st_size,
    )


class UnsafeDatasetManifestError(ValueError):
    """Raised when the root manifest is missing, aliased, or not a safe regular file."""


def resolve_dataset_manifest(root: Path) -> tuple[Path, Path]:
    """Return the resolved root and a non-aliased regular manifest inside it."""

    try:
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        raise UnsafeDatasetManifestError("dataset root does not exist") from error
    if not resolved_root.is_dir():
        raise UnsafeDatasetManifestError("dataset root is not a directory")

    try:
        manifest = resolve_owned_regular_file(resolved_root, "manifest.json")
    except UnsafeOwnedFileError as error:
        raise UnsafeDatasetManifestError(str(error).replace("artifact", "manifest.json")) from error
    return resolved_root, manifest.path
