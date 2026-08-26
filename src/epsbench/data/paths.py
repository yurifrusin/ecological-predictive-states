"""Fail-closed resolution for the dataset root manifest."""

from __future__ import annotations

import stat
from pathlib import Path


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

    manifest_path = resolved_root / "manifest.json"
    try:
        manifest_stat = manifest_path.lstat()
        resolved_manifest = manifest_path.resolve(strict=True)
    except OSError as error:
        raise UnsafeDatasetManifestError("missing manifest.json") from error
    if not resolved_manifest.is_relative_to(resolved_root):
        raise UnsafeDatasetManifestError("manifest.json resolves outside the dataset root")
    if stat.S_ISLNK(manifest_stat.st_mode):
        raise UnsafeDatasetManifestError("manifest.json must not be a symbolic-link alias")
    if not stat.S_ISREG(manifest_stat.st_mode):
        raise UnsafeDatasetManifestError("manifest.json must be a regular file")
    if manifest_stat.st_nlink != 1:
        raise UnsafeDatasetManifestError("manifest.json must not be a hard-link alias")
    return resolved_root, resolved_manifest
