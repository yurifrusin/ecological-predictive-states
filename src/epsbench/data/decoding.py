"""Shared decoded-artifact verification for validation and permissioned loading."""

from __future__ import annotations

import json
from io import BytesIO
from typing import Any, cast

import numpy as np
from PIL import Image

from epsbench.schema import ArtifactRecord
from epsbench.utils.canonical import (
    canonical_json_bytes,
    logical_array_hash,
    sha256_bytes,
)


class ArtifactDecodeError(ValueError):
    """Raised when immutable artifact bytes disagree with their logical declaration."""


def decode_json_artifact(snapshot: bytes, record: ArtifactRecord) -> dict[str, Any]:
    """Decode and verify one canonical JSON-object artifact snapshot."""

    try:
        payload = json.loads(snapshot.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ArtifactDecodeError(f"invalid JSON artifact: {record.path}") from error
    if not isinstance(payload, dict):
        raise ArtifactDecodeError(f"JSON artifact must contain an object: {record.path}")
    logical_bytes = canonical_json_bytes(payload)
    if sha256_bytes(logical_bytes) != record.logical_sha256:
        raise ArtifactDecodeError(f"artifact logical hash mismatch: {record.path}")
    if (
        record.media_type != "application/json"
        or record.dtype != "json"
        or record.shape != (len(logical_bytes),)
    ):
        raise ArtifactDecodeError(f"JSON artifact metadata mismatch: {record.path}")
    return payload


def _verify_array(
    record: ArtifactRecord,
    array: np.ndarray[Any, Any],
    expected_media_type: str,
) -> None:
    if record.media_type != expected_media_type:
        raise ArtifactDecodeError(f"array media type mismatch: {record.path}")
    if tuple(array.shape) != record.shape or str(array.dtype) != record.dtype:
        raise ArtifactDecodeError(f"array metadata mismatch: {record.path}")
    if logical_array_hash(array) != record.logical_sha256:
        raise ArtifactDecodeError(f"array logical hash mismatch: {record.path}")


def decode_rgb_artifact(
    snapshot: bytes,
    record: ArtifactRecord,
) -> np.ndarray[Any, Any]:
    """Decode and verify one RGB PNG artifact snapshot."""

    if record.media_type != "image/png":
        raise ArtifactDecodeError(f"array media type mismatch: {record.path}")
    try:
        with Image.open(BytesIO(snapshot)) as image:
            if image.mode != "RGB":
                raise ArtifactDecodeError(f"RGB artifact must use RGB mode: {record.path}")
            rgb = np.asarray(image, dtype=np.uint8).copy()
    except OSError as error:
        raise ArtifactDecodeError(f"unreadable RGB artifact: {record.path}") from error
    _verify_array(record, rgb, "image/png")
    return rgb


def decode_npy_artifact(
    snapshot: bytes,
    record: ArtifactRecord,
) -> np.ndarray[Any, Any]:
    """Decode and verify one NumPy artifact snapshot without permitting pickle."""

    if record.media_type != "application/x-npy":
        raise ArtifactDecodeError(f"array media type mismatch: {record.path}")
    try:
        array = cast(
            np.ndarray[Any, Any],
            np.load(BytesIO(snapshot), allow_pickle=False),
        )
    except (OSError, ValueError) as error:
        raise ArtifactDecodeError(f"unreadable NumPy artifact: {record.path}") from error
    _verify_array(record, array, "application/x-npy")
    return array
