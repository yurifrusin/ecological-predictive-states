"""Canonical serialisation and content-identity helpers."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel


def canonical_json_bytes(value: BaseModel | dict[str, Any] | list[Any]) -> bytes:
    """Return stable UTF-8 JSON with no volatile whitespace or key ordering."""

    payload: dict[str, Any] | list[Any]
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="json")
    else:
        payload = value
    text = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return text.encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logical_array_hash(array: np.ndarray[Any, Any]) -> str:
    """Hash array semantics separately from the bytes of its container file."""

    header = canonical_json_bytes(
        {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        }
    )
    contiguous = np.ascontiguousarray(array)
    return sha256_bytes(header + b"\0" + contiguous.tobytes(order="C"))


def write_canonical_json(path: Path, value: BaseModel | dict[str, Any] | list[Any]) -> None:
    """Write canonical JSON plus one final newline for readable deterministic files."""

    path.write_bytes(canonical_json_bytes(value) + b"\n")
