"""Byte-preserving serializers with optional synchronous retained publication."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Protocol

import numpy as np
from PIL import Image
from pydantic import BaseModel

from epsbench.utils.canonical import canonical_json_bytes


class ArtifactPublisher(Protocol):
    """Cooperative control-plane sink; completion returns only after durable flush."""

    staging_limit: int

    def start(self, path: str) -> None: ...
    def reserve_staging(self, size: int) -> None: ...
    def publish(self, path: Path, relative: str) -> None: ...
    def failure(self, reason: str) -> None: ...


class _BoundedStream:
    def __init__(self, stream: BinaryIO, limit: int, publisher: ArtifactPublisher) -> None:
        self.stream = stream
        self.limit = limit
        self.publisher = publisher

    def write(self, data: bytes) -> int:
        if self.stream.tell() + len(data) > self.limit:
            raise ValueError("staging byte limit exceeded before write")
        self.publisher.reserve_staging(len(data))
        return self.stream.write(data)

    def tell(self) -> int:
        return self.stream.tell()

    def flush(self) -> None:
        self.stream.flush()


class OutputWriter:
    def __init__(
        self,
        root: Path,
        publisher: ArtifactPublisher | None = None,
        *,
        publication_prefix: str = "",
    ) -> None:
        if publication_prefix and (
            not isinstance(publication_prefix, str)
            or PurePosixPath(publication_prefix).is_absolute()
            or len(publication_prefix.encode()) > 1024
            or any(char in publication_prefix for char in ("\\", ":", "\0"))
            or any(part in {"", ".", ".."} for part in publication_prefix.split("/"))
        ):
            raise ValueError("canonical publication prefix required")
        self.root = root.resolve()
        self.publisher = publisher
        self.publication_prefix = publication_prefix
        self.staged_bytes = 0

    def _write(self, path: Path, serialize: Callable[[Any], object]) -> None:
        if self.publisher is None:
            with path.open("wb") as stream:
                serialize(stream)
            return
        relative = path.absolute().relative_to(self.root).as_posix()
        if self.publication_prefix:
            relative = self.publication_prefix + "/" + relative
        # Existing ancestor links cannot redirect cooperative staging.
        for component in (path, *path.parents):
            if component.is_symlink():
                raise ValueError("linked staging path denied")
        self.publisher.start(relative)
        try:
            with path.open("xb") as stream:
                bounded = _BoundedStream(stream, self.publisher.staging_limit, self.publisher)
                try:
                    serialize(bounded)
                finally:
                    self.staged_bytes += stream.tell()
                stream.flush()
                os.fsync(stream.fileno())
            self.publisher.publish(path, relative)
        except Exception as error:
            self.publisher.failure(type(error).__name__)
            raise

    def array(self, path: Path, value: np.ndarray[Any, Any], *, allow_pickle: bool = False) -> None:
        if allow_pickle:
            raise ValueError("pickle output denied")
        self._write(path, lambda stream: np.save(stream, value, allow_pickle=False))

    def png(self, path: Path, rgb: np.ndarray[Any, Any]) -> None:
        self._write(
            path,
            lambda stream: Image.fromarray(rgb, mode="RGB").save(
                stream, format="PNG", compress_level=9, optimize=False
            ),
        )

    def canonical(self, path: Path, value: BaseModel | dict[str, Any] | list[Any]) -> None:
        self._write(path, lambda stream: stream.write(canonical_json_bytes(value) + b"\n"))

    def text(self, path: Path, value: str) -> None:
        # Match Path.write_text's platform newline translation.
        payload = value.replace("\n", os.linesep).encode("utf-8")
        self._write(path, lambda stream: stream.write(payload))
