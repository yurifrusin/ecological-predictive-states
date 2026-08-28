from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
import pytest
from PIL import Image

from epsbench.data.decoding import ArtifactDecodeError, decode_rgb_artifact
from epsbench.schema import ArtifactRecord, Modality
from epsbench.utils.canonical import logical_array_hash, sha256_bytes


@pytest.mark.parametrize(
    ("update", "expected"),
    (
        ({"logical_sha256": "0" * 64}, "array logical hash mismatch"),
        ({"dtype": "int8"}, "array metadata mismatch"),
        ({"shape": (2, 2, 4)}, "array metadata mismatch"),
        ({"media_type": "image/jpeg"}, "array media type mismatch"),
    ),
)
def test_shared_rgb_decoder_rejects_every_false_logical_declaration(
    update: dict[str, Any],
    expected: str,
) -> None:
    rgb = np.arange(12, dtype=np.uint8).reshape((2, 2, 3))
    stream = BytesIO()
    Image.fromarray(rgb, mode="RGB").save(stream, format="PNG")
    snapshot = stream.getvalue()
    record = ArtifactRecord(
        path="episode/rgb.png",
        modality=Modality.RGB,
        media_type="image/png",
        dtype="uint8",
        shape=(2, 2, 3),
        logical_sha256=logical_array_hash(rgb),
        file_sha256=sha256_bytes(snapshot),
        byte_count=len(snapshot),
    ).model_copy(update=update)

    with pytest.raises(ArtifactDecodeError, match=expected):
        decode_rgb_artifact(snapshot, record)
