"""Prospective aperture-only runtime admission; no SDK or scientific-input access."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints

from epsbench.diagnostics.corridor_aperture import config_root, encode

Hex40 = Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{40}$")]
Hex64 = Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$")]
Image = Annotated[str, StringConstraints(strict=True, pattern=r"^sha256:[0-9a-f]{64}$")]
PURPOSE = "corridor_aperture_native_v3"


class ApertureRuntimeBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    source_head: Hex40
    source_tree: Hex40
    configuration_root: Hex64
    purpose: Literal["corridor_aperture_native_v3"]

    image: Image
    manifest_sha256: Hex64


def aperture_rejection() -> str | None:
    """Public rejection category only; eligibility grants no execution approval."""
    if os.environ.get("EPS_APERTURE_RUNTIME") != "docker_candidate_v1":
        return "aperture_marker_absent"
    try:
        raw = os.environ.get("EPS_APERTURE_BINDING", "").encode()
        if len(raw) > 4096:
            return "aperture_binding_invalid"
        binding = ApertureRuntimeBinding.model_validate_json(raw)
        if raw != encode(binding.model_dump()):
            return "aperture_binding_invalid"
        if platform.system() != "Linux" or not Path("/.dockerenv").is_file():
            return "aperture_platform_unsupported"
        if any(
            os.environ.get(k)
            for k in (
                "WSL_INTEROP",
                "WSL_DISTRO_NAME",
                "EPS_A1_RUNTIME",
                "EPS_CAUSAL_RUNTIME",
                "EPS_PAIRED_APPEARANCE_RUNTIME",
                "EPS_RENDERER_DISCRIMINATOR_RUNTIME",
            )
        ):
            return "aperture_mixed_runtime"
        if (
            binding.configuration_root != config_root()
            or os.environ.get("EPS_APERTURE_IMAGE") != binding.image
        ):
            return "aperture_configuration_or_image_mismatch"
        manifest = Path("/invocation/manifest.json")
        if manifest.stat().st_size > 65536:
            return "aperture_preparation_mismatch"
        prepared_bytes = manifest.read_bytes()
        if len(prepared_bytes) > 65536:
            return "aperture_preparation_mismatch"
        prepared = json.loads(prepared_bytes)
        if (
            hashlib.sha256(prepared_bytes).hexdigest() != binding.manifest_sha256
            or type(prepared) is not dict
            or any(
                prepared.get(k) != getattr(binding, k)
                for k in ("source_head", "source_tree", "configuration_root")
            )
        ):
            return "aperture_preparation_mismatch"
        repository = Path(__file__).resolve().parents[3]

        def git(*args: str) -> str:
            return subprocess.run(
                ["git", "-C", str(repository), *args],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()

        if git("rev-parse", "HEAD", "HEAD^{tree}").splitlines() != [
            binding.source_head,
            binding.source_tree,
        ] or git("status", "--porcelain", "--untracked-files=normal"):
            return "aperture_source_mismatch"
        if any(
            os.environ.get(k) != v
            for k, v in {
                "MUJOCO_GL": "osmesa",
                "PYOPENGL_PLATFORM": "osmesa",
            }.items()
        ):
            return "aperture_backend_unsupported"
        root = Path("/sys/fs/cgroup")
        cpu = (root / "cpu.max").read_text().split()
        if not (
            (root / "memory.max").read_text().strip() == str(4 * 1024**3)
            and (root / "memory.swap.max").read_text().strip() == "0"
            and (root / "pids.max").read_text().strip() == "64"
            and len(cpu) == 2
            and int(cpu[1]) > 0
            and int(cpu[0]) == int(cpu[1])
        ):
            return "aperture_resources_unsupported"
        return None
    except (ValueError, OSError, AttributeError, TypeError, subprocess.SubprocessError):
        return "aperture_facts_unavailable"


def aperture_candidate() -> bool:
    return aperture_rejection() is None


def aperture_selected() -> bool:
    """Any aperture field selects its exclusive route, including empty values."""
    return any(
        k in os.environ
        for k in ("EPS_APERTURE_RUNTIME", "EPS_APERTURE_BINDING", "EPS_APERTURE_IMAGE")
    )
