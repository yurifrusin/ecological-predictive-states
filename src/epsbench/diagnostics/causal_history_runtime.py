"""Distinct prospective causal runtime admission; qualification remains a later decision."""

from __future__ import annotations

import os
import re
from pathlib import Path


def causal_candidate() -> bool:
    if (
        os.environ.get("EPS_CAUSAL_RUNTIME") != "docker_candidate_v1"
        or os.environ.get("EPS_A1_RUNTIME")
        or os.environ.get("WSL_INTEROP")
        or os.environ.get("WSL_DISTRO_NAME")
        or not Path("/.dockerenv").is_file()
        or os.environ.get("EPS_CAUSAL_PURPOSE") != "causal_history_native_v1"
        or any(
            re.fullmatch(pattern, os.environ.get(name, "")) is None
            for name, pattern in (
                ("EPS_CAUSAL_SOURCE_HEAD", r"[0-9a-f]{40}"),
                ("EPS_CAUSAL_SOURCE_TREE", r"[0-9a-f]{40}"),
                ("EPS_CAUSAL_IMAGE", r"sha256:[0-9a-f]{64}"),
            )
        )
    ):
        return False
    root = Path("/sys/fs/cgroup")
    affinity = "sched_getaffinity"
    filesystem = "statvfs"
    try:
        cpu = (root / "cpu.max").read_text().split()
        return (
            (root / "memory.max").read_text().strip() == str(8 * 1024**3)
            and (root / "memory.swap.max").read_text().strip() == "0"
            and (root / "pids.max").read_text().strip() == "64"
            and len(cpu) == 2
            and int(cpu[0]) == 2 * int(cpu[1])
            and sorted(getattr(os, affinity)(0)) == [0, 1]
            and os.environ.get("LP_NUM_THREADS") == "2"
            and os.environ.get("OMP_NUM_THREADS") == "2"
            and getattr(os, filesystem)("/output").f_blocks
            * getattr(os, filesystem)("/output").f_frsize
            == 255 * 1024**2
            and getattr(os, filesystem)("/dev/shm").f_blocks
            * getattr(os, filesystem)("/dev/shm").f_frsize
            == 1024**2
        )
    except (OSError, ValueError, AttributeError):
        return False
