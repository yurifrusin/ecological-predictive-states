"""Explicit prospective Docker candidate; does not impersonate the WSL profile."""

from __future__ import annotations

import os
from pathlib import Path


def docker_candidate() -> bool:
    if os.environ.get("EPS_A1_RUNTIME") != "docker_candidate_v1":
        return False
    if (
        not Path("/.dockerenv").is_file()
        or os.environ.get("WSL_INTEROP")
        or os.environ.get("WSL_DISTRO_NAME")
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
    except (OSError, ValueError):
        return False
