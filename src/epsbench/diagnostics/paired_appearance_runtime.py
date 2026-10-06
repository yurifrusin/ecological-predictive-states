"""Distinct appearance admission. Importing this module never accesses a native SDK."""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Annotated, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, StringConstraints, field_validator

from epsbench.diagnostics import paired_appearance as p
from epsbench.utils.canonical import canonical_json_bytes, sha256_bytes

Hex40 = Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{40}$")]
Hex64 = Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$")]
Token = Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{32}$")]
Image = Annotated[str, StringConstraints(strict=True, pattern=r"^sha256:[0-9a-f]{64}$")]
NATIVE = "paired_appearance_native_v1"
DUMMY = "paired_appearance_dummy_v1"
CONFIG_FILE: Literal["8a21410230a9f2ef1ce9b065671ff31e6ef97bcdcbb47ce936fbee5ba1afe9cd"] = (
    "8a21410230a9f2ef1ce9b065671ff31e6ef97bcdcbb47ce936fbee5ba1afe9cd"
)
VERSION: Literal["paired_appearance_execution_v1"] = "paired_appearance_execution_v1"


class Closed(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Preparation(Closed):
    version: Literal["paired_appearance_execution_v1"] = VERSION
    source_url: Literal["https://github.com/yurifrusin/ecological-predictive-states.git"]
    source_head: Hex40
    source_tree: Hex40
    config_file: Literal["8a21410230a9f2ef1ce9b065671ff31e6ef97bcdcbb47ce936fbee5ba1afe9cd"]
    config_root: Hex64
    asset_root: Hex64
    protection_root: Hex64
    membership_root: Hex64


class AppearanceExecutionBinding(Closed):
    preparation: Preparation
    image: Image
    purpose: Literal["paired_appearance_native_v1", "paired_appearance_dummy_v1"]
    output_id: Token
    token: Token
    policy: Literal["2g_2cpu_64pid_300s_64m_v1"] = "2g_2cpu_64pid_300s_64m_v1"

    @property
    def root(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.model_dump(mode="json")))


class LaunchDecision(Closed):
    binding_root: Hex64
    approved: Literal[True]
    purpose: Literal["paired_appearance_native_v1", "paired_appearance_dummy_v1"]
    # A substantive external decision, not a generated default or hash-only opt-in.
    authorization: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=4096)]

    @field_validator("approved", mode="before")
    @classmethod
    def strict_approval(cls, value: object) -> object:
        if value is not True:
            raise ValueError("approval must be the strict boolean true")
        return value


ClosedType = TypeVar("ClosedType", bound=Closed)


def parse(model: type[ClosedType], data: bytes) -> ClosedType:
    if type(data) is not bytes or len(data) > 16384:
        raise ValueError("bounded closed execution document required")
    return model.model_validate(p.strict_json(data))


def preparation(repository: Path, head: str, tree: str) -> Preparation:
    raw = (repository / "configs/development/paired_appearance_v1.json").read_bytes()
    p.fixed_config(raw)
    if sha256_bytes(raw) != CONFIG_FILE:
        raise ValueError("exact fixed config file bytes required")
    return Preparation(
        source_url="https://github.com/yurifrusin/ecological-predictive-states.git",
        source_head=head,
        source_tree=tree,
        config_file=CONFIG_FILE,
        config_root=sha256_bytes(p.config_bytes()),
        asset_root=sha256_bytes(
            canonical_json_bytes(
                [p.visual_plan(f, a).record for f in p.FAMILIES for a in p.APPEARANCES]
            )
        ),
        protection_root=sha256_bytes(canonical_json_bytes(p.protection_check(repository))),
        membership_root=sha256_bytes(canonical_json_bytes(list(p.contexts()))),
    )


def require_decision(binding: AppearanceExecutionBinding, decision: LaunchDecision) -> None:
    if (
        type(binding) is not AppearanceExecutionBinding
        or type(decision) is not LaunchDecision
        or decision.approved is not True
        or decision.binding_root != binding.root
        or decision.purpose != binding.purpose
        or not decision.authorization.strip()
    ):
        raise PermissionError("separate exact-binding execution decision required")


def environment_binding() -> AppearanceExecutionBinding:
    raw = os.environ.get("EPS_PAIRED_APPEARANCE_BINDING", "").encode()
    binding = parse(AppearanceExecutionBinding, raw)
    if not isinstance(binding, AppearanceExecutionBinding):
        raise ValueError("binding type")
    return binding


def paired_appearance_candidate() -> bool:
    """Native only: purpose, bindings and resource facts; common SDK checks remain upstream."""
    try:
        binding = environment_binding()
        if (
            binding.purpose != NATIVE
            or platform.system() != "Linux"
            or not Path("/.dockerenv").is_file()
            or any(
                os.environ.get(v)
                for v in ("EPS_A1_RUNTIME", "EPS_CAUSAL_RUNTIME", "WSL_INTEROP", "WSL_DISTRO_NAME")
            )
            or any(
                os.environ.get(k) != v
                for k, v in {
                    "EPS_PAIRED_APPEARANCE_RUNTIME": "docker_candidate_v1",
                    "EPS_PAIRED_APPEARANCE_PURPOSE": NATIVE,
                    "EPS_PAIRED_APPEARANCE_SOURCE_HEAD": binding.preparation.source_head,
                    "EPS_PAIRED_APPEARANCE_SOURCE_TREE": binding.preparation.source_tree,
                    "EPS_PAIRED_APPEARANCE_IMAGE": binding.image,
                    "EPS_PAIRED_APPEARANCE_ROOT": binding.root,
                    "MUJOCO_GL": "osmesa",
                    "PYOPENGL_PLATFORM": "osmesa",
                    "LP_NUM_THREADS": "2",
                    "OMP_NUM_THREADS": "2",
                }.items()
            )
        ):
            return False
        root = Path("/sys/fs/cgroup")
        affinity, filesystem = "sched_getaffinity", "statvfs"
        cpu = (root / "cpu.max").read_text().split()
        return (
            (root / "memory.max").read_text().strip() == str(2 * 1024**3)
            and (root / "memory.swap.max").read_text().strip() == "0"
            and (root / "pids.max").read_text().strip() == "64"
            and len(cpu) == 2
            and int(cpu[1]) > 0
            and int(cpu[0]) == 2 * int(cpu[1])
            and sorted(getattr(os, affinity)(0)) == [0, 1]
            and all(
                getattr(os, filesystem)(path).f_blocks * getattr(os, filesystem)(path).f_frsize
                == size * 1024**2
                for path, size in (("/tmp", 8), ("/dev/shm", 1))
            )
        )
    except (ValueError, OSError, AttributeError, TypeError):
        return False


def require_native_binding(repository: Path, binding: AppearanceExecutionBinding) -> None:
    if type(binding) is not AppearanceExecutionBinding or binding.purpose != NATIVE:
        raise PermissionError("appearance native purpose required before SDK access")
    if environment_binding() != binding or not paired_appearance_candidate():
        raise PermissionError("unsupported exact appearance runtime")
    if (
        preparation(repository, binding.preparation.source_head, binding.preparation.source_tree)
        != binding.preparation
    ):
        raise PermissionError("config/assets/protection binding differs")
    anchor_path = Path("/output/consumed.json")
    if anchor_path.stat().st_size > 16384:
        raise PermissionError("bounded consumed decision required")
    anchor = p.strict_json(anchor_path.read_bytes())
    if (
        set(anchor) != {"binding", "decision", "binding_root"}
        or anchor["binding"] != binding.model_dump(mode="json")
        or anchor["binding_root"] != binding.root
    ):
        raise PermissionError("consumed attempt binding differs")
    require_decision(binding, LaunchDecision.model_validate(anchor["decision"]))
    # The immutable prepared image manifest is checked without native import.
    manifest = parse(Preparation, Path("/preparation/appearance.json").read_bytes())
    if manifest != binding.preparation:
        raise PermissionError("image preparation manifest differs")
