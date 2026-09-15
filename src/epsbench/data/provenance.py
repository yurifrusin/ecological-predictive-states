"""Exact source and governing-document provenance collection."""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from epsbench import __version__
from epsbench.schema import GitAvailabilityStatus, SourceProvenance, sanitize_git_repository
from epsbench.utils.canonical import sha256_bytes, sha256_file

EXPECTED_GOVERNING_HASHES = {
    "docs/RESEARCH_CHARTER.md": (
        "4e15b97c6041e12a33bdb52fba5297672a2181df68bceb9a6df56395811b3b06"
    ),
    "docs/EPS_BENCH_V0.md": ("6c6cca701085a637e978d6e26113e7f90752e37d94773743e7be52b565219505"),
    "docs/MILESTONE_0.md": ("6b0f0d8e7248901defef2d90179ee2b8465d49be0bb13ac9754a9f6bd3c68c1b"),
    "CODEX_HANDOFF.md": ("829b96ee9938f65084ba2313d89b81c9e45ac0cacaf7befc8680657025d6e367"),
}


class ProvenanceError(RuntimeError):
    """Raised when required source provenance cannot be represented truthfully."""


class GitUnavailableError(ProvenanceError):
    """Raised by a provider that cannot inspect Git state."""


@dataclass(frozen=True)
class GitState:
    repository: str
    commit: str
    dirty: bool
    dirty_diff_sha256: str | None


class GitStateProvider(Protocol):
    def collect(self, source_root: Path) -> GitState: ...


def _git(source_root: Path, *arguments: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(source_root), *arguments],
            check=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", b"")
        message = detail.decode("utf-8", errors="replace").strip()
        raise GitUnavailableError(message or str(error)) from error
    return result.stdout


class SubprocessGitStateProvider:
    """Read exact Git state without modifying repository configuration."""

    def collect(self, source_root: Path) -> GitState:
        repository = _git(source_root, "remote", "get-url", "origin").decode().strip()
        commit = _git(source_root, "rev-parse", "HEAD").decode().strip()
        status = _git(
            source_root,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        )
        dirty = bool(status)
        dirty_hash: str | None = None
        if dirty:
            tracked_diff = _git(source_root, "diff", "--binary", "HEAD", "--")
            untracked_output = _git(
                source_root,
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
            )
            untracked_entries: list[bytes] = []
            for encoded_path in sorted(filter(None, untracked_output.split(b"\0"))):
                relative_path = encoded_path.decode("utf-8", errors="surrogateescape")
                file_path = source_root / relative_path
                if file_path.is_file():
                    untracked_entries.extend(
                        (encoded_path, b"\0", sha256_file(file_path).encode(), b"\0")
                    )
            dirty_hash = sha256_bytes(
                b"status\0"
                + status
                + b"tracked-diff\0"
                + tracked_diff
                + b"untracked-files\0"
                + b"".join(untracked_entries)
            )
        return GitState(
            repository=sanitize_git_repository(repository),
            commit=commit,
            dirty=dirty,
            dirty_diff_sha256=dirty_hash,
        )


def discover_source_root(start: Path) -> Path:
    """Find the nearest source tree containing the locked governing apparatus."""

    resolved = start.resolve()
    candidates = (resolved, *resolved.parents)
    for candidate in candidates:
        if (candidate / "uv.lock").is_file() and all(
            (candidate / relative_path).is_file() for relative_path in EXPECTED_GOVERNING_HASHES
        ):
            return candidate
    raise ProvenanceError("could not locate uv.lock and governing documents")


def collect_source_provenance(
    source_root: Path,
    git_provider: GitStateProvider | None = None,
) -> SourceProvenance:
    """Collect exact source identity while representing unavailable Git honestly."""

    root = discover_source_root(source_root)
    hashes = {
        relative_path: sha256_file(root / relative_path)
        for relative_path in EXPECTED_GOVERNING_HASHES
    }
    for relative_path, expected_hash in EXPECTED_GOVERNING_HASHES.items():
        if hashes[relative_path] != expected_hash:
            raise ProvenanceError(
                f"governing document hash mismatch for {relative_path}: "
                f"expected {expected_hash}, observed {hashes[relative_path]}"
            )

    provider = git_provider or SubprocessGitStateProvider()
    git_repository: str | None
    git_commit: str | None
    git_dirty: bool | None
    dirty_diff_sha256: str | None
    git_availability_status: GitAvailabilityStatus
    git_unavailable_reason: str | None
    try:
        git_state = provider.collect(root)
    except GitUnavailableError as error:
        git_repository = None
        git_commit = None
        git_dirty = None
        dirty_diff_sha256 = None
        git_availability_status = GitAvailabilityStatus.UNAVAILABLE
        git_unavailable_reason = str(error)
    else:
        git_repository = sanitize_git_repository(git_state.repository)
        git_commit = git_state.commit
        git_dirty = git_state.dirty
        dirty_diff_sha256 = git_state.dirty_diff_sha256
        git_availability_status = GitAvailabilityStatus.AVAILABLE
        git_unavailable_reason = None
    return SourceProvenance(
        git_repository=git_repository,
        git_commit=git_commit,
        git_dirty=git_dirty,
        dirty_diff_sha256=dirty_diff_sha256,
        git_availability_status=git_availability_status,
        git_unavailable_reason=git_unavailable_reason,
        uv_lock_sha256=sha256_file(root / "uv.lock"),
        research_charter_sha256=hashes["docs/RESEARCH_CHARTER.md"],
        eps_bench_spec_sha256=hashes["docs/EPS_BENCH_V0.md"],
        milestone_plan_sha256=hashes["docs/MILESTONE_0.md"],
        codex_handoff_sha256=hashes["CODEX_HANDOFF.md"],
        package_version=__version__,
        python_version=platform.python_version(),
    )
