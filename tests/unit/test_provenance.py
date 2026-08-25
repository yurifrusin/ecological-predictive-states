from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from epsbench.data.provenance import (
    EXPECTED_GOVERNING_HASHES,
    GitState,
    GitUnavailableError,
    ProvenanceError,
    collect_source_provenance,
)
from epsbench.schema import GitAvailabilityStatus


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
    )


@pytest.fixture
def provenance_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "source"
    repository.mkdir()
    for relative_path in (*EXPECTED_GOVERNING_HASHES, "uv.lock"):
        source = Path(relative_path)
        destination = repository / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    _git(repository, "init")
    _git(repository, "config", "user.email", "epsbench@example.invalid")
    _git(repository, "config", "user.name", "EPS Test")
    _git(repository, "remote", "add", "origin", "https://example.invalid/epsbench.git")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "fixture")
    return repository


def test_clean_and_dirty_git_provenance_is_truthful(provenance_repository: Path) -> None:
    clean = collect_source_provenance(provenance_repository)
    assert clean.git_availability_status == GitAvailabilityStatus.AVAILABLE
    assert clean.git_dirty is False
    assert clean.dirty_diff_sha256 is None

    (provenance_repository / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    dirty = collect_source_provenance(provenance_repository)
    assert dirty.git_dirty is True
    assert dirty.dirty_diff_sha256 is not None
    assert dirty.git_commit == clean.git_commit


def test_changed_lock_file_changes_recorded_lock_identity(provenance_repository: Path) -> None:
    before = collect_source_provenance(provenance_repository)
    lock_path = provenance_repository / "uv.lock"
    lock_path.write_text(lock_path.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")
    after = collect_source_provenance(provenance_repository)
    assert after.uv_lock_sha256 != before.uv_lock_sha256
    assert after.git_dirty is True


def test_changed_governing_document_is_rejected(provenance_repository: Path) -> None:
    charter = provenance_repository / "docs" / "RESEARCH_CHARTER.md"
    charter.write_text(charter.read_text(encoding="utf-8") + "changed\n", encoding="utf-8")
    with pytest.raises(ProvenanceError, match="governing document hash mismatch"):
        collect_source_provenance(provenance_repository)


def test_unavailable_git_is_not_reported_as_clean(provenance_repository: Path) -> None:
    class UnavailableProvider:
        def collect(self, source_root: Path) -> GitState:
            raise GitUnavailableError(f"Git unavailable for {source_root.name}")

    provenance = collect_source_provenance(
        provenance_repository,
        git_provider=UnavailableProvider(),
    )
    assert provenance.git_availability_status == GitAvailabilityStatus.UNAVAILABLE
    assert provenance.git_dirty is None
    assert provenance.git_commit is None
    assert provenance.git_unavailable_reason == "Git unavailable for source"
