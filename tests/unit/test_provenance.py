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
from epsbench.schema import (
    GitAvailabilityStatus,
    canonical_github_repository_identity,
    sanitize_git_repository,
)


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


@pytest.mark.parametrize(
    ("origin", "expected"),
    [
        (
            "https://user:secret@example.invalid/org/repo.git?access_token=secret#private",
            "https://example.invalid/org/repo.git",
        ),
        (
            "ssh://git:secret@example.invalid:2222/org/repo.git",
            "ssh://example.invalid:2222/org/repo.git",
        ),
        ("git@example.invalid:org/repo.git", "ssh://example.invalid/org/repo.git"),
        ("file:///C:/Users/private/repo", "local-repository-redacted"),
        ("C:/Users/private/repo", "local-repository-redacted"),
        ("../private/repo", "local-repository-redacted"),
    ],
)
def test_git_origin_is_sanitized_before_serialization(
    provenance_repository: Path,
    origin: str,
    expected: str,
) -> None:
    _git(provenance_repository, "remote", "set-url", "origin", origin)
    provenance = collect_source_provenance(provenance_repository)
    assert provenance.git_repository == expected
    serialized = provenance.model_dump_json()
    assert "secret" not in serialized
    assert "Users/private" not in serialized
    assert "../private" not in serialized


@pytest.mark.parametrize(
    "origin",
    [
        "https://github.com/yurifrusin/ecological-predictive-states",
        "https://github.com/yurifrusin/ecological-predictive-states.git",
        "HTTPS://GITHUB.COM/YuriFrusin/Ecological-Predictive-States.git",
    ],
)
def test_equivalent_github_https_origins_have_one_repository_identity(
    provenance_repository: Path,
    origin: str,
) -> None:
    _git(provenance_repository, "remote", "set-url", "origin", origin)
    provenance = collect_source_provenance(provenance_repository)
    assert provenance.git_repository == "yurifrusin/ecological-predictive-states"
    assert canonical_github_repository_identity(origin) == provenance.git_repository


@pytest.mark.parametrize(
    "origin",
    [
        "https://github.com/another-owner/ecological-predictive-states",
        "https://github.com/yurifrusin/another-repository",
    ],
)
def test_different_github_repositories_have_different_canonical_identities(origin: str) -> None:
    assert canonical_github_repository_identity(origin) != (
        "yurifrusin/ecological-predictive-states"
    )


@pytest.mark.parametrize(
    "origin",
    [
        "https://github.example/yurifrusin/ecological-predictive-states",
        "https://github.com/yurifrusin/ecological-predictive-states/extra",
        "https://user@github.com/yurifrusin/ecological-predictive-states",
        "https://token:secret@github.com/yurifrusin/ecological-predictive-states",
        "https://github.com/yurifrusin/ecological-predictive-states?token=secret",
        "https://github.com/yurifrusin/ecological-predictive-states#fragment",
        "https://github.com.evil.invalid/yurifrusin/ecological-predictive-states",
        "https://github.com/yurifrusin/ecological-predictive-states.git/extra",
    ],
)
def test_github_repository_identity_rejects_ambiguous_references(
    origin: str,
) -> None:
    with pytest.raises(ValueError):
        canonical_github_repository_identity(origin)
    assert sanitize_git_repository(origin) != "yurifrusin/ecological-predictive-states"
