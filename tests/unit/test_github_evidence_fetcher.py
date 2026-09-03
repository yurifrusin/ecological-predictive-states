"""Regressions for live GitHub repository visibility and artifact access."""

from __future__ import annotations

from copy import deepcopy

import pytest

from epsbench.github import github_repository_access
from scripts.fetch_github_actions_evidence import _require_stable_repository_resolution

REPOSITORY = "yurifrusin/ecological-predictive-states"


def _repository_metadata(
    *,
    visibility: str = "private",
    private: bool = True,
) -> dict[str, object]:
    return {
        "full_name": REPOSITORY,
        "url": f"https://api.github.com/repos/{REPOSITORY}",
        "html_url": f"https://github.com/{REPOSITORY}",
        "visibility": visibility,
        "private": private,
        "archived": False,
        "disabled": False,
    }


def test_connected_private_repository_has_private_authenticated_access() -> None:
    access = github_repository_access(_repository_metadata(), REPOSITORY)
    assert access == {
        "repository": REPOSITORY,
        "repository_api_url": f"https://api.github.com/repos/{REPOSITORY}",
        "repository_html_url": f"https://github.com/{REPOSITORY}",
        "repository_visibility": "private",
        "repository_private": True,
        "repository_archived": False,
        "repository_disabled": False,
        "artifact_access_mechanism": "connected_authenticated_github_actions",
    }


def test_public_repository_has_public_actions_access() -> None:
    access = github_repository_access(
        _repository_metadata(visibility="public", private=False),
        REPOSITORY,
    )
    assert access["repository_visibility"] == "public"
    assert access["repository_private"] is False
    assert access["artifact_access_mechanism"] == "public_github_actions"


def test_repository_visibility_change_during_production_resolution_is_rejected() -> None:
    before = _repository_metadata()
    after = _repository_metadata(visibility="public", private=False)
    with pytest.raises(RuntimeError, match="visibility or availability changed"):
        _require_stable_repository_resolution(before, after, REPOSITORY)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("full_name", "another-owner/ecological-predictive-states"),
        ("full_name", "yurifrusin/another-repository"),
        (
            "html_url",
            "https://github.example/yurifrusin/ecological-predictive-states",
        ),
        (
            "html_url",
            "https://user@github.com/yurifrusin/ecological-predictive-states",
        ),
        (
            "html_url",
            "https://github.com/yurifrusin/ecological-predictive-states?token=x",
        ),
        (
            "html_url",
            "https://github.com/yurifrusin/ecological-predictive-states#fragment",
        ),
        (
            "html_url",
            "https://github.com/yurifrusin/ecological-predictive-states/extra",
        ),
    ],
)
def test_live_repository_resolution_rejects_other_or_ambiguous_identity(
    field: str,
    value: str,
) -> None:
    metadata = deepcopy(_repository_metadata())
    metadata[field] = value
    with pytest.raises(ValueError, match="repository identity"):
        github_repository_access(metadata, REPOSITORY)
