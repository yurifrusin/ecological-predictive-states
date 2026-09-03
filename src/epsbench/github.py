"""Typed GitHub repository visibility and Actions artifact-access metadata."""

from __future__ import annotations

from typing import Any, Literal, TypedDict, cast

from epsbench.schema import canonical_github_repository_identity

RepositoryVisibility = Literal["private", "public", "internal"]
ArtifactAccessMechanism = Literal[
    "connected_authenticated_github_actions",
    "public_github_actions",
]


class GitHubRepositoryAccess(TypedDict):
    repository: str
    repository_api_url: str
    repository_html_url: str
    repository_visibility: RepositoryVisibility
    repository_private: bool
    repository_archived: bool
    repository_disabled: bool
    artifact_access_mechanism: ArtifactAccessMechanism


def artifact_access_mechanism_for_visibility(
    visibility: RepositoryVisibility,
) -> ArtifactAccessMechanism:
    """Return the review access mechanism implied by exact repository visibility."""

    if visibility == "public":
        return "public_github_actions"
    return "connected_authenticated_github_actions"


def github_repository_access(
    metadata: dict[str, Any],
    expected_repository: str,
) -> GitHubRepositoryAccess:
    """Validate live GitHub repository metadata and derive one typed access posture."""

    try:
        canonical_expected = canonical_github_repository_identity(expected_repository)
    except ValueError as error:
        raise ValueError("expected GitHub repository identity is invalid") from error
    if canonical_expected != expected_repository:
        raise ValueError("expected GitHub repository identity is not canonical")

    full_name = metadata.get("full_name")
    api_url = metadata.get("url")
    html_url = metadata.get("html_url")
    visibility = metadata.get("visibility")
    private = metadata.get("private")
    archived = metadata.get("archived")
    disabled = metadata.get("disabled")
    expected_api_url = f"https://api.github.com/repos/{expected_repository}"
    try:
        canonical_full_name = (
            canonical_github_repository_identity(full_name) if type(full_name) is str else None
        )
        canonical_html_url = (
            canonical_github_repository_identity(html_url) if type(html_url) is str else None
        )
    except ValueError as error:
        raise ValueError("live GitHub repository identity is malformed") from error
    if (
        canonical_full_name != expected_repository
        or canonical_html_url != expected_repository
        or api_url != expected_api_url
        or visibility not in {"private", "public", "internal"}
        or type(private) is not bool
        or type(archived) is not bool
        or type(disabled) is not bool
    ):
        raise ValueError("live GitHub repository identity or visibility is incomplete")
    if visibility == "public" and private:
        raise ValueError("public GitHub repository metadata cannot be private")
    if visibility == "private" and not private:
        raise ValueError("private GitHub repository metadata must carry private=true")
    if archived or disabled:
        raise ValueError("live GitHub repository is archived or disabled")

    typed_visibility = cast(RepositoryVisibility, visibility)
    return {
        "repository": expected_repository,
        "repository_api_url": expected_api_url,
        "repository_html_url": f"https://github.com/{expected_repository}",
        "repository_visibility": typed_visibility,
        "repository_private": private,
        "repository_archived": archived,
        "repository_disabled": disabled,
        "artifact_access_mechanism": artifact_access_mechanism_for_visibility(typed_visibility),
    }
