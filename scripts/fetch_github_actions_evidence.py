"""Resolve live GitHub Actions metadata and download exact raw artifact bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from epsbench.github import github_repository_access
from epsbench.utils.canonical import canonical_json_bytes

_REPOSITORY = "yurifrusin/ecological-predictive-states"
_API_HEADERS = {
    "Accept": "application/vnd.github+json",
    "User-Agent": "epsbench-github-evidence-fetcher",
    "X-GitHub-Api-Version": "2022-11-28",
}


def _fetch(url: str, token: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={**_API_HEADERS, "Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, urllib.error.HTTPError) as error:
        raise RuntimeError(f"GitHub Actions evidence is unavailable: {url}") from error
    if type(payload) is not dict:
        raise RuntimeError(f"GitHub Actions evidence is not an object: {url}")
    return payload


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


def _download_archive(url: str, token: str, expected_size: int) -> bytes:
    authenticated = urllib.request.Request(
        url,
        headers={**_API_HEADERS, "Authorization": f"Bearer {token}"},
    )
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        opener.open(authenticated, timeout=60)
    except urllib.error.HTTPError as redirect:
        if redirect.code not in {301, 302, 303, 307, 308}:
            raise RuntimeError("GitHub artifact download endpoint is unavailable") from redirect
        location = redirect.headers.get("Location")
    else:
        raise RuntimeError("GitHub artifact download did not use an isolated signed redirect")
    if type(location) is not str:
        raise RuntimeError("GitHub artifact download redirect is unavailable")
    parsed = urlsplit(location)
    if (
        parsed.scheme != "https"
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise RuntimeError("GitHub artifact download redirect is unsafe")
    unauthenticated = urllib.request.Request(
        location,
        headers={"User-Agent": _API_HEADERS["User-Agent"]},
    )
    try:
        with urllib.request.urlopen(unauthenticated, timeout=120) as response:
            payload: bytes = response.read(expected_size + 1)
    except (OSError, urllib.error.HTTPError) as error:
        raise RuntimeError("GitHub artifact archive bytes are unavailable") from error
    if len(payload) != expected_size:
        raise RuntimeError("downloaded artifact archive size differs from live metadata")
    return payload


def _artifact_identity(artifact: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "id",
        "name",
        "size_in_bytes",
        "digest",
        "expired",
        "created_at",
        "expires_at",
        "url",
        "archive_download_url",
        "workflow_run",
    )
    return {field: artifact.get(field) for field in fields}


def _require_stable_repository_resolution(
    before: dict[str, Any],
    after: dict[str, Any],
    repository: str,
) -> dict[str, Any]:
    try:
        before_access = github_repository_access(before, repository)
        after_access = github_repository_access(after, repository)
    except ValueError as error:
        raise RuntimeError("GitHub repository access metadata is invalid") from error
    if before_access != after_access:
        raise RuntimeError(
            "GitHub repository visibility or availability changed during live resolution"
        )
    return after


def _output_names(role: str) -> tuple[str, str, str, str, str]:
    if role == "publication":
        return (
            "artifact.zip",
            "live_artifact.json",
            "live_repository.json",
            "live_workflow_run.json",
            "live_workflow_jobs.json",
        )
    return (
        f"{role}_artifact.zip",
        f"{role}_live_artifact.json",
        f"{role}_live_repository.json",
        f"{role}_live_workflow_run.json",
        f"{role}_live_workflow_jobs.json",
    )


def _write_exclusive(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as output:
            output.write(payload)
    except FileExistsError:
        raise FileExistsError(f"live evidence already exists: {path}") from None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True)
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--artifact-id", required=True, type=int)
    parser.add_argument(
        "--role",
        choices=("publication", "packet", "evidence"),
        default="publication",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.repository != _REPOSITORY:
        raise RuntimeError("publication evidence repository is outside the authorised scope")
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN is required to resolve GitHub CI evidence")
    repository_endpoint = f"https://api.github.com/repos/{args.repository}"
    repository_before = _fetch(repository_endpoint, token)
    base = f"https://api.github.com/repos/{args.repository}/actions"
    artifact_endpoint = f"{base}/artifacts/{args.artifact_id}"
    artifact = _fetch(artifact_endpoint, token)
    run = _fetch(f"{base}/runs/{args.run_id}", token)
    run_attempt = run.get("run_attempt")
    head_sha = run.get("head_sha")
    artifact_run = artifact.get("workflow_run")
    if (
        type(run_attempt) is not int
        or run_attempt <= 0
        or type(head_sha) is not str
        or type(artifact_run) is not dict
        or artifact_run.get("id") != args.run_id
        or artifact_run.get("head_sha") != head_sha
        or artifact.get("expired") is not False
        or type(artifact.get("size_in_bytes")) is not int
        or artifact["size_in_bytes"] <= 0
        or type(artifact.get("archive_download_url")) is not str
    ):
        raise RuntimeError("workflow run or artifact lacks an exact live identity")
    archive = _download_archive(
        artifact["archive_download_url"],
        token,
        artifact["size_in_bytes"],
    )
    resolved_again = _fetch(artifact_endpoint, token)
    if resolved_again.get("expired") is not False or _artifact_identity(
        resolved_again
    ) != _artifact_identity(artifact):
        raise RuntimeError("artifact disappeared, expired, or changed during live resolution")
    digest = artifact.get("digest")
    if type(digest) is not str or not digest.startswith("sha256:"):
        raise RuntimeError("live artifact digest is unavailable")
    if hashlib.sha256(archive).hexdigest() != digest.removeprefix("sha256:"):
        raise RuntimeError("downloaded artifact archive digest differs from live metadata")
    jobs_endpoint = f"{base}/runs/{args.run_id}/attempts/{run_attempt}/jobs?per_page=100"
    jobs = _fetch(jobs_endpoint, token)
    jobs["resolved_workflow_run_id"] = args.run_id
    jobs["resolved_workflow_run_attempt"] = run_attempt
    jobs["resolved_head_sha"] = head_sha
    jobs["resolution_endpoint"] = jobs_endpoint
    repository_after = _fetch(repository_endpoint, token)
    repository = _require_stable_repository_resolution(
        repository_before,
        repository_after,
        args.repository,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    archive_name, artifact_name, repository_name, run_name, jobs_name = _output_names(args.role)
    for name, payload in (
        (archive_name, archive),
        (artifact_name, canonical_json_bytes(resolved_again) + b"\n"),
        (repository_name, canonical_json_bytes(repository) + b"\n"),
        (run_name, canonical_json_bytes(run) + b"\n"),
        (jobs_name, canonical_json_bytes(jobs) + b"\n"),
    ):
        _write_exclusive(args.output / name, payload)


if __name__ == "__main__":
    main()
