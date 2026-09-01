"""Fetch live GitHub Actions artifact, run, and job metadata without logging credentials."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from epsbench.utils.canonical import canonical_json_bytes


def _fetch(url: str, token: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "epsbench-public-evidence-fetcher",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, urllib.error.HTTPError) as error:
        raise RuntimeError(f"GitHub Actions evidence is unavailable: {url}") from error
    if type(payload) is not dict:
        raise RuntimeError(f"GitHub Actions evidence is not an object: {url}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True)
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--artifact-id", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.repository != "yurifrusin/ecological-predictive-states":
        raise RuntimeError("publication evidence repository is outside the authorised scope")
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN is required to resolve public CI evidence")
    base = f"https://api.github.com/repos/{args.repository}/actions"
    artifact = _fetch(f"{base}/artifacts/{args.artifact_id}", token)
    run = _fetch(f"{base}/runs/{args.run_id}", token)
    run_attempt = run.get("run_attempt")
    head_sha = run.get("head_sha")
    if type(run_attempt) is not int or run_attempt <= 0 or type(head_sha) is not str:
        raise RuntimeError("workflow run lacks an exact attempt or source commit")
    jobs_endpoint = f"{base}/runs/{args.run_id}/attempts/{run_attempt}/jobs?per_page=100"
    jobs = _fetch(jobs_endpoint, token)
    jobs["resolved_workflow_run_id"] = args.run_id
    jobs["resolved_workflow_run_attempt"] = run_attempt
    jobs["resolved_head_sha"] = head_sha
    jobs["resolution_endpoint"] = jobs_endpoint
    args.output.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("live_artifact.json", artifact),
        ("live_workflow_run.json", run),
        ("live_workflow_jobs.json", jobs),
    ):
        destination = args.output / name
        if destination.exists():
            raise FileExistsError(f"live evidence already exists: {destination}")
        destination.write_bytes(canonical_json_bytes(payload) + b"\n")


if __name__ == "__main__":
    main()
