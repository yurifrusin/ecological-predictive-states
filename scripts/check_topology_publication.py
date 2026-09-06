"""Bind complete Slice 6 sources to digest-verified live private GitHub evidence."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from epsbench.data.paths import open_owned_regular_file
from epsbench.freeze import _safe_extract_artifact_archive, _verify_artifact_archive_payload
from epsbench.github import github_repository_access
from epsbench.topology_qualification import compare_topology_packets, validate_topology_packet
from epsbench.utils.canonical import sha256_bytes, write_canonical_json

REPOSITORY = "yurifrusin/ecological-predictive-states"


def _read(root: Path, name: str) -> bytes:
    with open_owned_regular_file(root, name) as owned:
        return owned.payload


def check_publication(
    evidence: Path, output: Path, compare_with: Path | None = None
) -> dict[str, Any]:
    archive = _read(evidence, "artifact.zip")
    artifact = json.loads(_read(evidence, "live_artifact.json"))
    repository = json.loads(_read(evidence, "live_repository.json"))
    run = json.loads(_read(evidence, "live_workflow_run.json"))
    jobs = json.loads(_read(evidence, "live_workflow_jobs.json"))
    access = github_repository_access(repository, REPOSITORY)
    if access["repository_private"] is not True or access["repository_archived"] is not False:
        raise ValueError(
            "topology evidence requires the connected active private repository posture"
        )
    _verify_artifact_archive_payload(archive, artifact, "component topology")
    with tempfile.TemporaryDirectory(prefix="eps-topology-publication-") as temporary:
        extracted = Path(temporary) / "packet"
        _safe_extract_artifact_archive(archive, extracted, "component topology")
        packet = validate_topology_packet(extracted)
        name = "wgl" if packet.environment_id == "windows_wgl_locked" else "osmesa"
        expected_name = f"component-topology-{name}-packet-{packet.implementation_head}"
        run_id = run.get("id")
        workflow_binding = artifact.get("workflow_run", {})
        if (
            artifact.get("name") != expected_name
            or artifact.get("expired") is not False
            or run.get("head_sha") != packet.implementation_head
            or run.get("event") != "pull_request"
            or run.get("repository", {}).get("full_name") != REPOSITORY
            or workflow_binding.get("id") != run_id
            or workflow_binding.get("head_sha") != packet.implementation_head
            or artifact.get("url")
            != f"https://api.github.com/repos/{REPOSITORY}/actions/artifacts/{artifact.get('id')}"
        ):
            raise ValueError("live topology artifact/run/repository/exact-head binding differs")
        matching = [
            job
            for job in jobs.get("jobs", [])
            if job.get("name") == f"qualify-{name}"
            and job.get("run_id") == run_id
            and job.get("head_sha") == packet.implementation_head
        ]
        if len(matching) != 1:
            raise ValueError("live topology artifact requires exactly one matching renderer job")
        result = {
            "schema_version": "component_topology_publication_receipt_v1",
            "repository_access": access,
            "artifact_id": artifact["id"],
            "artifact_name": artifact["name"],
            "artifact_digest_sha256": sha256_bytes(archive),
            "artifact_size": len(archive),
            "run_id": run_id,
            "run_url": run["html_url"],
            "job_id": matching[0]["id"],
            "implementation_head": packet.implementation_head,
            "implementation_tree": packet.implementation_tree,
            "packet_sha256": packet.packet_sha256,
            "lock_root": packet.lock_root,
            "environment": packet.environment_id,
            "qualification": packet.local_qualification,
            "all_192_sources_reconstructed": True,
            "comparison": compare_topology_packets(extracted, compare_with)
            if compare_with is not None
            else None,
            "scientific_result": None,
            "gate_advancement": False,
        }
    if output.exists():
        raise FileExistsError("topology publication receipt already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-with", type=Path)
    args = parser.parse_args()
    result = check_publication(args.evidence, args.output, args.compare_with)
    print(f"Verified artifact {result['artifact_id']}; qualification {result['qualification']}")


if __name__ == "__main__":
    main()
