"""Synthetic live-evidence mutations; no frozen source rendering or topology derivation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import scripts.check_topology_publication as publication
from epsbench.schema import RendererProvenance
from epsbench.topology_qualification import TopologyPacket, packet_results
from epsbench.utils.canonical import sha256_bytes, write_canonical_json
from tests.unit.test_topology_qualification import synthetic_cells


@pytest.fixture
def synthetic_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, Any]]:
    rows = synthetic_cells()
    packet = TopologyPacket(
        schema_version="component_topology_qualification_packet_v1",
        implementation_head="a" * 40,
        implementation_tree="b" * 40,
        lock_commit="c" * 40,
        lock_tree="d" * 40,
        lock_root="1" * 64,
        remote_readback="a" * 40,
        environment_id="windows_wgl_locked",
        renderer=RendererProvenance(
            mujoco_version="3.12.0",
            numpy_version="2.4.6",
            renderer="mujoco.Renderer",
            backend="wgl-default",
            operating_system="Windows",
        ),
        cells=rows,
        **packet_results(rows),
        packet_sha256="2" * 64,
        scientific_result=None,
        gate_advancement=False,
    )
    monkeypatch.setattr(publication, "validate_topology_packet", lambda root: packet)
    monkeypatch.setattr(publication, "_safe_extract_artifact_archive", lambda *args: None)
    archive = b"synthetic archive; archive safety is independently covered by PR17 tests"
    (tmp_path / "artifact.zip").write_bytes(archive)
    repo = publication.REPOSITORY
    records: dict[str, Any] = {
        "live_artifact.json": {
            "id": 7,
            "name": f"component-topology-wgl-packet-{'a' * 40}",
            "digest": "sha256:" + sha256_bytes(archive),
            "size_in_bytes": len(archive),
            "expired": False,
            "workflow_run": {"id": 8, "head_sha": "a" * 40},
            "url": f"https://api.github.com/repos/{repo}/actions/artifacts/7",
        },
        "live_repository.json": {
            "full_name": repo,
            "url": f"https://api.github.com/repos/{repo}",
            "html_url": f"https://github.com/{repo}",
            "visibility": "private",
            "private": True,
            "archived": False,
            "disabled": False,
        },
        "live_workflow_run.json": {
            "id": 8,
            "head_sha": "a" * 40,
            "event": "pull_request",
            "repository": {"full_name": repo},
            "html_url": f"https://github.com/{repo}/actions/runs/8",
        },
        "live_workflow_jobs.json": {
            "jobs": [{"id": 9, "name": "qualify-wgl", "run_id": 8, "head_sha": "a" * 40}]
        },
    }
    for name, value in records.items():
        write_canonical_json(tmp_path / name, value)
    return tmp_path, records


@pytest.mark.parametrize(
    "mutation",
    [
        "digest",
        "size",
        "expiry",
        "artifact_name",
        "head",
        "run",
        "repository",
        "public",
        "job",
        "duplicate_job",
        "none",
    ],
)
def test_live_archive_identity_binding(
    synthetic_publication: tuple[Path, dict[str, Any]], mutation: str
) -> None:
    root, records = synthetic_publication
    if mutation == "digest":
        records["live_artifact.json"]["digest"] = "sha256:" + "0" * 64
    elif mutation == "size":
        records["live_artifact.json"]["size_in_bytes"] += 1
    elif mutation == "expiry":
        records["live_artifact.json"]["expired"] = True
    elif mutation == "artifact_name":
        records["live_artifact.json"]["name"] = "foreign"
    elif mutation == "head":
        records["live_workflow_run.json"]["head_sha"] = "b" * 40
    elif mutation == "run":
        records["live_artifact.json"]["workflow_run"]["id"] = 999
    elif mutation == "repository":
        records["live_workflow_run.json"]["repository"]["full_name"] = "foreign/repository"
    elif mutation == "public":
        records["live_repository.json"].update({"private": False, "visibility": "public"})
    elif mutation == "job":
        records["live_workflow_jobs.json"]["jobs"][0]["name"] = "foreign-job"
    elif mutation == "duplicate_job":
        records["live_workflow_jobs.json"]["jobs"] *= 2
    for name, value in records.items():
        write_canonical_json(root / name, value)
    if mutation == "none":
        result = publication.check_publication(root, root / "receipt.json")
        assert result["artifact_id"] == 7 and result["job_id"] == 9
        assert result["all_192_sources_reconstructed"] is True
        with pytest.raises(FileExistsError):
            publication.check_publication(root, root / "receipt.json")
    else:
        with pytest.raises(ValueError):
            publication.check_publication(root, root / "receipt.json")
        assert not (root / "receipt.json").exists()
