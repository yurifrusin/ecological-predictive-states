"""Attack the real packet-bound counterpart evidence path and require fail-closed rejection."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from epsbench.freeze import _domain_hash, validate_freeze_audit
from epsbench.utils.canonical import canonical_json_bytes, sha256_bytes


def _expect_rejected(label: str, validator: Callable[[], Any]) -> None:
    try:
        validator()
    except ValueError:
        print(f"counterpart adversarial case rejected: {label}", flush=True)
        return
    raise AssertionError(f"counterpart adversarial case was accepted: {label}")


@contextmanager
def _replace(path: Path, replacement: bytes) -> Iterator[None]:
    original = path.read_bytes()
    path.write_bytes(replacement)
    try:
        yield
    finally:
        path.write_bytes(original)


@contextmanager
def _temporarily_unavailable(path: Path) -> Iterator[None]:
    moved = path.with_name(f"{path.name}.temporarily-unavailable")
    path.rename(moved)
    try:
        yield
    finally:
        moved.rename(path)


def _rehash_packet(packet: dict[str, Any]) -> None:
    logical = dict(packet)
    logical.pop("complete_packet_root_sha256")
    packet["complete_packet_root_sha256"] = _domain_hash("complete_packet", logical)


def _reseal_receipt(receipt: dict[str, Any]) -> None:
    domain = dict(receipt)
    domain.pop("receipt_sha256")
    receipt["receipt_sha256"] = _domain_hash("renderer_receipt", domain)


def _reseal_publication(record: dict[str, Any]) -> None:
    domain = dict(record)
    domain.pop("record_sha256")
    record["record_sha256"] = sha256_bytes(canonical_json_bytes(domain))


@contextmanager
def _joint_snapshot_replacement(
    local_root: Path,
    evidence_root: Path,
    *,
    snapshot_name: str,
    evidence_name: str,
    payload: dict[str, Any],
) -> Iterator[None]:
    encoded = canonical_json_bytes(payload) + b"\n"
    packet_path = local_root / "freeze_candidate_packet.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet["snapshot_file_sha256"][snapshot_name] = sha256_bytes(encoded)
    _rehash_packet(packet)
    with (
        _replace(local_root / snapshot_name, encoded),
        _replace(evidence_root / evidence_name, encoded),
        _replace(packet_path, canonical_json_bytes(packet) + b"\n"),
    ):
        yield


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("local_packet", type=Path)
    parser.add_argument("counterpart_evidence", type=Path)
    args = parser.parse_args()
    local = args.local_packet
    evidence = args.counterpart_evidence

    def validate() -> dict[str, Any]:
        return validate_freeze_audit(local, evidence)

    validate()
    receipt_path = evidence / "renderer_receipt.json"
    original_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    publication_path = evidence / "publication_record.json"
    original_publication = json.loads(publication_path.read_text(encoding="utf-8"))

    receipt_mutations: tuple[tuple[str, Callable[[dict[str, Any]], None]], ...] = (
        (
            "source commit",
            lambda value: value.__setitem__("qualification_source_commit", "0" * 40),
        ),
        (
            "source tree",
            lambda value: value.__setitem__("qualification_source_tree", "0" * 40),
        ),
        (
            "complete packet root",
            lambda value: value.__setitem__("complete_packet_root_sha256", "0" * 64),
        ),
        (
            "packet file digest",
            lambda value: value["public_qualification_evidence"].__setitem__(
                "packet_file_sha256", "0" * 64
            ),
        ),
        (
            "packet tree digest",
            lambda value: value["public_qualification_evidence"].__setitem__(
                "packet_tree_root_sha256", "0" * 64
            ),
        ),
        (
            "outcome map",
            lambda value: value["selected_outcome_map"][0].__setitem__(
                "admission_status", "rejected"
            ),
        ),
        (
            "profile row",
            lambda value: value["profiles"][0].__setitem__("renderer_apparatus_qualified", False),
        ),
        (
            "portable root",
            lambda value: value["portable_apparatus_roots"].__setitem__(
                "procedural_asset_root_sha256", "0" * 64
            ),
        ),
        (
            "renderer-local root",
            lambda value: value["renderer_local_roots"].__setitem__(
                "renderer_specific_audit_root_sha256", "0" * 64
            ),
        ),
        (
            "threshold margin",
            lambda value: value["threshold_margin_summary"]["metrics"][
                "changed_controlled_pixel_fraction"
            ].__setitem__("minimum_margin", 999.0),
        ),
        (
            "authority flag",
            lambda value: value.__setitem__("benchmark_frozen", True),
        ),
        (
            "historical receipt schema",
            lambda value: value.__setitem__(
                "schema_version", "appearance_benchmark_renderer_qualification_receipt_v0"
            ),
        ),
    )
    for label, mutate in receipt_mutations:
        changed = json.loads(json.dumps(original_receipt))
        mutate(changed)
        _reseal_receipt(changed)
        with _joint_snapshot_replacement(
            local,
            evidence,
            snapshot_name="counterpart_renderer_receipt.json",
            evidence_name="renderer_receipt.json",
            payload=changed,
        ):
            _expect_rejected(f"jointly resealed receipt {label}", validate)

    changed = json.loads(json.dumps(original_receipt))
    changed["complete_packet_root_sha256"] = "1" * 64
    changed["selected_outcome_map"][0]["admission_status"] = "rejected"
    changed["profiles"][0]["renderer_apparatus_qualified"] = False
    changed["portable_apparatus_roots"]["procedural_asset_root_sha256"] = "2" * 64
    changed["renderer_local_roots"]["renderer_specific_audit_root_sha256"] = "3" * 64
    changed["threshold_margin_summary"]["metrics"]["changed_controlled_pixel_fraction"][
        "minimum_margin"
    ] = 999.0
    _reseal_receipt(changed)
    with _joint_snapshot_replacement(
        local,
        evidence,
        snapshot_name="counterpart_renderer_receipt.json",
        evidence_name="renderer_receipt.json",
        payload=changed,
    ):
        _expect_rejected("fully and jointly resealed renderer receipt", validate)

    publication_mutations: tuple[tuple[str, Callable[[dict[str, Any]], None]], ...] = (
        ("artifact ID", lambda value: value.__setitem__("artifact_id", value["artifact_id"] + 1)),
        ("artifact digest", lambda value: value.__setitem__("artifact_digest_sha256", "0" * 64)),
        ("workflow run", lambda value: value.__setitem__("workflow_run_id", 1)),
        ("workflow job", lambda value: value.__setitem__("job_database_id", 1)),
        ("source commit", lambda value: value.__setitem__("source_commit", "0" * 40)),
        ("source tree", lambda value: value.__setitem__("source_tree", "0" * 40)),
        (
            "packet tree",
            lambda value: value["packet_identity"].__setitem__("packet_tree_root_sha256", "0" * 64),
        ),
    )
    for label, mutate in publication_mutations:
        changed_record = json.loads(json.dumps(original_publication))
        mutate(changed_record)
        _reseal_publication(changed_record)
        with _joint_snapshot_replacement(
            local,
            evidence,
            snapshot_name="counterpart_publication_record.json",
            evidence_name="publication_record.json",
            payload=changed_record,
        ):
            _expect_rejected(f"jointly resealed publication {label}", validate)

    with _temporarily_unavailable(evidence / "packet"):
        _expect_rejected("unavailable public packet", validate)
    counterpart_packet = evidence / "packet"
    with _replace(counterpart_packet / "freeze_candidate_packet.json", b"{}\n"):
        _expect_rejected("substituted public packet", validate)
    with _temporarily_unavailable(counterpart_packet / "run.json"):
        _expect_rejected("incomplete public packet", validate)
    with _temporarily_unavailable(evidence / "live_artifact.json"):
        _expect_rejected("unavailable live artifact resolution", validate)

    live_artifact_path = evidence / "live_artifact.json"
    live_artifact = json.loads(live_artifact_path.read_text(encoding="utf-8"))
    live_artifact["expired"] = True
    with _replace(live_artifact_path, canonical_json_bytes(live_artifact) + b"\n"):
        _expect_rejected("expired public artifact", validate)

    validate()
    print("all packet-bound counterpart adversarial regressions passed", flush=True)


if __name__ == "__main__":
    main()
