"""Attack the raw-artifact counterpart path and require target-specific rejection."""

from __future__ import annotations

import argparse
import json
import stat
import tempfile
import warnings
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Any

from epsbench.freeze import (
    FreezeError,
    ImmutableArtifactBindingError,
    _domain_hash,
    _public_packet_evidence,
    _receipt_context,
    _safe_extract_artifact_archive,
    _validate_receipt_against_packet,
    validate_freeze_audit,
    validate_publication_record,
)
from epsbench.utils.canonical import canonical_json_bytes, sha256_bytes


def _expect_binding_rejected(
    label: str,
    expected_code: str,
    validator: Callable[[], Any],
) -> None:
    try:
        validator()
    except ImmutableArtifactBindingError as error:
        if error.code != expected_code:
            raise AssertionError(f"{label} reached {error.code}, not {expected_code}") from error
        print(f"counterpart artifact boundary rejected: {label} [{error.code}]", flush=True)
        return
    except Exception as error:
        raise AssertionError(
            f"{label} was rejected outside the immutable-artifact boundary: {error}"
        ) from error
    raise AssertionError(f"counterpart adversarial case was accepted: {label}")


def _expect_precise_rejection(
    label: str,
    expected_message: str,
    validator: Callable[[], Any],
) -> None:
    try:
        validator()
    except FreezeError as error:
        if expected_message not in str(error):
            raise AssertionError(
                f"{label} reached an unrelated validation error: {error}"
            ) from error
        print(f"counterpart exact validator rejected: {label} [{expected_message}]", flush=True)
        return
    except Exception as error:
        raise AssertionError(f"{label} reached an unrelated exception: {error}") from error
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


@contextmanager
def _temporary_extra_directory(root: Path) -> Iterator[None]:
    extra = root / "packet"
    extra.mkdir()
    try:
        yield
    finally:
        extra.rmdir()


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


def _zip_tree(root: Path) -> bytes:
    payload = BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            archive.write(path, path.relative_to(root).as_posix())
    return payload.getvalue()


def _zip_evidence(
    receipt: dict[str, Any],
    publication: dict[str, Any],
    *,
    compression: int = zipfile.ZIP_DEFLATED,
) -> bytes:
    payload = BytesIO()
    with zipfile.ZipFile(payload, "w", compression=compression) as archive:
        archive.writestr("publication_record.json", canonical_json_bytes(publication) + b"\n")
        archive.writestr("renderer_receipt.json", canonical_json_bytes(receipt) + b"\n")
    return payload.getvalue()


def _zip_special_member(name: str, file_type: int) -> bytes:
    payload = BytesIO()
    member = zipfile.ZipInfo(name)
    member.create_system = 3
    member.external_attr = (file_type | 0o644) << 16
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr(member, b"target")
    return payload.getvalue()


def _metadata_for_archive(path: Path, archive: bytes) -> bytes:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["size_in_bytes"] = len(archive)
    metadata["digest"] = f"sha256:{sha256_bytes(archive)}"
    return canonical_json_bytes(metadata) + b"\n"


@contextmanager
def _replace_local_counterpart_snapshots(
    local_root: Path,
    receipt: dict[str, Any],
    publication: dict[str, Any],
) -> Iterator[None]:
    receipt_bytes = canonical_json_bytes(receipt) + b"\n"
    publication_bytes = canonical_json_bytes(publication) + b"\n"
    packet_path = local_root / "freeze_candidate_packet.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet["snapshot_file_sha256"]["counterpart_renderer_receipt.json"] = sha256_bytes(
        receipt_bytes
    )
    packet["snapshot_file_sha256"]["counterpart_publication_record.json"] = sha256_bytes(
        publication_bytes
    )
    _rehash_packet(packet)
    with (
        _replace(local_root / "counterpart_renderer_receipt.json", receipt_bytes),
        _replace(local_root / "counterpart_publication_record.json", publication_bytes),
        _replace(packet_path, canonical_json_bytes(packet) + b"\n"),
    ):
        yield


def _archive_safety_cases() -> tuple[tuple[str, bytes, str], ...]:
    def archive(entries: list[tuple[str, bytes]]) -> bytes:
        payload = BytesIO()
        with zipfile.ZipFile(payload, "w") as zipped:
            for name, content in entries:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    zipped.writestr(name, content)
        return payload.getvalue()

    return (
        (
            "evidence archive extra member",
            archive(
                [
                    ("renderer_receipt.json", b"{}\n"),
                    ("publication_record.json", b"{}\n"),
                    ("extra.json", b"{}\n"),
                ]
            ),
            "artifact_archive_member_set_mismatch",
        ),
        (
            "evidence archive missing member",
            archive([("renderer_receipt.json", b"{}\n")]),
            "artifact_archive_member_set_mismatch",
        ),
        (
            "evidence archive traversal member",
            archive([("../publication_record.json", b"{}\n")]),
            "artifact_archive_unsafe_member",
        ),
        (
            "evidence archive absolute member",
            archive([("/publication_record.json", b"{}\n")]),
            "artifact_archive_unsafe_member",
        ),
        (
            "evidence archive case collision",
            archive(
                [
                    ("publication_record.json", b"{}\n"),
                    ("Publication_Record.json", b"{}\n"),
                ]
            ),
            "artifact_archive_colliding_member",
        ),
        (
            "evidence archive duplicate member",
            archive(
                [
                    ("publication_record.json", b"{}\n"),
                    ("publication_record.json", b'{"duplicate":true}\n'),
                ]
            ),
            "artifact_archive_colliding_member",
        ),
        (
            "evidence archive file/directory collision",
            archive([("publication_record.json", b"{}\n"), ("publication_record.json/x", b"x")]),
            "artifact_archive_colliding_member",
        ),
        (
            "evidence archive symbolic link",
            _zip_special_member("renderer_receipt.json", stat.S_IFLNK),
            "artifact_archive_unsafe_member",
        ),
    )


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
    packet_archive_path = evidence / "packet_artifact.zip"
    evidence_archive_path = evidence / "evidence_artifact.zip"
    packet_metadata_path = evidence / "packet_live_artifact.json"
    evidence_metadata_path = evidence / "evidence_live_artifact.json"
    packet_metadata = json.loads(packet_metadata_path.read_text(encoding="utf-8"))
    packet_archive = packet_archive_path.read_bytes()
    evidence_archive = evidence_archive_path.read_bytes()

    same_size_substitution = bytearray(packet_archive)
    same_size_substitution[-1] ^= 1
    with _replace(packet_archive_path, bytes(same_size_substitution)):
        _expect_binding_rejected(
            "correct packet artifact ID with incorrect same-size archive bytes",
            "artifact_archive_digest_mismatch",
            validate,
        )

    with _temporarily_unavailable(packet_archive_path):
        _expect_binding_rejected(
            "packet artifact disappearance during resolution",
            "counterpart_bundle_invalid",
            validate,
        )

    expired = dict(packet_metadata)
    expired["expired"] = True
    with _replace(packet_metadata_path, canonical_json_bytes(expired) + b"\n"):
        _expect_binding_rejected(
            "packet artifact expiry during resolution",
            "artifact_expired",
            validate,
        )

    with _replace(packet_archive_path, evidence_archive):
        _expect_binding_rejected(
            "packet and evidence archive cross-wiring",
            "artifact_archive_size_mismatch",
            validate,
        )

    with _temporary_extra_directory(evidence):
        _expect_binding_rejected(
            "caller-supplied extracted directory claiming the live artifact",
            "counterpart_bundle_invalid",
            validate,
        )

    with tempfile.TemporaryDirectory(prefix="epsbench-full-reseal-attack-") as temporary:
        root = Path(temporary)
        alternate_packet_root = root / "alternate-packet"
        original_evidence_root = root / "original-evidence"
        _safe_extract_artifact_archive(
            packet_archive,
            alternate_packet_root,
            "adversarial alternate packet setup",
        )
        _safe_extract_artifact_archive(
            evidence_archive,
            original_evidence_root,
            "adversarial evidence setup",
            expected_members={"renderer_receipt.json", "publication_record.json"},
        )
        run_path = alternate_packet_root / "run.json"
        run = json.loads(run_path.read_text(encoding="utf-8"))
        run["hostname"] = f"{run['hostname']}-independently-valid-alternate"
        run_path.write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        alternate_packet = validate_freeze_audit(alternate_packet_root)
        alternate_identity = _public_packet_evidence(alternate_packet_root, alternate_packet)

        receipt = json.loads(
            (original_evidence_root / "renderer_receipt.json").read_text(encoding="utf-8")
        )
        publication = json.loads(
            (original_evidence_root / "publication_record.json").read_text(encoding="utf-8")
        )
        if alternate_identity == publication["packet_identity"]:
            raise AssertionError("alternate packet did not obtain a distinct exact tree identity")
        receipt["public_qualification_evidence"] = alternate_identity
        _reseal_receipt(receipt)
        publication["packet_identity"] = alternate_identity
        _reseal_publication(publication)

        definition, seeds, lock = _receipt_context(alternate_packet_root)
        _validate_receipt_against_packet(
            receipt,
            alternate_packet_root,
            alternate_packet,
            definition,
            seeds,
            lock,
        )
        validate_publication_record(
            publication,
            alternate_packet_root,
            alternate_packet,
            packet_metadata,
            json.loads((evidence / "packet_live_workflow_run.json").read_text()),
            json.loads((evidence / "packet_live_workflow_jobs.json").read_text()),
        )
        print(
            "alternate packet, receipt, publication record, packet tree, and readiness "
            "dependencies independently validate before substitution",
            flush=True,
        )

        alternate_packet_archive = _zip_tree(alternate_packet_root)
        alternate_evidence_archive = _zip_evidence(receipt, publication)
        with (
            _replace(packet_archive_path, alternate_packet_archive),
            _replace(evidence_archive_path, alternate_evidence_archive),
            _replace_local_counterpart_snapshots(local, receipt, publication),
        ):
            _expect_binding_rejected(
                "different valid packet with jointly resealed evidence and retained live digest",
                "artifact_archive_size_mismatch"
                if len(alternate_packet_archive) != len(packet_archive)
                else "artifact_archive_digest_mismatch",
                validate,
            )

        reencoded_evidence = _zip_evidence(
            json.loads(
                (original_evidence_root / "renderer_receipt.json").read_text(encoding="utf-8")
            ),
            json.loads(
                (original_evidence_root / "publication_record.json").read_text(encoding="utf-8")
            ),
            compression=zipfile.ZIP_STORED,
        )
        if reencoded_evidence == evidence_archive:
            raise AssertionError("substituted evidence archive did not change raw bytes")
        with _replace(evidence_archive_path, reencoded_evidence):
            _expect_binding_rejected(
                "substituted evidence archive with the same valid evidence records",
                "artifact_archive_size_mismatch"
                if len(reencoded_evidence) != len(evidence_archive)
                else "artifact_archive_digest_mismatch",
                validate,
            )

    for label, malicious_archive, expected_code in _archive_safety_cases():
        with (
            _replace(evidence_archive_path, malicious_archive),
            _replace(
                evidence_metadata_path,
                _metadata_for_archive(evidence_metadata_path, malicious_archive),
            ),
        ):
            _expect_binding_rejected(label, expected_code, validate)

    original_receipt: dict[str, Any]
    original_publication: dict[str, Any]
    with tempfile.TemporaryDirectory(prefix="epsbench-record-pairing-") as temporary:
        extracted = Path(temporary) / "evidence"
        _safe_extract_artifact_archive(
            evidence_archive,
            extracted,
            "record-pairing setup",
            expected_members={"renderer_receipt.json", "publication_record.json"},
        )
        original_receipt = json.loads((extracted / "renderer_receipt.json").read_text())
        original_publication = json.loads((extracted / "publication_record.json").read_text())

    changed_publication = json.loads(json.dumps(original_publication))
    changed_publication["packet_identity"]["packet_tree_root_sha256"] = "0" * 64
    _reseal_publication(changed_publication)
    mismatched_records = _zip_evidence(original_receipt, changed_publication)
    with (
        _replace(evidence_archive_path, mismatched_records),
        _replace(
            evidence_metadata_path,
            _metadata_for_archive(evidence_metadata_path, mismatched_records),
        ),
        _replace_local_counterpart_snapshots(local, original_receipt, changed_publication),
    ):
        _expect_precise_rejection(
            "correct receipt paired with publication record for a different packet",
            "public packet publication differs from validated packet",
            validate,
        )

    validate()
    print("all target-specific raw-artifact counterpart regressions passed", flush=True)


if __name__ == "__main__":
    main()
