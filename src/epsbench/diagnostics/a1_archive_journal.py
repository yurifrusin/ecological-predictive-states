"""Small evaluator backing on the existing retained application-byte ledger."""

from __future__ import annotations

from epsbench.diagnostics.a1_retention import RetainedArchive


class ArchiveJournal:
    def __init__(self, archive: RetainedArchive) -> None:
        self.archive = archive

    def exists(self, name: str) -> bool:
        return any(path == "journal/" + name for path, _, _ in self.archive.entries())

    def read(self, name: str) -> str:
        path = "journal/" + name
        entries = {p: n for p, n, _ in self.archive.entries()}
        size = entries[path]
        if size > 4 * 1024 * 1024:
            raise ValueError("journal read exceeds bound")
        self.archive.reserve_copy(f"journal-reads/{self.archive.sequence:08d}", size)
        return b"".join(self.archive.read_chunks(path)).decode()

    def write(self, name: str, content: str) -> None:
        self.archive.put("journal/" + name, content.encode())

    def append(self, name: str, content: str) -> None:
        prefix = "journal/" + name + "/"
        index = sum(path.startswith(prefix) for path, _, _ in self.archive.entries())
        self.archive.put(f"{prefix}{index:08d}", content.encode())
