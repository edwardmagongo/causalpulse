"""Records keyed by what produced them, one JSON file each.

A paid step stores each result under a digest of everything that produced
it, and skips any digest already stored, so an interrupted run resumes
without paying twice. A record is never overwritten: the store holds what was
measured, and a second, different result for the same key would be a defect
in the caller.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path


def digest(fields: Mapping[str, object]) -> str:
    """A short, stable digest of a record's key fields."""
    canonical = json.dumps(dict(fields), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


class Store:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, kind: str, key: str) -> Path:
        return self.root / kind / f"{key}.json"

    def has(self, kind: str, key: str) -> bool:
        return self._path(kind, key).exists()

    def put(self, kind: str, key: str, record: Mapping[str, object]) -> None:
        path = self._path(kind, key)
        if path.exists():
            raise FileExistsError(f"{kind}/{key} is already stored")
        path.parent.mkdir(parents=True, exist_ok=True)
        # Written beside the target and renamed into place, so a reader (or a
        # resume after a crash) never sees half a record.
        partial = path.with_suffix(f".{os.getpid()}.partial")
        partial.write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(partial, path)

    def get(self, kind: str, key: str) -> dict[str, object]:
        loaded = json.loads(self._path(kind, key).read_text(encoding="utf-8"))
        assert isinstance(loaded, dict)
        return loaded

    def records(self, kind: str) -> list[dict[str, object]]:
        """Every record of `kind`, ordered by key."""
        directory = self.root / kind
        if not directory.exists():
            return []
        return [self.get(kind, path.stem) for path in sorted(directory.glob("*.json"))]

    def export(self, kind: str, path: Path) -> int:
        """Write every record of `kind` as gzipped JSON lines; return the count.

        The bytes depend on the records alone: the order is by key and the
        gzip header carries no timestamp, so an unchanged store exports to an
        unchanged file.
        """
        records = self.records(kind)
        lines = "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as stream:
            stream.write(lines.encode("utf-8"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(buffer.getvalue())
        return len(records)

    def restore(self, kind: str, path: Path, key: Callable[[dict[str, object]], str]) -> int:
        """Store every record of an export not already held; return how many were added.

        `key` recomputes each record's key from its own fields, so a committed
        bundle can be graded again without the store it came from.
        """
        added = 0
        for record in read_export(path):
            record_key = key(record)
            if not self.has(kind, record_key):
                self.put(kind, record_key, record)
                added += 1
        return added


def read_export(path: Path) -> list[dict[str, object]]:
    """The records `Store.export` wrote, in order."""
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]
