"""A benchmark run on disk: one shard per cell, and a ledger written last.

A run is a directory holding `manifest.json`, `ledger.jsonl` and `cells/`. Each
cell's rows are written to their own Parquet shard, atomically, by whichever
process fitted the cell. Only afterwards does the one parent process append that
cell's line to the ledger. So a ledger line always points at a complete shard,
and a kill at any instant leaves the directory resumable: a shard with no line
is an orphan, redone rather than trusted.

A cell counts as done only if its ledger line agrees with the file on disk --
same digest, same row count. That is a property of the bytes, not a trust in the
log: a shard truncated or edited after the fact stops counting, and is redone.

The run's *identity* is everything that decides what a shard holds -- the grid
axes, the estimators, the fit configuration, the root entropy -- except the
replication count. That one exception is what makes a run extendable: a cell's
generator comes from its own identity (`grid.cell_rng`), so replications 0 to 49
draw the same data whether the run asks for 50 replications or 100, and asking
for more only adds cells.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from cp_core.results import git_commit, package_versions
from cp_core.values import EstimatorConfig

from cp_evaluation.grid import (
    CateAxes,
    Cell,
    CrossSectionAxes,
    DonorPoolAxes,
    GridConfig,
    PanelAxes,
    cell_key,
    expand_cells,
)
from cp_evaluation.run import ROW_COLUMNS, STRING_COLUMNS, sort_result_frame

MANIFEST_NAME = "manifest.json"
LEDGER_NAME = "ledger.jsonl"
CELLS_DIR = "cells"
SCHEMA_VERSION = 1


class RunMismatchError(ValueError):
    """The run on disk was made under a different identity than the one asked for."""


class IncompleteRunError(RuntimeError):
    """A reader wanted every cell and some are not done."""


class CorruptLedgerError(RuntimeError):
    """A ledger line other than a torn final one could not be read."""


def _jsonable(value: object) -> Any:
    """`value` as plain JSON data: tuples become lists, keys are sorted."""
    return json.loads(json.dumps(value, sort_keys=True))


def _differences(prefix: str, left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    """Dotted names of every leaf field two nested mappings disagree on."""
    names: list[str] = []
    for key in sorted(set(left) | set(right)):
        a, b = left.get(key), right.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            names.extend(_differences(f"{prefix}.{key}", a, b))
        elif a != b:
            names.append(f"{prefix}.{key}")
    return names


@dataclass(frozen=True)
class RunIdentity:
    """Everything that decides what a run's shards hold, except how many replications."""

    tier: str
    grid: dict[str, Any]
    estimators: tuple[str, ...]
    fit_config: dict[str, Any]
    root_entropy: int

    @classmethod
    def of(
        cls,
        tier: str,
        config: GridConfig,
        estimators: tuple[str, ...] | list[str],
        fit_config: EstimatorConfig,
        root_entropy: int,
    ) -> RunIdentity:
        grid = {
            "cross_section": None
            if config.cross_section is None
            else asdict(config.cross_section),
            "panel": None if config.panel is None else asdict(config.panel),
        }
        if config.donor_pool is not None:
            # Only when present: a grid without a donor arm keeps the identity
            # (and so the manifest) it always had.
            grid["donor_pool"] = asdict(config.donor_pool)
        if config.cate is not None:
            # Only when present, for the same reason as the donor arm: a grid
            # without it keeps the identity (and so the manifest) it always had.
            grid["cate"] = asdict(config.cate)
        return cls(
            tier=tier,
            grid=_jsonable(grid),
            estimators=tuple(sorted(estimators)),
            fit_config=_jsonable(asdict(fit_config)),
            root_entropy=root_entropy,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "grid": self.grid,
            "estimators": list(self.estimators),
            "fit_config": self.fit_config,
            "root_entropy": self.root_entropy,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> RunIdentity:
        return cls(
            tier=str(raw["tier"]),
            grid=dict(raw["grid"]),
            estimators=tuple(raw["estimators"]),
            fit_config=dict(raw["fit_config"]),
            root_entropy=int(raw["root_entropy"]),
        )

    def differences(self, other: RunIdentity) -> list[str]:
        """The fields the two identities disagree on, named for a human to act on."""
        names = [
            name
            for name in ("tier", "estimators", "root_entropy")
            if getattr(self, name) != getattr(other, name)
        ]
        names.extend(_differences("grid", self.grid, other.grid))
        names.extend(_differences("fit_config", self.fit_config, other.fit_config))
        return names


def _axes(block: dict[str, Any]) -> dict[str, tuple[Any, ...]]:
    """JSON gives lists; every axis field on a grid dataclass is a tuple."""
    return {name: tuple(values) for name, values in block.items()}


def grid_config_of(grid: dict[str, Any], replications: int) -> GridConfig:
    """Rebuild the `GridConfig` an identity's `grid` block describes."""
    cross_section = grid["cross_section"]
    panel = grid["panel"]
    donor_pool = grid.get("donor_pool")
    cate = grid.get("cate")
    return GridConfig(
        cross_section=(
            None if cross_section is None else CrossSectionAxes(**_axes(cross_section))
        ),
        panel=None if panel is None else PanelAxes(**_axes(panel)),
        donor_pool=None if donor_pool is None else DonorPoolAxes(**_axes(donor_pool)),
        cate=None if cate is None else CateAxes(**_axes(cate)),
        replications=replications,
    )


@dataclass(frozen=True)
class LedgerEntry:
    """What the parent process records once a cell's shard is complete."""

    key: str
    rows: int
    sha256: str

    def to_line(self) -> str:
        fields = {"key": self.key, "rows": self.rows, "sha256": self.sha256}
        return json.dumps(fields, sort_keys=True)


def shard_path(directory: Path, key: str) -> Path:
    return directory / CELLS_DIR / f"{key}.parquet"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_shard(directory: Path, key: str, rows: list[dict[str, object]]) -> LedgerEntry:
    """Write one cell's rows atomically and describe the file that resulted.

    Written to a sibling temporary name and renamed into place, so a reader (or
    a kill) never sees half a shard under the real name. Safe to call from a
    worker process: it touches nothing but its own shard.
    """
    frame = pd.DataFrame(rows, columns=list(ROW_COLUMNS))
    final = shard_path(directory, key)
    temporary = final.with_name(final.name + ".tmp")
    frame.to_parquet(temporary, engine="pyarrow", index=False)
    os.replace(temporary, final)
    return LedgerEntry(key=key, rows=len(frame), sha256=_sha256(final))


def _row_count(path: Path) -> int:
    return len(pd.read_parquet(path, columns=[ROW_COLUMNS[0]]))


def _write_text_atomically(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


@dataclass(frozen=True)
class RunStatus:
    """How far a run has got, counted against the cells its grid expands to."""

    expected: int
    done: int
    orphans: int
    by_scenario: dict[str, tuple[int, int]]

    @property
    def remaining(self) -> int:
        return self.expected - self.done


class RunStore:
    def __init__(
        self,
        directory: Path,
        identity: RunIdentity,
        replications: int,
        provenance: dict[str, Any],
    ) -> None:
        self.directory = directory
        self.identity = identity
        self.replications = replications
        self.provenance = provenance

    # -- lifecycle ---------------------------------------------------------

    @classmethod
    def create(cls, directory: Path, identity: RunIdentity, replications: int) -> RunStore:
        if (directory / MANIFEST_NAME).exists():
            raise FileExistsError(f"{directory} already holds a run; open it to resume it")
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(
                f"{directory} is not empty and holds no run; refusing to write into it"
            )
        (directory / CELLS_DIR).mkdir(parents=True, exist_ok=True)
        (directory / LEDGER_NAME).touch()
        store = cls(
            directory,
            identity,
            replications,
            {"git_commit": git_commit(), "package_versions": package_versions()},
        )
        store._write_manifest()
        return store

    @classmethod
    def open(cls, directory: Path) -> RunStore:
        path = directory / MANIFEST_NAME
        if not path.exists():
            raise FileNotFoundError(f"{directory} has no {MANIFEST_NAME}; it is not a run")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("schema") != SCHEMA_VERSION:
            raise ValueError(
                f"{path} is schema {raw.get('schema')!r}; this code reads {SCHEMA_VERSION}"
            )
        return cls(
            directory,
            RunIdentity.from_json(raw["identity"]),
            int(raw["replications"]),
            dict(raw["provenance"]),
        )

    def _write_manifest(self) -> None:
        manifest = {
            "schema": SCHEMA_VERSION,
            "identity": self.identity.to_json(),
            "replications": self.replications,
            "provenance": self.provenance,
        }
        _write_text_atomically(
            self.directory / MANIFEST_NAME, json.dumps(manifest, sort_keys=True, indent=2)
        )

    def check_identity(self, requested: RunIdentity) -> None:
        """Refuse to continue this run under a different identity, naming the fields."""
        differing = self.identity.differences(requested)
        if differing:
            raise RunMismatchError(
                f"{self.directory} was made under a different configuration; "
                f"these fields differ: {', '.join(differing)}. Mixing shards from "
                "two configurations would make the run describe neither."
            )

    def set_replications(self, replications: int) -> None:
        if replications < self.replications:
            raise ValueError(
                f"{self.directory} already asks for {self.replications} replications; "
                f"lowering that to {replications} would silently ignore finished shards"
            )
        if replications != self.replications:
            self.replications = replications
            self._write_manifest()

    # -- what the run should contain ---------------------------------------

    def grid_config(self) -> GridConfig:
        return grid_config_of(self.identity.grid, self.replications)

    def expected_cells(self) -> dict[str, Cell]:
        return {cell_key(cell): cell for cell in expand_cells(self.grid_config())}

    # -- the ledger --------------------------------------------------------

    def record(self, entry: LedgerEntry) -> None:
        """Append one completed cell. Called only by the parent, after the shard exists."""
        with (self.directory / LEDGER_NAME).open("a", encoding="utf-8") as handle:
            handle.write(entry.to_line() + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _ledger_entries(self) -> dict[str, LedgerEntry]:
        path = self.directory / LEDGER_NAME
        lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
        entries: dict[str, LedgerEntry] = {}
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                entry = LedgerEntry(str(raw["key"]), int(raw["rows"]), str(raw["sha256"]))
            except (ValueError, KeyError, TypeError) as exc:
                if number == len(lines):
                    # A kill mid-append leaves a torn last line. It names no
                    # finished cell, so it is dropped and that cell is redone.
                    continue
                raise CorruptLedgerError(f"{path}:{number} is not a ledger line") from exc
            entries[entry.key] = entry
        return entries

    def intact_entries(self) -> dict[str, LedgerEntry]:
        """Ledger entries whose shard is on disk and agrees with them."""
        intact: dict[str, LedgerEntry] = {}
        for key, entry in self._ledger_entries().items():
            path = shard_path(self.directory, key)
            if not path.exists() or _sha256(path) != entry.sha256:
                continue
            if _row_count(path) != entry.rows:
                continue
            intact[key] = entry
        return intact

    def sweep_temporary(self) -> None:
        """Delete half-written shards a kill left behind."""
        for leftover in (self.directory / CELLS_DIR).glob("*.tmp"):
            leftover.unlink()

    # -- reading -----------------------------------------------------------

    def status(self) -> RunStatus:
        expected = self.expected_cells()
        intact = set(self.intact_entries())
        done = intact & set(expected)
        on_disk = {p.stem for p in (self.directory / CELLS_DIR).glob("*.parquet")}
        by_scenario: dict[str, tuple[int, int]] = {}
        for key, cell in expected.items():
            finished, total = by_scenario.get(cell.scenario, (0, 0))
            by_scenario[cell.scenario] = (finished + (key in done), total + 1)
        return RunStatus(
            expected=len(expected),
            done=len(done),
            orphans=len(on_disk - done),
            by_scenario=dict(sorted(by_scenario.items())),
        )

    def read_frame(self, *, partial: bool = False) -> pd.DataFrame:
        """Every finished cell's rows, in the one order every reader sees.

        Refuses an incomplete run unless asked for a partial one: a table over a
        silently truncated run would read as a complete result.
        """
        expected = self.expected_cells()
        intact = self.intact_entries()
        done = sorted(set(intact) & set(expected))
        missing = len(expected) - len(done)
        if missing and not partial:
            raise IncompleteRunError(
                f"{self.directory} has {missing} of {len(expected)} cells not done; "
                "finish it with --resume, or ask for a partial read"
            )
        if not done:
            return pd.DataFrame(columns=list(ROW_COLUMNS))
        frames = [pd.read_parquet(shard_path(self.directory, key)) for key in done]
        frame = pd.concat(frames, ignore_index=True)
        text = list(STRING_COLUMNS)
        frame[text] = frame[text].astype("str")
        return sort_result_frame(frame)
