"""Running a benchmark tier, and reporting a stored run.

`TIERS` maps a tier name to the very objects the test suite gates on, not to a
copy of their contents. The `ci` grid the suite asserts tolerance bands against
and the `ci` grid this command runs must be the same object, or the committed
bands describe neither.

`report` reads only what `run` wrote and never recomputes, so a report cannot
silently disagree with the run it claims to describe.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from cp_core.results import git_commit, package_versions
from cp_evaluation.grid import CrossSectionAxes, PanelAxes
from cp_evaluation.metrics import summarise
from cp_evaluation.run import run_grid
from cp_evaluation.tiers import (
    CI_ESTIMATORS,
    CI_FIT_CONFIG,
    FULL_ESTIMATORS,
    FULL_FIT_CONFIG,
    ci_config,
    full_config,
)

from cp_cli.io import read_benchmark, write_benchmark

TIERS = {
    "ci": (ci_config, CI_ESTIMATORS, CI_FIT_CONFIG),
    "full": (full_config, FULL_ESTIMATORS, FULL_FIT_CONFIG),
}


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tier", required=True, help="One of: ci, full.")
    parser.add_argument("--out", required=True, help="Directory to write results to.")
    parser.add_argument("--root-entropy", type=int, default=0, help="Seed root.")


def run_tier(args: argparse.Namespace) -> int:
    tier = str(args.tier)
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; available: {sorted(TIERS)}")
    make_config, estimators, fit_config = TIERS[tier]
    config = make_config()

    frame = run_grid(
        config,
        estimators=list(estimators),
        fit_config=fit_config,
        root_entropy=args.root_entropy,
    )

    sidecar = {
        "tier": tier,
        "estimators": list(estimators),
        "bootstrap_reps": fit_config.bootstrap_reps,
        "random_seed": fit_config.random_seed,
        "root_entropy": args.root_entropy,
        "replications": config.replications,
        # Both axis blocks are read off the dataclass rather than transcribed
        # field by field. The sidecar is the run's provenance record -- `bench
        # report` hands it straight back to a reader as the description of the
        # grid that produced these rows -- so an axis the dataclass has and
        # this block does not is a run.json that actively misdescribes its own
        # parquet. Phase 18 added four axes to `CrossSectionAxes`; a
        # transcribed block would have kept reporting the pre-Phase-18 grid
        # for every `--tier full` run, silently, since nothing downstream
        # cross-checks the two. Deriving means a fifth axis cannot repeat it.
        "cross_section": (
            None if config.cross_section is None else _axis_block(config.cross_section)
        ),
        "panel": None if config.panel is None else _axis_block(config.panel),
        # What the grid was *asked* for is not what it necessarily produced.
        # A scenario is crossed only against the axes it declares
        # (`SCENARIOS[id].axes`), so an axis it does not declare is pinned to
        # `ScenarioParams`' default in its cells -- including when the config's
        # own axis does not contain that default. A run at
        # `confounding_strength=(0.5, 2.0)` over a randomly-assigning scenario
        # writes rows reading 1.0 and none at either requested strength, and
        # the two blocks above would describe a grid this parquet does not
        # contain. These blocks record what actually landed, so `bench report`
        # can say when the two disagree.
        #
        # Derived from `frame`, not from a second walk of the declaration
        # rule: a reimplementation here could drift from `expand_cells`, and
        # the claim being made is specifically about the rows in the parquet
        # beside this file, so the rows are what it is read off.
        "cross_section_realised": (
            None if config.cross_section is None else _realised_block(frame, config.cross_section)
        ),
        "panel_realised": (
            None if config.panel is None else _realised_block(frame, config.panel)
        ),
        "package_versions": package_versions(),
        "git_commit": git_commit(),
    }

    out = Path(args.out)
    write_benchmark(frame, out, sidecar)
    fitted = int((frame["status"] == "fitted").sum())
    print(f"wrote {len(frame)} rows ({fitted} fitted) to {out}")
    return 0


def _axis_block(axes: CrossSectionAxes | PanelAxes) -> dict[str, list[object]]:
    """One axis dataclass, as JSON-ready lists, every field included."""
    return {name: list(values) for name, values in asdict(axes).items()}


def _realised_block(
    frame: pd.DataFrame, axes: CrossSectionAxes | PanelAxes
) -> dict[str, list[object]]:
    """The values of each of `axes`' fields that actually appear in `frame`.

    Keyed by the same field names `_axis_block` uses, so the requested and
    realised blocks can be compared entry by entry. Each entry is the union of
    an axis's values over *every* scenario in the frame, so the block cannot
    express per-scenario gating: `full`'s `confounding_strength` reads
    `[0.5, 1.0, 2.0]` because D, G and H supply the outer two and A, B, C and
    F supply only 1.0. The parquet's own rows are the only per-scenario
    record. Rows are filtered to the
    shape these axes describe, since the other shape leaves every one of these
    columns null -- `confounding_strength` is NaN on a panel row and
    `n_periods` on a cross-sectional one -- and an unfiltered read would report
    each axis as empty.

    Values are sorted and deduplicated: this records *which* values a run
    produced, not how often or in what order, and sorting makes two runs of the
    same grid compare equal regardless of how their axes were written down.
    `.tolist()` rather than the numpy scalars `unique()` returns, so the result
    is JSON-serialisable.

    Integral values are rendered in the requested block's own type. `n_periods`
    is a float64 column in the frame -- it is NaN on every cross-sectional row,
    which widens the whole column -- so a run asking for `n_periods=(7,)` would
    otherwise record `[7]` beside `[7.0]` in one `run.json`. That costs nothing
    in `_axis_discrepancies`, which compares numerically, but these two blocks
    exist to be read side by side, and one value rendered two ways invites
    exactly the "do these differ?" question the blocks were added to answer.
    Only an exactly-integral float is narrowed, so a genuinely fractional value
    cannot be silently truncated into agreement.
    """
    shape = "cross_section" if isinstance(axes, CrossSectionAxes) else "panel"
    rows = frame[frame["shape"] == shape]
    block: dict[str, list[object]] = {}
    for name, requested in asdict(axes).items():
        # `CrossSectionAxes.scenarios`/`PanelAxes.scenarios` name the column
        # `scenario` in the singular; every other field matches its column.
        column = "scenario" if name == "scenarios" else name
        values = sorted(rows[column].dropna().unique().tolist())
        block[name] = [_matching_requested_type(value, requested) for value in values]
    return block


def _matching_requested_type(value: object, requested: Sequence[object]) -> object:
    """`value`, narrowed to `int` when that is how the axis was asked for.

    Deliberately narrow: it fires only for a float that is exactly integral on
    an axis whose requested values are all `int`. Anything else -- a fractional
    value, a string, an axis requested as floats, an empty axis -- is returned
    untouched, so this can correct pandas' NaN-driven int-to-float widening
    without being able to change what a value means.
    """
    if not isinstance(value, float) or not value.is_integer():
        return value
    if not requested or not all(isinstance(r, int) for r in requested):
        return value
    return int(value)


def _axis_discrepancies(sidecar: Mapping[str, object]) -> list[str]:
    """One line per axis whose realised values differ from the requested ones.

    Empty for a run where every axis produced exactly what was asked for, which
    is both shipped tiers: `ci` and `full` each carry `ScenarioParams`' default
    confounding strength on their axis, so the scenarios that do not declare
    the axis collapse onto a value that was requested anyway. Checked for both,
    `full` included -- from `expand_cells` rather than from a run, since the
    claim is about which cells are enumerated and that tier takes ~3.1 hours
    (`tests/cli/test_bench.py::test_the_full_tier_produces_no_discrepancy_either`,
    which also carries a negative control, since `full`'s silence depends on
    scenario F being present to declare `heterogeneity_scale`).

    Compared as sets, not as lists: the requested block preserves the order the
    axis was written in while the realised block is sorted, and neither order
    nor a repeated value is a discrepancy worth reporting.

    Returns nothing for a `run.json` written before these blocks existed, which
    simply has no realised block to compare against.
    """
    lines: list[str] = []
    for shape in ("cross_section", "panel"):
        requested = sidecar.get(shape)
        realised = sidecar.get(f"{shape}_realised")
        if not isinstance(requested, dict) or not isinstance(realised, dict):
            continue
        for name, asked in requested.items():
            got = realised.get(name)
            if got is None or set(asked) == set(got):
                continue
            lines.append(f"  {shape}.{name}: requested {list(asked)}, realised {list(got)}")
    return lines


def add_report_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("directory", help="A directory written by `bench run`.")


def report(args: argparse.Namespace) -> int:
    frame, sidecar = read_benchmark(Path(args.directory))
    summary = summarise(frame)
    print(f"tier: {sidecar.get('tier')}  rows: {len(frame)}")
    # Printed before the summary, not after: it qualifies every row below it.
    # Silent when the grid produced what it was asked for, so a reader who sees
    # nothing here can read the requested axes as the grid's description.
    discrepancies = _axis_discrepancies(sidecar)
    if discrepancies:
        print(
            "note: some axis values differ from the requested grid -- a scenario is "
            "crossed only against the axes it declares, and takes ScenarioParams' "
            "default for the rest:"
        )
        for line in discrepancies:
            print(line)
    print(summary.to_string(index=False))
    return 0
