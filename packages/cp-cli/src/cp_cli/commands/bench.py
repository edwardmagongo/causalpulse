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
from dataclasses import asdict
from pathlib import Path

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


def add_report_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("directory", help="A directory written by `bench run`.")


def report(args: argparse.Namespace) -> int:
    frame, sidecar = read_benchmark(Path(args.directory))
    summary = summarise(frame)
    print(f"tier: {sidecar.get('tier')}  rows: {len(frame)}")
    print(summary.to_string(index=False))
    return 0
