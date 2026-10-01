"""Inspecting, compacting and analysing a sharded benchmark run.

`status` and `compact` read the store; `study` turns a finished one into the
tables, figures and expectations check the research report is written from. All
three read only what `bench run` wrote and recompute nothing a fit produced.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from cp_evaluation.expectations import evaluate_expectations, expectations_markdown
from cp_evaluation.grid import SCENARIO_AXES
from cp_evaluation.metrics import summarise
from cp_evaluation.store import RunStore
from cp_evaluation.study import (
    baseline_slice,
    coverage_of_study_markdown,
    headline_tables,
    markdown_table,
)
from cp_figures.benchmark import (
    PROFILE_AXES,
    axis_profile_data,
    draw_axis_profile,
    draw_heatmap,
    heatmap_data,
)
from cp_report.benchmark import (
    benchmark_evidence_from_run,
    render_html_benchmark_report,
    write_markdown_benchmark_report,
)
from matplotlib.figure import Figure

from cp_cli.commands.bench import load_run
from cp_cli.io import write_benchmark

HEATMAP_METRICS = (
    "bias",
    "rmse",
    "coverage_analytic",
    "coverage_bootstrap",
    "se_calibration",
    "false_positive_rate",
)
PROFILE_METRICS = ("bias", "coverage_analytic")


def add_status_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("directory", help="A sharded run written by `bench run`.")


def status(args: argparse.Namespace) -> int:
    store = RunStore.open(Path(args.directory))
    run = store.status()
    print(f"tier: {store.identity.tier}  replications: {store.replications}")
    print(
        f"cells: {run.done}/{run.expected} done, {run.remaining} remaining, "
        f"{run.orphans} orphan shards"
    )
    for scenario, (done, total) in run.by_scenario.items():
        print(f"  {scenario}: {done}/{total}")
    return 0


def add_compact_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("directory", help="A finished sharded run.")
    parser.add_argument("--out", required=True, help="Directory for the single-file copy.")


def compact(args: argparse.Namespace) -> int:
    RunStore.open(Path(args.directory))  # a compact copy is only meaningful for a sharded run
    frame, sidecar = load_run(Path(args.directory))
    write_benchmark(frame, Path(args.out), sidecar)
    print(f"wrote {len(frame)} rows to {args.out}")
    return 0


def add_study_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("directory", help="A sharded run written by `bench run`.")
    parser.add_argument("--out", required=True, help="Directory for the analysis outputs.")
    parser.add_argument(
        "--partial", action="store_true", help="Analyse an unfinished run, labelled as such."
    )


def _save(figure: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")


def _write_figures(frame: pd.DataFrame, out: Path) -> list[str]:
    """Heatmaps over the baseline slice and axis profiles per scenario and axis.

    Returns one line per figure the plotting code refused, with its reason: a
    profile along an axis a scenario does not vary is not drawn as a flat line.
    """
    refusals: list[str] = []
    pooled = summarise(baseline_slice(frame), by=["scenario"])
    for metric in HEATMAP_METRICS:
        data = heatmap_data(pooled, metric)
        figure = Figure(figsize=(1.0 + 0.9 * len(data.scenarios), 1.0 + 0.45 * len(data.methods)))
        draw_heatmap(data, figure.subplots())
        _save(figure, out / "figures" / f"heatmap_{metric}.png")

    per_cell = summarise(frame)
    scenarios = sorted(s for s in frame["scenario"].unique() if s in SCENARIO_AXES)
    for scenario in scenarios:
        for axis in PROFILE_AXES:
            for metric in PROFILE_METRICS:
                try:
                    profile = axis_profile_data(per_cell, metric, axis=axis, scenario=scenario)
                except ValueError as exc:
                    refusals.append(f"{scenario} / {axis} / {metric}: {exc}")
                    continue
                figure = Figure(figsize=(5.0, 3.5))
                draw_axis_profile(profile, figure.subplots())
                _save(figure, out / "figures" / f"profile_{scenario}_{axis}_{metric}.png")
    return refusals


def run_study(args: argparse.Namespace) -> int:
    """Write a finished run's tables, figures and expectations check under `--out`.

    `tables/*.csv` must be read with `pandas.read_csv(..., keep_default_na=False)`:
    the CATE scenario `NULL` is otherwise read back as a missing value and its rows
    silently drop out of any grouping.
    """
    directory = Path(args.directory)
    store = RunStore.open(directory)  # the analysis reads a sharded run only
    frame, _ = load_run(directory, partial=args.partial)
    run = store.status()
    out = Path(args.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    tables = headline_tables(frame)
    tables["per_cell_summary"] = summarise(frame)
    for name, table in tables.items():
        table.to_csv(out / "tables" / f"{name}.csv", index=False)
        (out / "tables" / f"{name}.md").write_text(markdown_table(table), encoding="utf-8")

    results = evaluate_expectations(frame, run)
    (out / "expectations.md").write_text(expectations_markdown(results), encoding="utf-8")
    (out / "coverage-of-study.md").write_text(
        coverage_of_study_markdown(frame, run, store.replications), encoding="utf-8"
    )
    refusals = _write_figures(frame, out)
    (out / "figures").mkdir(exist_ok=True)
    (out / "figures" / "refusals.md").write_text(
        "\n".join(["# Figures not drawn", "", *(f"- {line}" for line in refusals), ""]),
        encoding="utf-8",
    )

    record = {
        "run": str(directory),
        "tier": store.identity.tier,
        "replications": store.replications,
        "git_commit": store.provenance.get("git_commit"),
        "cells": {"expected": run.expected, "done": run.done, "orphans": run.orphans},
        "partial": bool(args.partial and run.remaining),
        "expectations": [asdict(r) for r in results],
        "tables": sorted(tables),
    }
    (out / "study.json").write_text(json.dumps(record, indent=2, sort_keys=True, default=str))
    verdicts = ", ".join(f"{r.id}={r.verdict}" for r in results)
    print(f"wrote the study analysis to {out}: {verdicts}")
    return 0


_HTML_REPORT_NAME = "benchmark-report.html"


def add_study_report_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("directory", help="A finished sharded run written by `bench run`.")
    parser.add_argument("--out", required=True, help="An existing directory for the report.")
    parser.add_argument("--format", choices=("html", "markdown"), default="markdown")
    parser.add_argument(
        "--limitation",
        action="append",
        default=[],
        metavar="TEXT",
        help="A limitation to state in the report; repeat for several.",
    )
    parser.add_argument(
        "--narratives",
        metavar="PATH",
        help="A JSON object mapping an expectation id to the paragraph that explains it.",
    )
    parser.add_argument(
        "--cate-run",
        metavar="DIR",
        help="A finished `bench run --tier cate` run, reported in its own CATE section.",
    )


def _read_narratives(path: str | None) -> dict[str, str]:
    if path is None:
        return {}
    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in loaded.items()
    ):
        raise ValueError(f"{path} must hold one JSON object mapping expectation ids to text")
    return loaded


def run_study_report(args: argparse.Namespace) -> int:
    out = Path(args.out)
    if not out.is_dir():
        raise FileNotFoundError(f"--out must be an existing directory, got {out}")
    evidence = benchmark_evidence_from_run(
        Path(args.directory),
        cate_directory=None if args.cate_run is None else Path(args.cate_run),
        limitations=tuple(args.limitation),
        narratives=_read_narratives(args.narratives),
    )
    if args.format == "html":
        path = out / _HTML_REPORT_NAME
        path.write_text(render_html_benchmark_report(evidence), encoding="utf-8")
    else:
        path = write_markdown_benchmark_report(evidence, out)
    print(f"wrote the benchmark study report to {path}")
    return 0


__all__: Sequence[str] = ("status", "compact", "run_study", "run_study_report")
