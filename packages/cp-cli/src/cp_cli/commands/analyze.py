"""Fit estimators against a dataset, and record how to reproduce the run.

Two dataset sources with different provenance. A `--scenario` is named by a
`ScenarioReference`, which regenerates it; a `--data` CSV is named by a
`FixtureReference`, which records the file's sha256 so a later replay refuses
if it changed. Neither embeds the frame.

Only cross-sectional scenarios are accepted here. `resolve_scenario_reference`
(the function `replay` calls to regenerate a scenario dataset) supports
`n_units` plus every `ScenarioParams` axis, and refuses every other `params`
key by name -- a panel scenario needs `n_periods`, which that resolver has no
way to carry. Writing a manifest `replay` cannot honour would defeat the point
of a manifest, so panel scenario ids are refused here rather than accepted and
left to fail later.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from cp_causal.manifest import run_analysis
from cp_core.dataset import CausalDataset
from cp_core.manifest import DatasetReference, FixtureReference, ScenarioReference
from cp_core.results import EffectEstimate
from cp_core.values import EstimatorConfig
from cp_llm.extract import fact_table
from cp_llm.facts import FACTS_FILENAME
from cp_report.pipeline import AnalysisOutcome, analyse
from cp_simulation.panel_scenarios import PANEL_SCENARIOS
from cp_simulation.params import ScenarioParams
from cp_simulation.scenarios import SCENARIOS, build_scenario

from cp_cli.io import write_estimates
from cp_cli.schema_args import add_schema_arguments, schema_from_args


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scenario", help="Cross-sectional scenario id to generate and analyse.")
    parser.add_argument("--n-units", type=int, help="Units, with --scenario.")
    parser.add_argument("--seed", type=int, default=0, help="Seed, with --scenario.")
    parser.add_argument(
        "--confounding-strength", type=float, default=1.0, help="With --scenario."
    )
    parser.add_argument("--data", help="Path to a CSV to analyse.")
    parser.add_argument(
        "--estimators",
        help="Comma-separated names. Optional with --report, where the router chooses.",
    )
    parser.add_argument("--out", required=True, help="Directory to write results to.")
    parser.add_argument(
        "--random-seed", type=int, default=0, help="EstimatorConfig random_seed."
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Also write one evidence report per estimate, a summary.json and a facts.json.",
    )
    add_schema_arguments(parser)


def _load(args: argparse.Namespace) -> tuple[CausalDataset, DatasetReference]:
    if args.scenario is not None and args.data is not None:
        raise ValueError("--scenario and --data were given both; pass one or the other")
    if args.scenario is None and args.data is None:
        raise ValueError("one of --scenario or --data is required")

    if args.scenario is not None:
        scenario = str(args.scenario)
        if scenario in PANEL_SCENARIOS:
            raise ValueError(
                f"scenario {scenario!r} is a panel scenario; analyze accepts only "
                f"cross-sectional scenarios ({sorted(SCENARIOS)}). A panel "
                "ScenarioReference would need n_periods, which the scenario "
                "resolver used at replay time cannot carry, so it is refused here "
                "rather than written into a manifest that cannot be replayed."
            )
        if scenario not in SCENARIOS:
            raise ValueError(
                f"unknown scenario {scenario!r}; available: {sorted(SCENARIOS)}"
            )
        if args.n_units is None:
            raise ValueError("--scenario requires --n-units")
        # Routed through `ScenarioParams`, not the bare-float facade -- see
        # `cp_evaluation.resolve.resolve_cell` for why.
        scenario_params = ScenarioParams(confounding_strength=args.confounding_strength)
        dataset = build_scenario(scenario, scenario_params).generate(
            n_units=args.n_units, rng=np.random.default_rng(args.seed)
        ).dataset
        # Every axis field is recorded explicitly, via `asdict`, rather than
        # only the ones this command exposes a flag for -- so a manifest
        # written here does not lean on `resolve_scenario_reference` filling
        # anything in from `ScenarioParams`' defaults at replay time, and
        # keeps replaying the same way even if those defaults ever change.
        reference: DatasetReference = ScenarioReference(
            scenario=scenario,
            seed=args.seed,
            params={"n_units": args.n_units, **asdict(scenario_params)},
        )
        return dataset, reference

    schema = schema_from_args(args)
    frame = pd.read_csv(args.data)
    return CausalDataset(frame=frame, schema=schema), FixtureReference.from_path(str(args.data))


def _write_outcome(outcome: AnalysisOutcome, out: Path) -> None:
    for report in outcome.reports:
        directory = out / "reports" / report.estimate.method
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "report.html").write_text(report.html)
        (directory / "report.md").write_text(report.markdown)
        for filename, svg in report.assets.items():
            (directory / filename).write_text(svg)
    summary = {
        "routing": [
            {"estimator": d.estimator_name, "eligible": d.eligible, "reasons": list(d.reasons)}
            for d in outcome.decisions
        ],
        "failures": [{"method": f.method, "error": f.error} for f in outcome.failures],
        "comparisons": [group.to_dict() for group in outcome.comparisons],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    # Everything `causalpulse interpret` reads. Written on every report run:
    # building it calls no model and needs no credentials.
    (out / FACTS_FILENAME).write_text(fact_table(outcome).to_json(), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    dataset, reference = _load(args)
    names = (
        None
        if args.estimators is None
        else [n for n in str(args.estimators).split(",") if n]
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    estimates: tuple[EffectEstimate, ...]
    if args.report:
        outcome = analyse(dataset, reference, names, random_seed=args.random_seed)
        _write_outcome(outcome, out)
        manifest = outcome.manifest
        estimates = tuple(report.estimate for report in outcome.reports)
        for failure in outcome.failures:
            print(f"{failure.method}: failed: {failure.error}")
    else:
        if names is None:
            raise ValueError(
                "--estimators is required unless --report is given, in which case "
                "the router chooses"
            )
        config = EstimatorConfig(random_seed=args.random_seed)
        manifest, estimates = run_analysis(dataset, reference, [(n, config) for n in names])
    write_estimates(estimates, out / "estimates.csv")
    (out / "manifest.json").write_text(manifest.to_json())
    for estimate in estimates:
        print(
            f"{estimate.method}: {estimate.point:.4f} "
            f"[{estimate.ci.lower:.4f}, {estimate.ci.upper:.4f}]"
        )
    return 0
