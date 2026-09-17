"""Generate a scenario to CSV, alongside the reference that produced it.

The reference file is the point of this command rather than a nicety. A CSV of
simulated data is indistinguishable from real data once it is on disk, and
without a record of which scenario, seed and parameters produced it the
provenance survives only in whoever ran the command.

Only the frame is written to CSV. A generated result also carries `.truth`
(the ground truth used to grade estimators against) -- that is exactly the
thing a real analysis never has, and writing it into a CSV a user will later
treat as "data" would let it leak into an analysis that is supposed to be
honest about not knowing it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from cp_core.manifest import ScenarioReference
from cp_simulation.panel_scenarios import PANEL_SCENARIOS, build_panel_scenario
from cp_simulation.panel_truth import SimulatedPanel
from cp_simulation.params import ScenarioParams
from cp_simulation.scenarios import SCENARIOS, build_scenario
from cp_simulation.truth import SimulatedDataset


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scenario", required=True, help="Scenario id, e.g. D or P1.")
    parser.add_argument("--n-units", type=int, required=True, help="Number of units.")
    parser.add_argument("--seed", type=int, required=True, help="Replication seed.")
    parser.add_argument("--out", required=True, help="Path to write the CSV to.")
    parser.add_argument(
        "--confounding-strength",
        type=float,
        default=None,
        help="Cross-sectional scenarios only.",
    )
    parser.add_argument("--n-periods", type=int, default=None, help="Panel scenarios only.")


def run(args: argparse.Namespace) -> int:
    scenario = str(args.scenario)
    is_panel = scenario in PANEL_SCENARIOS
    if not is_panel and scenario not in SCENARIOS:
        raise ValueError(
            f"unknown scenario {scenario!r}; available: "
            f"{sorted(SCENARIOS)} (cross-section) and {sorted(PANEL_SCENARIOS)} (panel)"
        )

    if is_panel and args.n_periods is None:
        raise ValueError(f"panel scenario {scenario!r} requires --n-periods")
    if not is_panel and args.n_periods is not None:
        raise ValueError(
            f"cross-sectional scenario {scenario!r} has no period axis; --n-periods "
            "does not apply to it"
        )
    if is_panel and args.confounding_strength is not None:
        raise ValueError(
            f"panel scenario {scenario!r} has no confounding parameter; --confounding-strength "
            "does not apply to it"
        )

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    params: dict[str, object]
    simulated: SimulatedDataset | SimulatedPanel
    if is_panel:
        simulated = build_panel_scenario(scenario).generate(
            n_units=args.n_units, n_periods=args.n_periods, rng=rng
        )
        params = {"n_units": args.n_units, "n_periods": args.n_periods}
    else:
        confounding_strength = args.confounding_strength or 1.0
        # Routed through `ScenarioParams`, not the bare-float facade -- see
        # `cp_evaluation.resolve.resolve_cell` for why: `build_scenario` is
        # the one entry point every caller goes through, and this command has
        # no flags yet for the other four Phase 18 axes, so it fills in
        # their defaults explicitly rather than through the float shorthand.
        scenario_params = ScenarioParams(confounding_strength=confounding_strength)
        simulated = build_scenario(scenario, scenario_params).generate(
            n_units=args.n_units, rng=rng
        )
        # Every axis field is recorded explicitly, via `asdict`, mirroring
        # `analyze` -- not just the two keys (`n_units`,
        # `confounding_strength`) that predate Phase 18. Without this, a
        # `.reference.json` for a widened run would replay the other four
        # axes from `ScenarioParams`' defaults regardless of what actually
        # generated the CSV on disk.
        params = {"n_units": args.n_units, **asdict(scenario_params)}

    simulated.dataset.frame.to_csv(out, index=False)

    reference = ScenarioReference(scenario=scenario, seed=args.seed, params=params)
    # A single `with_suffix` call replaces only the final suffix, which is
    # exactly what turning `d.csv` into `d.reference.json` needs. A path with
    # no suffix at all (`d`) still gets one appended, and a multi-dot name
    # (`my.data.csv`) keeps its non-final dots: `my.data.reference.json`, not
    # `my.reference.json`.
    reference_path = out.with_suffix(".reference.json")
    reference_path.write_text(
        json.dumps({"kind": "scenario", **asdict(reference)}, sort_keys=True, indent=2)
    )
    print(f"wrote {len(simulated.dataset.frame)} rows to {out} and {reference_path}")
    return 0
