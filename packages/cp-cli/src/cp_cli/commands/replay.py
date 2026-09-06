"""Reconstruct a recorded analysis and check the numbers still match.

This is the one place in the system that composes both dataset resolvers.
`cp_causal` cannot import the simulation package, so `replay` takes its
resolver by injection; supplying it is the whole reason `dataset_resolver`
exists, and the CLI sits high enough in the layers contract to do it.

What to compare against is a judgement call the task brief flags explicitly.
`AnalysisManifest` records the inputs to an analysis, not its outputs, so a
replay has nothing of its own to check itself against. Comparing two replays
of the same manifest in the same process (the brief's draft) is weak by
construction: it would pass even if replay reconstructed entirely the wrong
dataset, as long as it did so consistently both times, because both replays
share the same resolver, the same process, and the same code path.

`analyze` always writes `estimates.csv` beside `manifest.json`, and those are
the numbers a user actually remembers and cares about matching. So the
primary comparison here is against that recorded file: does replaying the
manifest reproduce what `analyze` actually reported last time. That is a
strictly stronger check -- it catches a replay that is wrong but
self-consistent, which the two-replay check cannot -- and it is what makes a
perturbed replay meaningfully wrong rather than just meaningfully different
from itself. The two-replay comparison is kept only as a fallback for a
manifest whose `estimates.csv` is missing (moved, or never written), where
there is genuinely nothing recorded to compare against.

The default tolerance is exact equality, not a rounding allowance. The test
suite already establishes that replay reproduces estimates bit for bit, so a
difference is a finding rather than noise. `--tolerance` exists for a user
whose environment genuinely differs -- a different BLAS, say -- and the output
states which comparison was used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from cp_causal.manifest import dataset_resolver, replay
from cp_core.manifest import AnalysisManifest
from cp_core.results import EffectEstimate
from cp_simulation.scenarios import resolve_scenario_reference


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("manifest", help="Path to a manifest.json written by `analyze`.")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Absolute tolerance on the point estimate. Default 0, meaning exact.",
    )


def _compare_to_recorded(
    replayed: tuple[EffectEstimate, ...], recorded: pd.DataFrame, tolerance: float
) -> list[str]:
    """Compare replayed points against `estimates.csv`'s recorded ones.

    Positional, not name-keyed: `estimates.csv` is written in the same order
    as `manifest.estimators`, which `replay` iterates in order, so position
    is the correspondence both files actually share. A method-name mismatch
    at a position is itself reported as a difference rather than silently
    realigned by name.
    """
    if len(replayed) != len(recorded):
        raise ValueError(
            f"recorded estimates.csv has {len(recorded)} row(s) but replay produced "
            f"{len(replayed)} estimate(s); the manifest and the recorded run disagree"
        )
    messages = []
    for position, estimate in enumerate(replayed):
        row = recorded.iloc[position]
        recorded_method = str(row["method"])
        if estimate.method != recorded_method:
            messages.append(
                f"position {position}: replay produced {estimate.method!r} but "
                f"estimates.csv recorded {recorded_method!r} there"
            )
            continue
        recorded_point = float(row["point"])
        if abs(estimate.point - recorded_point) > tolerance:
            messages.append(
                f"{estimate.method}: replayed and recorded estimates differ, "
                f"replayed {estimate.point:.10g} against recorded {recorded_point:.10g}"
            )
    return messages


def _compare_replays(
    first: tuple[EffectEstimate, ...], second: tuple[EffectEstimate, ...], tolerance: float
) -> list[str]:
    messages = []
    for one, two in zip(first, second, strict=True):
        if abs(one.point - two.point) > tolerance:
            messages.append(
                f"{one.method}: two replays of the same manifest differ, "
                f"{one.point:.10g} against {two.point:.10g}"
            )
    return messages


def run(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    manifest = AnalysisManifest.from_json(manifest_path.read_text())
    resolve = dataset_resolver(resolve_scenario_reference)
    replayed = replay(manifest, resolve)
    tolerance = float(args.tolerance)

    comparison = "exactly" if tolerance == 0.0 else f"within {tolerance}"
    print(f"comparing {comparison}")
    for estimate in replayed:
        print(f"{estimate.method}: {estimate.point:.10g}")

    estimates_path = manifest_path.parent / "estimates.csv"
    if estimates_path.exists():
        print(f"comparing against recorded estimates at {estimates_path}")
        # pandas' default C float parser is not round-trip safe -- it can
        # return a float64 one ULP away from the one that was written, which
        # a tolerance of exactly 0.0 would then wrongly flag as a mismatch.
        # `float_precision="round_trip"` parses the same way `float()` does.
        recorded = pd.read_csv(estimates_path, float_precision="round_trip")
        differences = _compare_to_recorded(replayed, recorded, tolerance)
    else:
        print(f"{estimates_path} not found; comparing two replays of the manifest instead")
        again = replay(manifest, resolve)
        differences = _compare_replays(replayed, again, tolerance)

    if differences:
        for message in differences:
            print(message)
        print(f"{len(differences)} estimator(s) differ")
        return 1
    print(f"all {len(replayed)} estimator(s) match {comparison}")
    return 0
