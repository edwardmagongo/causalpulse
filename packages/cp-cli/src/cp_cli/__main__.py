"""Argument parsing and dispatch, and nothing else.

Every subcommand lives in its own module under `commands/` and exposes a
function taking parsed arguments and returning an exit code. No command
module reads `sys.argv` or calls `sys.exit`, so each is callable directly
from a test.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from cp_cli.commands import analyze, bench, replay, simulate, study


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="causalpulse", description="Causal inference and experimentation platform."
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    simulate_parser = subcommands.add_parser("simulate", help="Generate a scenario to CSV.")
    simulate.add_arguments(simulate_parser)
    simulate_parser.set_defaults(handler=simulate.run)

    analyze_parser = subcommands.add_parser(
        "analyze", help="Fit estimators and record how to redo it."
    )
    analyze.add_arguments(analyze_parser)
    analyze_parser.set_defaults(handler=analyze.run)

    replay_parser = subcommands.add_parser("replay", help="Refit from a manifest and compare.")
    replay.add_arguments(replay_parser)
    replay_parser.set_defaults(handler=replay.run)

    bench_parser = subcommands.add_parser("bench", help="Run and report the benchmark grid.")
    bench_commands = bench_parser.add_subparsers(dest="bench_command", required=True)

    bench_run = bench_commands.add_parser("run", help="Run a tier and write its results.")
    bench.add_run_arguments(bench_run)
    bench_run.set_defaults(handler=bench.run_tier)

    bench_report = bench_commands.add_parser("report", help="Summarise a stored run.")
    bench.add_report_arguments(bench_report)
    bench_report.set_defaults(handler=bench.report)

    bench_status = bench_commands.add_parser("status", help="Count a sharded run's finished cells.")
    study.add_status_arguments(bench_status)
    bench_status.set_defaults(handler=study.status)

    bench_compact = bench_commands.add_parser(
        "compact", help="Fold a finished sharded run into one results file."
    )
    study.add_compact_arguments(bench_compact)
    bench_compact.set_defaults(handler=study.compact)

    bench_study = bench_commands.add_parser(
        "study", help="Write the study's tables, figures and expectations check."
    )
    study.add_study_arguments(bench_study)
    bench_study.set_defaults(handler=study.run_study)
    bench_study_report = bench_commands.add_parser(
        "study-report", help="Write the benchmark study's research report."
    )
    study.add_study_report_arguments(bench_study_report)
    bench_study_report.set_defaults(handler=study.run_study_report)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.error("no handler for that command")
    result = handler(args)
    assert isinstance(result, int)
    return result
