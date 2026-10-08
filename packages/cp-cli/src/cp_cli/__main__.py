"""Argument parsing and dispatch, and nothing else.

Every subcommand lives in its own module under `commands/` and exposes a
function taking parsed arguments and returning an exit code. No command
module reads `sys.argv` or calls `sys.exit`, so each is callable directly
from a test.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from cp_cli.commands import analyze, bench, interpret, llm_eval, replay, simulate, study


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

    interpret_parser = subcommands.add_parser(
        "interpret", help="Write a verified interpretation of an analyze --report run."
    )
    interpret.add_arguments(interpret_parser)
    interpret_parser.set_defaults(handler=interpret.run)

    llm_eval_parser = subcommands.add_parser(
        "llm-eval", help="Measure how often a verified interpretation misreads its analysis."
    )
    llm_eval_commands = llm_eval_parser.add_subparsers(dest="llm_eval_command", required=True)
    for name, help_text, add, handler in (
        ("build", "Fit the corpus and write its fact tables.", llm_eval.add_build_arguments,
         llm_eval.run_build),
        ("run", "Interpret the corpus (billed).", llm_eval.add_run_arguments, llm_eval.run_run),
        ("grade", "Judge the verified narratives (billed).", llm_eval.add_grade_arguments,
         llm_eval.run_grade),
        ("report", "Write the report and the bundle of records.", llm_eval.add_report_arguments,
         llm_eval.run_report),
        ("restore", "Load a committed bundle into a store.", llm_eval.add_restore_arguments,
         llm_eval.run_restore),
    ):
        command = llm_eval_commands.add_parser(name, help=help_text)
        add(command)
        command.set_defaults(handler=handler)
    label_parser = llm_eval_commands.add_parser("label", help="Export or import the labels.")
    label_commands = label_parser.add_subparsers(dest="label_command", required=True)
    label_export = label_commands.add_parser("export", help="Write a sample to label, as CSV.")
    llm_eval.add_label_export_arguments(label_export)
    label_export.set_defaults(handler=llm_eval.run_label_export)
    label_import = label_commands.add_parser("import", help="Validate and store a labelled CSV.")
    llm_eval.add_label_import_arguments(label_import)
    label_import.set_defaults(handler=llm_eval.run_label_import)

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
        "study",
        help=(
            "Write the study's tables (CSV, Markdown and Parquet), figures and "
            "expectations check; prefer the Parquet tables, or read the CSVs with "
            "keep_default_na=False."
        ),
    )
    study.add_study_arguments(bench_study)
    bench_study.set_defaults(handler=study.run_study)
    bench_study_report = bench_commands.add_parser(
        "study-report", help="Write the benchmark study's research report."
    )
    study.add_study_report_arguments(bench_study_report)
    bench_study_report.set_defaults(handler=study.run_study_report)
    bench_track = bench_commands.add_parser(
        "track", help="Log a finished run to MLflow for comparison across runs."
    )
    bench.add_track_arguments(bench_track)
    bench_track.set_defaults(handler=bench.track)

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
