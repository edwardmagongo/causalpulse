"""Argument parsing and dispatch, and nothing else.

Every subcommand lives in its own module under `commands/` and exposes a
function taking parsed arguments and returning an exit code. No command
module reads `sys.argv` or calls `sys.exit`, so each is callable directly
from a test.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="causalpulse", description="Causal inference and experimentation platform."
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    subcommands.add_parser("simulate", help="Generate a scenario to CSV.")
    subcommands.add_parser("analyze", help="Fit estimators and record how to redo it.")
    subcommands.add_parser("replay", help="Refit from a manifest and compare.")

    bench = subcommands.add_parser("bench", help="Run and report the benchmark grid.")
    bench_commands = bench.add_subparsers(dest="bench_command", required=True)
    bench_commands.add_parser("run", help="Run a tier and write its results.")
    bench_commands.add_parser("report", help="Summarise a stored run.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    return 0
