"""Write a plain-language interpretation of a stored analysis, or refuse.

Reads the `facts.json` that `analyze --report` wrote, so interpreting never
refits anything and can be repeated. The model call is the only step here
that needs credentials or a network; `build_model` is the one place it is
constructed, and a test replaces it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import cast, get_args

from cp_llm.facts import FACTS_FILENAME, FactTable
from cp_llm.interpret import InterpretationRefusedError, ModelDidNotAnswerError, interpret
from cp_llm.model import DEFAULT_EFFORT, DEFAULT_MODEL, Effort, ModelCallError, NarrativeModel


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("run_dir", help="A directory written by `analyze --report`.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model id.")
    parser.add_argument(
        "--effort", default=DEFAULT_EFFORT, choices=get_args(Effort), help="Reasoning effort."
    )


def build_model(model: str, effort: str) -> NarrativeModel:
    # Imported here, so that no other command loads the SDK.
    from cp_llm.client import ClaudeModel

    return ClaudeModel(model=model, effort=cast(Effort, effort))


def run(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    facts_path = run_dir / FACTS_FILENAME
    if not facts_path.exists():
        print(
            f"{run_dir} has no {FACTS_FILENAME}; run `causalpulse analyze --report` first",
            file=sys.stderr,
        )
        return 1
    raw = facts_path.read_text(encoding="utf-8")
    facts = FactTable.from_json(raw)

    try:
        result = interpret(facts, build_model(args.model, args.effort))
    except InterpretationRefusedError as exc:
        print(str(exc), file=sys.stderr)
        for violation in exc.attempts[-1].violations:
            print(f"  [{violation.rule}] {violation.reason}", file=sys.stderr)
            if violation.statement:
                print(f"    {violation.statement}", file=sys.stderr)
        return 1
    except (ModelDidNotAnswerError, ModelCallError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    record = {
        **result.to_dict(),
        "facts_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    }
    markdown = run_dir / "interpretation.md"
    markdown.write_text(result.rendered.to_markdown(), encoding="utf-8")
    (run_dir / "interpretation.json").write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote {markdown}")
    return 0
