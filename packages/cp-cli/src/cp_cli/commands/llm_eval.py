"""`causalpulse llm-eval`: build the corpus, run, judge, label and report the study.

`run` and `grade` spend money and take `--max-calls`, which is required: a
paid step never runs without a ceiling. `build_model` and `build_judge` are
the only places a real model is constructed, and a test replaces them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast, get_args

from cp_llm.model import DEFAULT_EFFORT, DEFAULT_MODEL, Effort, NarrativeModel
from cp_llm_eval import corpus, labels
from cp_llm_eval.graded import GradedNarrative, grade_records
from cp_llm_eval.judge import VERDICTS, JudgeModel, judge_all, judge_jobs, verdict_key
from cp_llm_eval.report import build_report, write_report
from cp_llm_eval.runner import INTERPRETATIONS, CallBudget, interpretation_key, plan_jobs, run
from cp_llm_eval.store import Store
from cp_llm_eval.variants import VARIANTS

LABELS_FILENAME = "labels.json"
SAMPLE_FILENAME = "sample.json"
INTERPRETATIONS_BUNDLE = "interpretations.jsonl.gz"
VERDICTS_BUNDLE = "verdicts.jsonl.gz"


def build_model(model: str, effort: str) -> NarrativeModel:
    # Imported here, so that no other command loads the SDK.
    from cp_llm.client import ClaudeModel

    return ClaudeModel(model=model, effort=cast(Effort, effort))


def build_judge(model: str, effort: str) -> JudgeModel:
    from cp_llm_eval.judge import ClaudeJudge

    return ClaudeJudge(model=model, effort=cast(Effort, effort))


def _say(message: str) -> None:
    print(message, flush=True)


def _model_arguments(parser: argparse.ArgumentParser, model_help: str) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL, help=model_help)
    parser.add_argument(
        "--effort", default=DEFAULT_EFFORT, choices=get_args(Effort), help="Reasoning effort."
    )


def add_build_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", required=True, help="Directory to write the corpus to.")
    parser.add_argument(
        "--case", action="append", help="Build only this case id (repeatable)."
    )


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus", required=True, help="A directory written by `build`.")
    parser.add_argument("--store", required=True, help="Directory the records are kept in.")
    parser.add_argument("--max-calls", type=int, required=True, help="Most model calls to make.")
    parser.add_argument(
        "--variant", action="append", choices=sorted(VARIANTS), help="Prompt (repeatable)."
    )
    parser.add_argument("--replicates", type=int, default=4, help="Replies per fact table.")
    parser.add_argument("--case", action="append", help="Only this case id (repeatable).")
    parser.add_argument("--workers", type=int, default=4, help="Calls in flight at once.")
    _model_arguments(parser, "Model that writes the narratives.")


def add_grade_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus", required=True, help="A directory written by `build`.")
    parser.add_argument("--store", required=True, help="The store `run` wrote.")
    parser.add_argument("--max-calls", type=int, required=True, help="Most judge calls to make.")
    parser.add_argument("--workers", type=int, default=4, help="Calls in flight at once.")
    _model_arguments(parser, "Model that judges.")


def add_label_export_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--out", required=True, help="CSV file to write.")
    parser.add_argument("--size", type=int, default=labels.DEFAULT_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--judge-model", default=DEFAULT_MODEL)


def add_label_import_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--csv", required=True, help="The labelled CSV.")
    parser.add_argument("--judge-model", default=DEFAULT_MODEL)


def add_report_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--out", required=True, help="Directory for the report and bundle.")
    parser.add_argument("--judge-model", default=DEFAULT_MODEL)


def add_restore_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bundle", required=True, help="A directory `report` wrote.")
    parser.add_argument("--store", required=True, help="Store to restore the records into.")


def run_build(args: argparse.Namespace) -> int:
    cases = corpus.default_cases()
    if args.case:
        unknown = set(args.case) - {c.case_id for c in cases}
        if unknown:
            print(f"unknown case id(s): {sorted(unknown)}", file=sys.stderr)
            return 1
        cases = tuple(c for c in cases if c.case_id in args.case)
    try:
        entries = corpus.build(cases, Path(args.out), Path.cwd(), progress=_say)
    except corpus.CorpusPropertyError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    _say(f"wrote {len(entries)} case(s) to {args.out}")
    return 0


def _entries(args: argparse.Namespace) -> tuple[corpus.CorpusEntry, ...]:
    entries = corpus.load(Path(args.corpus))
    chosen = getattr(args, "case", None)
    return tuple(e for e in entries if not chosen or e.case_id in chosen)


def run_run(args: argparse.Namespace) -> int:
    entries = _entries(args)
    if not entries:
        print("no case selected", file=sys.stderr)
        return 1
    variants = [VARIANTS[name] for name in (args.variant or ["baseline"])]
    jobs = plan_jobs(entries, variants, args.replicates, args.model, args.effort)
    summary = run(
        jobs,
        lambda: build_model(args.model, args.effort),
        Store(Path(args.store)),
        CallBudget(args.max_calls),
        workers=args.workers,
        progress=_say,
    )
    _say(
        f"{summary.stored} stored, {summary.skipped} already stored, "
        f"{summary.transport_failures} transport failure(s), {summary.not_started} not started "
        f"(budget); {summary.calls} call(s), ${summary.cost:.2f}"
    )
    return 0 if summary.transport_failures == 0 and summary.not_started == 0 else 1


def run_grade(args: argparse.Namespace) -> int:
    store = Store(Path(args.store))
    facts = {e.case_id: e.facts for e in corpus.load(Path(args.corpus))}
    jobs = judge_jobs(store.records(INTERPRETATIONS), facts, args.model)
    summary = judge_all(
        jobs,
        lambda: build_judge(args.model, args.effort),
        store,
        CallBudget(args.max_calls),
        workers=args.workers,
        progress=_say,
    )
    _say(
        f"{summary.stored} judged ({summary.judge_errors} unusable), {summary.skipped} already "
        f"judged, {summary.transport_failures} transport failure(s), {summary.not_started} not "
        f"started (budget); {summary.calls} call(s), ${summary.cost:.2f}"
    )
    return 0 if summary.transport_failures == 0 and summary.not_started == 0 else 1


def _graded(
    args: argparse.Namespace,
) -> tuple[tuple[corpus.CorpusEntry, ...], tuple[GradedNarrative, ...]]:
    store = Store(Path(args.store))
    entries = corpus.load(Path(args.corpus))
    graded = grade_records(
        entries, store.records(INTERPRETATIONS), store.records(VERDICTS), args.judge_model
    )
    return entries, graded


def run_label_export(args: argparse.Namespace) -> int:
    entries, graded = _graded(args)
    chosen = labels.sample(graded, args.size, args.seed)
    if not chosen:
        print("no judged narrative to sample from; run `grade` first", file=sys.stderr)
        return 1
    labels.export_csv(chosen, {e.case_id: e.facts for e in entries}, Path(args.out))
    labels.save_sample(chosen, Path(args.store) / SAMPLE_FILENAME)
    _say(f"wrote {len(chosen)} statement(s) to {args.out}")
    return 0


def run_label_import(args: argparse.Namespace) -> int:
    sample_path = Path(args.store) / SAMPLE_FILENAME
    if not sample_path.exists():
        print(
            f"no exported sample at {sample_path}; run `label export` first", file=sys.stderr
        )
        return 1
    try:
        imported = labels.import_csv(Path(args.csv), labels.load_sample(sample_path))
    except labels.LabelError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    labels.save(imported, Path(args.store) / LABELS_FILENAME)
    _say(f"stored {len(imported)} label(s)")
    return 0


def run_report(args: argparse.Namespace) -> int:
    store = Store(Path(args.store))
    entries, graded = _graded(args)
    report = build_report(
        entries,
        graded,
        labels.load(Path(args.store) / LABELS_FILENAME),
        store.records(INTERPRETATIONS),
        store.records(VERDICTS),
    )
    out = Path(args.out)
    markdown, _data = write_report(report, out)
    store.export(INTERPRETATIONS, out / INTERPRETATIONS_BUNDLE)
    store.export(VERDICTS, out / VERDICTS_BUNDLE)
    _say(f"wrote {markdown}")
    return 0


def run_restore(args: argparse.Namespace) -> int:
    store, bundle = Store(Path(args.store)), Path(args.bundle)
    interpretations = store.restore(
        INTERPRETATIONS, bundle / INTERPRETATIONS_BUNDLE, interpretation_key
    )
    verdicts = store.restore(VERDICTS, bundle / VERDICTS_BUNDLE, verdict_key)
    _say(f"restored {interpretations} interpretation(s) and {verdicts} verdict(s)")
    return 0
