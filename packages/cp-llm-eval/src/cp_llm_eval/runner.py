"""Running interpretations over the corpus, within a budget, resumably.

Each (case, variant, replicate) is one call to phase 1's `interpret`. Its
result is stored under a key of everything that produced it, and a rerun
skips every stored key. A transport failure stores nothing, so a rerun
retries it. Every model call counts against the budget, a failed one too, and
an interpretation is started only when the budget can absorb its largest
possible number of calls.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

from cp_llm.facts import FactTable
from cp_llm.interpret import InterpretationRefusedError, ModelDidNotAnswerError, interpret
from cp_llm.model import ModelCallError, ModelReply, NarrativeModel, Turn
from cp_llm.prompt import prompt_version

from cp_llm_eval.corpus import CorpusEntry
from cp_llm_eval.store import Store, digest
from cp_llm_eval.variants import Variant

INTERPRETATIONS = "interpretations"
MAX_ATTEMPTS = 2
# claude-opus-5-5's price per token, as phase 1 measured cost (design §16).
INPUT_PRICE = 4e-6
OUTPUT_PRICE = 20e-6


@dataclass(frozen=True)
class Call:
    """One model call that returned. A call that raised has no record here."""

    input_tokens: int
    output_tokens: int
    stop_reason: str
    model: str


def cost(calls: Sequence[Call]) -> float:
    return sum(c.input_tokens * INPUT_PRICE + c.output_tokens * OUTPUT_PRICE for c in calls)


class CallBudget:
    """At most `max_calls` model calls, shared by every worker.

    `reserve` holds room for an interpretation's worst case before it starts;
    `spend` turns one held call into a used one, just before the call is made,
    so a call that then fails is still counted; `release` returns what the
    interpretation did not use.
    """

    def __init__(self, max_calls: int) -> None:
        if max_calls < 0:
            raise ValueError(f"max_calls must not be negative, got {max_calls}")
        self.max_calls = max_calls
        self._used = 0
        self._held = 0
        self._lock = threading.Lock()

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    def reserve(self, calls: int) -> bool:
        with self._lock:
            if self._used + self._held + calls > self.max_calls:
                return False
            self._held += calls
            return True

    def spend(self) -> None:
        with self._lock:
            if self._held < 1:
                raise RuntimeError("a model call was made without a reservation")
            self._held -= 1
            self._used += 1

    def release(self, calls: int) -> None:
        with self._lock:
            if calls > self._held:
                raise RuntimeError("released more calls than were held")
            self._held -= calls


class RecordingModel:
    """Passes calls through to `inner`, charging the budget and noting each reply."""

    def __init__(self, inner: NarrativeModel, budget: CallBudget) -> None:
        self._inner = inner
        self._budget = budget
        self.calls: list[Call] = []
        self.made = 0

    def complete(self, system: str, turns: Sequence[Turn]) -> ModelReply:
        self._budget.spend()
        self.made += 1
        reply = self._inner.complete(system, turns)
        self.calls.append(
            Call(
                reply.usage.input_tokens,
                reply.usage.output_tokens,
                reply.stop_reason,
                reply.model,
            )
        )
        return reply


@dataclass(frozen=True)
class Job:
    entry: CorpusEntry
    variant: Variant
    replicate: int
    model: str
    effort: str

    def key_fields(self) -> dict[str, object]:
        return {
            "case_id": self.entry.case_id,
            "facts_sha256": self.entry.facts_sha256,
            "variant": self.variant.name,
            "prompt_version": prompt_version(self.variant.system()),
            "model": self.model,
            "effort": self.effort,
            "replicate": self.replicate,
        }

    def key(self) -> str:
        return digest(self.key_fields())


def interpretation_key(record: Mapping[str, Any]) -> str:
    """A stored interpretation's key, recomputed from the key fields it carries."""
    return digest(record["key"])


@dataclass(frozen=True)
class Outcome:
    """One job's result: its record (None after a transport failure) and its calls."""

    record: dict[str, object] | None
    made: int
    calls: tuple[Call, ...]


def interpretation_record(
    job: Job, facts: FactTable, model: NarrativeModel, budget: CallBudget
) -> Outcome:
    """Interpret one job, charging every model call to `budget`."""
    recording = RecordingModel(model, budget)
    record: dict[str, object] = {"key": job.key_fields()}
    try:
        result = interpret(facts, recording, MAX_ATTEMPTS, system=job.variant.system())
    except ModelCallError:
        return Outcome(None, recording.made, tuple(recording.calls))
    except InterpretationRefusedError as exc:
        record |= {"outcome": "refused", "attempts": [a.to_dict() for a in exc.attempts]}
    except ModelDidNotAnswerError as exc:
        record |= {
            "outcome": "no_answer",
            "stop_reason": exc.stop_reason,
            "attempts": [a.to_dict() for a in exc.attempts],
        }
    else:
        interpretation = result.to_dict()
        record |= {
            "outcome": "verified",
            "interpretation": interpretation,
            "attempts": interpretation["attempts"],
        }
    record["calls"] = [asdict(call) for call in recording.calls]
    return Outcome(record, recording.made, tuple(recording.calls))


@dataclass
class RunSummary:
    planned: int = 0
    skipped: int = 0
    stored: int = 0
    transport_failures: int = 0
    not_started: int = 0
    calls: int = 0
    cost: float = 0.0


def plan_jobs(
    entries: Sequence[CorpusEntry],
    variants: Sequence[Variant],
    replicates: int,
    model: str,
    effort: str,
) -> list[Job]:
    """Every job, in report order: variant, then case, then replicate."""
    if replicates < 1:
        raise ValueError(f"replicates must be at least 1, got {replicates}")
    return [
        Job(entry, variant, replicate, model, effort)
        for variant in variants
        for entry in entries
        for replicate in range(1, replicates + 1)
    ]


def run(
    jobs: Sequence[Job],
    model_factory: Callable[[], NarrativeModel],
    store: Store,
    budget: CallBudget,
    workers: int = 4,
    progress: Callable[[str], None] = lambda _message: None,
) -> RunSummary:
    """Run every job not already stored, `workers` at a time, within `budget`."""
    summary = RunSummary(planned=len(jobs))
    pending = [job for job in jobs if not store.has(INTERPRETATIONS, job.key())]
    summary.skipped = len(jobs) - len(pending)
    lock = threading.Lock()
    model = model_factory()

    def work(job: Job) -> None:
        if not budget.reserve(MAX_ATTEMPTS):
            with lock:
                summary.not_started += 1
            return
        outcome = interpretation_record(job, job.entry.facts, model, budget)
        budget.release(MAX_ATTEMPTS - outcome.made)
        with lock:
            summary.calls += outcome.made
            summary.cost += cost(outcome.calls)
            if outcome.record is None:
                summary.transport_failures += 1
                progress(
                    f"{job.entry.case_id} {job.variant.name} r{job.replicate}: transport failure"
                )
                return
            store.put(INTERPRETATIONS, job.key(), outcome.record)
            summary.stored += 1
            progress(
                f"{job.entry.case_id} {job.variant.name} r{job.replicate}: "
                f"{outcome.record['outcome']} ({outcome.made} call(s), ${summary.cost:.2f} so far)"
            )

    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in [pool.submit(work, job) for job in pending]:
            future.result()
    return summary
