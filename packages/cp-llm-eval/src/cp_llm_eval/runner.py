"""Running interpretations over the corpus, within a budget, resumably.

Each (case, arm, replicate) is one interpretation: on a grounded arm a call
to phase 1's `interpret`, which may send a rejected reply back once; on the
ungrounded arm one call to `ungrounded.write_unverified`, which has no
verifier to send anything back to. Its result is stored under a key of
everything that produced it, and a rerun skips every stored key. A transport
failure stores nothing, so a rerun retries it. Every model call counts
against the budget, a failed one too, and an interpretation is started only
when the budget can absorb its largest possible number of calls -- which is
the arm's, not the module's.
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

from cp_llm_eval.arms import Arm
from cp_llm_eval.corpus import CorpusEntry
from cp_llm_eval.store import Store, digest
from cp_llm_eval.ungrounded import CALLS as UNGROUNDED_CALLS
from cp_llm_eval.ungrounded import UNVERIFIED, write_unverified

INTERPRETATIONS = "interpretations"
# The most attempts a grounded interpretation gets: the first reply, and one
# more after the verifier's complaints are put to the model.
MAX_ATTEMPTS = 2
# claude-opus-5-5's price per token, as phase 1 measured cost.
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


def max_calls(arm: Arm) -> int:
    """The most model calls one of this arm's narratives can cost.

    A grounded arm may put the verifier's complaints to the model once. The
    ungrounded arm has no verifier, so there is nothing to retry against and
    one call is all it can spend.
    """
    return MAX_ATTEMPTS if arm.grounded else UNGROUNDED_CALLS


@dataclass(frozen=True)
class Job:
    entry: CorpusEntry
    # Named for the key field it writes, which the 199 stored records of the
    # phase 2 study hash under that name.
    variant: Arm
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


def _verified(job: Job, facts: FactTable, model: NarrativeModel) -> dict[str, object]:
    """A grounded arm's outcome: verified, refused, or no answer at all."""
    try:
        result = interpret(facts, model, MAX_ATTEMPTS, system=job.variant.system())
    except InterpretationRefusedError as exc:
        return {"outcome": "refused", "attempts": [a.to_dict() for a in exc.attempts]}
    except ModelDidNotAnswerError as exc:
        return {
            "outcome": "no_answer",
            "stop_reason": exc.stop_reason,
            "attempts": [a.to_dict() for a in exc.attempts],
        }
    interpretation = result.to_dict()
    return {
        "outcome": "verified",
        "interpretation": interpretation,
        "attempts": interpretation["attempts"],
    }


def _unverified(job: Job, facts: FactTable, model: NarrativeModel) -> dict[str, object]:
    """The ungrounded arm's outcome: the one reply, or no answer at all.

    There is no refusal here and there cannot be: nothing is verified, so
    nothing can fail verification. A no-answer is stored exactly as a
    grounded arm's is, so the two arms' denominators are built the same way.
    """
    try:
        result = write_unverified(facts, model, system=job.variant.system())
    except ModelDidNotAnswerError as exc:
        return {"outcome": "no_answer", "stop_reason": exc.stop_reason, "attempts": []}
    unverified = result.to_dict()
    return {
        "outcome": UNVERIFIED,
        "interpretation": unverified,
        "attempts": unverified["attempts"],
    }


def interpretation_record(
    job: Job, facts: FactTable, model: NarrativeModel, budget: CallBudget
) -> Outcome:
    """Interpret one job by its arm's run path, charging every call to `budget`."""
    recording = RecordingModel(model, budget)
    record: dict[str, object] = {"key": job.key_fields()}
    write = _verified if job.variant.grounded else _unverified
    try:
        record |= write(job, facts, recording)
    except ModelCallError:
        return Outcome(None, recording.made, tuple(recording.calls))
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
    arms: Sequence[Arm],
    replicates: int,
    model: str,
    effort: str,
) -> list[Job]:
    """Every job, in report order: arm, then case, then replicate.

    A job listed twice (an arm or case given twice) is kept once, at its
    first place, so it is never run twice.
    """
    if replicates < 1:
        raise ValueError(f"replicates must be at least 1, got {replicates}")
    jobs: dict[str, Job] = {}
    for arm in arms:
        for entry in entries:
            for replicate in range(1, replicates + 1):
                job = Job(entry, arm, replicate, model, effort)
                jobs.setdefault(job.key(), job)
    return list(jobs.values())


class StopOnError:
    """Stops a pool after a worker's first unexpected error, then raises it.

    A job checks `stopped` before it reserves any calls, so once an error is
    noted no new call starts; calls already in flight may finish.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._error: Exception | None = None

    @property
    def stopped(self) -> bool:
        with self._lock:
            return self._error is not None

    def note(self, error: Exception) -> None:
        with self._lock:
            if self._error is None:
                self._error = error

    def raise_if_stopped(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise error


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

    stop = StopOnError()

    def work(job: Job) -> None:
        if stop.stopped:
            return
        try:
            attempt(job)
        except Exception as exc:  # anything a transport failure is not
            stop.note(exc)

    def attempt(job: Job) -> None:
        calls = max_calls(job.variant)
        if not budget.reserve(calls):
            with lock:
                summary.not_started += 1
            return
        outcome = interpretation_record(job, job.entry.facts, model, budget)
        budget.release(calls - outcome.made)
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
    stop.raise_if_stopped()
    return summary
