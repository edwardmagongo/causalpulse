"""The one place the analyses worker is built: `python -m cp_worker` and the
tests both call `make_worker`, so the time limit the tests exercise is the one
that runs.

RQ enforces a job's `job_timeout` with a one-shot `SIGALRM` whose handler
raises `rq.timeouts.JobTimeoutException` -- an `Exception`. `fit_each` isolates
each estimator with `except Exception`, so that exception would be recorded as
one estimator's failure, and the remaining estimators would then run with no
limit left at all. `EscapingDeathPenalty` raises `JobTimeLimitExceeded`, a
`BaseException`, instead: it passes through every `except Exception` to
`run_analysis_job`, which catches it by name and fails the analysis.

The alarm interrupts Python code only. A single call into compiled code (one
long XLA or LAPACK call) finishes before the handler can run.
"""

from __future__ import annotations

import contextlib
from types import FrameType

from cp_api.db import QUEUE_NAME
from redis import Redis
from rq import SimpleWorker
from rq.timeouts import JobTimeoutException, UnixSignalDeathPenalty


class JobTimeLimitExceeded(BaseException):
    """The job ran past its `job_timeout`; deliberately not an `Exception`."""

    def __init__(self, seconds: int) -> None:
        super().__init__(f"the analysis exceeded the {seconds}-second job limit and was stopped")
        self.seconds = seconds


class EscapingDeathPenalty(UnixSignalDeathPenalty):
    """`UnixSignalDeathPenalty`, raising `JobTimeLimitExceeded` for a job's limit.

    RQ's worker passes `JobTimeoutException` explicitly when it times a job
    (`BaseWorker.perform_job`), so the exception is swapped where the alarm
    fires rather than via the constructor's argument. Every limit RQ sets
    with `JobTimeoutException` -- the job's own, and a success or failure
    callback's, which these jobs do not register -- raises
    `JobTimeLimitExceeded`; any other exception passed in is raised unchanged.
    """

    def handle_death_penalty(self, signum: int, frame: FrameType | None) -> None:
        if issubclass(self._exception, JobTimeoutException):
            raise JobTimeLimitExceeded(self._timeout)
        # rq ships `py.typed` but leaves these two methods unannotated.
        super().handle_death_penalty(signum, frame)  # type: ignore[no-untyped-call]

    def cancel_death_penalty(self) -> None:
        # RQ's own `__exit__` ignores an alarm that fires as the job returns,
        # but catches only its `Exception`-based timeouts; do the same here.
        with contextlib.suppress(JobTimeLimitExceeded):
            super().cancel_death_penalty()  # type: ignore[no-untyped-call]


class AnalysisWorker(SimpleWorker):
    """`SimpleWorker` runs each job in this process rather than a fork: JAX,
    which `bsts` and `hierarchical_did` use, is not safe in a child forked
    after it has initialised."""

    death_penalty_class = EscapingDeathPenalty


def make_worker(connection: Redis) -> AnalysisWorker:
    # By name, so the worker builds the queue with its own death-penalty class.
    return AnalysisWorker([QUEUE_NAME], connection=connection)
