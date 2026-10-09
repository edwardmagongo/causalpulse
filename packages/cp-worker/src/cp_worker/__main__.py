"""`python -m cp_worker`: one RQ worker on the analyses queue.

`cp_worker.worker.make_worker` builds it: an in-process `SimpleWorker`, with
the time limit that `run_analysis_job` can catch.
"""

from __future__ import annotations

import logging
import os

from cp_api.settings import require_production_settings
from redis import Redis

from cp_worker.jobs import require_interpretation_settings
from cp_worker.worker import make_worker


def main() -> None:
    # First, before Redis is contacted or a job can be taken.
    require_production_settings()
    require_interpretation_settings()
    # A root handler, so `cp_worker.jobs`' tracebacks reach the container log
    # with a time and a name; RQ then adds no handler of its own.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    make_worker(Redis.from_url(os.environ["REDIS_URL"])).work()


if __name__ == "__main__":
    main()
