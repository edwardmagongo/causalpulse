"""`python -m cp_worker`: one RQ worker on the analyses queue.

`SimpleWorker` runs each job in this process rather than a fork. JAX, which
`bsts` and `hierarchical_did` use, is not safe to use in a child forked after
it has initialised.
"""

from __future__ import annotations

import os

from cp_api.db import QUEUE_NAME
from redis import Redis
from rq import Queue, SimpleWorker


def main() -> None:
    connection = Redis.from_url(os.environ["REDIS_URL"])
    SimpleWorker([Queue(QUEUE_NAME, connection=connection)], connection=connection).work()


if __name__ == "__main__":
    main()
