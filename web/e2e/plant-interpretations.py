"""Plant three interpretations of one analysis, for the browser tests; no test calls a model.

`make smoke-web` and `make smoke-prod` run it inside the api container, the script on
standard input: `python - <analysis id>`. Oldest first, it writes a failed interpretation
and a verified one from claude-opus-5-5 at medium effort, the verified one under the
server's current prompt and rules, then a newer failed one from claude-sonnet-5-5 at low
effort. The stack the browser tests run against writes no interpretations, so this is how
they reach an analysis page with earlier interpretations.
"""

import sys
import uuid
from datetime import UTC, datetime, timedelta

from cp_api.app import CURRENT_INTERPRETER_VERSION
from cp_api.db import Interpretation, make_engine, make_sessionmaker

analysis_id = uuid.UUID(sys.argv[1])
now = datetime.now(UTC)


def planted(minutes_ago: int, **fields: object) -> Interpretation:
    at = now - timedelta(minutes=minutes_ago)
    return Interpretation(
        analysis_id=analysis_id,
        job_id=f"planted-{uuid.uuid4().hex[:12]}",
        interpreter_version=CURRENT_INTERPRETER_VERSION,
        created_at=at,
        started_at=at,
        finished_at=at + timedelta(seconds=20),
        **fields,
    )


engine = make_engine()
with make_sessionmaker(engine)() as session:
    session.add_all([
        planted(
            30, status="failed", model="claude-opus-5-5", effort="medium",
            error="planted: the model API call failed",
        ),
        planted(
            20, status="verified", model="claude-opus-5-5", effort="medium",
            returned_model="claude-opus-5-5", prompt_version="planted",
            rendered={
                "sections": [
                    {"name": "summary", "heading": "Summary",
                     "statements": ["A planted narrative, written by no model."]},
                ],
                "limits": "Planted limits.",
            },
            notice="Planted notice.",
            attempts=[],
        ),
        planted(
            10, status="failed", model="claude-sonnet-5-5", effort="low",
            error="planted: the newer interpretation failed",
        ),
    ])
    session.commit()
engine.dispose()
