"""The API's OpenAPI document, as the text committed at `web/openapi.json`.

The dashboard's TypeScript types are generated from that file. A test compares
the file with `render()`, so a route or a response model cannot change without
the file changing with it.

Building the app here contacts nothing: an unbound `sessionmaker` and a Redis
client that connects only on first use stand in for the real ones, and
listing routes uses neither.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from redis import Redis
from sqlalchemy.orm import sessionmaker

from cp_api.app import create_app


def document() -> dict[str, Any]:
    return create_app(sessions=sessionmaker(), redis=Redis()).openapi()


def render() -> str:
    """Sorted keys and a trailing newline, so the committed file diffs cleanly."""
    return json.dumps(document(), indent=2, sort_keys=True) + "\n"


def main() -> None:
    sys.stdout.write(render())
