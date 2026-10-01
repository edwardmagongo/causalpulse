"""Request-scoped dependencies shared by the routes and by `cp_api.auth`.

Split out of `cp_api.app` so that `cp_api.auth` can depend on the session
without importing the app, which imports it.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.orm import Session, sessionmaker


def get_session(request: Request) -> Iterator[Session]:
    """A module-level dependency, not a local closure inside `create_app`.

    `from __future__ import annotations` turns every annotation into a
    string; FastAPI resolves `SessionDep` via `get_type_hints`, which needs a
    name reachable from the function's `__globals__`. A `SessionDep` defined
    inside `create_app` is a local, invisible to that lookup, so route
    parameters typed with it silently stop being recognised as
    dependencies -- they are treated as ordinary (query) parameters instead.
    Reading the session factory off `app.state` keeps `SessionDep` a genuine
    module global.
    """
    sessions: sessionmaker[Session] = request.app.state.sessions
    with sessions() as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]
