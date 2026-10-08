# One image for the api, worker and migrate services; each overrides the command.
#
# The interpreter is the one `.python-version` names, which the suite's
# thresholds were measured on; `tests/test_image.py` fails if the two disagree.
FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
WORKDIR /app

# Bytecode is compiled here, as root, because the user the services run as
# cannot write beside the sources: without it every process would compile the
# whole environment again each time it starts.
ENV UV_COMPILE_BYTECODE=1

# Dependencies before code. The lockfile and the root manifest are enough to
# install everything but the workspace's own packages, so a change under
# `packages/` reuses this layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-workspace

# The workspace's packages are installed editable, so their sources stay under
# `packages/` and are compiled there.
COPY packages ./packages
RUN uv sync --frozen --no-dev \
    && /app/.venv/bin/python -m compileall -q packages

# The services run as `app`, uid and gid 10001: a fixed number, so the owner of
# a mounted volume's files is known in advance. It has no home directory and
# owns nothing under /app, which it can read and not write.
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /nonexistent \
        --shell /usr/sbin/nologin app

# What the services write while running goes under /tmp, each through its own
# setting:
#   MPLCONFIGDIR  Matplotlib's font cache, written when the worker first draws
#                 a figure. Unset, Matplotlib looks for it under the home
#                 directory, which this user does not have.
#   TMPDIR        `tempfile`: the report's figures while they are rendered, and
#                 an upload's body once it outgrows memory.
ENV PATH="/app/.venv/bin:$PATH" MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib TMPDIR=/tmp
USER app
