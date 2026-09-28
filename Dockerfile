# One image for the api, worker and migrate services; each overrides the command.
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY packages ./packages
RUN uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH" MPLBACKEND=Agg
