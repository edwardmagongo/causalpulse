.PHONY: install fmt lint types imports test test-fast cov check check-services

install:
	uv sync

fmt:
	uv run ruff format .

lint:
	uv run ruff check .

types:
	uv run mypy packages

imports:
	uv run lint-imports

test:
	uv run pytest -v -n auto

test-fast:
	uv run pytest -v -m "not slow" -n auto

cov:
	uv run pytest --cov=packages --cov-report=term-missing

check: lint types imports test

SERVICES_ENV = TEST_DATABASE_URL=postgresql+psycopg://causalpulse:causalpulse@localhost:5434/causalpulse TEST_REDIS_URL=redis://localhost:6381/0
TEST_STACK = CP_POSTGRES_PORT=5434 CP_REDIS_PORT=6381 docker compose -p causalpulse-test

# Serial, not -n auto: the tests share one database. Runs against its own
# compose project on 5434/6381, never the demo stack's 5433/6380, so it can
# never wipe a running demo.
check-services:
	$(TEST_STACK) up -d --wait postgres redis || { $(TEST_STACK) down -v; exit 1; }
	$(SERVICES_ENV) uv run pytest -v -m "services and not slow"; status=$$?; $(TEST_STACK) down -v; exit $$status
