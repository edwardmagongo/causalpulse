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

SERVICES_ENV = DATABASE_URL=postgresql+psycopg://causalpulse:causalpulse@localhost:5433/causalpulse REDIS_URL=redis://localhost:6380/0

# Serial, not -n auto: the tests share one database.
check-services:
	docker compose up -d --wait postgres redis
	$(SERVICES_ENV) uv run pytest -v -m "services and not slow"; status=$$?; docker compose down -v; exit $$status
