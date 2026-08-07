.PHONY: install fmt lint types imports test test-fast cov check

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
	uv run pytest -v

test-fast:
	uv run pytest -v -m "not slow"

cov:
	uv run pytest --cov=packages --cov-report=term-missing

check: lint types imports test
