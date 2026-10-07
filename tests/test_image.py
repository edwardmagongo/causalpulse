"""The Python image, read from its `Dockerfile`. Nothing here needs Docker.

The suite's thresholds were measured on the interpreter `.python-version`
names, so an image built from another one runs code the suite never ran.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# `FROM [--platform=...] python:<tag> [AS name]`.
_FROM_PYTHON = re.compile(r"^FROM\s+(?:--\S+\s+)*python:(\S+)", re.IGNORECASE)


def _instructions() -> list[tuple[str, str]]:
    """Each instruction as (keyword, arguments), with continued lines joined."""
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    joined = re.sub(r"\\\n", " ", text)
    instructions = []
    for line in joined.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        keyword, _, arguments = stripped.partition(" ")
        instructions.append((keyword.upper(), arguments.strip()))
    return instructions


def _minor(version: str) -> str:
    return ".".join(version.split(".")[:2])


def test_every_python_stage_is_built_on_the_interpreter_the_suite_runs():
    pinned = _minor((ROOT / ".python-version").read_text(encoding="utf-8").strip())
    tags = [
        match.group(1)
        for keyword, arguments in _instructions()
        if (match := _FROM_PYTHON.match(f"{keyword} {arguments}"))
    ]

    assert tags, "the Dockerfile has no `FROM python:` line"
    # `3.13-slim` and `3.13.2-slim` both name 3.13; a bare `3` or `slim` does not.
    assert [_minor(tag.split("-")[0]) for tag in tags] == [pinned] * len(tags), tags


def test_the_image_does_not_run_as_root():
    instructions = _instructions()
    keywords = [keyword for keyword, _ in instructions]

    assert "RUN" in keywords
    last_run = len(keywords) - 1 - keywords[::-1].index("RUN")
    users = [arguments for keyword, arguments in instructions[last_run + 1 :] if keyword == "USER"]

    assert users, "no USER line after the last RUN: the image's processes run as root"
    # The last one is the user the containers run as. `user:group` is allowed.
    assert users[-1].split(":")[0] not in ("root", "0"), users
