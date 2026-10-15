"""The API's copy of the interpreter version is the interpreter's own.

`cp_api` may not import `cp_llm`, so it keeps the version as a constant; this
test is what keeps the two equal.
"""

from cp_api.app import CURRENT_INTERPRETER_VERSION
from cp_llm.version import interpreter_version


def test_the_api_holds_the_interpreters_current_version():
    assert interpreter_version() == CURRENT_INTERPRETER_VERSION, (
        "the prompt or the verifier's rules changed: set cp_api.app.CURRENT_INTERPRETER_VERSION "
        f"to {interpreter_version()!r}"
    )
