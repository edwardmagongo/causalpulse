"""What an interpretation is written and verified under, as one short id."""

import hashlib

from cp_llm import prompt, verify


def interpreter_version() -> str:
    """A short hash of the prompt version and the verifier's rule names.

    Two interpretations with the same version were written under the same
    system prompt and reply schema and verified against rules of the same
    names. A rule whose behaviour changes while its name and the prompt stay
    the same does not change it; `tests/llm/test_version.py` pins the rule
    names so that a rule added, removed or renamed is a decision, not an
    accident. The model and the effort are recorded beside it on each
    interpretation, not in it; the token limit is a constant of the code,
    `cp_llm.model.MAX_TOKENS`, and is not recorded on it.
    """
    rules = ",".join(sorted([*verify.STATEMENT_RULES, *verify.NARRATIVE_RULES]))
    payload = f"{prompt.prompt_version()}|{rules}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
