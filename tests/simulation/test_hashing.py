"""Pin `_hashing.py`'s own contract directly, not through a consumer.

`config_hash` exists to be a complete function of a component's
configuration: two configurations that differ must hash differently, and
that has to hold even when the difference is "a field exists now that
didn't before," because the shape of a component is part of its
configuration. This lives beside the module it pins rather than inside
`test_dgp.py`, which exercises `CrossSectionDGP` rather than `_hashing.py`
directly.
"""

import dataclasses

from cp_simulation._hashing import component_hash, describe_components


def test_adding_a_field_to_a_hashed_component_changes_the_hash():
    # Phase 18 Task 2 briefly taught `_describe()` to omit one opt-in field
    # from the hash while it sat at its own default, specifically to avoid
    # this outcome for `PropensityAssignment.intercept`. That was reverted
    # (see the Phase 18 Task 2 fix report): a recorded hash that stays
    # byte-identical after a component gains a field is the exact failure
    # `config_hash` exists to prevent -- a config_hash that no longer
    # distinguishes "field doesn't exist" from "field exists at its no-op
    # value" is not a complete function of the configuration any more.
    #
    # Even at the new field's own default, the hash must move -- and the new
    # field carries the very metadata key the reverted mechanism keyed off.
    # That tag is load-bearing: an untagged field changes the hash whether or
    # not the suppression mechanism exists, so a test using one would pass
    # either way and pin nothing. With the tag, this test passes today
    # (metadata is ignored) and fails the moment anyone reintroduces the
    # omission -- which is the whole point of keeping it.
    without_field = dataclasses.make_dataclass("Component", [("x", float, 1.0)], frozen=True)
    with_field = dataclasses.make_dataclass(
        "Component",
        [
            ("x", float, 1.0),
            (
                "y",
                float,
                dataclasses.field(default=0.0, metadata={"hash_omit_at_default": True}),
            ),
        ],
        frozen=True,
    )

    before = component_hash(describe_components(component=without_field(x=1.0)))
    after = component_hash(describe_components(component=with_field(x=1.0)))

    assert before != after
