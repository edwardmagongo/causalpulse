"""Plain-language interpretation of an analysis, refused unless it can be traced.

An analysis becomes a table of facts. A model writes a narrative in which
every value is a placeholder naming a fact. A deterministic verifier refuses
any narrative it cannot trace back to the table, and only then are the
placeholders replaced.

`cp_llm.extract` is the one module that imports the result types, and
`cp_llm.client` the one that imports the model SDK.
"""

__version__ = "0.1.0"
