"""The evidence report over this project's result types.

Sits between the figures package and the command line in `.importlinter`'s
layers contract: it may import the result types and the figures, and nothing
below it may import this.

Every section here is two pieces -- a pure function returning exactly what
will be written, and a thin renderer per output format. The split is what
makes the report testable against structure rather than against rendered
strings, which break on every wording change.

Two rules are reused rather than restated. An effect estimate is never
rendered without its interval, via the figures package's own guard. And no
document is emitted whose prose fails this project's language lint --
including text interpolated from a caller's data, which no other check in
this repository covers.
"""

__version__ = "0.1.0"
