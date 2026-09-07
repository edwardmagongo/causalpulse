"""Static publication-quality figures over this project's result types.

Sits between the evaluation package and the command line in
`.importlinter`'s layers contract: it may import the result types, and
nothing below it may import this. That placement keeps estimation free of a
plotting dependency, and leaves the layer above free to embed figures in a
report.

Every figure here is two pieces -- a pure function returning exactly what will
be plotted, and a thin function drawing it onto a caller-supplied `Axes`. The
split is what makes the figures testable against numbers rather than pixels,
which matters in a project with no continuous integration to regenerate
baseline images against.
"""

import matplotlib

# Selected before pyplot is imported anywhere. The test suite is headless and
# turns warnings into errors, so a backend that tried to open a window would
# fail in whichever test happened to draw first rather than here.
matplotlib.use("Agg")

__version__ = "0.1.0"
