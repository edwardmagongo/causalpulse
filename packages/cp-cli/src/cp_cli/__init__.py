"""The `causalpulse` command line interface.

Sits at the top of `.importlinter`'s layers contract: it may import every
other package and nothing may import it. That placement is not tidiness. The
evaluation package is allowed to see ground truth, and is itself unimportable
for exactly that reason; a command line that imports the evaluator has to
close the same door one level higher, or an estimator could reach truth by
importing the CLI.
"""

__version__ = "0.1.0"
