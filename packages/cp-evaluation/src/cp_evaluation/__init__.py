"""Benchmark grid over the simulated scenarios.

The one package that imports both the ground-truth generators and the
estimators, because that combination is what a benchmark is. It therefore sits
at the top of `.importlinter`'s layers contract: it may import everything, and
nothing may import it. Without that placement the ground-truth firewall would
be reachable transitively, through this package, by exactly the packages the
firewall exists to keep away from truth.
"""

__version__ = "0.1.0"
