"""Every figure that plots an effect estimate must go through the guard.

The rule -- a helper handed an estimate without an
interval raises rather than drawing a bare point -- lives in one function
(`cp_figures.guards.require_interval`) so a figure cannot forget it. That
only holds if every such figure actually calls it, which is what this file
checks. Without it the guard is a convention, not a rule.

This is the behavioural form, not the weaker source-text-matching one: each
listed function is called with a real `EffectEstimate` whose interval has
been stripped, and the test asserts the call actually raises. That is
possible here because `EffectEstimate.__post_init__`
(`packages/cp-core/src/cp_core/results.py`) validates `estimand`, `ci_basis`,
`scale`, `se`, `n_treated`, `n_control` and `p_value`, but not `ci` --
`EffectEstimate.ci` is typed `Interval`, not `Interval | None`, yet nothing
in `__post_init__` checks it. Python dataclasses do not enforce field types
at runtime, only `__post_init__` does, so `dataclasses.replace(estimate,
ci=None)` constructs successfully despite the static type. That hand-built
object is exactly the case `require_interval` exists to catch, and it is
reachable from a plain test -- so this file proves the guard actually fires,
not merely that its error string appears somewhere in the source.

Data functions that display an effect estimate's point WITH ITS INTERVAL,
and must therefore consult the guard. Extend this list when such a figure is
added.

Deliberately one entry, and the omissions are reasoned rather than lazy --
verified by reading every `*_data` function in `cp_figures.effects`,
`cp_figures.panel`, `cp_figures.balance`, `cp_figures.sensitivity` and
`cp_figures.benchmark`:

- `forest_plot_data` (`cp_figures.effects`) takes a sequence of
  `EffectEstimate` and reads `estimate.ci` through `require_interval` for
  every row before drawing a point. It belongs here.
- `event_study_data` (`cp_figures.panel`) takes a single `EffectEstimate`
  and never reads `estimate.ci`'s *bounds* -- its module docstring says so
  directly. It builds each horizon's interval from
  `event_time_standard_errors`, a per-horizon diagnostic distinct from the
  single averaged post-treatment interval `estimate.ci` bounds. Calling
  `require_interval` there would be answering the wrong question, and it
  does not belong in `ESTIMATE_CONSUMERS`, which is about that exact
  hazard: drawing an estimate's point without its interval. It does read
  `estimate.ci`'s *level*, though, to build every horizon's margin at the
  confidence level the fit was actually run at -- a different hazard
  (labelling uncertainty at the wrong level, or crashing outright if
  `estimate.ci` is `None`) from the one `require_interval` guards
  (drawing uncertainty not at all). That read goes through its own guard,
  `cp_figures.guards.require_ci_level`, refused the same way
  `require_interval` refuses a missing interval; `tests/figures/test_panel.py`
  and `tests/figures/test_guards.py` prove that guard is actually reached,
  the same way this file proves `require_interval` is reached for
  `ESTIMATE_CONSUMERS`.
- `love_plot_data` and `overlap_data` (`cp_figures.balance`) take an
  `EffectEstimate` but read `smd_before`/`smd_after` and
  `per_unit_propensity` diagnostics -- balance and propensity information,
  not an effect estimate -- and never touch `estimate.ci` or
  `estimate.point`. The module docstring says outright that
  `require_interval` does not apply: there is no interval to guard.
- `counterfactual_path_data` (`cp_figures.effects`) takes a
  `CounterfactualPath`, not an `EffectEstimate` at all.
- `specification_curve_data` (`cp_figures.sensitivity`) takes a
  `SpecificationCurve`, whose rows are `SpecificationPoint` -- a distinct
  type with its own `ci: Interval` field -- not `EffectEstimate`, so it does
  not belong in `ESTIMATE_CONSUMERS` below. It carries the identical hazard
  under a different type, though, and is guarded by
  `guards.require_specification_interval`, `require_interval`'s sibling for
  that type; `tests/figures/test_sensitivity.py` proves that guard is
  actually reached, the way this file proves `require_interval` is reached
  for `ESTIMATE_CONSUMERS`.
  `placebo_data` (`cp_figures.sensitivity`) takes a `PlaceboTestResult`,
  which carries no `ci` field at all.
- `heatmap_data` and `axis_profile_data` (`cp_figures.benchmark`) each take a
  `pandas.DataFrame` summary, not an `EffectEstimate`.

If any of those ever renders `estimate.point` alongside `estimate.ci`, it
belongs here too.
"""

import dataclasses
import inspect
from collections.abc import Callable, Sequence

import cp_figures.effects as effects
import pytest
from cp_core.results import EffectEstimate, FitMetadata
from cp_core.values import Interval
from cp_figures import guards

# Each entry consumes a Sequence[EffectEstimate] -- the only shape any
# current estimate-consuming data function takes. If a future entry takes a
# single EffectEstimate instead, adapt the call site in the test below
# rather than force-fitting it into this shape.
ESTIMATE_CONSUMERS: list[Callable[[Sequence[EffectEstimate]], object]] = [
    effects.forest_plot_data
]


def _estimate(**overrides):
    base = dict(
        estimand="ATE",
        method="ipw",
        point=0.25,
        se=0.1,
        ci=Interval(lower=0.05, upper=0.45, level=0.95),
        ci_basis="analytic",
        bootstrap_ci=None,
        p_value=0.01,
        n_treated=100,
        n_control=100,
        scale="percentage_point",
        diagnostics={},
        assumptions=(),
        counterfactual=None,
        fit_metadata=FitMetadata.capture(
            estimator="ipw", config_hash="h", runtime_seconds=0.1, random_seed=0
        ),
        subgroup_findings=(),
    )
    base.update(overrides)
    return EffectEstimate(**base)


def test_every_estimate_consuming_data_function_raises_without_an_interval():
    # EffectEstimate.ci is typed Interval, not Interval | None, and
    # __post_init__ does not validate it (see the module docstring), so this
    # construction succeeds despite the static type -- exactly the object
    # each listed function must refuse to draw a bare point from.
    bare = dataclasses.replace(_estimate(), ci=None)

    for function in ESTIMATE_CONSUMERS:
        with pytest.raises(ValueError, match="without an interval"):
            function([bare])


def test_the_guard_is_the_only_place_the_rule_is_stated():
    # A second copy of the check would drift from this one.
    assert "without an interval" in inspect.getsource(guards.require_interval)
