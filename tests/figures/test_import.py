def test_importing_the_package_selects_a_non_interactive_backend():
    # The suite runs headless and treats warnings as errors. A backend that
    # tries to open a window would either fail or warn, and either way the
    # failure would appear in whichever test happened to draw first.
    import cp_figures  # noqa: F401
    import matplotlib

    assert matplotlib.get_backend().lower() == "agg"


def test_the_figures_package_may_import_the_result_types():
    import cp_figures  # noqa: F401
    from cp_core.results import EffectEstimate  # noqa: F401
