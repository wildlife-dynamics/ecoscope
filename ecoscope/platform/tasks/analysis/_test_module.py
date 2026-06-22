from wt_registry import register


@register()
def test_function() -> str:
    from ecoscope.analysis.test_deep.folder.breaks_windows.module import test_func

    return test_func()
