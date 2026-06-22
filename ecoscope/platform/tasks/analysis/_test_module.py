from wt_registry import register

from ecoscope.analysis.test_deep.folder.breaks_windows.module import test_func


@register()
def test_function():
    return test_func()
