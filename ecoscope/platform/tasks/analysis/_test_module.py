from wt_registry import register


@register()
def test_function(name: str) -> str:
    from ecoscope.analysis.test_deep.folder.breaks_windows.module import test_func

    return f"{name}_{test_func()}"
