from ecoscope.distributed.decorators import distributed


def f(a: int, b: int) -> int:
    return a + b


def test_call_simple():
    d = distributed(f)
    assert f(1, 2) == 3
    assert d(1, 2) == 3


def test_replace():
    d0 = distributed(f)
    assert d0.arg_prevalidators == {}
    assert d0(1, 2) == 3

    arg_prevalidators = {"a": lambda x: int(x)}
    d1 = d0.replace(arg_prevalidators=arg_prevalidators)
    assert d1.arg_prevalidators == arg_prevalidators
    # although we assigned arg_prevalidators, this is not
    # testing that they *work*, just that we can assign them
    assert d1(1, 2) == 3
