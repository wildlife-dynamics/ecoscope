from dataclasses import FrozenInstanceError
from typing import Annotated

import pytest
from pydantic.functional_validators import BeforeValidator

from ecoscope.distributed.decorators import distributed


def test_call_simple():
    @distributed
    def f(a: int, b: int) -> int:
        return a + b
    
    assert f.func(1, 2) == 3
    assert f(1, 2) == 3


def test_frozen_instance():
    @distributed
    def f(a: int) -> int:
        return a

    assert f.validate == False
    with pytest.raises(FrozenInstanceError):
        f.validate = True

    f_new = f.replace(validate=True)
    assert f_new.validate == True


def test_call_with_arg_prevalidators():
    @distributed
    def f(a: Annotated[int, "some metadata field"]) -> int:
        return a
    
    def a_prevalidator(x):
        return x + 1

    # FIXME: __annotations__ is a shared reference between `f` and `f_new`, even though
    # f_new is a different object. this `.replace` call _should_ give us a totally different
    # `.func` attribute, but it doesn't, because of that shared reference.
    f_new = f.replace(arg_prevalidators={"a": a_prevalidator})
    # FIXME: this assert should pass, but it does not:
    # assert f_new.func.__annotations__ != f.func.__annotations__
    f_func_meta: list[BeforeValidator] = f_new.func.__annotations__["a"].__metadata__
    assert a_prevalidator(1) == 2
    assert f_func_meta[0].func(1) == 2  # calling prevalidator directly, we see its behavior
    assert f(1) == 1   # but without `validate=True` we still get normal behavior
    # only when we set validate=True do we finally see the prevalidator is invoked
    assert f.replace(validate=True)(1) == 2
