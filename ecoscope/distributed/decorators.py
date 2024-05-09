import functools
from dataclasses import dataclass, field, replace
from typing import Callable, NewType

ArgName = NewType("ArgName", str)


@dataclass
class distributed:
    """
    Parameters
    ----------
    func : Callable
        The function (or other callable) to wrap.
    arg_prevalidators : dict[ArgName, Callable]
        ...
    """
    func: Callable
    arg_prevalidators: dict[ArgName, Callable] = field(default_factory=dict)
    return_postvalidator: Callable | None = None

    def __post_init__(self):
        functools.update_wrapper(self, self.func)
        # NOTE: the validation steps described below do suggest BaseModel could be useful,
        # but because this class does not need to be serialized or deserialized, and is not
        # designed to directly handle user input, i do think keeping the implementation as
        # lightweight as possible is desirable.

        # TODO: make sure the keys of arg_prevalidators are all arg_names on self.func
        # TODO: `strict=True` requires the callable values of arg_prevalidators to be
        # hinted with a return type, and for the return type of the callable to match
        # the input type of the matching arg on self.func

        # TODO: make sure return_postvalidator is a single-argument callable
        # TODO: `strict=True` requires return_postvalidator to be type-hinted and for
        # the type of it's single argument to be the same as the return type of self.func

    def replace(self, /, **changes: dict) -> "distributed":
        return replace(self, **changes)

    def __call__(self, *args, **kwargs):
        # TODO: apply arg_prevalidators and arg_postvalidators to self.func annotations
        return self.func(*args, **kwargs)
