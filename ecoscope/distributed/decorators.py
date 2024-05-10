import functools
import types
from dataclasses import FrozenInstanceError, dataclass, field, replace
from typing import Callable, NewType

from pydantic import validate_call
from pydantic.functional_validators import BeforeValidator

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
    func: types.FunctionType
    arg_prevalidators: dict[ArgName, Callable] = field(default_factory=dict)
    return_postvalidator: Callable | None = None
    validate: bool = False
    _initialized: bool = False

    def __post_init__(self):
        # TODO: make sure the keys of arg_prevalidators are all arg_names on self.func
        # TODO: `strict=True` requires the callable values of arg_prevalidators to be
        # hinted with a return type, and for the return type of the callable to match
        # the input type of the matching arg on self.func
        for arg_name in self.arg_prevalidators:
            # assumes there is an annotation and that is of type typing.Annotated
            # TODO: handle case of no annotation, or non-Annotated annotation
            meta = list(self.func.__annotations__[arg_name].__metadata__)
            # if there are is an existing BeforeValidator instance in the metadata,
            # we will overwrite it by re-assigning to its index. if not, we will just
            # add our new BeforeValidator to the end of the list (i.e., index it as -1)
            bf_idx = (
                -1
                if not any([isinstance(m, BeforeValidator) for m in meta])
                else [i for i, m in enumerate(meta) if isinstance(m, BeforeValidator)][0]
            )
            meta[bf_idx] = BeforeValidator(func=self.arg_prevalidators[arg_name])
            self.func.__annotations__[arg_name].__metadata__ = tuple(meta)
        if self.validate:
            self.func = validate_call(self.func, validate_return=True)
        # TODO: make sure return_postvalidator is a single-argument callable
        # TODO: `strict=True` requires return_postvalidator to be type-hinted and for
        # the type of it's single argument to be the same as the return type of self.func
        functools.update_wrapper(self, self.func)
        self._initialized = True

    def replace(self, /, **changes: dict) -> "distributed":
        self._initialized = False
        return replace(self, **changes)

    def __setattr__(self, name, value):
        if self._initialized and name != "_initialized":
            raise FrozenInstanceError(
                "Re-assignment of attributes not permitted post-init. "
                "Use `self.replace` to create a new instance instead."
            )
        return super().__setattr__(name, value)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
