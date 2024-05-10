import functools
import types
from dataclasses import FrozenInstanceError, dataclass, field, replace
from typing import Callable, NewType

from pydantic import validate_call
from pydantic.functional_validators import AfterValidator, BeforeValidator

ArgName = NewType("ArgName", str)


def _get_validator_index(
    existing_meta: dict, validator_type: AfterValidator | BeforeValidator,
) -> int:
    """If there are is an existing validator instance of the specified type in the metadata,
    we will overwrite it by re-assigning to its index. if not, we will just add our new
    validator to the end of the list (i.e., index it as -1)
    """
    return (
        -1
        if not any([isinstance(m, validator_type) for m in existing_meta])
        else [i for i, m in enumerate(existing_meta) if isinstance(m, validator_type)][0]
    )


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
            # TODO: assumes there is an annotation and that is of type typing.Annotated,
            # handle case of no annotation, or non-Annotated annotation.
            arg_meta = list(self.func.__annotations__[arg_name].__metadata__)
            bv_idx = _get_validator_index(arg_meta, validator_type=BeforeValidator)
            arg_meta[bv_idx] = BeforeValidator(func=self.arg_prevalidators[arg_name])
            self.func.__annotations__[arg_name].__metadata__ = tuple(arg_meta)
        # TODO: make sure return_postvalidator is a single-argument callable
        # TODO: `strict=True` requires return_postvalidator to be type-hinted and for
        # the type of it's single argument to be the same as the return type of self.func
        if self.return_postvalidator:
            # TODO: assumes there is an annotation and that is of type typing.Annotated,
            # handle case of no annotation, or non-Annotated annotation.
            return_meta = list(self.func.__annotations__["return"].__metadata__)
            av_idx = _get_validator_index(return_meta, validator_type=AfterValidator)
            return_meta[av_idx] = AfterValidator(func=self.return_postvalidator)
            self.func.__annotations__["return"].__metadata__ = tuple(return_meta)
        if self.validate:
            self.func = validate_call(self.func, validate_return=True)
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
