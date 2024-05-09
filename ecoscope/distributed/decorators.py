from dataclasses import dataclass, replace
from typing import Callable, NewType

import functools

ArgName = NewType("ArgName", str)


@dataclass
class distributed:
    func: Callable
    arg_loaders: dict[ArgName, Callable] | None = None
    dump_return: Callable | None = None
    # TODO memoize: bool
    # TODO arg_hashers: dict (for memoize) - need to be stable across processes

    def __post_init__(self):
        functools.update_wrapper(self, self.func)

    def override(self, changes: dict) -> "distributed":
        return replace(self, changes)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
