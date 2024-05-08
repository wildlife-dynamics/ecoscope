from dataclasses import dataclass, replace
from typing import Callable

import functools


@dataclass
class serde:
    func: Callable
    serializer: Callable | None = None
    deserializer: Callable | None = None

    def __post_init__(self):
        functools.update_wrapper(self, self.func)

    def override(self, changes: dict):
        return replace(self, changes)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
