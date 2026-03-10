from ecoscope.platform.tasks.config import (
    default_if_string_is_empty,
    default_if_string_is_none_or_skip,
)


def test_default_if_string_is_none_or_skip():
    value = None
    default = "Hello"
    assert default == default_if_string_is_none_or_skip(value, default)

    value = "Use Me"
    default = "Hello"
    assert value == default_if_string_is_none_or_skip(value, default)


def test_default_if_string_is_empty():
    value = ""
    default = "Hello"
    assert default == default_if_string_is_empty(value, default)

    value = "Use Me"
    default = "Hello"
    assert value == default_if_string_is_empty(value, default)
