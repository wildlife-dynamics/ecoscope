import pandas as pd
from wt_task.skip import SkipSentinel

from ecoscope.platform.tasks.config import (
    concat_string_vars,
    default_if_string_is_empty,
    default_if_string_is_none_or_skip,
    get_column_names_from_dataframe,
    prefix_string_var,
    set_bool_var,
    set_list_of_string_vars,
    set_string_var,
    title_case_var,
)


def test_set_string_var() -> None:
    assert set_string_var("hi") == "hi"


def test_set_bool_var() -> None:
    assert set_bool_var(True) is True
    assert set_bool_var(False) is False


def test_set_list_of_string_vars() -> None:
    assert set_list_of_string_vars(["a", "b"]) == ["a", "b"]


def test_prefix_string_var() -> None:
    assert prefix_string_var("bar", "foo_") == "foo_bar"


def test_default_if_string_is_none_or_skip_with_skip() -> None:
    assert default_if_string_is_none_or_skip(SkipSentinel(), "fallback") == "fallback"


def test_default_if_string_is_none_or_skip_with_none() -> None:
    assert default_if_string_is_none_or_skip(None, "fallback") == "fallback"


def test_default_if_string_is_none_or_skip_with_value() -> None:
    assert default_if_string_is_none_or_skip("keep", "fallback") == "keep"


def test_default_if_string_is_empty() -> None:
    assert default_if_string_is_empty("", "fallback") == "fallback"
    assert default_if_string_is_empty("keep", "fallback") == "keep"


def test_concat_string_vars_skips_sentinels() -> None:
    assert concat_string_vars(["a", SkipSentinel(), "b"]) == "ab"


def test_title_case_var() -> None:
    assert title_case_var("hello_world") == "Hello World"


def test_get_column_names_from_dataframe_with_excludes() -> None:
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})

    assert get_column_names_from_dataframe(df, exclude_column_names=["b"]) == ["a", "c"]


def test_get_column_names_from_dataframe_no_excludes() -> None:
    df = pd.DataFrame({"a": [1], "b": [2]})

    assert get_column_names_from_dataframe(df, exclude_column_names=None) == ["a", "b"]
