import pandas as pd
from wt_task.skip import SkipSentinel

from ecoscope.platform.tasks.config import (
    concat_string_vars,
    default_if_string_is_empty,
    default_if_string_is_none_or_skip,
    get_column_names_from_dataframe,
    title_case_var,
)


def test_default_if_string_is_none_or_skip() -> None:
    assert default_if_string_is_none_or_skip(SkipSentinel(), "fallback") == "fallback"
    assert default_if_string_is_none_or_skip(None, "fallback") == "fallback"
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
