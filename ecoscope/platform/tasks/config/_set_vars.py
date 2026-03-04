from typing import Annotated

from ecoscope.platform.annotations import AnyDataFrame
from pydantic import Field
from wt_registry import register
from wt_task.skip import SkipSentinel


@register()
def set_string_var(
    var: Annotated[str, Field(title="")],
) -> str:
    return var


@register()
def set_bool_var(
    var: Annotated[bool, Field(title="")],
) -> bool:
    return var


@register()
def set_list_of_string_vars(
    vars: Annotated[list[str], Field(title="")],
) -> list[str]:
    return vars


@register()
def prefix_string_var(
    var: Annotated[str, Field(title="")],
    prefix: Annotated[str, Field(title="")],
) -> str:
    return f"{prefix}{var}"


@register()
def default_if_string_is_none_or_skip(
    value: Annotated[str | None | SkipSentinel, Field(description="The value to passthrough")],
    default: Annotated[str, Field(description="Default if `value` is None or Skip")],
) -> str:
    return default if value is None or isinstance(value, SkipSentinel) else value


@register()
def default_if_string_is_empty(
    value: Annotated[str, Field(description="The value to passthrough")],
    default: Annotated[str | None | SkipSentinel, Field(description="Default if `value` is None")],
) -> str | None | SkipSentinel:
    return default if value == "" else value


@register()
def concat_string_vars(
    values: Annotated[list[str | SkipSentinel], Field(description="The values to concatenate")],
) -> str:
    values_to_concat: list[str] = [v for v in values if not isinstance(v, SkipSentinel)]
    return "".join(values_to_concat)


@register()
def title_case_var(
    var: Annotated[str, Field(...)],
) -> str:
    return var.replace("_", " ").title()


@register()
def get_column_names_from_dataframe(
    df: Annotated[AnyDataFrame, Field(...)],
    exclude_column_names: Annotated[list[str] | None, Field(...)],
) -> list[str]:
    columns = df.columns.to_list()

    if exclude_column_names:
        columns = [col for col in columns if col not in exclude_column_names]

    return columns
