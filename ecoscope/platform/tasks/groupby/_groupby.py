from typing import Annotated, Any, cast

import pandas as pd
from pydantic import Field
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register
from wt_task.skip import SkippedDependencyFallback, SkipSentinel

from ecoscope.platform.annotations import AnyDataFrame
from ecoscope.platform.indexes import (
    AllGrouper,
    CompositeFilter,
    IndexName,
    IndexValue,
    UserDefinedGroupers,
    ValueGrouper,
)


def _groupkey_to_composite_filter(groupers: list[IndexName], index_values: tuple[IndexValue, ...]) -> CompositeFilter:
    """Given the list of `groupers` used to group a dataframe, convert a group key
    tuple (the pandas native representation) to a composite filter (our representation).

    Examples:

    ```python
    >>> groupers = ["month", "year"]
    >>> index_values = (1, 2021)
    >>> _groupkey_to_composite_filter(groupers, index_values)
    (('month', '=', 1), ('year', '=', 2021))
    >>> groupers = ["animal_name", "species"]
    >>> index_values = ("Jo", "Elephas maximus")
    >>> _groupkey_to_composite_filter(groupers, index_values)
    (('animal_name', '=', 'Jo'), ('species', '=', 'Elephas maximus'))

    ```

    """
    return tuple((index, "=", value) for index, value in zip(groupers, index_values))


def _groupers_field_json_schema_extra(schema: dict) -> None:
    """Edit the JSON schema for the `groupers` field to make it more user-friendly."""
    schema["items"]["title"] = "Group by"


@register()
def set_groupers(
    groupers: Annotated[
        UserDefinedGroupers | SkipJsonSchema[None],
        Field(
            default=None,
            title=" ",  # deliberately a single empty space, to hide the field in the UI
            json_schema_extra=_groupers_field_json_schema_extra,
            description="""\
            Specify how the data should be grouped to create the views for your dashboard.
            This field is optional; if left blank, all the data will appear in a single view.
            """,
        ),
    ] = None,
) -> Annotated[
    AllGrouper | UserDefinedGroupers,
    Field(
        description="""\
        Passthrough of the input groupers, for use in downstream tasks.
        If no groupers are given, the `AllGrouper` is returned instead.
        """,
    ),
]:
    return groupers if groupers else AllGrouper()


KeyedIterableOfDataFrames = list[tuple[CompositeFilter, AnyDataFrame]]
KeyedIterableOfAny = list[tuple[CompositeFilter, Any]]
CombinedKeyedIterable = list[tuple[CompositeFilter, list[Any]]]


@register()
def split_groups(
    df: AnyDataFrame,
    groupers: Annotated[
        AllGrouper | UserDefinedGroupers,
        Field(description="Index(es) and/or column(s) to group by"),
    ],
) -> Annotated[
    KeyedIterableOfDataFrames,
    Field(
        description="""\
        List of 2-tuples of key:value pairs. Each key:value pair consists of a composite
        filter (the key) and the corresponding subset of the input dataframe (the value).
        """
    ),
]:
    if isinstance(groupers, AllGrouper):
        # if the special-cased AllGrouper is given, then just passthrough the `df`, wrapped
        # in the container expected by downstream tasks. we don't actually have to apply the
        # "all" index to the dataframe, because that is not used downstream.
        return [(((groupers.index_name, "=", "True"),), df)]

    value_groupers = [vg for vg in groupers if isinstance(vg, ValueGrouper)]

    for vg in value_groupers:
        key = vg.index_name
        if key in df.index.names:
            idx_level = df.index.names.index(key)
            # Coerce any None/NA values into string "None"
            if isinstance(df.index, pd.MultiIndex):
                index_frame = df.index.to_frame()
                index_frame.fillna({key: "None"}, inplace=True)
                df.index = pd.MultiIndex.from_frame(index_frame)
            else:
                index_update = df.index.get_level_values(idx_level).fillna("None")
                df.index = index_update
            if not pd.api.types.is_string_dtype(df.index.get_level_values(idx_level)):
                raise ValueError(
                    "All indexes used as categorical value groupers must contain "
                    f"only string data (with no null values); got {df.dtypes}."
                )
        elif key in df.columns:
            # Coerce any None/NA values into string "None"
            df.fillna({key: "None"}, inplace=True)
            if not pd.api.types.is_string_dtype(df[key]):
                raise ValueError(
                    "All columns used as categorical value groupers must contain "
                    f"only string data (with no null values); got {df.dtypes}."
                )
        else:
            raise ValueError(f"Value grouper '{key}' is neither a column nor an index in the DataFrame")

    # TODO: configurable cardinality constraint with a default?
    grouper_index_names = [g.index_name for g in groupers]
    grouped = df.groupby(grouper_index_names)
    return [
        (_groupkey_to_composite_filter(grouper_index_names, index_value), group)  # type: ignore[misc,arg-type]
        for index_value, group in grouped
    ]


def _drop_skip_sentinels(
    iterables: list[KeyedIterableOfAny],
) -> list[KeyedIterableOfAny]:
    """Drop any SkipSentinel values from the keyed iterable."""
    return [i for i in iterables for elem in i if not isinstance(elem[1], SkipSentinel)]


@register()
def groupbykey(
    iterables: Annotated[
        list[KeyedIterableOfAny],
        Field(description="List of keyed iterables"),
        SkippedDependencyFallback(_drop_skip_sentinels),
    ],
) -> Annotated[
    CombinedKeyedIterable,
    Field(
        description="""
        Flattened collection of keyed iterables with values associated with matching keys combined.
        """
    ),
]:
    seen = set()
    out: dict[CompositeFilter, list] = {}
    for i in iterables:
        for key, value in i:
            if key in seen:
                out[key].append(value)
            else:
                seen.add(key)
                out[key] = [value]
    return list(out.items())


@register()
def merge_df(
    iterables: Annotated[
        KeyedIterableOfDataFrames,
        Field(
            description="""\
        List of 2-tuples of (key:value) pairs. Each (key:value) pair consists of a composite
        filter (the key) and the corresponding subset of the input dataframe (the value).
        """
        ),
    ],
) -> Annotated[
    AnyDataFrame,
    Field(
        description="""
        The merged dataframe, with the index set to the composite filter keys.
        """
    ),
]:
    index_names = [filter[0] for filter in iterables[0][0]]

    dfs_with_index = []
    for filter, df in iterables:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("All values must be pandas DataFrames.")
        if len(filter) != len(index_names):
            raise ValueError(f"Filter length {len(filter)} does not match index names length {len(index_names)}.")
        temp_df = df.copy()
        for i, index_name in enumerate(index_names):
            temp_df[index_name] = filter[i][2]
        temp_df.set_index(index_names, inplace=True)
        dfs_with_index.append(temp_df)
    result_df = pd.concat(dfs_with_index, axis=0)
    return cast(
        AnyDataFrame,
        result_df,
    )
