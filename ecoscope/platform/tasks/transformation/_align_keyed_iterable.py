from typing import Annotated, Any, cast

import pandas as pd
from pydantic import Field
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register

from ecoscope.platform.annotations import AnyDataFrame
from ecoscope.platform.indexes import CompositeFilter

KeyedIterableOfDataFrames = list[tuple[CompositeFilter, AnyDataFrame]]
KeyedIterableOfAny = list[tuple[CompositeFilter, Any]]


@register()
def align_keyed_iterable_to_reference(
    target: Annotated[
        KeyedIterableOfDataFrames,
        Field(
            description="Keyed iterable of DataFrames to reindex to the reference keys.",
            exclude=True,
        ),
    ],
    reference: Annotated[
        KeyedIterableOfAny,
        Field(
            description="Keyed iterable whose keys define the key universe to align to.",
            exclude=True,
        ),
    ],
    fill_value: Annotated[
        AnyDataFrame | SkipJsonSchema[None],
        Field(
            description=(
                "DataFrame used to fill keys missing from the target. Defaults to "
                "an empty DataFrame matching the schema of the target's first entry."
            ),
            exclude=True,
        ),
    ] = None,
) -> KeyedIterableOfDataFrames:
    """
    Reindex ``target`` to include every key in ``reference``.

    Keys present in the reference but missing from the target are filled with
    ``fill_value`` (default: an empty DataFrame matching the schema of the
    target's first entry, or a schema-less empty DataFrame if the target is
    empty). Keys present only in the target are kept, so no data is dropped.
    Result order is reference keys first, then any target-only keys.

    Args:
        target: Keyed iterable of DataFrames to reindex.
        reference: Keyed iterable providing the key universe.
        fill_value: DataFrame used for keys missing from the target.

    Returns:
        Keyed iterable covering the union of reference and target keys.
    """
    target_by_key = dict(target)
    fill: pd.DataFrame
    if fill_value is not None:
        fill = fill_value
    elif target:
        fill = target[0][1].iloc[0:0]
    else:
        fill = pd.DataFrame()

    out = []
    seen = set()
    for key, _ in reference:
        if key in seen:
            continue
        seen.add(key)
        out.append((key, target_by_key[key] if key in target_by_key else fill.copy()))
    for key, value in target:
        if key not in seen:
            seen.add(key)
            out.append((key, value))
    return cast(KeyedIterableOfDataFrames, out)
