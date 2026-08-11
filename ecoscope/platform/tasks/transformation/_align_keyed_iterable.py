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
    target's first entry). Keys present only in the target are kept, so no
    data is dropped. Result order is reference keys first, then any
    target-only keys.

    Raises:
        ValueError: If the target contains duplicate keys, or if the target is
            empty and no ``fill_value`` is given (there is no schema to fill
            the reference keys with).

    Args:
        target: Keyed iterable of DataFrames to reindex.
        reference: Keyed iterable providing the key universe.
        fill_value: DataFrame used for keys missing from the target.

    Returns:
        Keyed iterable covering the union of reference and target keys.
    """
    target_by_key: dict[CompositeFilter, pd.DataFrame] = {}
    for key, value in target:
        if key in target_by_key:
            raise ValueError(f"Duplicate key in target: {key!r}")
        target_by_key[key] = value
    fill: pd.DataFrame
    if fill_value is not None:
        fill = fill_value
    elif target:
        fill = target[0][1].iloc[0:0]
    else:
        raise ValueError(
            "target is empty and no fill_value was given; pass fill_value to define the schema for filled keys."
        )

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
