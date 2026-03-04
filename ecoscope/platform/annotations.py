from typing import Annotated, Any, Callable, ParamSpec, TypeVar, Union, get_origin

import pandas as pd
import pandera.pandas as pa
import pandera.typing as pa_typing
from pydantic import Field, GetJsonSchemaHandler
from pydantic.fields import FieldInfo
from pydantic.json_schema import JsonSchemaValue, WithJsonSchema
from pydantic_core import core_schema as cs

FP = ParamSpec("FP")


def create_custom_field(
    field_name: str,
    always_set: dict,
    require_default: bool,
    *,
    __f: Callable[FP, FieldInfo] = Field,  # type: ignore[assignment]
) -> Callable[FP, FieldInfo]:
    """A factory function which creates a custom Field with `always_set` kwargs always set.
    Note that the `always_set` kwargs will not be omitted from the signature of the returned
    custom Field as understood by static type checkers, but they will raise `ValueError` at
    runtime if overridden. If `require_default` is `True`, the user must provide a default when
    calling the custom Field.

    Args:
        always_set (dict): The keyword arguments to always set on Field.
        require_default (bool): If True, the user must provide a default when calling
            the custom Field.
        __f (Callable[FP, FR], optional): Defaults to Field. This is included in the signature
            for editor support only; the default should not be overridden.
    """
    if not __f == Field:
        raise ValueError(
            "This factory only supports `pydantic.Field` as the value for `__f`; `__f` "
            "is included in the signature for editor support only; the default should "
            "not be overridden."
        )

    def wrapper(*args, **kwargs):
        # Take a copy of always_set since we may want to mutate it
        fixed_kwargs = always_set.copy()
        if require_default and (not args and "default" not in kwargs):
            raise ValueError(f"A default is required for fields constructed with '{field_name}'.")
        # In the specific case of json_schema_extra, we want to merge anything in
        # always_set into the user provided kwargs, and raise on any conflicts
        if always_json_schema_extra := always_set.get("json_schema_extra", False):
            user_json_schema_extra = kwargs.pop("json_schema_extra", {})
            if any(key in user_json_schema_extra for key in always_json_schema_extra):
                raise ValueError(
                    f"Fields constructed with '{field_name}' cannot override "
                    f"json_schema_extra: {always_json_schema_extra.keys()}"
                )
            fixed_kwargs["json_schema_extra"] = user_json_schema_extra | always_json_schema_extra
        if any(key in kwargs for key in always_set):
            raise ValueError(f"Fields constructed with '{field_name}' cannot override: {always_set.keys()}")
        kwargs.update(fixed_kwargs)
        return __f(*args, **kwargs)

    return wrapper


AdvancedField = create_custom_field(  # type: ignore[var-annotated]
    "AdvancedField",
    always_set={"json_schema_extra": {"ecoscope:advanced": True}},
    require_default=True,
)


class JsonSerializableDataFrameModel(pa.DataFrameModel):
    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: cs.CoreSchema, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
        return cls.to_json_schema()


class EmptyDataFrameModel(JsonSerializableDataFrameModel):
    @pa.dataframe_check
    def is_empty(cls, df: pd.DataFrame):  # type: ignore[misc]
        return df.empty


DataFrameSchema = TypeVar("DataFrameSchema", bound=JsonSerializableDataFrameModel)

DataFrame = Annotated[
    pa_typing.DataFrame[DataFrameSchema],
    # pa.typing.DataFrame is very hard to meaningfully serialize to JSON. Pandera itself
    # does not yet support this, see: https://github.com/unionai-oss/pandera/issues/421.
    # The "ideal" workaround I think involves overriding `__get_pydantic_json_schema__`,
    # as worked for `JsonSerializableDataFrameModel` above, however the meaningful data
    # we would want to access within that classmethod is the subscripted `DataframeSchema`
    # type passed by the user. This *is* accessible outside classmethods (in the context
    # in which the subscription happens) with `typing.get_args`. However, that does not work
    # on the `cls` object we have in the classmethod. It *feels* like this SO post somehow
    # points towards a solution: https://stackoverflow.com/a/65064882, but I really struggled
    # to make it work. So in the interim, we will just always use the generic schema declared
    # below, which will not contain any schema-specific information. This *will not* affect
    # validation behavior, only JSON Schema generation.
    WithJsonSchema({"type": "ecoscope_workflows.annotations.DataFrame"}),
]


class GeoDataFrameBaseSchema(JsonSerializableDataFrameModel):
    # pandera does support geopandas types (https://pandera.readthedocs.io/en/stable/geopandas.html)
    # but this would require this module depending on geopandas, which we are trying to avoid. so
    # unless we come up with another solution, for now we are letting `geometry` contain anything.
    geometry: pa_typing.Series[Any] = pa.Field(nullable=True)


class StrictGeoDataFrameBaseSchema(JsonSerializableDataFrameModel):
    geometry: pa_typing.Series[Any] = pa.Field(nullable=False)


AnyDataFrame = DataFrame[JsonSerializableDataFrameModel]
AnyGeoDataFrame = DataFrame[GeoDataFrameBaseSchema]
EmptyDataFrame = DataFrame[EmptyDataFrameModel]
AnyDataFrameOrEmpty = Union[AnyDataFrame, EmptyDataFrame]
AnyGeoDataFrameOrEmpty = Union[AnyGeoDataFrame, EmptyDataFrame]


def is_subscripted_pandera_dataframe(obj):
    if hasattr(obj, "__origin__") and hasattr(obj, "__args__"):
        if get_origin(obj) == pa_typing.DataFrame:
            return True
    return False
