from inspect import signature
from typing import Annotated, get_args

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from pydantic.fields import FieldInfo
from pydantic.json_schema import GenerateJsonSchema

import ecoscope.distributed.types as etd

class MatchingSchema(GenerateJsonSchema):
    def generate(self, schema, mode='validation'):
        json_schema = super().generate(schema, mode=mode)
        if "title" in json_schema:
            # TypeAdapter(func).json_schema() has no "title"
            del json_schema["title"]
        # By default, this is False for TypeAdapter(func).json_schema(),
        # and omitted for BaseModel.model_json_schema(). Another way to
        # get this to match is to use ConfigDict(extra=False) as config
        # for the BaseModel, since we're subclassing GenerateJsonSchema
        # anyway, this works too...
        json_schema["additionalProperties"] = False
        return json_schema


def test_jsonschema_from_signature_basic():
    class FuncSignature(BaseModel):
        foo: int
        bar: str

    def func(foo: int, bar: str): ...

    from_func = TypeAdapter(func).json_schema(schema_generator=MatchingSchema)
    from_model = FuncSignature.model_json_schema(schema_generator=MatchingSchema)
    assert from_func == from_model


class SurfacesDescriptionSchema(MatchingSchema):
    def generate(self, schema, mode='validation'):
        json_schema = super().generate(schema, mode=mode)
        if "function" in schema:
            for p in json_schema["properties"]:
                annotation_args = get_args(signature(schema["function"]).parameters[p].annotation)
                if any([isinstance(arg, FieldInfo) for arg in annotation_args]):
                    Field: FieldInfo = [arg for arg in annotation_args if isinstance(arg, FieldInfo)][0]
                    if Field.description:
                        json_schema["properties"][p]["description"] = Field.description
        return json_schema


def test_jsonschema_from_signature_nontrivial():
    config_dict = ConfigDict(arbitrary_types_allowed=True)

    class TimeDensityConfig(BaseModel):
        model_config = config_dict

        input_df: etd.InputDataframe
        pixel_size: Annotated[
            float,
            Field(default=250.0, description="Pixel size for raster profile."),
        ]
        crs: Annotated[str, Field(default="ESRI:102022")]
        nodata_value: Annotated[float, Field(default=float("nan"), allow_inf_nan=True)]
        band_count: Annotated[int, Field(default=1)]
        max_speed_factor: Annotated[float, Field(default=1.05)]
        expansion_factor: Annotated[float, Field(default=1.3)]
        percentiles: Annotated[list[float], Field(default=[50.0, 60.0, 70.0, 80.0, 90.0, 95.0])]

    def calculate_time_density(
        input_df: etd.InputDataframe,
        pixel_size: Annotated[
            float,
            Field(default=250.0, description="Pixel size for raster profile."),
        ],
        crs: Annotated[str, Field(default="ESRI:102022")],
        nodata_value: Annotated[float, Field(default=float("nan"), allow_inf_nan=True)],
        band_count: Annotated[int, Field(default=1)],
        max_speed_factor: Annotated[float, Field(default=1.05)],
        expansion_factor: Annotated[float, Field(default=1.3)],
        percentiles: Annotated[list[float], Field(default=[50.0, 60.0, 70.0, 80.0, 90.0, 95.0])],
    ): ...

    schema_kws = dict(schema_generator=SurfacesDescriptionSchema)
    from_func = TypeAdapter(
        calculate_time_density,
        config=config_dict,
    ).json_schema(**schema_kws)
    from_model = TimeDensityConfig.model_json_schema(**schema_kws)
    # `nodata_value` defaults to `nan`; numpy evals `nan == nan` as true
    np.testing.assert_equal(from_func, from_model)
