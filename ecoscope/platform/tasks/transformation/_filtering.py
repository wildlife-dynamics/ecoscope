import logging
from typing import Annotated, cast

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register

from ecoscope.platform.annotations import (
    AdvancedField,
    AnyDataFrame,
    AnyGeoDataFrame,
)

logger = logging.getLogger(__name__)


class Coordinate(BaseModel):
    y: Annotated[float, Field(title="Latitude", description="Example -0.15293")]
    x: Annotated[float, Field(title="Longitude", description="Example 37.30906")]


class BoundingBox(BaseModel):
    min_y: Annotated[float, AdvancedField(default=-90.0, title="Min Latitude")] = -90.0
    max_y: Annotated[float, AdvancedField(default=90.0, title="Max Latitude")] = 90.0
    min_x: Annotated[float, AdvancedField(default=-180.0, title="Min Longitude")] = -180.0
    max_x: Annotated[float, AdvancedField(default=180.0, title="Max Longitude")] = 180.0


@register()
def apply_reloc_coord_filter(
    df: AnyGeoDataFrame,
    bounding_box: Annotated[
        BoundingBox | SkipJsonSchema[None],
        AdvancedField(
            default=BoundingBox(),
            description="Filter events to inside these bounding coordinates.",
        ),
    ] = None,
    filter_point_coords: Annotated[
        list[Coordinate] | SkipJsonSchema[None],
        AdvancedField(
            default=[],
            title="Filter Exact Point Coordinates",
            description=(
                "By adding a filter, the workflow will not include events recorded at the specified coordinates."
            ),
        ),
    ] = None,
    roi_gdf: Annotated[
        AnyGeoDataFrame | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="The ROI geopandas dataframe, in EPSG: 4326, indexed by ROI name",
        ),
    ] = None,
    roi_name: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(default=None, description="The ROI name"),
    ] = None,
    reset_index: Annotated[
        bool | SkipJsonSchema[None],
        AdvancedField(default=True, description="Reset index after filtering"),
    ] = True,
) -> AnyGeoDataFrame:
    import geopandas  # type: ignore[import-untyped]
    import shapely  # type: ignore[import-untyped]
    import shapely.wkt  # type: ignore[import-untyped]

    if filter_point_coords is None:
        filter_point_coords = []
    if bounding_box is None:
        bounding_box = BoundingBox()

    # TODO: move it to ecoscope core
    filter_point_coords = geopandas.GeoSeries(shapely.geometry.Point(coord.x, coord.y) for coord in filter_point_coords)

    def envelope_reloc_filter(geometry) -> bool:
        # We want to 'pass-through' null geometry here
        if geometry is None:
            return True

        return (
            geometry.x > bounding_box.min_x
            and geometry.x < bounding_box.max_x
            and geometry.y > bounding_box.min_y
            and geometry.y < bounding_box.max_y
            and geometry not in filter_point_coords  # type: ignore[operator]
        )

    filtered_df = df.loc[df["geometry"].apply(envelope_reloc_filter), :]  # type: ignore[index,assignment]

    if roi_gdf is not None and roi_name is not None:
        roi = roi_gdf.loc[roi_name, "geometry"]
        filtered_df = filtered_df.loc[filtered_df.intersects(roi), :]  # type: ignore[operator,index,assignment]

    if reset_index:
        filtered_df = filtered_df.reset_index(drop=True)
    return cast(AnyGeoDataFrame, filtered_df)


@register()
def drop_nan_values_by_column(
    df: AnyDataFrame,
    column_name: Annotated[str, Field(description="The column to check")],
) -> AnyDataFrame:
    import numpy as np

    return cast(AnyDataFrame, df[~np.isnan(df[column_name])])


@register()
def drop_null_geometry(
    gdf: AnyGeoDataFrame,
) -> AnyGeoDataFrame:
    return cast(AnyGeoDataFrame, gdf.loc[(~gdf.geometry.isna()) & (~gdf.geometry.is_empty)])
