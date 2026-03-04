from typing import Annotated, TypeAlias

from pydantic import Field
from wt_registry import register

from ecoscope.platform.annotations import AdvancedField, AnyGeoDataFrame
from ecoscope.platform.tasks.analysis._time_density import (
    AutoScaleGridCellSize,
    AutoScaleOrCustomAnnotation,
    CrsAnnotation,
    CustomGridCellSize,
)

IntersectingOnlyAnnotation: TypeAlias = Annotated[
    bool,
    AdvancedField(
        default=False,
        description="Whether to return only grid cells intersecting with the aoi.",
    ),
]
AoiAnnotation: TypeAlias = Annotated[
    AnyGeoDataFrame,
    Field(description="The area to create a meshgrid of.", exclude=True),
]


@register()
def create_meshgrid(
    aoi: AoiAnnotation,
    auto_scale_or_custom_cell_size: AutoScaleOrCustomAnnotation = None,
    crs: CrsAnnotation = "EPSG:3857",
    intersecting_only: IntersectingOnlyAnnotation = False,
) -> AnyGeoDataFrame:
    """
    Create a grid from the provided area of interest.
    """
    import os

    import geopandas as gpd  # type: ignore[import-untyped]
    from shapely.geometry import box

    from ecoscope.analysis.UD import (  # type: ignore[import-untyped]
        grid_size_from_geographic_extent,
    )
    from ecoscope.base.utils import create_meshgrid  # type: ignore[import-untyped]

    if auto_scale_or_custom_cell_size is None:
        auto_scale_or_custom_cell_size = AutoScaleGridCellSize()

    if isinstance(auto_scale_or_custom_cell_size, CustomGridCellSize):
        cell_size = auto_scale_or_custom_cell_size.grid_cell_size  # type: ignore[union-attr]

        # Approximate the number of grid cells we'll generate
        # and error if it's above the acceptable threshold
        CONTAINER_MEMORY = int(os.getenv("ECOSCOPE_WORKFLOWS_CONTAINER_MEMORY", 32e10))
        # Roughly, 75% of container mem / 10 columns of traj data / 8 bytes per dataframe 'cell'
        MAX_CELL_COUNT = CONTAINER_MEMORY * 0.75 / 80

        bounds = aoi.to_crs(crs).unary_union.bounds  # type: ignore[operator]

        extent_lat = bounds[3] - bounds[1]
        extent_lon = bounds[2] - bounds[0]
        num_cells_lat = extent_lat / cell_size
        num_cells_lon = extent_lon / cell_size

        if num_cells_lat * num_cells_lon > MAX_CELL_COUNT:
            raise ValueError("Custom grid cell size is too small for the extent of the area of interest")
    else:
        cell_size = grid_size_from_geographic_extent(aoi)

    result = create_meshgrid(
        box(*aoi.total_bounds),
        in_crs=aoi.crs,
        out_crs=crs,
        xlen=cell_size,
        ylen=cell_size,
        return_intersecting_only=intersecting_only,
    )

    return gpd.GeoDataFrame(geometry=result)
