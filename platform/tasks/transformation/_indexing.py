from typing import Annotated, TypeAlias, cast

from pydantic import Field, TypeAdapter
from wt_registry import register

from ecoscope.platform.annotations import AdvancedField, AnyDataFrame, AnyGeoDataFrame
from ecoscope.platform.indexes import (
    AllGrouper,
    SpatialGrouper,
    TemporalGrouper,
    UserDefinedGroupers,
)
from ecoscope.platform.schemas import (
    EmptyDataFrame,
    EventGDF,
    EventsWithDisplayNamesGDF,
    RegionsGDF,
    TrajectoryGDF,
)

FeatureGroupId: TypeAlias = str


@register()
def add_temporal_index(
    df: Annotated[
        AnyDataFrame, Field(description="The dataframe to add the temporal index to.")
    ],
    time_col: Annotated[
        str, Field(description="The name of the column containing time data.")
    ],
    groupers: Annotated[
        AllGrouper | UserDefinedGroupers,
        Field(
            description="""\
            A list of groupers which may contain TemporalGroupers. If TemporalGroupers are present,
            additional indexes will be added to the `df` by formatting the `time_col` according to
            the `index_name` attribute of each TemporalGrouper. If no TemporalGroupers are present,
            this task will return the input `df` unchanged. This parameter is excluded from the
            generated RJSF because it should only be set programmatically in the `spec.yaml` file.
            Note also that the type of this parameter is `AllGrouper | UserDefinedGroupers` to allow
            passing a list of any type of Grouper from upstream tasks in the DAG; any elements of
            the list which are not TemporalGroupers will simply be ignored here.
            """,
            exclude=True,
        ),
    ],
    cast_to_datetime: Annotated[
        bool,
        AdvancedField(
            default=True,
            description="Whether to attempt casting `time_col` to datetime.",
        ),
    ] = True,
    format: Annotated[
        str,
        AdvancedField(
            default="mixed",
            description="""\
            If `cast_to_datetime=True`, the format to pass to `pd.to_datetime`
            when attempting to cast `time_col` to datetime. Defaults to "mixed".
            """,
        ),
    ] = "mixed",
) -> AnyDataFrame:
    import pandas as pd

    if cast_to_datetime:
        df[time_col] = pd.to_datetime(df[time_col], format=format)

    if not isinstance(groupers, AllGrouper):
        temporal_groupers = [g for g in groupers if isinstance(g, TemporalGrouper)]
        for tg in temporal_groupers:
            df[tg.index_name] = df[time_col].dt.strftime(tg.temporal_index.directive)
            df = df.set_index(tg.index_name, append=True)  # type: ignore[assignment]

    return cast(AnyDataFrame, df)


@register()
def extract_spatial_grouper_feature_group_names(
    groupers: AllGrouper | UserDefinedGroupers,
) -> list[FeatureGroupId]:
    """If there are spatial groupers, extract and return feature group names"""
    if isinstance(groupers, AllGrouper):
        return []
    return [
        grouper.spatial_index_name
        for grouper in groupers
        if isinstance(grouper, SpatialGrouper)
    ]


@register()
def resolve_spatial_feature_groups_for_spatial_groupers(
    groupers: AllGrouper | UserDefinedGroupers,
    spatial_feature_groups: list[RegionsGDF | EmptyDataFrame]
    | RegionsGDF
    | EmptyDataFrame,
) -> AllGrouper | UserDefinedGroupers:
    """Resolves feature groups for SpatialGroupers, if necessary"""
    if not isinstance(groupers, AllGrouper) and spatial_feature_groups is not None:
        if not isinstance(spatial_feature_groups, list):
            spatial_feature_groups = [spatial_feature_groups]

        lookup: dict[str, RegionsGDF | EmptyDataFrame] = {
            sfg["metadata"].iloc[0]["display_name"]: sfg
            for sfg in spatial_feature_groups
            if not sfg.empty
        }
        for grouper in groupers:
            if isinstance(grouper, SpatialGrouper):
                sfg = lookup.get(grouper.spatial_index_name)
                if sfg is not None:
                    # We want to filter out non-polys since lines/points don't make sense as spatial groupers
                    sfg = sfg[sfg.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
                    if sfg.empty:  # type: ignore[union-attr]
                        raise ValueError(
                            f"There are no polygons in Feature Group {grouper.display_name}, you must select a feature collection that contains at least 1 polygon."
                        )
                    grouper.resolve(spatial_regions=sfg)  # type: ignore[arg-type]

    return groupers


@register()
def add_spatial_index(
    gdf: Annotated[
        TrajectoryGDF | EventGDF | EventsWithDisplayNamesGDF,
        Field(description="The dataframe to add the spatial index to."),
    ],
    groupers: Annotated[
        AllGrouper | UserDefinedGroupers,
        Field(
            description="""\
            A list of groupers which may contain SpatialGroupers. If SpatialGroupers are present,
            additional indexes will be added to the `gdf` by taking a spatial join of each region in the SpatialGrouper,
            and adding the joined region name
            If no SpatialGroupers are present, this task will return the input `gdf` unchanged.
            This parameter is excluded from the generated RJSF because it should only be set programmatically in the `spec.yaml` file.
            Note also that the type of this parameter is `AllGrouper | UserDefinedGroupers` to allow
            passing a list of any type of Grouper from upstream tasks in the DAG; any elements of
            the list which are not SpatialGrouper will simply be ignored here.
            """,
            exclude=True,
        ),
    ],
) -> AnyGeoDataFrame:
    import geopandas as gpd  # type: ignore[import-untyped]
    import numpy as np
    from ecoscope.trajectory import Trajectory  # type: ignore[import-untyped]

    if not isinstance(groupers, AllGrouper):
        spatial_groupers = [g for g in groupers if isinstance(g, SpatialGrouper)]
        for sg in spatial_groupers:
            if sg.is_resolved and sg.spatial_regions is not None:
                # spatial_regions is typed as AnyGeoDataFrame,
                # but in our opinionated use here we expect a RegionsGDF
                TypeAdapter(RegionsGDF).validate_python(sg.spatial_regions)

                regions = sg.spatial_regions.copy()
                # Build the column to be used as the spatial index
                # Region display names can be empty, so use the UUID as a fallback
                regions[sg.index_name] = np.where(
                    regions["name"] != "",
                    regions["name"],
                    regions["pk"],
                )
                # Reduce down to only the spatial index and the geometry
                # This results in just a single column (the index) being added after the overlay/sjoin
                regions = regions.reset_index(drop=True)
                regions = regions.set_index(sg.index_name)
                regions = regions.drop(
                    columns=["pk", "name", "short_name", "feature_type", "metadata"]
                )

                if all(col in gdf.columns for col in ["segment_start", "segment_end"]):
                    traj = Trajectory(gdf=gdf)
                    overlay = traj.apply_spatial_classification(
                        spatial_regions=regions, output_column_name=sg.index_name
                    )
                else:
                    # Overlay has a call to reset_index,
                    # and we can't control the behavior of it from the params available in overlay
                    # So, drop any default columns that may have been created by earlier calls to reset_index
                    # In order to ensure that the reset_index inside overlay doesn't raise
                    gdf = gdf.drop(columns=["index", "level_0"], errors="ignore")
                    overlay = gpd.sjoin(
                        gdf,
                        regions,
                        how="inner",
                        predicate="intersects",
                    )
                gdf = overlay.set_index(sg.index_name, append=True)

    return cast(AnyGeoDataFrame, gdf)
