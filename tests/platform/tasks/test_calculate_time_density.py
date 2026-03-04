from importlib.resources import files

import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
import pytest
from ecoscope.platform.tasks.analysis import (
    calculate_elliptical_time_density,
    calculate_linear_time_density,
)
from ecoscope.platform.tasks.analysis._create_meshgrid import (
    create_meshgrid,
)
from ecoscope.platform.tasks.analysis._time_density import (
    AutoScaleGridCellSize,
    CustomGridCellSize,
)


def test_calculate_elliptical_time_density_custom_cell_size():
    example_input_df_path = (
        files("ecoscope.platform.tasks.preprocessing") / "relocations-to-trajectory.example-return.parquet"
    )
    input_df = gpd.read_parquet(example_input_df_path)
    kws = dict(
        auto_scale_or_custom_cell_size=CustomGridCellSize(grid_cell_size=250.0),
        crs="ESRI:53042",
        band_count=1,
        nodata_value="nan",
        max_speed_factor=1.05,
        expansion_factor=1.3,
        percentiles=[50.0, 60.0, 70.0, 80.0, 90.0, 95.0],
    )
    result = calculate_elliptical_time_density(input_df, **kws)

    assert result.shape == (6, 3)
    assert all([column in result for column in ["percentile", "geometry", "area_sqkm"]])
    assert list(result["area_sqkm"]) == pytest.approx(
        [
            600.4375,
            463.6875,
            308.6875,
            215.125,
            152.0,
            106.6875,
        ]
    )


@pytest.mark.parametrize(
    "percentiles",
    [
        [50.0, 60.0, 70.0, 80.0, 90.0, 95.0],
        [
            90.0,
            80.0,
            70.0,
            70.0,
            80.0,
            90.0,
            95.0,
            50.0,
            60.0,
        ],
    ],
)
def test_calculate_elliptical_time_density_custom_auto_scale(percentiles):
    example_input_df_path = (
        files("ecoscope.platform.tasks.preprocessing") / "relocations-to-trajectory.example-return.parquet"
    )
    input_df = gpd.read_parquet(example_input_df_path)
    kws = dict(
        auto_scale_or_custom_cell_size=AutoScaleGridCellSize(),
        crs="ESRI:53042",
        band_count=1,
        nodata_value="nan",
        max_speed_factor=1.05,
        expansion_factor=1.3,
        percentiles=percentiles,
    )
    result = calculate_elliptical_time_density(input_df, **kws)

    assert result.shape == (6, 3)
    assert all([column in result for column in ["percentile", "geometry", "area_sqkm"]])
    assert list(result["area_sqkm"]) == pytest.approx(
        [626.837667, 499.715083, 354.804824, 263.25759, 195.664425, 143.487245]
    )


@pytest.mark.parametrize(
    "percentiles",
    [
        [50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        [90.0, 100.0, 100.0, 70.0, 60.0, 50.0, 80.0],
    ],
)
def test_calculate_linear_time_density_custom_auto_scale(percentiles):
    example_input_df_path = (
        files("ecoscope.platform.tasks.preprocessing") / "relocations-to-trajectory.example-return.parquet"
    )
    input_df = gpd.read_parquet(example_input_df_path)
    grid = create_meshgrid(
        aoi=input_df,
        auto_scale_or_custom_cell_size=AutoScaleGridCellSize(),
    )
    percentile_levels = percentiles

    result = calculate_linear_time_density(
        trajectory_gdf=input_df,
        meshgrid=grid,
        percentiles=percentile_levels,
    )
    assert all([column in result for column in ["percentile", "density"]])

    density_without_nans = result[~np.isnan(result["density"])]["density"]
    percentile_without_nans = result[~np.isnan(result["percentile"])]["percentile"]
    assert len(density_without_nans) == len(percentile_without_nans)
    assert set(percentile_without_nans.unique()) == set(percentile_levels)
    # Density values should be normalized
    assert round(result.density.sum(), 1) == 1.0


def test_calculate_elliptical_time_density_custom_auto_scale_raises_empty_percentiles():
    example_input_df_path = (
        files("ecoscope.platform.tasks.preprocessing") / "relocations-to-trajectory.example-return.parquet"
    )
    input_df = gpd.read_parquet(example_input_df_path)

    with pytest.raises(ValueError):
        calculate_elliptical_time_density(
            input_df,
            auto_scale_or_custom_cell_size=AutoScaleGridCellSize(),
            crs="ESRI:53042",
            band_count=1,
            nodata_value="nan",
            max_speed_factor=1.05,
            expansion_factor=1.3,
            percentiles=[],
        )


def test_calculate_linear_time_density_custom_auto_scale_raises_empty_percentiles():
    example_input_df_path = (
        files("ecoscope.platform.tasks.preprocessing") / "relocations-to-trajectory.example-return.parquet"
    )
    input_df = gpd.read_parquet(example_input_df_path)
    grid = create_meshgrid(
        aoi=input_df,
        auto_scale_or_custom_cell_size=AutoScaleGridCellSize(),
    )

    with pytest.raises(ValueError):
        calculate_linear_time_density(
            trajectory_gdf=input_df,
            meshgrid=grid,
            percentiles=[],
        )
