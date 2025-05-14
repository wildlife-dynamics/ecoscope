import pytest
import pandas as pd
import geopandas as gpd
from ecoscope.base.utils import hex_to_rgba
from ecoscope.base.utils import (
    create_meshgrid,
    groupby_intervals,
    create_interval_index,
    create_modis_interval_index,
    add_val_index,
    add_temporal_index,
    grid_size_from_geographic_extent,
    utm_grid_size_from_geographic_extent,
    ModisBegin,
)


def test_create_meshgrid(aoi_gdf):
    aoi = aoi_gdf.dissolve().iloc[0]["geometry"]

    mesh = create_meshgrid(
        aoi,
        "EPSG:4326",
        "EPSG:4326",
        xlen=100000,
        ylen=100000,
    )

    # at 100kms resolution, we should have 4 big squares
    assert len(mesh) == 4
    assert mesh.intersects(aoi).all()


def test_create_meshgrid_aligned(aoi_gdf):
    aoi = aoi_gdf.dissolve().iloc[0]["geometry"]

    mesh = create_meshgrid(aoi, "EPSG:4326", "EPSG:4326", xlen=100000, ylen=100000, align_to_existing=aoi_gdf)

    assert len(mesh) == 6
    assert mesh.intersects(aoi).all()


def test_groupby_intervals(movebank_relocations):
    movebank_relocations.gdf = movebank_relocations.gdf.dropna(subset="extra__external-temperature")

    start_temp = movebank_relocations.gdf["extra__external-temperature"].min()
    end_temp = movebank_relocations.gdf["extra__external-temperature"].max()

    intervals = pd.interval_range(start=start_temp, end=end_temp)
    groupby = groupby_intervals(movebank_relocations.gdf, "extra__external-temperature", intervals)

    # since our range is from min to max the sum of the length of each group should be equivalent to value_counts() -1
    assert len(groupby) == end_temp - start_temp - 1
    assert (
        sum(len(group) for _, group in groupby)
        == sum(count for count in movebank_relocations.gdf["extra__external-temperature"].value_counts()) - 1
    )


def test_create_interval_index():
    intervals = create_interval_index(
        start=pd.Timestamp("2010-01-01 12:00:00"), intervals=6, freq=pd.Timedelta(hours=12)
    )

    assert len(intervals) == 6
    for index, value in enumerate(intervals):
        assert (value.right - value.left).components.hours == 12
        if index > 0:
            # start of the current interval should be end of the previous
            assert value.left == intervals[index - 1].right


def test_create_interval_index_rounded_overlap():
    intervals = create_interval_index(
        start=pd.Timestamp("2010-01-01 12:05:00"),
        intervals=6,
        freq=pd.Timedelta(hours=48),
        overlap=pd.Timedelta(hours=24),
        round_down_to_freq=True,
    )

    assert len(intervals) == 6
    # Check that we actually rounded
    assert intervals[0].left.hour == 0
    for index, value in enumerate(intervals):
        assert (value.right - value.left).components.days == 2
        if index > 0:
            # start of current interval should be the previous end -1
            assert (value.left - intervals[index - 1].right).components.days == -1


def test_create_modis_interval():
    modis_intervals = create_modis_interval_index(
        start=pd.Timestamp("2010-01-01 12:05:00", tz="Africa/Nairobi"),
        intervals=6,
    )

    assert len(modis_intervals) == 6
    for index, value in enumerate(modis_intervals):
        assert (value.right - value.left).components.days == 16
        if index > 0:
            # start of the current interval should be end of the previous
            assert value.left == modis_intervals[index - 1].right


def test_add_val_index():
    df = pd.DataFrame({"colA": [1, 2, 3, 4], "colB": ["A", "B", "C", "D"]})
    df.set_index("colA", inplace=True)

    df = add_val_index(df, "new_index", "colB")
    assert df.index[0] == (1, "A")
    assert df.index.names == ["colA", "new_index"]

    df = add_val_index(df, "newer_index", "value")
    assert df.index[0] == (1, "A", "value")
    assert df.index.names == ["colA", "new_index", "newer_index"]


def test_add_temporal_index(movebank_relocations):
    with_temporal_index = add_temporal_index(
        df=movebank_relocations.gdf, index_name="temporal", time_col="fixtime", directive="%d/%m/%y"
    )

    assert with_temporal_index.index.names == ["event-id", "temporal"]
    assert with_temporal_index.index[0][1] == movebank_relocations.gdf["fixtime"].iloc[0].strftime("%d/%m/%y")


def test_modis_offset():
    ts1 = pd.Timestamp("2022-01-13 17:00:00+0")
    ts2 = pd.Timestamp("2022-12-26 17:00:00+0")
    modis = ModisBegin()
    assert modis.apply(ts1) == pd.Timestamp("2022-01-17 00:00:00+0")
    assert modis.apply(ts2) == pd.Timestamp("2023-01-01 00:00:00+0")


@pytest.mark.parametrize(
    "hex_str,expected",
    [
        ("#000000", (0, 0, 0, 255)),
        ("FFFFFF00", (255, 255, 255, 0)),
        ("#4444AABB", (68, 68, 170, 187)),
        ("#123456", (18, 52, 86, 255)),
    ],
)
def test_hex_to_rgba(hex_str, expected):
    assert hex_to_rgba(hex_str) == expected


@pytest.mark.parametrize(
    "hex_str",
    ["hello", "", "#FF00FNFF", None],
)
def test_hex_to_rgba_invalid(hex_str):
    with pytest.raises(ValueError):
        hex_to_rgba(hex_str)


def test_grid_size_from_geographic_extent(movebank_relocations, aoi_gdf):
    relocs_gdf = movebank_relocations.gdf
    points_worldwide = gpd.read_file("tests/sample_data/vector/points_worldwide.geojson")
    points_worldwide.set_crs("EPSG:4326", inplace=True)
    lines_worldwide = gpd.read_file("tests/sample_data/vector/lines-worldwide.geojson")
    lines_worldwide.set_crs("EPSG:4326", inplace=True)

    relocs_cell_size = grid_size_from_geographic_extent(relocs_gdf)
    aoi_gdf_cell_size = grid_size_from_geographic_extent(aoi_gdf)
    points_worldwide_cell_size = grid_size_from_geographic_extent(points_worldwide)
    lines_worldwide_cell_size = grid_size_from_geographic_extent(lines_worldwide)

    utm_relocs_cell_size = utm_grid_size_from_geographic_extent(relocs_gdf)
    utm_aoi_gdf_cell_size = utm_grid_size_from_geographic_extent(aoi_gdf)
    utm_points_worldwide_cell_size = utm_grid_size_from_geographic_extent(points_worldwide)
    utm_lines_worldwide_cell_size = utm_grid_size_from_geographic_extent(lines_worldwide)

    assert relocs_cell_size < aoi_gdf_cell_size < points_worldwide_cell_size

    relocs_grid = create_meshgrid(
        aoi=relocs_gdf.union_all(),
        in_crs=relocs_gdf.crs,
        out_crs=relocs_gdf.crs,
        xlen=relocs_cell_size,
        ylen=relocs_cell_size,
    )
    aoi_gdf_grid = create_meshgrid(
        aoi=aoi_gdf.union_all(), in_crs=aoi_gdf.crs, out_crs=aoi_gdf.crs, xlen=aoi_gdf_cell_size, ylen=aoi_gdf_cell_size
    )
    points_worldwide_grid = create_meshgrid(
        aoi=points_worldwide.union_all(),
        in_crs=points_worldwide.crs,
        out_crs=points_worldwide.crs,
        xlen=points_worldwide_cell_size,
        ylen=points_worldwide_cell_size,
    )
    lines_worldwide_grid = create_meshgrid(
        aoi=points_worldwide.union_all(),
        in_crs=lines_worldwide.crs,
        out_crs=lines_worldwide.crs,
        xlen=lines_worldwide_cell_size,
        ylen=lines_worldwide_cell_size,
    )
    open("TESTS/relocs_grid.geojson", "w").write(relocs_grid.to_json())
    open("TESTS/aoi_gdf_grid.geojson", "w").write(aoi_gdf_grid.to_json())
    open("TESTS/points_worldwide_grid.geojson", "w").write(points_worldwide_grid.to_json())
    open("TESTS/lines_worldwide_grid.geojson", "w").write(lines_worldwide_grid.to_json())

    utm_relocs_grid = create_meshgrid(
        aoi=relocs_gdf.union_all(),
        in_crs=relocs_gdf.crs,
        out_crs=relocs_gdf.crs,
        xlen=utm_relocs_cell_size,
        ylen=utm_relocs_cell_size,
    )
    utm_aoi_gdf_grid = create_meshgrid(
        aoi=aoi_gdf.union_all(),
        in_crs=aoi_gdf.crs,
        out_crs=aoi_gdf.crs,
        xlen=utm_aoi_gdf_cell_size,
        ylen=utm_aoi_gdf_cell_size,
    )
    utm_points_worldwide_grid = create_meshgrid(
        aoi=points_worldwide.union_all(),
        in_crs=points_worldwide.crs,
        out_crs=points_worldwide.crs,
        xlen=utm_points_worldwide_cell_size,
        ylen=utm_points_worldwide_cell_size,
    )
    utm_lines_worldwide_grid = create_meshgrid(
        aoi=points_worldwide.union_all(),
        in_crs=lines_worldwide.crs,
        out_crs=lines_worldwide.crs,
        xlen=utm_lines_worldwide_cell_size,
        ylen=utm_lines_worldwide_cell_size,
    )
    open("TESTS/utm_relocs_grid.geojson", "w").write(utm_relocs_grid.to_json())
    open("TESTS/utm_aoi_gdf_grid.geojson", "w").write(utm_aoi_gdf_grid.to_json())
    open("TESTS/utm_points_worldwide_grid.geojson", "w").write(utm_points_worldwide_grid.to_json())
    open("TESTS/utm_lines_worldwide_grid.geojson", "w").write(utm_lines_worldwide_grid.to_json())
