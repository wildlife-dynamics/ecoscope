import pandas as pd

from ecoscope.base.utils import (
    create_meshgrid,
    groupby_intervals,
    create_interval_index,
    create_modis_interval_index,
    add_val_index,
    add_temporal_index,
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
    movebank_relocations = movebank_relocations.dropna(subset="extra__external-temperature")

    start_temp = movebank_relocations["extra__external-temperature"].min()
    end_temp = movebank_relocations["extra__external-temperature"].max()

    intervals = pd.interval_range(start=start_temp, end=end_temp)
    groupby = groupby_intervals(movebank_relocations, "extra__external-temperature", intervals)

    # since our range is from min to max the sum of the length of each group should be equivalent to value_counts() -1
    assert len(groupby) == end_temp - start_temp - 1
    assert (
        sum(len(group) for _, group in groupby)
        == sum(count for count in movebank_relocations["extra__external-temperature"].value_counts()) - 1
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
        df=movebank_relocations, index_name="temporal", time_col="fixtime", directive="%d/%m/%y"
    )

    assert with_temporal_index.index.names == ["event-id", "temporal"]
    assert with_temporal_index.index[0][1] == movebank_relocations["fixtime"].iloc[0].strftime("%d/%m/%y")


def test_modis_offset():
    ts1 = pd.Timestamp("2022-01-13 17:00:00+0")
    ts2 = pd.Timestamp("2022-12-26 17:00:00+0")
    modis = ModisBegin()
    assert modis.apply(ts1) == pd.Timestamp("2022-01-17 00:00:00+0")
    assert modis.apply(ts2) == pd.Timestamp("2023-01-01 00:00:00+0")
