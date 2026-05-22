import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from ecoscope import plotting
from ecoscope.analysis import seasons


@pytest.mark.io
def test_seasons():
    gdf = gpd.read_file("tests/sample_data/vector/AOI_sites.gpkg").to_crs(4326)

    aoi = gdf.geometry.iat[0]

    # Extract the standardized NDVI ndvi_vals within the AOI
    ndvi_vals = seasons.std_ndvi_vals(
        aoi,
        img_coll="MODIS/061/MCD43A4",
        nir_band="Nadir_Reflectance_Band2",
        red_band="Nadir_Reflectance_Band1",
        start="2010-01-01",
        end="2021-01-01",
    )

    # Calculate the seasonal transition point
    cuts = seasons.val_cuts(ndvi_vals, 2)

    # Determine the seasonal time windows
    windows = seasons.seasonal_windows(ndvi_vals, cuts, season_labels=["dry", "wet"])

    plotting.plot_seasonal_dist(ndvi_vals["NDVI"], cuts)

    assert len(windows) > 0


@pytest.fixture
def bimodal_ndvi_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    per_mode = 50
    low = rng.normal(0.2, 0.05, per_mode)
    high = rng.normal(0.7, 0.05, per_mode)
    ndvi = np.concatenate([low, high])
    dates = pd.date_range("2020-01-01", periods=ndvi.size, freq="D", tz="UTC")
    return pd.DataFrame({"NDVI": ndvi, "img_date": dates})


def test_min_max_scaler_normalizes_to_unit_range() -> None:
    out = seasons._min_max_scaler([0.0, 5.0, 10.0])
    assert out.min() == 0.0
    assert out.max() == 1.0


def test_val_cuts_two_seasons_returns_sentinel_bracketed_cut(bimodal_ndvi_frame: pd.DataFrame) -> None:
    np.random.seed(0)

    cuts = seasons.val_cuts(bimodal_ndvi_frame, num_seasons=2)

    assert cuts[0] == float("-inf")
    assert cuts[-1] == float("inf")
    assert len(cuts) == 3
    assert 0.0 < cuts[1] < 1.0


def test_val_cuts_three_seasons_returns_two_internal_cuts() -> None:
    # np.random.seed seeds sklearn.mixture.GaussianMixture's default random_state
    np.random.seed(0)
    rng = np.random.default_rng(7)
    ndvi = np.concatenate(
        [
            rng.normal(0.15, 0.03, 50),
            rng.normal(0.5, 0.03, 50),
            rng.normal(0.85, 0.03, 50),
        ]
    )
    dates = pd.date_range("2020-01-01", periods=ndvi.size, freq="D", tz="UTC")
    df = pd.DataFrame({"NDVI": ndvi, "img_date": dates})

    cuts = seasons.val_cuts(df, num_seasons=3)

    assert cuts[0] == float("-inf")
    assert cuts[-1] == float("inf")
    assert len(cuts) == 4
    assert cuts[1] < cuts[2]


def test_seasonal_windows_assigns_labels_and_aggregates(bimodal_ndvi_frame: pd.DataFrame) -> None:
    np.random.seed(0)
    cuts = seasons.val_cuts(bimodal_ndvi_frame, num_seasons=2)

    windows = seasons.seasonal_windows(bimodal_ndvi_frame, cuts, season_labels=["dry", "wet"])

    assert list(windows.columns) == ["start", "end", "season"]
    assert set(windows["season"].astype(str)) == {"dry", "wet"}
    assert windows.index.name == "unique_season"
    assert (windows["end"] >= windows["start"]).all()


def test_add_seasonal_index_attaches_season_to_index(bimodal_ndvi_frame: pd.DataFrame) -> None:
    np.random.seed(0)
    fixtimes = pd.date_range("2020-01-15", periods=8, freq="10D", tz="UTC")
    df = pd.DataFrame({"fixtime": fixtimes, "subject": ["x"] * 8}).set_index("subject")

    out = seasons.add_seasonal_index(
        df=df,
        index_name="season",
        start_date=fixtimes.min(),
        end_date=fixtimes.max(),
        time_col="fixtime",
        ndvi_vals=bimodal_ndvi_frame,
        seasons=2,
        season_labels=["dry", "wet"],
    )

    assert "season" in out.index.names
    season_level = out.index.get_level_values("season").astype(str)
    # Fixtimes span both halves of the bimodal NDVI date range, so we expect both labels
    assert set(season_level) == {"dry", "wet"}


def test_add_seasonal_index_defaults_to_dry_wet_labels(bimodal_ndvi_frame: pd.DataFrame) -> None:
    np.random.seed(0)
    fixtimes = pd.date_range("2020-01-15", periods=8, freq="10D", tz="UTC")
    df = pd.DataFrame({"fixtime": fixtimes}, index=pd.Index(list("abcdefgh"), name="row"))

    out = seasons.add_seasonal_index(
        df=df,
        index_name="season",
        start_date=fixtimes.min(),
        end_date=fixtimes.max(),
        time_col="fixtime",
        ndvi_vals=bimodal_ndvi_frame,
        seasons=2,
    )

    assert "season" in out.index.names
    season_level = out.index.get_level_values("season").astype(str)
    assert set(season_level) == {"dry", "wet"}


def test_add_seasonal_index_mismatched_labels_raises(bimodal_ndvi_frame: pd.DataFrame) -> None:
    np.random.seed(0)
    fixtimes = pd.date_range("2020-01-15", periods=4, freq="10D", tz="UTC")
    df = pd.DataFrame({"fixtime": fixtimes})

    with pytest.raises(Exception, match="must match the number of 'season_labels'"):
        seasons.add_seasonal_index(
            df=df,
            index_name="season",
            start_date=fixtimes.min(),
            end_date=fixtimes.max(),
            time_col="fixtime",
            ndvi_vals=bimodal_ndvi_frame,
            seasons=3,
            season_labels=["dry", "wet"],
        )
