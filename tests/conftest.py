import os
import warnings

import ee
import geopandas as gpd
import pandas as pd
import pytest

import ecoscope
from ecoscope.base import Relocations

os.environ["USE_PYGEOS"] = "0"


def pytest_configure(config):
    ecoscope.init()

    os.makedirs("tests/outputs", exist_ok=True)

    if config.inicfg.get("markers") == ["io"]:
        try:
            EE_ACCOUNT = os.getenv("EE_ACCOUNT")
            EE_PRIVATE_KEY_DATA = os.getenv("EE_PRIVATE_KEY_DATA")
            if EE_ACCOUNT and EE_PRIVATE_KEY_DATA:
                credentials = ee.ServiceAccountCredentials(EE_ACCOUNT, key_data=EE_PRIVATE_KEY_DATA)
                ee.Initialize(credentials=credentials, project=credentials.project_id)
                pytest.earthengine = True
        except Exception:
            warnings.warn(Warning("Earth Engine can not be initialized."))


@pytest.mark.io
@pytest.fixture(scope="session")
def er_io():
    ER_SERVER = "https://mep-dev.pamdas.org"
    ER_USERNAME = os.getenv("ER_USERNAME")
    ER_PASSWORD = os.getenv("ER_PASSWORD")
    er_io = ecoscope.io.EarthRangerIO(server=ER_SERVER, username=ER_USERNAME, password=ER_PASSWORD)

    er_io.GROUP_NAME = "Elephants"
    er_io.SUBJECT_IDS = er_io.get_subjects(subject_group_name=er_io.GROUP_NAME).id.tolist()
    er_io.SUBJECTSOURCE_IDS, er_io.SOURCE_IDS = er_io.get_subjectsources(subjects=",".join(er_io.SUBJECT_IDS))[
        ["id", "source"]
    ].values.T.tolist()

    return er_io


@pytest.mark.io
@pytest.fixture(scope="session")
def er_events_io():
    ER_SERVER = "https://mep-dev.pamdas.org"
    ER_USERNAME = os.getenv("ER_USERNAME")
    ER_PASSWORD = os.getenv("ER_PASSWORD")
    er_events_io = ecoscope.io.EarthRangerIO(
        server=ER_SERVER, username=ER_USERNAME, password=ER_PASSWORD, tcp_limit=5, sub_page_size=100
    )

    er_events_io.GROUP_NAME = "Elephants"
    er_events_io.SUBJECT_IDS = er_events_io.get_subjects(subject_group_name=er_events_io.GROUP_NAME).id.tolist()
    er_events_io.SUBJECTSOURCE_IDS, er_events_io.SOURCE_IDS = er_events_io.get_subjectsources(
        subjects=",".join(er_events_io.SUBJECT_IDS)
    )[["id", "source"]].values.T.tolist()

    return er_events_io


@pytest.fixture
def movebank_relocations():
    df = pd.read_feather("tests/sample_data/vector/movebank_data.feather")
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.pop("location-long"), df.pop("location-lat")),
        crs=4326,
    )
    gdf["timestamp"] = pd.to_datetime(gdf["timestamp"], utc=True)
    return Relocations.from_gdf(
        gdf,
        groupby_col="individual-local-identifier",
        time_col="timestamp",
        uuid_col="event-id",
    )


@pytest.fixture
def aoi_gdf():
    AOI_FILE = "tests/sample_data/vector/maec_4zones_UTM36S.gpkg"
    regions_gdf = gpd.GeoDataFrame.from_file(AOI_FILE).to_crs(4326)
    regions_gdf.set_index("ZONE", drop=True, inplace=True)
    return regions_gdf


@pytest.fixture
def sample_relocs():
    gdf = gpd.read_parquet("tests/sample_data/vector/sample_relocs.parquet")
    gdf = ecoscope.io.earthranger_utils.clean_time_cols(gdf)

    return ecoscope.base.Relocations.from_gdf(gdf)
