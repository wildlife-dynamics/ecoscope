"""Tests for the load_spatial_features_group task."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd  # type: ignore[import-untyped]
import pytest
from shapely.geometry import Point, Polygon

from ecoscope.platform.tasks.io import (
    EarthRangerSpatialFeatures,
    LocalFileSpatialFeatures,
    RemoteFileSpatialFeatures,
    load_spatial_features_group,
)

FIXTURE_DIR = Path(__file__).resolve().parent.parent.parent / ("ecoscope_workflows_ext_ecoscope/tasks/io")

# Local file tests


def test_load_geoparquet_polygons():
    """Test loading polygons from a geoparquet file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        geoparquet_path = Path(tmpdir) / "regions.parquet"

        gdf = gpd.GeoDataFrame(
            {"name": ["Region A", "Region B"], "area": [100, 200]},
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),
            ],
            crs="EPSG:4326",
        )
        gdf.to_parquet(geoparquet_path)

        config = LocalFileSpatialFeatures(
            file_path=str(geoparquet_path),
            name_column="name",
        )
        result = load_spatial_features_group(config)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 2
        assert "pk" in result.columns
        assert "name" in result.columns
        assert "short_name" in result.columns
        assert "feature_type" in result.columns
        assert "metadata" in result.columns
        assert result.crs.to_epsg() == 4326
        assert set(result["name"]) == {"Region A", "Region B"}


def test_load_geopackage_with_layer():
    """Test loading from a geopackage with specific layer."""
    with tempfile.TemporaryDirectory() as tmpdir:
        gpkg_path = Path(tmpdir) / "regions.gpkg"

        gdf1 = gpd.GeoDataFrame(
            {"name": ["Layer1 Region"]},
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            crs="EPSG:4326",
        )
        gdf2 = gpd.GeoDataFrame(
            {"name": ["Layer2 Region A", "Layer2 Region B"]},
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),
            ],
            crs="EPSG:4326",
        )

        gdf1.to_file(gpkg_path, layer="layer1", driver="GPKG")
        gdf2.to_file(gpkg_path, layer="layer2", driver="GPKG")

        config = LocalFileSpatialFeatures(
            file_path=str(gpkg_path),
            layer="layer2",
            name_column="name",
        )
        result = load_spatial_features_group(config)

        assert len(result) == 2
        assert "Layer2 Region A" in result["name"].values


def test_filters_non_polygon_geometries():
    """Test that non-polygon geometries are silently dropped."""
    with tempfile.TemporaryDirectory() as tmpdir:
        geoparquet_path = Path(tmpdir) / "mixed.parquet"

        gdf = gpd.GeoDataFrame(
            {"name": ["Polygon", "Point", "Another Polygon"]},
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Point(0.5, 0.5),
                Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),
            ],
            crs="EPSG:4326",
        )
        gdf.to_parquet(geoparquet_path)

        config = LocalFileSpatialFeatures(
            file_path=str(geoparquet_path),
            name_column="name",
        )
        result = load_spatial_features_group(config)

        assert len(result) == 2
        assert set(result["name"]) == {"Polygon", "Another Polygon"}


def test_rejects_duplicate_names():
    """Test that duplicate region names raise ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        geoparquet_path = Path(tmpdir) / "dupes.parquet"

        gdf = gpd.GeoDataFrame(
            {"name": ["Region A", "Region A", "Region B"]},
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),
                Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
            ],
            crs="EPSG:4326",
        )
        gdf.to_parquet(geoparquet_path)

        config = LocalFileSpatialFeatures(
            file_path=str(geoparquet_path),
            name_column="name",
        )

        with pytest.raises(ValueError, match="Region names must be unique.*Region A"):
            load_spatial_features_group(config)


def test_raises_on_missing_crs():
    """Test that missing CRS raises ValueError instead of silently assuming 4326."""
    with tempfile.TemporaryDirectory() as tmpdir:
        geoparquet_path = Path(tmpdir) / "no_crs.parquet"

        gdf = gpd.GeoDataFrame(
            {"name": ["Region A"]},
            geometry=[Polygon([(500000, 0), (500100, 0), (500100, 100), (500000, 100)])],
        )
        gdf.to_parquet(geoparquet_path)

        config = LocalFileSpatialFeatures(
            file_path=str(geoparquet_path),
            name_column="name",
        )

        with pytest.raises(ValueError, match="no CRS information"):
            load_spatial_features_group(config)


def test_converts_crs_to_4326():
    """Test that CRS is converted to EPSG:4326."""
    with tempfile.TemporaryDirectory() as tmpdir:
        geoparquet_path = Path(tmpdir) / "regions_utm.parquet"

        gdf = gpd.GeoDataFrame(
            {"name": ["Region A"]},
            geometry=[Polygon([(500000, 0), (500100, 0), (500100, 100), (500000, 100)])],
            crs="EPSG:32633",
        )
        gdf.to_parquet(geoparquet_path)

        config = LocalFileSpatialFeatures(
            file_path=str(geoparquet_path),
            name_column="name",
        )
        result = load_spatial_features_group(config)

        assert result.crs.to_epsg() == 4326


def test_custom_name_column():
    """Test using a custom name column."""
    with tempfile.TemporaryDirectory() as tmpdir:
        geoparquet_path = Path(tmpdir) / "regions.parquet"

        gdf = gpd.GeoDataFrame(
            {"region_name": ["Custom Region"], "id": [1]},
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            crs="EPSG:4326",
        )
        gdf.to_parquet(geoparquet_path)

        config = LocalFileSpatialFeatures(
            file_path=str(geoparquet_path),
            name_column="region_name",
        )
        result = load_spatial_features_group(config)

        assert result["name"].iloc[0] == "Custom Region"


def test_invalid_file_extension():
    """Test that invalid file extension raises ValueError at config construction."""
    with pytest.raises(ValueError, match="geoparquet.*geopackage"):
        LocalFileSpatialFeatures(
            file_path="data.csv",
            name_column="name",
        )


def test_missing_name_column():
    """Test that missing name column raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        geoparquet_path = Path(tmpdir) / "regions.parquet"

        gdf = gpd.GeoDataFrame(
            {"region": ["Region A"]},
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            crs="EPSG:4326",
        )
        gdf.to_parquet(geoparquet_path)

        config = LocalFileSpatialFeatures(
            file_path=str(geoparquet_path),
            name_column="name",
        )

        with pytest.raises(ValueError, match="Column 'name' not found"):
            load_spatial_features_group(config)


def test_metadata_contains_display_name():
    """Test that metadata contains display_name derived from filename."""
    with tempfile.TemporaryDirectory() as tmpdir:
        geoparquet_path = Path(tmpdir) / "conservation_areas.parquet"

        gdf = gpd.GeoDataFrame(
            {"name": ["Area A"]},
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            crs="EPSG:4326",
        )
        gdf.to_parquet(geoparquet_path)

        config = LocalFileSpatialFeatures(
            file_path=str(geoparquet_path),
            name_column="name",
        )
        result = load_spatial_features_group(config)

        assert result["metadata"].iloc[0]["display_name"] == "Conservation Areas"


# Remote file tests


@patch("ecoscope.io.download_file")
def test_downloads_and_loads_remote_file(mock_download):
    """Test downloading and loading a remote geoparquet file."""
    import shutil

    with tempfile.TemporaryDirectory() as srcdir, tempfile.TemporaryDirectory():
        # Create a source parquet file
        src_path = Path(srcdir) / "regions.parquet"
        gdf = gpd.GeoDataFrame(
            {"name": ["Remote Region"]},
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            crs="EPSG:4326",
        )
        gdf.to_parquet(src_path)

        # Mock download_file to copy the file into the target directory
        def fake_download(url, dest_dir):
            shutil.copy(src_path, Path(dest_dir) / "regions.parquet")

        mock_download.side_effect = fake_download

        config = RemoteFileSpatialFeatures(
            url="https://example.com/data/regions.parquet",
            name_column="name",
        )
        result = load_spatial_features_group(config)

        assert mock_download.call_count == 1
        assert len(result) == 1
        assert result["name"].iloc[0] == "Remote Region"
        assert result["metadata"].iloc[0]["display_name"] == "Regions"


# EarthRanger tests


def test_earthranger_requires_data_source():
    """Test that EarthRanger source requires a data_source field."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EarthRangerSpatialFeatures(
            spatial_features_group_name="Test Group",
        )


@patch("ecoscope_workflows_ext_ecoscope.connections.EarthRangerConnection.client_from_named_connection")
@patch("ecoscope_workflows_ext_ecoscope.tasks.io._earthranger.get_spatial_features_group")
def test_loads_from_earthranger(mock_get_sfg, mock_get_client):
    """Test loading spatial features from EarthRanger via the old task."""
    fixture = gpd.read_parquet(FIXTURE_DIR / "get-spatial-features-group.example-return.parquet")
    mock_get_sfg.return_value = fixture
    mock_get_client.return_value = MagicMock()

    config = EarthRangerSpatialFeatures(
        data_source="test_connection",
        spatial_features_group_name="Test Group",
    )
    result = load_spatial_features_group(config)

    mock_get_client.assert_called_once_with("test_connection")
    mock_get_sfg.assert_called_once_with(
        client=mock_get_client.return_value,
        spatial_features_group_name="Test Group",
    )
    assert len(result) == 3
    assert result["metadata"].iloc[0]["display_name"] == "SpatialGrouperTest"
    assert result["metadata"].iloc[0]["id"] == "efddee80-8072-4bb1-8078-b391c1b39dac"


@patch("ecoscope_workflows_ext_ecoscope.connections.EarthRangerConnection.client_from_named_connection")
@patch("ecoscope_workflows_ext_ecoscope.tasks.io._earthranger.get_spatial_features_group")
def test_filters_non_polygons_from_earthranger(mock_get_sfg, mock_get_client):
    """Test that non-polygon geometries from EarthRanger are silently dropped."""
    mock_get_sfg.return_value = gpd.GeoDataFrame(
        {
            "pk": ["pk1", "pk2"],
            "name": ["Polygon", "Point"],
            "short_name": ["P", "Pt"],
            "feature_type": ["polygon", "point"],
            "metadata": [
                {"id": "1", "display_name": "G"},
                {"id": "2", "display_name": "G"},
            ],
        },
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Point(0.5, 0.5),
        ],
        crs="EPSG:4326",
    )
    mock_get_client.return_value = MagicMock()

    config = EarthRangerSpatialFeatures(
        data_source="test_connection",
        spatial_features_group_name="Mixed Group",
    )
    result = load_spatial_features_group(config)

    assert len(result) == 1
    assert result["name"].iloc[0] == "Polygon"


@patch("ecoscope_workflows_ext_ecoscope.connections.EarthRangerConnection.client_from_named_connection")
@patch("ecoscope_workflows_ext_ecoscope.tasks.io._earthranger.get_spatial_features_group")
def test_rejects_duplicate_names_from_earthranger(mock_get_sfg, mock_get_client):
    """Test that duplicate names from EarthRanger raise ValueError."""
    mock_get_sfg.return_value = gpd.GeoDataFrame(
        {
            "pk": ["pk1", "pk2"],
            "name": ["Same Name", "Same Name"],
            "short_name": ["S", "S"],
            "feature_type": ["polygon", "polygon"],
            "metadata": [
                {"id": "1", "display_name": "G"},
                {"id": "2", "display_name": "G"},
            ],
        },
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),
        ],
        crs="EPSG:4326",
    )
    mock_get_client.return_value = MagicMock()

    config = EarthRangerSpatialFeatures(
        data_source="test_connection",
        spatial_features_group_name="Dupe Group",
    )

    with pytest.raises(ValueError, match="Region names must be unique.*Same Name"):
        load_spatial_features_group(config)
