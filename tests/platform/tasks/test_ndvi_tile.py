"""Unit tests for create_ndvi_tile (moved from ecoscope-workflows-ext-custom).

Earth Engine and HTTP calls are mocked, so these run without credentials.
"""

import inspect
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import ee
import geopandas as gpd  # type: ignore[import-untyped]
import pytest
from shapely.geometry import Polygon

from ecoscope.platform.tasks.analysis import create_ndvi_tile
from ecoscope.platform.tasks.io._earthengine import (
    _DETAILED_QA_BAND,
    _MODIS_CALCULATED,
    _MODIS_PRECOMPUTED,
    _MODIS_WATER_MASK,
    _NIR_BAND,
    _RED_BAND,
)
from ecoscope.platform.tasks.results._pydeck import BitmapLayerDefinition, LegendSegment

_PRECOMPUTED = "MODIS MYD13A1 16-Day Composite"
_CALCULATED = "MODIS MCD43A4 Daily NBAR"
_PERSIST = "ecoscope.platform.tasks.analysis._ndvi_tile._persist_bytes"


@pytest.fixture
def roi_gdf():
    poly = Polygon([(36.0, -1.5), (37.0, -1.5), (37.0, -0.5), (36.0, -0.5)])
    return gpd.GeoDataFrame({"name": ["test_roi"], "geometry": [poly]}, crs="EPSG:4326")


@pytest.fixture
def mock_time_range():
    tr = MagicMock()
    tr.since = datetime(2023, 1, 1, tzinfo=timezone.utc)
    tr.until = datetime(2023, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    return tr


@pytest.fixture
def mock_ee_api():
    """Patch real ee module attributes to avoid needing GEE authentication."""
    mock_img = MagicMock()
    mock_collection = MagicMock()
    mock_collection.select.return_value = mock_collection
    mock_collection.filterDate.return_value = mock_collection
    mock_collection.map.return_value = mock_collection
    mock_collection.sort.return_value = mock_collection
    mock_collection.reduce.return_value = mock_img
    mock_img.clip.return_value = mock_img

    mock_img.reduceRegion.return_value = MagicMock()
    mock_img.reduceRegion.return_value.getInfo.return_value = {
        "NDVI_mean_min": 0.1,
        "NDVI_mean_max": 0.9,
    }

    mock_map_image = MagicMock()
    mock_map_image.getThumbURL.return_value = "https://earthengine.googleapis.com/thumb/fake"

    mock_reducer = MagicMock()
    for name in ("mean", "median", "min", "max", "minMax"):
        getattr(mock_reducer, name).return_value = MagicMock()

    # Mock requests.get to return fake PNG bytes
    mock_response = MagicMock()
    mock_response.content = b"\x89PNG\r\n\x1a\nfake-png-data"
    mock_response.raise_for_status.return_value = None

    with (
        patch.object(ee, "ImageCollection", return_value=mock_collection) as mock_ic,
        patch.object(ee, "Geometry", return_value=MagicMock()),
        patch.object(ee, "Image", return_value=mock_map_image),
        patch.object(ee, "Reducer", mock_reducer),
        patch("requests.get", return_value=mock_response),
    ):
        yield {
            "ImageCollection": mock_ic,
            "Image_instance": mock_map_image,
            "collection": mock_collection,
            "img": mock_img,
        }


def _apply_mapped_fn(collection):
    """Run the function passed to ImageCollection.map on a mock image and return it."""
    (fn,) = [c.args[0] for c in collection.map.call_args_list]
    img = MagicMock()
    fn(img)
    return img


# --- behaviour carried over from ext-custom ---------------------------------


def test_create_ndvi_tile_returns_tile_layer_definition(roi_gdf, mock_time_range, mock_ee_api):
    result = create_ndvi_tile(client=MagicMock(), roi=roi_gdf, time_range=mock_time_range)
    assert isinstance(result, BitmapLayerDefinition)
    assert result.image.startswith("data:image/png;base64,")
    assert result.bounds is not None
    assert len(result.bounds) == 4


def test_create_ndvi_tile_has_legend(roi_gdf, mock_time_range, mock_ee_api):
    result = create_ndvi_tile(client=MagicMock(), roi=roi_gdf, time_range=mock_time_range)
    assert result.legend is not None
    assert isinstance(result.legend, LegendSegment)
    assert result.legend.title == "NDVI"
    # Default palette has 6 colors, so 6 legend values
    assert len(result.legend.values) == 6
    assert result.legend.values[0].label == "0.10"
    assert result.legend.values[-1].label == "0.90"


def test_create_ndvi_tile_precomputed_uses_myd13a1(roi_gdf, mock_time_range, mock_ee_api):
    create_ndvi_tile(
        client=MagicMock(),
        roi=roi_gdf,
        time_range=mock_time_range,
        ndvi_method=_PRECOMPUTED,
    )
    mock_ee_api["ImageCollection"].assert_called_once_with(_MODIS_PRECOMPUTED)
    mock_ee_api["collection"].select.assert_called_once_with(["NDVI"])
    # Precomputed path applies .map() for 0.0001 scaling
    mock_ee_api["collection"].map.assert_called_once()


def test_create_ndvi_tile_calculated_uses_mcd43a4(roi_gdf, mock_time_range, mock_ee_api):
    create_ndvi_tile(
        client=MagicMock(),
        roi=roi_gdf,
        time_range=mock_time_range,
        ndvi_method=_CALCULATED,
    )
    mock_ee_api["ImageCollection"].assert_called_once_with(_MODIS_CALCULATED)
    mock_ee_api["collection"].select.assert_called_once_with([_NIR_BAND, _RED_BAND])


def test_create_ndvi_tile_custom_palette(roi_gdf, mock_time_range, mock_ee_api):
    custom_palette = ["#ff0000", "#00ff00", "#0000ff"]
    result = create_ndvi_tile(
        client=MagicMock(),
        roi=roi_gdf,
        time_range=mock_time_range,
        palette=custom_palette,
    )
    thumb_params = mock_ee_api["Image_instance"].getThumbURL.call_args[0][0]
    assert thumb_params["palette"] == custom_palette
    assert len(result.legend.values) == 3
    assert result.legend.values[0].color == "#ff0000"
    assert result.legend.values[-1].color == "#0000ff"


def test_create_ndvi_tile_reducer_median(roi_gdf, mock_time_range, mock_ee_api):
    create_ndvi_tile(
        client=MagicMock(),
        roi=roi_gdf,
        time_range=mock_time_range,
        reducer="median",
    )
    mock_ee_api["collection"].reduce.assert_called_once()


def test_create_ndvi_tile_exports_geotiff(roi_gdf, mock_time_range, mock_ee_api):
    with patch(_PERSIST) as mock_persist:
        result = create_ndvi_tile(
            client=MagicMock(),
            roi=roi_gdf,
            time_range=mock_time_range,
            root_path="/tmp/results",
            filename_prefix="test_ndvi",
        )
        mock_persist.assert_called_once()
        args = mock_persist.call_args
        assert args[0][1] == "/tmp/results"
        assert args[0][2].startswith("test_ndvi_")
        assert args[0][2].endswith(".tif")
        assert isinstance(result, BitmapLayerDefinition)


def test_create_ndvi_tile_skips_geotiff_by_default(roi_gdf, mock_time_range, mock_ee_api):
    with patch(_PERSIST) as mock_persist:
        create_ndvi_tile(client=MagicMock(), roi=roi_gdf, time_range=mock_time_range)
        mock_persist.assert_not_called()


# --- water mask --------------------------------------------------------------


def test_create_ndvi_tile_mask_water_defaults_to_false():
    assert inspect.signature(create_ndvi_tile).parameters["mask_water"].default is False


def test_precomputed_without_mask_does_not_mask(roi_gdf, mock_time_range, mock_ee_api):
    create_ndvi_tile(client=MagicMock(), roi=roi_gdf, time_range=mock_time_range, ndvi_method=_PRECOMPUTED)

    img = _apply_mapped_fn(mock_ee_api["collection"])
    img.select.return_value.multiply.return_value.updateMask.assert_not_called()
    img.select.assert_called_once_with("NDVI")


def test_precomputed_with_mask_keeps_qa_band_and_masks(roi_gdf, mock_time_range, mock_ee_api):
    create_ndvi_tile(
        client=MagicMock(),
        roi=roi_gdf,
        time_range=mock_time_range,
        ndvi_method=_PRECOMPUTED,
        mask_water=True,
    )

    mock_ee_api["collection"].select.assert_called_once_with(["NDVI", _DETAILED_QA_BAND])
    img = _apply_mapped_fn(mock_ee_api["collection"])
    img.select.return_value.multiply.return_value.updateMask.assert_called_once()
    img.select.assert_any_call(_DETAILED_QA_BAND)
    # Masking happens before the temporal reduction.
    mock_ee_api["collection"].reduce.assert_called_once()


def test_calculated_without_mask_does_not_use_mod44w(roi_gdf, mock_time_range, mock_ee_api):
    create_ndvi_tile(client=MagicMock(), roi=roi_gdf, time_range=mock_time_range, ndvi_method=_CALCULATED)

    _apply_mapped_fn(mock_ee_api["collection"])
    assert _MODIS_WATER_MASK not in [c.args[0] for c in mock_ee_api["ImageCollection"].call_args_list]


def test_calculated_with_mask_uses_mod44w(roi_gdf, mock_time_range, mock_ee_api):
    create_ndvi_tile(
        client=MagicMock(),
        roi=roi_gdf,
        time_range=mock_time_range,
        ndvi_method=_CALCULATED,
        mask_water=True,
    )

    img = _apply_mapped_fn(mock_ee_api["collection"])
    ndvi = img.select.return_value.subtract.return_value.divide.return_value.rename.return_value
    ndvi.updateMask.assert_called_once()
    mock_ee_api["ImageCollection"].assert_any_call(_MODIS_WATER_MASK)
