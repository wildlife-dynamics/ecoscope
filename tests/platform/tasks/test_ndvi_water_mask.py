"""Unit tests for the optional NDVI water mask.

These tests mock Earth Engine, so they run without credentials. The end-to-end
behaviour against real MODIS data is covered by the ``io``-marked tests in
``test_io_earthengine.py``.
"""

import inspect
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import ee
import geopandas as gpd  # type: ignore[import-untyped]
import pandas as pd
import pytest
from shapely.geometry import Polygon

from ecoscope.platform.tasks.filter._filter import UTC_TIMEZONEINFO, TimeRange
from ecoscope.platform.tasks.io import calculate_ndvi_range
from ecoscope.platform.tasks.io._earthengine import (
    _DETAILED_QA_BAND,
    _LAND_CLASS,
    _LAND_WATER_BIT_OFFSET,
    _LAND_WATER_BIT_WIDTH_MASK,
    _MODIS_WATER_MASK,
    _NIR_BAND,
    _RED_BAND,
    _WATER_MASK_BAND,
    _land_mask,
)

_PRECOMPUTED = "MODIS MYD13A1 16-Day Composite"
_CALCULATED = "MODIS MCD43A4 Daily NBAR"
_LABEL_FN = "ecoscope.io.eetools.label_gdf_with_temporal_image_collection_by_feature"


# --- _land_mask -------------------------------------------------------------


@pytest.mark.parametrize(
    "land_water_class, is_land",
    [
        (0, False),  # shallow ocean
        (1, True),  # land
        (2, False),  # ocean coastlines and lake shorelines
        (3, False),  # shallow inland water
        (4, False),  # ephemeral water
        (5, False),  # deep inland water
        (6, False),  # moderate or continental ocean
        (7, False),  # deep ocean
    ],
)
@pytest.mark.parametrize("other_bits", [0, 0b11_1111_1111, 1 << 14])
def test_land_water_bit_decoding(land_water_class, is_land, other_bits):
    """The shift/and/eq constants isolate bits 11-13 regardless of the other QA bits."""
    qa = (land_water_class << _LAND_WATER_BIT_OFFSET) | other_bits
    decoded = (qa >> _LAND_WATER_BIT_OFFSET) & _LAND_WATER_BIT_WIDTH_MASK
    assert (decoded == _LAND_CLASS) is is_land


def test_land_mask_myd13a1_reads_detailed_qa_bits():
    img = MagicMock()
    result = _land_mask(img, _PRECOMPUTED)

    img.select.assert_called_once_with(_DETAILED_QA_BAND)
    shift = img.select.return_value.rightShift
    shift.assert_called_once_with(_LAND_WATER_BIT_OFFSET)
    bit_and = shift.return_value.bitwiseAnd
    bit_and.assert_called_once_with(_LAND_WATER_BIT_WIDTH_MASK)
    bit_and.return_value.eq.assert_called_once_with(_LAND_CLASS)
    assert result is bit_and.return_value.eq.return_value


def test_land_mask_mcd43a4_uses_static_mod44w():
    img = MagicMock()
    with patch.object(ee, "ImageCollection") as mock_ic:
        result = _land_mask(img, _CALCULATED)

    mock_ic.assert_called_once_with(_MODIS_WATER_MASK)
    sort = mock_ic.return_value.sort
    sort.assert_called_once_with("system:time_start", False)
    select = sort.return_value.first.return_value.select
    select.assert_called_once_with(_WATER_MASK_BAND)
    select.return_value.eq.assert_called_once_with(0)
    assert result is select.return_value.eq.return_value
    img.select.assert_not_called()


# --- calculate_ndvi_range wiring --------------------------------------------


@pytest.fixture
def roi():
    poly = Polygon([(29.3, -0.7), (29.95, -0.7), (29.95, 0.0), (29.3, 0.0)])
    return gpd.GeoDataFrame({"name": ["test_roi"], "geometry": [poly]}, crs="EPSG:4326")


@pytest.fixture
def time_range():
    return TimeRange(
        since=datetime(2023, 1, 1, tzinfo=timezone.utc),
        until=datetime(2023, 12, 31, tzinfo=timezone.utc),
        timezone=UTC_TIMEZONEINFO,
    )


def _labelled(ndvi_method):
    """Shape of what eetools returns for one ROI: one current and one historical image."""
    index = pd.Index(["test_roi", "test_roi"], name="name")
    dates = ["2023-06-10T00:00:00", "2022-06-10T00:00:00"]
    if ndvi_method == _PRECOMPUTED:
        return pd.DataFrame({"img_date": dates, "NDVI": [0.5, 0.4]}, index=index)
    return pd.DataFrame({"img_date": dates, _NIR_BAND: [0.3, 0.3], _RED_BAND: [0.1, 0.1]}, index=index)


@pytest.fixture
def run(roi, time_range):
    """Call calculate_ndvi_range with Earth Engine and eetools mocked out."""

    def _run(ndvi_method, mask_water):
        coll = MagicMock()
        for method in ("filterDate", "select", "map", "sort"):
            getattr(coll, method).return_value = coll
        with (
            patch.object(ee, "ImageCollection", return_value=coll) as mock_ic,
            patch.object(ee, "Reducer"),
            patch(_LABEL_FN, return_value=_labelled(ndvi_method)),
        ):
            result = calculate_ndvi_range(
                client=MagicMock(),
                roi=roi,
                time_range=time_range,
                ndvi_method=ndvi_method,
                mask_water=mask_water,
            )
            mapped = [c.args[0] for c in coll.map.call_args_list]
            # Evaluate any mapped functions while ee is still patched.
            mapped_imgs = []
            for fn in mapped:
                img = MagicMock()
                fn(img)
                mapped_imgs.append(img)
        return result, coll, mock_ic, mapped_imgs

    return _run


def test_mask_water_defaults_to_false():
    assert inspect.signature(calculate_ndvi_range).parameters["mask_water"].default is False


def test_precomputed_without_mask_is_unchanged(run):
    result, coll, _, mapped_imgs = run(_PRECOMPUTED, mask_water=False)

    coll.select.assert_called_once_with(["NDVI"])
    (img,) = mapped_imgs
    img.select.return_value.multiply.return_value.updateMask.assert_not_called()
    assert list(result.columns) == ["min", "max", "mean", "NDVI", "img_date"]


def test_precomputed_with_mask_keeps_qa_band_and_masks(run):
    result, coll, _, mapped_imgs = run(_PRECOMPUTED, mask_water=True)

    coll.select.assert_called_once_with(["NDVI", _DETAILED_QA_BAND])
    (img,) = mapped_imgs
    img.select.return_value.multiply.return_value.updateMask.assert_called_once()
    img.select.assert_any_call(_DETAILED_QA_BAND)
    assert list(result.columns) == ["min", "max", "mean", "NDVI", "img_date"]


def test_calculated_without_mask_is_unchanged(run):
    result, coll, mock_ic, _ = run(_CALCULATED, mask_water=False)

    coll.map.assert_not_called()
    assert _MODIS_WATER_MASK not in [c.args[0] for c in mock_ic.call_args_list]
    assert list(result.columns) == ["min", "max", "mean", "NDVI", "img_date"]


def test_calculated_with_mask_uses_mod44w(run):
    result, coll, mock_ic, mapped_imgs = run(_CALCULATED, mask_water=True)

    coll.map.assert_called_once()
    (img,) = mapped_imgs
    img.updateMask.assert_called_once()
    mock_ic.assert_any_call(_MODIS_WATER_MASK)
    assert list(result.columns) == ["min", "max", "mean", "NDVI", "img_date"]
