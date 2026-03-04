import json

import geopandas as gpd  # type: ignore[import-untyped]
import pandas as pd
import pytest
from ecoscope.platform.tasks.transformation import assign_subject_colors
from shapely.geometry import Point


@pytest.fixture
def sample_subject_observations():
    """Create sample subject observations data similar to EarthRanger output."""
    data = {
        "groupby_col": ["F1", "F1", "F2", "F2", "M1", "M1", "M2", "M2", "M3", "M3"],
        "subject__additional": [
            json.dumps(
                {
                    "rgb": "255, 0, 0",
                    "sex": "female",
                    "region": "Gourma",
                    "tm_animal_id": "F1",
                }
            ),
            json.dumps(
                {
                    "rgb": "255, 0, 0",
                    "sex": "female",
                    "region": "Gourma",
                    "tm_animal_id": "F1",
                }
            ),
            json.dumps(
                {
                    "rgb": "0, 255, 0",
                    "sex": "female",
                    "region": "Gourma",
                    "tm_animal_id": "F2",
                }
            ),
            json.dumps(
                {
                    "rgb": "0, 255, 0",
                    "sex": "female",
                    "region": "Gourma",
                    "tm_animal_id": "F2",
                }
            ),
            json.dumps({"sex": "male", "region": "Sahel", "tm_animal_id": "M1"}),  # No rgb
            json.dumps({"sex": "male", "region": "Sahel", "tm_animal_id": "M1"}),  # No rgb
            json.dumps(
                {
                    "rgb": "255, 0, 0",  # Duplicate rgb with F1
                    "sex": "male",
                    "region": "Sahel",
                    "tm_animal_id": "M2",
                }
            ),
            json.dumps(
                {
                    "rgb": "255, 0, 0",  # Duplicate rgb with F1
                    "sex": "male",
                    "region": "Sahel",
                    "tm_animal_id": "M2",
                }
            ),
            None,  # Missing additional data
            None,  # Missing additional data
        ],
        "fixtime": pd.date_range("2024-01-01", periods=10, freq="H", tz="UTC"),
        "geometry": [Point(0, 0) for _ in range(10)],
    }
    return gpd.GeoDataFrame(data, crs="EPSG:4326")


def test_assign_subject_colors_basic(sample_subject_observations):
    """Test basic color assignment functionality."""
    result = assign_subject_colors(
        df=sample_subject_observations,
        subject_id_column="groupby_col",
        additional_column="subject__additional",
        output_column="subject_color",
    )

    # Check that color column was added
    assert "subject_color" in result.columns

    # Check that all subjects have colors
    assert result["subject_color"].notna().all()

    # Check that colors are RGBA tuples with values in 0-255 range
    assert all(
        isinstance(color, tuple) and len(color) == 4 and all(isinstance(c, int) and 0 <= c <= 255 for c in color)
        for color in result["subject_color"]
    )


def test_assign_subject_colors_unique_rgb_preserved(sample_subject_observations):
    """Test that subjects with unique rgb values keep their assigned colors."""
    result = assign_subject_colors(
        df=sample_subject_observations,
        subject_id_column="groupby_col",
        additional_column="subject__additional",
        output_column="subject_color",
    )

    # F2 has unique rgb "0, 255, 0" which should be preserved as green
    f2_color = result[result["groupby_col"] == "F2"]["subject_color"].iloc[0]
    assert f2_color == (0, 255, 0, 255)  # Pure green as RGBA tuple


def test_assign_subject_colors_duplicate_rgb_uses_palette(sample_subject_observations):
    """Test that subjects with duplicate rgb values get palette colors when fallback_strategy='palette'."""
    result = assign_subject_colors(
        df=sample_subject_observations,
        subject_id_column="groupby_col",
        additional_column="subject__additional",
        output_column="subject_color",
        fallback_strategy="palette",
    )

    # F1 and M2 both have "255, 0, 0" (red) - they should get different palette colors
    f1_color = result[result["groupby_col"] == "F1"]["subject_color"].iloc[0]
    m2_color = result[result["groupby_col"] == "M2"]["subject_color"].iloc[0]

    # They should NOT be the original red color (since it's duplicated)
    # and they should be different from each other
    assert f1_color != m2_color


def test_assign_subject_colors_duplicate_rgb_keeps_original(
    sample_subject_observations,
):
    """Test that subjects with duplicate rgb values keep original color when fallback_strategy='default_color'."""
    result = assign_subject_colors(
        df=sample_subject_observations,
        subject_id_column="groupby_col",
        additional_column="subject__additional",
        output_column="subject_color",
        fallback_strategy="default_color",
    )

    # F1 and M2 both have "255, 0, 0" (red) - they should both keep the original red color
    f1_color = result[result["groupby_col"] == "F1"]["subject_color"].iloc[0]
    m2_color = result[result["groupby_col"] == "M2"]["subject_color"].iloc[0]

    # Both should be the original red color
    assert f1_color == (255, 0, 0, 255)
    assert m2_color == (255, 0, 0, 255)


def test_assign_subject_colors_missing_rgb_uses_palette(sample_subject_observations):
    """Test that subjects without rgb get palette colors when fallback_strategy='palette'."""
    result = assign_subject_colors(
        df=sample_subject_observations,
        subject_id_column="groupby_col",
        additional_column="subject__additional",
        output_column="subject_color",
        fallback_strategy="palette",
    )

    # M1 and M3 have no rgb - they should get different palette colors
    m1_color = result[result["groupby_col"] == "M1"]["subject_color"].iloc[0]
    m3_color = result[result["groupby_col"] == "M3"]["subject_color"].iloc[0]

    assert m1_color is not None
    assert m3_color is not None
    assert isinstance(m1_color, tuple) and len(m1_color) == 4
    assert isinstance(m3_color, tuple) and len(m3_color) == 4
    # They should get different palette colors
    assert m1_color != m3_color


def test_assign_subject_colors_missing_rgb_uses_default_color(
    sample_subject_observations,
):
    """Test that subjects without rgb get default_color when fallback_strategy='default_color'."""
    result = assign_subject_colors(
        df=sample_subject_observations,
        subject_id_column="groupby_col",
        additional_column="subject__additional",
        output_column="subject_color",
        fallback_strategy="default_color",
    )

    # M1 and M3 have no rgb - they should both get the default color (#FFFF00 = yellow)
    m1_color = result[result["groupby_col"] == "M1"]["subject_color"].iloc[0]
    m3_color = result[result["groupby_col"] == "M3"]["subject_color"].iloc[0]

    assert m1_color == (255, 255, 0, 255)  # #FFFF00 as RGBA
    assert m3_color == (255, 255, 0, 255)


def test_assign_subject_colors_custom_default_color(sample_subject_observations):
    """Test using a custom default_color."""
    result = assign_subject_colors(
        df=sample_subject_observations,
        subject_id_column="groupby_col",
        additional_column="subject__additional",
        output_column="subject_color",
        fallback_strategy="default_color",
        default_color="#FF00FF",  # Magenta
    )

    # M1 and M3 have no rgb - they should both get the custom default color
    m1_color = result[result["groupby_col"] == "M1"]["subject_color"].iloc[0]
    m3_color = result[result["groupby_col"] == "M3"]["subject_color"].iloc[0]

    assert m1_color == (255, 0, 255, 255)  # #FF00FF as RGBA
    assert m3_color == (255, 0, 255, 255)


def test_assign_subject_colors_consistent_per_subject(sample_subject_observations):
    """Test that all observations of the same subject get the same color."""
    result = assign_subject_colors(
        df=sample_subject_observations,
        subject_id_column="groupby_col",
        additional_column="subject__additional",
        output_column="subject_color",
    )

    # Check each subject has consistent color across all their observations
    for subject_id in result["groupby_col"].unique():
        subject_colors = result[result["groupby_col"] == subject_id]["subject_color"]
        assert len(subject_colors.unique()) == 1, f"Subject {subject_id} has inconsistent colors"


def test_assign_subject_colors_missing_additional_column_with_palette():
    """Test behavior when additional column is missing with palette strategy."""
    df = gpd.GeoDataFrame(
        {
            "groupby_col": ["S1", "S2", "S3"],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
        },
        crs="EPSG:4326",
    )

    result = assign_subject_colors(
        df=df,
        subject_id_column="groupby_col",
        additional_column="subject__additional",  # This column doesn't exist
        output_column="subject_color",
        fallback_strategy="palette",
    )

    # Should still work and assign palette colors to all subjects
    assert "subject_color" in result.columns
    assert result["subject_color"].notna().all()
    assert len(result["subject_color"].unique()) == 3  # 3 unique subjects = 3 different colors


def test_assign_subject_colors_missing_additional_column_with_default_color():
    """Test behavior when additional column is missing with default_color strategy."""
    df = gpd.GeoDataFrame(
        {
            "groupby_col": ["S1", "S2", "S3"],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
        },
        crs="EPSG:4326",
    )

    result = assign_subject_colors(
        df=df,
        subject_id_column="groupby_col",
        additional_column="subject__additional",  # This column doesn't exist
        output_column="subject_color",
        fallback_strategy="default_color",
    )

    # Should still work and assign default_color to all subjects
    assert "subject_color" in result.columns
    assert result["subject_color"].notna().all()
    # All subjects should get the same default color
    assert len(result["subject_color"].unique()) == 1
    assert result["subject_color"].iloc[0] == (255, 255, 0, 255)  # #FFFF00


def test_assign_subject_colors_custom_palette():
    """Test using a custom color palette."""
    df = gpd.GeoDataFrame(
        {
            "groupby_col": ["S1", "S2", "S3"],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
        },
        crs="EPSG:4326",
    )

    result = assign_subject_colors(
        df=df,
        subject_id_column="groupby_col",
        additional_column="subject__additional",
        output_column="subject_color",
        fallback_strategy="palette",
        default_palette="tab10",  # Different palette
    )

    assert "subject_color" in result.columns
    assert result["subject_color"].notna().all()
    # All subjects should get different palette colors
    assert len(result["subject_color"].unique()) == 3


def test_assign_subject_colors_invalid_subject_column():
    """Test that appropriate error is raised when subject column doesn't exist."""
    df = gpd.GeoDataFrame(
        {"other_col": ["S1", "S2"], "geometry": [Point(0, 0), Point(1, 1)]},
        crs="EPSG:4326",
    )

    with pytest.raises(ValueError, match="Subject ID column .* not found"):
        assign_subject_colors(
            df=df,
            subject_id_column="groupby_col",  # This column doesn't exist
            additional_column="subject__additional",
            output_column="subject_color",
        )


def test_assign_subject_colors_with_dict_additional_data():
    """Test that function works when additional data is already a dict (not JSON string)."""
    df = gpd.GeoDataFrame(
        {
            "groupby_col": ["S1", "S2"],
            "subject__additional": [
                {"rgb": "100, 150, 200"},
                {"rgb": "50, 75, 100"},
            ],
            "geometry": [Point(0, 0), Point(1, 1)],
        },
        crs="EPSG:4326",
    )

    result = assign_subject_colors(
        df=df,
        subject_id_column="groupby_col",
        additional_column="subject__additional",
        output_column="subject_color",
    )

    assert "subject_color" in result.columns
    assert result["subject_color"].notna().all()
    # Both should have unique colors
    s1_color = result[result["groupby_col"] == "S1"]["subject_color"].iloc[0]
    s2_color = result[result["groupby_col"] == "S2"]["subject_color"].iloc[0]
    assert s1_color == (100, 150, 200, 255)  # RGBA tuple
    assert s2_color == (50, 75, 100, 255)  # RGBA tuple


def test_assign_subject_colors_malformed_rgb():
    """Test handling of malformed rgb values."""
    df = gpd.GeoDataFrame(
        {
            "groupby_col": ["S1", "S2"],
            "subject__additional": [
                json.dumps({"rgb": "invalid"}),
                json.dumps({"rgb": "255, 0"}),  # Missing third value
            ],
            "geometry": [Point(0, 0), Point(1, 1)],
        },
        crs="EPSG:4326",
    )

    # Should not crash, should fall back to default_color
    result = assign_subject_colors(
        df=df,
        subject_id_column="groupby_col",
        additional_column="subject__additional",
        output_column="subject_color",
        fallback_strategy="default_color",
    )

    assert "subject_color" in result.columns
    assert result["subject_color"].notna().all()
    # Both should get the default color since their rgb is malformed
    assert result["subject_color"].iloc[0] == (255, 255, 0, 255)
    assert result["subject_color"].iloc[1] == (255, 255, 0, 255)


def test_assign_subject_colors_nan_handling():
    """Test that NaN subject IDs get NAN_COLOR (transparent black)."""
    df = gpd.GeoDataFrame(
        {
            "groupby_col": ["S1", None, "S2", None],
            "subject__additional": [
                json.dumps({"rgb": "255, 0, 0"}),
                json.dumps({"rgb": "0, 255, 0"}),
                json.dumps({"rgb": "0, 0, 255"}),
                None,
            ],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3)],
        },
        crs="EPSG:4326",
    )

    result = assign_subject_colors(
        df=df,
        subject_id_column="groupby_col",
        additional_column="subject__additional",
        output_column="subject_color",
    )

    # Check that NaN subject IDs get NAN_COLOR (0, 0, 0, 0)
    nan_colors = result[result["groupby_col"].isna()]["subject_color"]
    assert all(color == (0, 0, 0, 0) for color in nan_colors)

    # Check that valid subjects still get proper colors
    valid_colors = result[result["groupby_col"].notna()]["subject_color"]
    assert all(color != (0, 0, 0, 0) for color in valid_colors)
