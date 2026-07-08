import pandas as pd

from ecoscope.platform.tasks.transformation import (
    drop_column_prefix,
    drop_duplicate_columns,
)


class TestDropColumnPrefix:
    """Tests for drop_column_prefix task."""

    def test_basic_prefix_removal(self):
        """Test basic prefix removal with various scenarios."""
        # Normal case with prefix removal
        df = pd.DataFrame({"prefix_col1": [1, 2], "prefix_col2": [3, 4], "other": [5, 6]})
        result = drop_column_prefix(df, "prefix_")
        assert list(result.columns) == ["col1", "col2", "other"]

        # No columns match prefix
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        result = drop_column_prefix(df, "prefix_")
        assert list(result.columns) == ["col1", "col2"]

        # Empty dataframe
        df = pd.DataFrame()
        result = drop_column_prefix(df, "prefix_")
        assert len(result.columns) == 0

    def test_suffix_strategy(self):
        """Test suffix strategy handles conflicts, preserves data and order."""
        df = pd.DataFrame(
            {
                "prefix_x": [1],
                "y": [2],
                "x": [3],  # Conflict with prefix_x
                "prefix_z": [4],
            }
        )
        result = drop_column_prefix(df, "prefix_", duplicate_strategy="suffix")

        # Check column names and order
        assert list(result.columns) == ["x_1", "y", "x", "z"]

        # Check data integrity
        assert result["x_1"].tolist() == [1]
        assert result["y"].tolist() == [2]
        assert result["x"].tolist() == [3]
        assert result["z"].tolist() == [4]

    def test_error_strategy(self):
        """Test error strategy raises on conflict, succeeds otherwise."""
        # Should raise on conflict
        df = pd.DataFrame({"prefix_name": [1, 2], "name": [3, 4]})
        try:
            drop_column_prefix(df, "prefix_", duplicate_strategy="error")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "duplicate columns" in str(e).lower()
            assert "name" in str(e)

        # Should succeed without conflict
        df = pd.DataFrame({"prefix_col1": [1, 2], "prefix_col2": [3, 4], "other": [5, 6]})
        result = drop_column_prefix(df, "prefix_", duplicate_strategy="error")
        assert list(result.columns) == ["col1", "col2", "other"]

    def test_keep_original_strategy(self):
        """Test keep_original strategy keeps conflicts, renames non-conflicts, preserves order."""
        df = pd.DataFrame(
            {
                "prefix_a": [1, 2],
                "b": [3, 4],
                "prefix_c": [5, 6],
                "c": [7, 8],  # Conflict with prefix_c
                "prefix_d": [9, 10],
            }
        )
        result = drop_column_prefix(df, "prefix_", duplicate_strategy="keep_original")

        # Check columns: a renamed, b unchanged, prefix_c kept (conflict), c unchanged, d renamed
        assert list(result.columns) == ["a", "b", "prefix_c", "c", "d"]

        # Check data integrity
        assert result["a"].tolist() == [1, 2]
        assert result["prefix_c"].tolist() == [5, 6]
        assert result["c"].tolist() == [7, 8]


def test_drop_column_prefix_suffix_case_insensitive():
    """Columns differing only in case should be treated as duplicates (GPKG/SQLite compat)."""
    df = pd.DataFrame(
        {
            "prefix_Vessel ID": ["a", "b"],
            "prefix_Vessel Id": ["c", "d"],
            "other": [1, 2],
        }
    )
    result = drop_column_prefix(df, "prefix_", duplicate_strategy="suffix")
    assert "Vessel ID" in result.columns
    assert "Vessel Id_1" in result.columns
    assert result["Vessel ID"].tolist() == ["a", "b"]
    assert result["Vessel Id_1"].tolist() == ["c", "d"]


def test_drop_column_prefix_error_case_insensitive():
    """Error strategy should detect case-insensitive collisions."""
    df = pd.DataFrame({"prefix_Name": [1], "prefix_name": [2]})
    try:
        drop_column_prefix(df, "prefix_", duplicate_strategy="error")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "duplicate" in str(e).lower()


def test_drop_column_prefix_keep_original_case_insensitive():
    """Keep_original should not rename if case-insensitive collision with existing col."""
    df = pd.DataFrame({"prefix_Name": [1], "name": [2]})
    result = drop_column_prefix(df, "prefix_", duplicate_strategy="keep_original")
    assert "prefix_Name" in result.columns
    assert "name" in result.columns


def test_drop_column_prefix_no_false_positive_case_insensitive():
    """Columns that don't collide case-insensitively should still be renamed normally."""
    df = pd.DataFrame({"prefix_Alpha": [1], "prefix_Beta": [2], "gamma": [3]})
    result = drop_column_prefix(df, "prefix_", duplicate_strategy="suffix")
    assert list(result.columns) == ["Alpha", "Beta", "gamma"]


def test_drop_duplicate_columns_no_duplicates():
    df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
    result = drop_duplicate_columns(df)
    assert list(result.columns) == ["A", "B", "C"]


def test_drop_duplicate_columns_drop_last():
    df = pd.DataFrame([[1, 2, 3]], columns=["A", "B", "A"])
    result = drop_duplicate_columns(df, strategy="drop_last")
    assert list(result.columns) == ["A", "B"]
    assert result["A"].iloc[0] == 1  # keeps first occurrence


def test_drop_duplicate_columns_drop_first():
    df = pd.DataFrame([[1, 2, 3]], columns=["A", "B", "A"])
    result = drop_duplicate_columns(df, strategy="drop_first")
    assert list(result.columns) == ["B", "A"]
    assert result["A"].iloc[0] == 3  # keeps last occurrence


def test_drop_duplicate_columns_suffix():
    df = pd.DataFrame([[1, 2, 3, 4]], columns=["A", "B", "A", "A"])
    result = drop_duplicate_columns(df, strategy="suffix")
    assert list(result.columns) == ["A", "B", "A_1", "A_2"]
    assert result["A"].iloc[0] == 1
    assert result["A_1"].iloc[0] == 3
    assert result["A_2"].iloc[0] == 4
