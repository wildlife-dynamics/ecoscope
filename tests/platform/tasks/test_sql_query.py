import geopandas as gpd  # type: ignore[import-untyped]
import pandas as pd
import pytest
from pandasql.sqldf import PandaSQLException  # type: ignore[import-untyped]
from shapely.geometry import Point
from sqlalchemy.exc import ProgrammingError

from ecoscope.platform.tasks.transformation._sql_query import (
    apply_sql_query,
    validate_sql_query,
)


class TestValidateSQLQuery:
    """Test SQL query validation for security."""

    def test_valid_simple_query(self):
        """Test basic valid SELECT query."""
        result = validate_sql_query("SELECT * FROM df")
        assert result == "SELECT * FROM df"

    def test_valid_complex_query(self):
        """Test complex valid query with multiple clauses."""
        query_str = """
            SELECT category, COUNT(*) as count, SUM(amount) as total
            FROM df
            WHERE amount > 10
            GROUP BY category
            HAVING COUNT(*) > 1
            ORDER BY total DESC
            LIMIT 10
        """
        result = validate_sql_query(query_str)
        assert result.startswith("SELECT")

    def test_valid_with_trailing_semicolon(self):
        """Test that trailing semicolon is allowed."""
        result = validate_sql_query("SELECT * FROM df;")
        assert result == "SELECT * FROM df;"

    def test_reject_dangerous_keywords(self):
        """Test rejection of dangerous SQL operations."""
        dangerous_queries = [
            "DROP TABLE df",
            "DELETE FROM df WHERE x = 1",
            "INSERT INTO df VALUES (1, 2)",
            "UPDATE df SET x = 1",
            "CREATE TABLE new_table AS SELECT * FROM df",
            "ALTER TABLE df ADD COLUMN x INT",
            "TRUNCATE TABLE df",
            "PRAGMA table_info(df)",
        ]

        for query in dangerous_queries:
            with pytest.raises(ValueError, match="Forbidden SQL keyword"):
                validate_sql_query(query)

    def test_reject_multiple_statements(self):
        """Test rejection of multiple statements."""
        with pytest.raises(ValueError, match="Multiple SQL statements"):
            validate_sql_query("SELECT * FROM df; DROP TABLE df")

    def test_reject_non_select(self):
        """Test rejection of non-SELECT statements."""
        with pytest.raises(ValueError, match="Only SELECT queries"):
            validate_sql_query("SHOW TABLES")

    def test_reject_too_long_query(self):
        """Test rejection of excessively long queries."""
        long_query = "SELECT * FROM df WHERE x = " + "1" * 10_000
        with pytest.raises(ValueError, match="exceeds maximum length"):
            validate_sql_query(long_query)


class TestApplySQLQuery:
    """Test SQL query execution with composite real-world scenarios."""

    def test_empty_query(self):
        """Test that empty query returns original dataframe."""
        df = pd.DataFrame({"x": [1, 2, 3]})

        # Empty query should return original df
        result = apply_sql_query(df, "")
        pd.testing.assert_frame_equal(result, df)

        # Whitespace-only query should also return original df
        result = apply_sql_query(df, "   ")
        pd.testing.assert_frame_equal(result, df)

    def test_filter_and_select_columns(self):
        """Test filtering rows and selecting specific columns."""
        df = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie", "David"],
                "age": [25, 30, 35, 40],
                "city": ["NYC", "LA", "NYC", "SF"],
            }
        )

        result = apply_sql_query(df, "SELECT name, age FROM df WHERE age > 30 AND city = 'NYC'")

        assert len(result) == 1
        assert result["name"].iloc[0] == "Charlie"
        assert result["age"].iloc[0] == 35
        assert list(result.columns) == ["name", "age"]

    def test_aggregate_with_group_by_and_having(self):
        """Test aggregation with GROUP BY, HAVING, and ORDER BY."""
        df = pd.DataFrame(
            {
                "category": ["A", "A", "B", "B", "C", "C", "C"],
                "amount": [10, 20, 15, 25, 5, 10, 30],
            }
        )

        result = apply_sql_query(
            df,
            """
            SELECT
                category,
                COUNT(*) as count,
                SUM(amount) as total,
                AVG(amount) as average
            FROM df
            GROUP BY category
            HAVING COUNT(*) >= 2
            ORDER BY total DESC
            """,
        )

        assert len(result) == 3
        # First row should be C (total = 45)
        assert result["category"].iloc[0] == "C"
        assert result["count"].iloc[0] == 3
        assert result["total"].iloc[0] == 45
        assert result["average"].iloc[0] == 15.0

    def test_case_when_with_ordering(self):
        """Test CASE WHEN for computed columns with sorting."""
        df = pd.DataFrame({"name": ["A", "B", "C", "D"], "score": [85, 92, 78, 95]})

        result = apply_sql_query(
            df,
            """
            SELECT
                name,
                score,
                CASE
                    WHEN score >= 90 THEN 'Excellent'
                    WHEN score >= 80 THEN 'Good'
                    ELSE 'Fair'
                END as grade
            FROM df
            WHERE score > 75
            ORDER BY score DESC
            """,
        )

        assert len(result) == 4
        assert result["name"].iloc[0] == "D"  # Highest score
        assert result["grade"].iloc[0] == "Excellent"
        assert result["grade"].iloc[1] == "Excellent"  # B with 92
        assert result["grade"].iloc[2] == "Good"  # A with 85

    def test_subquery_and_distinct(self):
        """Test subquery with DISTINCT."""
        df = pd.DataFrame(
            {
                "product": ["A", "A", "B", "B", "C"],
                "price": [10, 10, 20, 20, 15],
                "quantity": [5, 3, 2, 4, 6],
            }
        )

        # AVG(price) = (10+10+20+20+15)/5 = 15, so only product B (price 20) is > avg
        result = apply_sql_query(
            df,
            """
            SELECT DISTINCT product, price
            FROM df
            WHERE price > (SELECT AVG(price) FROM df)
            ORDER BY price DESC
            """,
        )

        assert len(result) == 1
        assert result["product"].iloc[0] == "B"
        assert result["price"].iloc[0] == 20

    def test_limit_and_offset(self):
        """Test LIMIT and OFFSET for pagination."""
        df = pd.DataFrame({"id": range(1, 11), "value": range(10, 20)})

        result = apply_sql_query(df, "SELECT * FROM df ORDER BY id LIMIT 3 OFFSET 2")

        assert len(result) == 3
        assert list(result["id"]) == [3, 4, 5]

    def test_empty_result_set(self):
        """Test query that returns no rows."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        result = apply_sql_query(df, "SELECT * FROM df WHERE x > 100")

        assert len(result) == 0
        assert list(result.columns) == ["x", "y"]

    def test_null_handling(self):
        """Test handling of NULL values in queries."""
        df = pd.DataFrame({"name": ["A", "B", None, "D"], "value": [10, None, 30, 40]})

        result = apply_sql_query(
            df,
            """
            SELECT
                name,
                value,
                CASE WHEN value IS NULL THEN 0 ELSE value END as clean_value
            FROM df
            WHERE name IS NOT NULL
            """,
        )

        assert len(result) == 3
        assert result["clean_value"].iloc[1] == 0  # B has NULL value

    def test_geodataframe_preserves_geometry(self):
        """Test that GeoDataFrame geometry is preserved."""
        gdf = gpd.GeoDataFrame(
            {
                "name": ["A", "B", "C"],
                "value": [10, 20, 30],
                "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
            },
            crs="EPSG:4326",
        )

        result = apply_sql_query(gdf, "SELECT name, value, geometry FROM df WHERE value > 15")

        assert isinstance(result, gpd.GeoDataFrame)
        assert result.crs == gdf.crs
        assert len(result) == 2
        assert result["geometry"].iloc[0].equals(Point(1, 1))

    def test_geodataframe_without_geometry_selection(self):
        """Test GeoDataFrame when geometry is not selected."""
        gdf = gpd.GeoDataFrame(
            {
                "name": ["A", "B", "C"],
                "value": [10, 20, 30],
                "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
            },
            crs="EPSG:4326",
        )

        result = apply_sql_query(gdf, "SELECT name, value FROM df WHERE value > 15")

        # Result should be a regular DataFrame since geometry wasn't selected
        assert isinstance(result, pd.DataFrame)
        assert not isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 2
        assert "geometry" not in result.columns

    def test_multiple_aggregations(self):
        """Test multiple different aggregation functions."""
        df = pd.DataFrame(
            {
                "category": ["X", "X", "X", "Y", "Y"],
                "value": [10, 20, 30, 40, 50],
            }
        )

        result = apply_sql_query(
            df,
            """
            SELECT
                category,
                COUNT(*) as count,
                MIN(value) as min_val,
                MAX(value) as max_val,
                AVG(value) as avg_val,
                SUM(value) as sum_val
            FROM df
            GROUP BY category
            """,
        )

        assert len(result) == 2
        x_row = result[result["category"] == "X"].iloc[0]
        assert x_row["count"] == 3
        assert x_row["min_val"] == 10
        assert x_row["max_val"] == 30
        assert x_row["avg_val"] == 20.0
        assert x_row["sum_val"] == 60

    def test_invalid_sql_syntax(self):
        """Test that invalid SQL raises error."""

        df = pd.DataFrame({"x": [1, 2, 3]})

        with pytest.raises(PandaSQLException):
            apply_sql_query(df, "SELECT * FORM df")  # Typo: FORM instead of FROM

    def test_nonexistent_column(self):
        """Test that querying non-existent column raises error."""

        df = pd.DataFrame({"x": [1, 2, 3]})

        with pytest.raises(PandaSQLException):
            apply_sql_query(df, "SELECT missing_column FROM df")

    def test_columns_parameter_filters_columns(self):
        """Test that columns parameter filters which columns are available in query."""
        df = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie"],
                "age": [25, 30, 35],
                "city": ["NYC", "LA", "SF"],
                "salary": [50000, 60000, 70000],
            }
        )

        # Only make name and age available to SQL query
        result = apply_sql_query(df, "SELECT name, age FROM df WHERE age > 25", columns=["name", "age"])

        assert len(result) == 2
        assert list(result.columns) == ["name", "age"]
        assert result["name"].tolist() == ["Bob", "Charlie"]

    def test_columns_parameter_excludes_inaccessible_columns(self):
        """Test that excluded columns cannot be queried."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "value": [10, 20, 30],
                "excluded": ["a", "b", "c"],
            }
        )

        # Try to query excluded column - should fail
        with pytest.raises(PandaSQLException):
            apply_sql_query(df, "SELECT id, excluded FROM df", columns=["id", "value"])

    def test_columns_parameter_with_geodataframe(self):
        """Test columns parameter works with GeoDataFrame and geometry."""
        gdf = gpd.GeoDataFrame(
            {
                "name": ["A", "B", "C"],
                "value": [10, 20, 30],
                "extra": ["x", "y", "z"],
                "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
            },
            crs="EPSG:4326",
        )

        # Include only name, value, and geometry
        result = apply_sql_query(
            gdf,
            "SELECT name, value, geometry FROM df WHERE value > 15",
            columns=["name", "value", "geometry"],
        )

        assert isinstance(result, gpd.GeoDataFrame)
        assert result.crs == gdf.crs
        assert len(result) == 2
        assert list(result.columns) == ["name", "value", "geometry"]
        # 'extra' column should not be accessible
        assert "extra" not in result.columns


class TestSQLQueryIntegration:
    """Integration tests for real-world workflow scenarios."""

    def test_data_cleaning_pipeline(self):
        """Test a complete data cleaning and transformation workflow."""
        # Simulate messy real-world data
        df = pd.DataFrame(
            {
                "customer_id": [1, 2, 2, 3, 3, 3, 4],
                "purchase_date": [
                    "2024-01-01",
                    "2024-01-05",
                    "2024-01-10",
                    "2024-01-15",
                    "2024-01-20",
                    "2024-01-25",
                    "2024-01-30",
                ],
                "amount": [100.0, 50.0, 75.0, 200.0, None, 150.0, 80.0],
                "status": [
                    "completed",
                    "completed",
                    "pending",
                    "completed",
                    "completed",
                    "completed",
                    "refunded",
                ],
            }
        )

        # Complex query: filter, clean, aggregate
        result = apply_sql_query(
            df,
            """
            SELECT
                customer_id,
                COUNT(*) as total_orders,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_orders,
                SUM(CASE WHEN amount IS NOT NULL THEN amount ELSE 0 END) as total_spent
            FROM df
            WHERE status IN ('completed', 'pending')
            GROUP BY customer_id
            HAVING COUNT(*) >= 2
            ORDER BY total_spent DESC
            """,
        )

        assert len(result) == 2  # Customers 2 and 3
        assert result["customer_id"].iloc[0] == 3  # Highest total_spent
        # Customer 3: 3 total orders (2024-01-15 completed, 2024-01-20 completed, 2024-01-25 completed)
        assert result["total_orders"].iloc[0] == 3
        assert result["completed_orders"].iloc[0] == 3
        # Customer 3 total: 200 + 0 (NULL) + 150 = 350
        assert result["total_spent"].iloc[0] == 350.0

    def test_geospatial_filtering_with_attributes(self):
        """Test filtering GeoDataFrame by attributes while preserving geometry."""
        # Simulate wildlife tracking data
        gdf = gpd.GeoDataFrame(
            {
                "animal_id": ["Lion1", "Lion1", "Lion2", "Lion2", "Elephant1"],
                "timestamp": pd.date_range("2024-01-01", periods=5, freq="h"),
                "speed_kmh": [5.2, 12.8, 3.1, 8.5, 2.0],
                "geometry": [Point(35.0 + i * 0.01, -1.5 + i * 0.01) for i in range(5)],
            },
            crs="EPSG:4326",
        )

        # Find high-speed movements for each animal
        result = apply_sql_query(
            gdf,
            """
            SELECT
                animal_id,
                COUNT(*) as observations,
                AVG(speed_kmh) as avg_speed,
                MAX(speed_kmh) as max_speed,
                geometry
            FROM df
            WHERE speed_kmh > 3.0
            GROUP BY animal_id
            ORDER BY max_speed DESC
            """,
        )

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 2  # Lion1 and Lion2 (Elephant filtered out)
        assert result["animal_id"].iloc[0] == "Lion1"  # Highest max speed
        assert result["max_speed"].iloc[0] == 12.8

    def test_ranking_and_percentiles(self):
        """Test ranking and top-N queries."""
        df = pd.DataFrame(
            {
                "region": ["North", "North", "South", "South", "East", "West"],
                "sales": [1000, 1500, 800, 1200, 2000, 900],
            }
        )

        # Get regions with above-average sales
        result = apply_sql_query(
            df,
            """
            SELECT
                region,
                SUM(sales) as total_sales,
                COUNT(*) as num_stores
            FROM df
            GROUP BY region
            HAVING SUM(sales) > (SELECT AVG(sales) FROM df)
            ORDER BY total_sales DESC
            """,
        )

        assert len(result) == 3  # North, East, South (above avg)
        assert result["region"].iloc[0] == "North"  # Highest total
        assert result["total_sales"].iloc[0] == 2500


def test_sanitize_default_allows_complex_columns():
    """With sanitize=True (default), list/dict columns can be queried without a whitelist."""
    df = pd.DataFrame(
        {
            "name": ["A", "B", "C"],
            "value": [10, 20, 30],
            "tags": [["x", "y"], ["z"], []],
            "meta": [{"k": 1}, {"k": 2}, {"k": 3}],
        }
    )

    # Without sanitize this would raise PandaSQLException; sanitize makes it work.
    result = apply_sql_query(df, "SELECT name, value, tags FROM df WHERE value > 15")

    assert len(result) == 2
    assert list(result.columns) == ["name", "value", "tags"]
    # Complex column came through as a JSON string
    assert result["tags"].iloc[0] == '["z"]'


def test_sanitize_false_passes_complex_columns_through():
    """With sanitize=False, complex columns reach SQLite and raise (legacy behavior)."""
    df = pd.DataFrame(
        {
            "name": ["A", "B"],
            "tags": [["x", "y"], ["z"]],
        }
    )

    with pytest.raises((ProgrammingError, PandaSQLException)):
        apply_sql_query(df, "SELECT * FROM df", sanitize=False)


def test_sanitize_false_with_columns_whitelist_still_works():
    """sanitize=False preserves the legacy columns-whitelist workaround."""
    df = pd.DataFrame(
        {
            "name": ["A", "B", "C"],
            "value": [10, 20, 30],
            "tags": [["x"], ["y"], ["z"]],
        }
    )

    result = apply_sql_query(
        df,
        "SELECT name, value FROM df WHERE value > 15",
        columns=["name", "value"],
        sanitize=False,
    )

    assert len(result) == 2
    assert list(result.columns) == ["name", "value"]


def test_sanitize_empty_query_returns_sanitized_df():
    """sanitize runs before the empty-query early-return, so complex cols are stringified."""
    df = pd.DataFrame(
        {
            "name": ["A", "B"],
            "tags": [["x", "y"], ["z"]],
        }
    )

    result = apply_sql_query(df, "")

    assert result["tags"].iloc[0] == '["x", "y"]'
    assert result["tags"].iloc[1] == '["z"]'


def test_sanitize_empty_query_preserves_simple_df():
    """sanitize=True is a no-op for simple typed columns (backward compatibility)."""
    df = pd.DataFrame({"x": [1, 2, 3]})

    result = apply_sql_query(df, "")
    pd.testing.assert_frame_equal(result, df)


def test_sanitize_geodataframe_preserves_geometry_with_complex_columns():
    """GeoDataFrame: complex non-geometry columns are sanitized, geometry preserved."""
    gdf = gpd.GeoDataFrame(
        {
            "name": ["A", "B", "C"],
            "value": [10, 20, 30],
            "tags": [["x", "y"], ["z"], []],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
        },
        crs="EPSG:4326",
    )

    result = apply_sql_query(gdf, "SELECT name, value, tags, geometry FROM df WHERE value > 15")

    assert isinstance(result, gpd.GeoDataFrame)
    assert result.crs == gdf.crs
    assert len(result) == 2
    assert result["geometry"].iloc[0].equals(Point(1, 1))
    assert result["tags"].iloc[0] == '["z"]'
