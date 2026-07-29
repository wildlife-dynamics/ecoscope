"""Unit tests for row-order handling on the ERWarehouseClient observations paths
(ERDW-247).

The warehouse API makes no row-order guarantee -- its DuckDB query path has no
``ORDER BY`` and its parallel scan order varies between runs -- whereas the
EarthRanger API path is sorted by ``recorded_at``. ``Trajectory.from_relocations``
pairs adjacent rows via ``shift(-1)``, so unordered input silently produces
trajectory segments built from unrelated fixes.

These use a mocked ERWarehouseClient (no live server), so unlike the integration
tests in test_io_earthranger.py they are NOT marked ``io`` and therefore run in the
default (non-io) test job.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import geopandas as gpd  # type: ignore[import-untyped]
import pytest

from ecoscope.platform.tasks.filter._filter import UTC_TIMEZONEINFO, TimeRange
from ecoscope.platform.tasks.io import get_subjectgroup_observations
from ecoscope.platform.tasks.io._earthranger import _sort_warehouse_relocations

_TIME_RANGE = TimeRange(
    since=datetime(2015, 1, 1, tzinfo=timezone.utc),
    until=datetime(2015, 1, 31, tzinfo=timezone.utc),
    timezone=UTC_TIMEZONEINFO,
)


def _make_observations_arrow_table(fixtimes, subject_ids, sources=None):
    """Build a pa.Table matching OBSERVATIONS_SCHEMA__ECOSCOPE_SLIM_V1.

    ``fixtimes`` are emitted in the given order, which is what lets these tests
    simulate the warehouse returning rows in an arbitrary order.
    """
    import geoarrow.pyarrow as ga  # type: ignore[import-untyped]
    import pyarrow as pa
    from ecoscope_earthranger_io_core.arrow import OBSERVATIONS_SCHEMA__ECOSCOPE_SLIM_V1
    from shapely.geometry import Point

    n = len(fixtimes)
    sources = sources if sources is not None else ["source-1"] * n
    geometries = [Point(37.5 + i * 0.001, -2.5).wkb for i in range(n)]

    return pa.table(
        {
            "geometry": ga.array(geometries),
            "fixtime": pa.array(fixtimes, type=pa.timestamp("ns", tz="UTC")),
            "groupby_col": pa.array(subject_ids, type=pa.string()),
            "extra__subject__name": pa.array([f"subj-{s}" for s in subject_ids], type=pa.string()),
            "extra__subject__subject_subtype": pa.array(["elephant"] * n, type=pa.string()),
            "extra__source": pa.array(sources, type=pa.string()),
            "junk_status": pa.array([False] * n, type=pa.bool_()),
        },
        schema=OBSERVATIONS_SCHEMA__ECOSCOPE_SLIM_V1,
    )


def _fixtimes(minutes):
    return [datetime(2015, 1, 1, tzinfo=timezone.utc).replace(minute=m % 60, hour=m // 60) for m in minutes]


class TestSortWarehouseRelocations:
    """Direct tests for the _sort_warehouse_relocations helper."""

    def test_sorts_by_groupby_col_then_fixtime(self):
        gdf = gpd.GeoDataFrame(
            {
                "groupby_col": ["b", "a", "b", "a"],
                "fixtime": _fixtimes([30, 40, 10, 20]),
                "extra__source": ["s1"] * 4,
            }
        )
        result = _sort_warehouse_relocations(gdf)

        assert list(result.groupby_col) == ["a", "a", "b", "b"]
        assert result.groupby("groupby_col")["fixtime"].is_monotonic_increasing.all()

    def test_resets_index(self):
        gdf = gpd.GeoDataFrame({"groupby_col": ["a", "a"], "fixtime": _fixtimes([20, 10])})
        result = _sort_warehouse_relocations(gdf)

        assert list(result.index) == [0, 1]

    def test_source_breaks_fixtime_ties_deterministically(self):
        """Two sources reporting the same instant must order the same way every run."""
        same = _fixtimes([10, 10])
        first = _sort_warehouse_relocations(
            gpd.GeoDataFrame({"groupby_col": ["a", "a"], "fixtime": same, "extra__source": ["s2", "s1"]})
        )
        second = _sort_warehouse_relocations(
            gpd.GeoDataFrame({"groupby_col": ["a", "a"], "fixtime": same, "extra__source": ["s1", "s2"]})
        )

        assert list(first.extra__source) == ["s1", "s2"]
        assert list(first.extra__source) == list(second.extra__source)

    def test_missing_required_keys_raises(self):
        """Degrading to a partial sort would silently reintroduce the corruption."""
        with pytest.raises(ValueError, match="missing required sort keys"):
            _sort_warehouse_relocations(gpd.GeoDataFrame({"fixtime": _fixtimes([10])}))

        with pytest.raises(ValueError, match="missing required sort keys"):
            _sort_warehouse_relocations(gpd.GeoDataFrame({"groupby_col": ["a"]}))

    def test_optional_source_key_absent_is_fine(self):
        gdf = gpd.GeoDataFrame({"groupby_col": ["a", "a"], "fixtime": _fixtimes([20, 10])})
        result = _sort_warehouse_relocations(gdf)

        assert result.fixtime.is_monotonic_increasing

    def test_empty_frame(self):
        gdf = gpd.GeoDataFrame({"groupby_col": [], "fixtime": [], "extra__source": []})

        assert _sort_warehouse_relocations(gdf).empty


class TestGetSubjectgroupObservationsRowOrder:
    """The warehouse branch of get_subjectgroup_observations must return sorted rows."""

    def _call_with_table(self, table):
        mock_legacy_client = MagicMock()
        mock_warehouse_client = MagicMock()
        mock_warehouse_client.get_subjectgroup_observations.return_value = table

        with patch(
            "ecoscope.platform.tasks.io._earthranger._make_warehouse_client_from_env",
            return_value=mock_warehouse_client,
        ):
            result = get_subjectgroup_observations(
                client=mock_legacy_client,
                time_range=_TIME_RANGE,
                subject_group_name="Ecoscope-5-Subs",
                raise_on_empty=False,
            )

        mock_warehouse_client.get_subjectgroup_observations.assert_called_once()
        mock_legacy_client.get_subjectgroup_observations.assert_not_called()
        return result

    def test_unordered_warehouse_rows_are_sorted(self):
        """Rows arriving in arbitrary order come back time-ordered per subject."""
        table = _make_observations_arrow_table(
            fixtimes=_fixtimes([50, 10, 30, 20, 40]),
            subject_ids=["s1", "s2", "s1", "s1", "s2"],
        )
        result = self._call_with_table(table)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 5
        assert result.groupby("groupby_col")["fixtime"].is_monotonic_increasing.all()

    def test_row_order_does_not_change_content(self):
        """Sorting must reorder rows only -- never add, drop, or alter them."""
        ordered = _fixtimes([10, 20, 30, 40, 50])
        subjects = ["s1"] * 5
        forward = self._call_with_table(_make_observations_arrow_table(ordered, subjects))
        reversed_ = self._call_with_table(_make_observations_arrow_table(list(reversed(ordered)), subjects))

        assert list(forward.fixtime) == list(reversed_.fixtime)
        assert len(forward) == len(reversed_) == 5
