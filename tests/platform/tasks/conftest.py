"""Shared fixtures for the platform task tests.

The warehouse observations table factory lives here rather than in the individual
warehouse test modules: it is built against ``OBSERVATIONS_SCHEMA__ECOSCOPE_SLIM_V1``,
and ``pa.table(..., schema=...)`` raises ``KeyError`` for any schema field it is not
given an array for. So every column io-core adds to that schema breaks every copy of
this helper at once -- which is exactly what happened when v0.0.22 added
``subject_additional``.
"""

from datetime import datetime, timezone

import pytest


@pytest.fixture
def warehouse_observations_table():
    """Callable building a ``pa.Table`` shaped like ``ERWarehouseClient`` observations.

    ``fixtimes`` are emitted in the order given, which is what lets callers simulate the
    warehouse returning rows in an arbitrary order; omit them for one fix per subject a
    minute apart. ``additional`` is one value per row for ``extra__subject__additional``
    -- pass None for an all-null column, which is what the warehouse serves when the
    caller does not set ``include_subject_additional``.
    """

    def _make(subject_ids=("s1", "s2"), fixtimes=None, sources=None, additional=None):
        import geoarrow.pyarrow as ga  # type: ignore[import-untyped]
        import pyarrow as pa
        from ecoscope_earthranger_io_core.arrow import OBSERVATIONS_SCHEMA__ECOSCOPE_SLIM_V1
        from shapely.geometry import Point

        subject_ids = list(subject_ids)
        n = len(subject_ids)
        if fixtimes is None:
            fixtimes = [datetime(2015, 1, 1, minute=i, tzinfo=timezone.utc) for i in range(n)]
        if sources is None:
            sources = [f"source-{s}" for s in subject_ids]

        return pa.table(
            {
                "geometry": ga.array([Point(37.5 + i * 0.001, -2.5).wkb for i in range(n)]),
                "fixtime": pa.array(list(fixtimes), type=pa.timestamp("ns", tz="UTC")),
                "groupby_col": pa.array(subject_ids, type=pa.string()),
                "extra__subject__name": pa.array([f"subj-{s}" for s in subject_ids], type=pa.string()),
                "extra__subject__subject_subtype": pa.array(["elephant"] * n, type=pa.string()),
                "extra__subject__additional": pa.array(
                    list(additional) if additional is not None else [None] * n, type=pa.string()
                ),
                "extra__source": pa.array(list(sources), type=pa.string()),
                "junk_status": pa.array([False] * n, type=pa.bool_()),
            },
            schema=OBSERVATIONS_SCHEMA__ECOSCOPE_SLIM_V1,
        )

    return _make
