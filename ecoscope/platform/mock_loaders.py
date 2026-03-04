"""Mock data loaders for wt-task testing.

This module provides a parquet loader registered via the ``wt_task.mock_loaders``
entry point group, enabling ``wt_task.testing.create_task_magicmock`` to load
``.parquet`` example-return files without wt-task depending on geopandas/pandas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_parquet(path: Path) -> Any:
    """Load a parquet or geoparquet file.

    Attempts to read as a GeoDataFrame first (via geopandas); falls back to
    a plain DataFrame (via pandas) if the file doesn't contain geometry data.

    Args:
        path: Path to the parquet file.

    Returns:
        A GeoDataFrame or DataFrame with the file contents.
    """
    import geopandas as gpd  # type: ignore[import-untyped]
    import pandas as pd

    try:
        return gpd.read_parquet(path.as_uri())
    except ValueError:
        return pd.read_parquet(path.as_uri())
