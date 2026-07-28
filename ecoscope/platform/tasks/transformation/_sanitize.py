import json

import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
import pandas as pd


def _isnull(v):
    """Check if value is null, avoiding ambiguous truth value errors."""
    # None check first
    if v is None:
        return True

    # Handle float NaN
    if isinstance(v, float) and pd.isna(v):
        return True

    # Avoid ambiguous truth value errors with DataFrames/Series
    if isinstance(v, (pd.DataFrame, pd.Series)):
        return False

    # For all other types, try pd.isna
    try:
        result = pd.isna(v)
        # Ensure result is a scalar boolean
        if isinstance(result, (bool, np.bool_)):
            return result
        return False
    except (ValueError, TypeError):
        return False


def _decode_bytes(v):
    if isinstance(v, (bytes, bytearray)):
        try:
            return v.decode("utf-8", errors="replace")
        except Exception:
            return str(v)
    return v


def _json_default(obj):
    """
    Default JSON serializer for objects not serializable by default json module.

    This is used as the 'default' parameter in json.dumps() to handle numpy types
    and other non-serializable objects nested within lists/dicts.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _jsonify(v):
    """
    Convert Python objects to JSON strings for DataFrame cells.

    This function is called via pandas Series.map() to convert complex Python objects
    (lists, dicts, sets, numpy arrays) into JSON strings for Arrow compatibility.
    Uses _json_default to handle nested numpy types.
    """
    if isinstance(v, (np.ndarray, set, list, dict)):
        # Convert sets to lists, then serialize everything with numpy support
        val = list(v) if isinstance(v, set) else v
        return json.dumps(val, ensure_ascii=False, default=_json_default)
    return v


def _looks_numeric_series(s):
    nn = s.dropna()
    if nn.empty:
        return False
    return nn.map(lambda x: isinstance(x, (int, np.integer, float, np.floating))).all()


def _has_bytes(s):
    return s.dropna().map(lambda x: isinstance(x, (bytes, bytearray))).any()


def _has_collections(s):
    return s.dropna().map(lambda x: isinstance(x, (list, dict, set))).any()


def _has_strings(s):
    return s.dropna().map(lambda x: isinstance(x, str)).any()


def _has_numbers(s):
    return s.dropna().map(lambda x: isinstance(x, (int, np.integer, float, np.floating))).any()


def sanitize_for_arrow(
    df: pd.DataFrame | gpd.GeoDataFrame, force_string_cols: set[str] | None = None
) -> pd.DataFrame | gpd.GeoDataFrame:
    """Return a copy where all columns are Arrow-friendly."""
    out = df.copy()
    force_string_cols = force_string_cols or set()

    for col in out.columns:
        s = out[col]

        # Keep datetimes and booleans as-is
        if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue

        # Categoricals are OK; leave or optionally .astype('category')
        if isinstance(s.dtype, pd.CategoricalDtype):
            continue

        # Numeric dtypes are OK
        if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
            continue

        # Everything else (incl. 'object' and mixed)
        if col in force_string_cols:
            s = s.map(_decode_bytes).map(_jsonify)
            out[col] = s.astype("string[pyarrow]")
            continue

        if pd.api.types.is_object_dtype(s):
            # First, normalize bytes and JSON-unsafe Python objects
            if _has_bytes(s) or _has_collections(s):
                s = s.map(_decode_bytes).map(_jsonify)

            # Decide target dtype
            has_str = _has_strings(s)
            has_num = _has_numbers(s)
            if has_num and not has_str and _looks_numeric_series(s):
                # Pure numeric (after cleaning): cast to numeric with pandas nullable types
                out[col] = pd.to_numeric(s, errors="coerce")
                # If it's really integral, upgrade to Int64
                # Use a lambda to handle both int and float safely
                if out[col].dropna().map(lambda x: float(x).is_integer() if pd.notna(x) else False).all():
                    out[col] = out[col].astype("Int64")
                continue

            if has_num and has_str:
                # Mixed numbers + strings → coerce to string
                out[col] = s.map(lambda v: None if _isnull(v) else str(v)).astype("string[pyarrow]")
                continue

            # If it’s neither clearly numeric nor clearly something else, default to string
            out[col] = s.map(lambda v: None if _isnull(v) else str(v)).astype("string[pyarrow]")
            continue

        # Fallback: for extension/unknown dtypes, try pandas' convert_dtypes
        try:
            out[col] = out[col].convert_dtypes()
        except Exception:
            out[col] = out[col].astype("string[pyarrow]")

    return out
