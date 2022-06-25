import email
import os
import typing

import pandas as pd
import requests
from tqdm.auto import tqdm


def to_hex(val, default="#ff0000"):
    if val and not pd.isnull(val):
        return "#{:02X}{:02X}{:02X}".format(*[int(i) for i in val.split(",")])
    return default


def pack_columns(dataframe: pd.DataFrame, columns: typing.List):
    """This method would add all extra columns to single column"""
    metadata_cols = list(set(dataframe.columns).difference(set(columns)))

    # To prevent additional column from being dropped, name the column metadata (rename it back).
    if metadata_cols:
        dataframe["metadata"] = dataframe[metadata_cols].to_dict(orient="records")
        dataframe.drop(metadata_cols, inplace=True, axis=1)
        dataframe.rename(columns={"metadata": "additional"}, inplace=True)
    return dataframe


def extract_voltage(s: typing.Dict):
    """
    Extracts voltage from different source-provider in EarthRanger
    Parameters
    ----------
    s: typing.Dict

    Returns
    -------
    typing.Any

    """
    additional = s["extra__observation_details"] or {}
    voltage = additional.get("battery", None)  # savannah tracking
    if not voltage:
        voltage = additional.get("mainVoltage", None)  # vectronics
    if not voltage:
        voltage = additional.get("batt", None)  # AWT
    if not voltage:
        voltage = additional.get("power", None)  # Followit
    return voltage


def download_file(url, path, overwrite_existing=False, chunk_size=1024, **request_kwargs):
    r = requests.get(url, stream=True, **request_kwargs)

    if os.path.isdir(path):
        m = email.message.Message()
        m["content-type"] = r.headers["content-disposition"]
        filename = m.get_param("filename")
        if filename is None:
            raise ValueError("URL has no RFC 6266 filename.")
        path = os.path.join(path, filename)

    if os.path.exists(path) and not overwrite_existing:
        print(f"{path} exists. Skipping...")
        return

    with open(path, "wb") as f:
        with tqdm.wrapattr(f, "write", total=int(r.headers["Content-Length"])) as fout:
            for chunk in r.iter_content(chunk_size=chunk_size):
                fout.write(chunk)
