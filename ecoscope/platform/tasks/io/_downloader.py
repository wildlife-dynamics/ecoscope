from typing import Annotated, cast

from pydantic import Field
from wt_registry import register

from ecoscope.platform.annotations import AnyGeoDataFrame


@register(tags=["io"])
def download_roi(
    url: Annotated[str, Field(description="The path to ROI gpkg file")],
    roi_column: Annotated[str | None, Field(description="The column name of the ROI name")] = "name",
    roi_name: Annotated[str | None, Field(description="The ROI name")] = None,
    layer_name: Annotated[str | None, Field(description="The layer name")] = None,
) -> AnyGeoDataFrame:
    """Download ROI from a URL."""
    import tempfile

    import geopandas as gpd  # type: ignore[import-untyped]

    from ecoscope.io import download_file  # type: ignore[import-untyped]

    tmp_roi_path = tempfile.NamedTemporaryFile(suffix=".gpkg").name
    download_file(
        url=url,
        path=tmp_roi_path,
        overwrite_existing=True,
    )

    roi = gpd.read_file(tmp_roi_path, layer=layer_name).to_crs(4326)
    if roi_column is not None:
        roi = roi.rename(columns={roi_column: "name"})
        roi.set_index("name", inplace=True)

    if roi_name:
        roi = roi.loc[roi_name]

    return cast(
        AnyGeoDataFrame,
        roi,
    )
