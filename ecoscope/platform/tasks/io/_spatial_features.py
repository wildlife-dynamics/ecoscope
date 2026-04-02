"""Task for loading spatial feature groups from multiple sources."""

import os
import uuid
from typing import Annotated, Literal, cast

import geopandas as gpd  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register

from ecoscope.platform.annotations import AdvancedField
from ecoscope.platform.connections import EarthRangerConnection
from ecoscope.platform.schemas import EmptyDataFrame, RegionsGDF
from ecoscope.platform.tasks.io._earthranger import (
    get_spatial_features_group,
)

GEOPARQUET_EXTENSIONS = (".parquet", ".geoparquet")
GEOPACKAGE_EXTENSIONS = (".gpkg",)
VALID_EXTENSIONS = GEOPARQUET_EXTENSIONS + GEOPACKAGE_EXTENSIONS


class LocalFileSpatialFeatures(BaseModel):
    model_config = ConfigDict(json_schema_extra={"title": "Local File"})
    source: Annotated[Literal["local_file"], Field(exclude=True)] = "local_file"
    file_path: Annotated[
        str,
        Field(description="Path to geoparquet (.parquet) or geopackage (.gpkg) file"),
    ]
    name_column: Annotated[
        str,
        Field(description="Column to use as region name"),
    ]
    layer: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="Layer name (only applicable to geopackage files)",
        ),
    ] = None

    @model_validator(mode="after")
    def validate_file_extension(self) -> "LocalFileSpatialFeatures":
        path = self.file_path.lower()
        if not path.endswith(VALID_EXTENSIONS):
            raise ValueError(
                f"File must be geoparquet {GEOPARQUET_EXTENSIONS} or "
                "geopackage {GEOPACKAGE_EXTENSIONS}. Got: {self.file_path}"
            )
        if self.layer is not None and not path.endswith(GEOPACKAGE_EXTENSIONS):
            raise ValueError("Layer can only be specified for geopackage (.gpkg) files")
        return self


class RemoteFileSpatialFeatures(BaseModel):
    model_config = ConfigDict(json_schema_extra={"title": "Remote URL"})
    source: Annotated[Literal["remote_file"], Field(exclude=True)] = "remote_file"
    url: Annotated[
        str,
        Field(description="URL to geoparquet (.parquet) or geopackage (.gpkg) file"),
    ]
    name_column: Annotated[
        str,
        Field(description="Column to use as region name"),
    ]
    layer: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="Layer name (only applicable for geopackage files)",
        ),
    ] = None

    @model_validator(mode="after")
    def validate_url_extension(self) -> "RemoteFileSpatialFeatures":
        path = self.url.split("?")[0].lower()
        if not path.endswith(VALID_EXTENSIONS):
            raise ValueError(
                f"URL must point to geoparquet {GEOPARQUET_EXTENSIONS} or "
                "geopackage {GEOPACKAGE_EXTENSIONS}. Got: {self.url}"
            )
        if self.layer is not None and not path.endswith(GEOPACKAGE_EXTENSIONS):
            raise ValueError("Layer can only be specified for geopackage (.gpkg) files")
        return self


class EarthRangerSpatialFeatures(BaseModel):
    model_config = ConfigDict(json_schema_extra={"title": "EarthRanger"})
    source: Annotated[Literal["earthranger"], Field(exclude=True)] = "earthranger"
    data_source: Annotated[
        str,
        Field(
            title="Data Source",
            description="Select one of your configured EarthRanger data sources.",
        ),
    ]
    spatial_features_group_name: Annotated[str, Field(description="Name of the spatial features group in EarthRanger")]


SpatialFeaturesConfig = LocalFileSpatialFeatures | RemoteFileSpatialFeatures | EarthRangerSpatialFeatures


def _validate_regions(regions_gdf: gpd.GeoDataFrame) -> None:
    """Validate name uniqueness of a regions GeoDataFrame.

    Args:
        regions_gdf: GeoDataFrame with 'name' and 'geometry' columns.

    Raises:
        ValueError: If names are not unique.
    """
    duplicated = regions_gdf["name"][regions_gdf["name"].duplicated()]
    if not duplicated.empty:
        raise ValueError(f"Region names must be unique. Duplicates: {sorted(duplicated.unique())}")


def _normalize_crs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Convert GeoDataFrame to CRS 4326."""
    if gdf.crs is None:
        raise ValueError(
            "GeoDataFrame has no CRS information. "
            "Cannot safely convert to EPSG:4326 without knowing the source CRS. "
            "Please ensure the source file includes CRS metadata."
        )
    return gdf.to_crs(4326)


def _load_spatial_regions_from_file(
    file_path: str,
    layer: str | None,
    name_column: str,
    display_name: str,
) -> RegionsGDF | EmptyDataFrame:
    """Load and convert a file to RegionsGDF format."""
    # Load file
    if file_path.lower().endswith(GEOPACKAGE_EXTENSIONS):
        gdf = gpd.read_file(file_path, layer=layer)
    else:
        gdf = gpd.read_parquet(file_path)

    # Validate required columns
    if name_column not in gdf.columns:
        raise ValueError(f"Column '{name_column}' not found in file. Available: {list(gdf.columns)}")
    if "geometry" not in gdf.columns:
        raise ValueError("File must have a 'geometry' column")

    gdf = _normalize_crs(gdf)

    # Filter to polygon geometries only
    polygon_mask = gdf.geometry.geom_type.isin({"Polygon", "MultiPolygon"})
    gdf = gdf[polygon_mask].reset_index(drop=True)

    if gdf.empty:
        raise ValueError("No Polygon or MultiPolygon geometries found in the file")

    # Generate UUIDs for each row
    uuids = [str(uuid.uuid4()) for _ in range(len(gdf))]

    # Convert to RegionsGDF schema
    regions_gdf = gpd.GeoDataFrame(
        {
            "pk": uuids,
            "name": gdf[name_column].astype(str),
            "short_name": gdf[name_column].astype(str),
            "feature_type": "polygon",
            "geometry": gdf.geometry,
            "metadata": [{"id": uid, "display_name": display_name} for uid in uuids],
        },
        crs=4326,
    )

    _validate_regions(regions_gdf)
    return cast(RegionsGDF, regions_gdf)


@register(tags=["io"])
def load_spatial_features_group(
    config: Annotated[
        SpatialFeaturesConfig,
        Field(title="Spatial Feature Data Source"),
    ],
) -> RegionsGDF | EmptyDataFrame:
    """
    Load spatial feature group from local file, remote URL, or EarthRanger.

    Supports geoparquet and geopackage files. All geometries are converted to
    CRS 4326 and filtered to polygons only.

    Args:
        config: Configuration specifying the source type and parameters.

    Returns:
        RegionsGDF with columns: pk, name, short_name, feature_type, geometry, metadata.
        Returns EmptyDataFrame if no polygons found.

    Raises:
        ValueError: If file format is invalid or required columns are missing.
    """

    if isinstance(config, EarthRangerSpatialFeatures):
        client = EarthRangerConnection.client_from_named_connection(config.data_source)
        regions_gdf = get_spatial_features_group(
            client=client,
            spatial_features_group_name=config.spatial_features_group_name,
        )

        # Filter to polygon geometries only
        polygon_mask = regions_gdf.geometry.geom_type.isin({"Polygon", "MultiPolygon"})
        regions_gdf = regions_gdf[polygon_mask].reset_index(drop=True)

        if regions_gdf.empty:
            raise ValueError("No Polygon or MultiPolygon geometries found")

        _validate_regions(regions_gdf)
        return cast(RegionsGDF, regions_gdf)

    elif isinstance(config, RemoteFileSpatialFeatures):
        import tempfile

        from ecoscope.io import download_file  # type: ignore[import-untyped]

        # Derive display_name from URL filename
        display_name = config.url.split("/")[-1].rsplit(".", 1)[0].replace("-", " ").replace("_", " ").title()

        # Download to temp directory
        tmp_dir = tempfile.mkdtemp()
        download_file(config.url, tmp_dir)

        # Find the downloaded file
        downloaded = os.listdir(tmp_dir)
        if not downloaded:
            raise ValueError(f"No file downloaded from {config.url}")
        temp_path = os.path.join(tmp_dir, downloaded[0])

        return _load_spatial_regions_from_file(
            file_path=temp_path,
            layer=config.layer,
            name_column=config.name_column,
            display_name=display_name,
        )

    else:  # LocalFileSpatialFeatures
        # Derive display_name from filename
        display_name = os.path.basename(config.file_path).rsplit(".", 1)[0].replace("-", " ").replace("_", " ").title()

        return _load_spatial_regions_from_file(
            file_path=config.file_path,
            layer=config.layer,
            name_column=config.name_column,
            display_name=display_name,
        )
