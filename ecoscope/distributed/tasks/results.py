from typing import Annotated

from pydantic import Field

from ecoscope.distributed.decorators import distributed
from ecoscope.distributed.types import DataFrame, JsonSerializableDataFrameModel

class AnyGDFSchema(JsonSerializableDataFrameModel):
    pass

# NEXT: make a map
# NEXT: summary stats
# NEXT: dag defintion and parsing

@distributed
def gdf_to_ecomap(
    gdf: DataFrame[AnyGDFSchema],
    /,
    title
):
    from ecoscope.mapping import EcoMap

    m = EcoMap(**map_props.general_props.dict())

    m.add_title(title=title, **map_props.title_props.dict())

    # Add tile layers
    for tl in map_props.tile_layers:
        m.add_tile_layer(**tl.dict())

    # add north arrow
    m.add_north_arrow(**map_props.north_arrow_props.dict())

    # add the data
    m.add_gdf(gdf, **map_props.geo_dataframes_props.dict())

    # zoom to the dataset
    m.zoom_to_gdf(gdf)

    return m


@distributed
def summary_statistics():
    ...
