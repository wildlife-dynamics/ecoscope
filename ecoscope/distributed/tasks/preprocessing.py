from typing import Annotated

from pydantic import Field

from ecoscope.distributed.decorators import distributed
from ecoscope.distributed.tasks.io import SubjectGroupObservationsGDFSchema
from ecoscope.distributed.types import DataFrame


class RelocationsGDFSchema(SubjectGroupObservationsGDFSchema):
    # FIXME: how does this differ from `SubjectGroupObservationsGDFSchema`?
    pass


@distributed
def process_relocations(
    observations: DataFrame[SubjectGroupObservationsGDFSchema],
    /,
    filter_point_coords: Annotated[list[list[float]], Field()],   
    relocs_columns: Annotated[list[str], Field()],
) -> DataFrame[RelocationsGDFSchema]:
    from ecoscope.base import RelocsCoordinateFilter, Relocations
    
    relocs = Relocations(observations)

    # filter relocations based on the config
    relocs.apply_reloc_filter(
        RelocsCoordinateFilter(filter_point_coords=filter_point_coords),
        inplace=True,
    )
    relocs.remove_filtered(inplace=True)

    # subset columns
    relocs = relocs[relocs_columns]

    # rename columns
    relocs.columns = [i.replace("extra__", "") for i in relocs.columns]
    relocs.columns = [i.replace("subject__", "") for i in relocs.columns]
    return relocs


# Trajectory filter
    # # trajectory filter
    # min_length_meters: float = 0.001,
    # max_length_meters: float = 10000,
    # max_time_secs: float = 3600,
    # min_time_secs: float = 1,
    # max_speed_kmhr: float = 120,
    # min_speed_kmhr: float = 0.0,
