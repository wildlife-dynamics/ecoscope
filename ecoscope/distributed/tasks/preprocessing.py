from ecoscope.distributed.decorators import distributed


@distributed
def relocations_filters(
    observations: ...,
    relocs_filter_coords,   
):
    from ecoscope.base import RelocsCoordinateFilter, Relocations
    
    relocs = Relocations(observations)
    relocs.apply_reloc_filter(
        RelocsCoordinateFilter(filter_point_coords=relocs_filter_coords),
        inplace=True,
    )
    relocs.remove_filtered(inplace=True)


# Trajectory filter
    # # trajectory filter
    # min_length_meters: float = 0.001,
    # max_length_meters: float = 10000,
    # max_time_secs: float = 3600,
    # min_time_secs: float = 1,
    # max_speed_kmhr: float = 120,
    # min_speed_kmhr: float = 0.0,
