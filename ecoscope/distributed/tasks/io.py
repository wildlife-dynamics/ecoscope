import os

from ecoscope.distributed.decorators import distributed


@distributed
def get_trajectories_from_earthranger(
    # client
    server,
    username,
    tcp_limit,
    sub_page_size,
    # get_subjectgroup_observations
    group_name,
    include_inactive: bool,
    since,
    until,
):
    from ecoscope.io import EarthRangerIO

    earthranger_io = EarthRangerIO(
        server=server,
        username=username,
        password=os.getenv("ER_PASSWORD"),
        tcp_limit=tcp_limit,
        sub_page_size=sub_page_size,
    )
    return earthranger_io.get_subjectgroup_observations(
        group_name=group_name,
        include_subject_details=True,
        include_inactive=include_inactive,
        since=since,
        until=until,
    )
