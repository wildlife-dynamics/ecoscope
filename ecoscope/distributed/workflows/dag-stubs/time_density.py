import graphlib

from ecoscope.distributed.tasks.io import _get_subjectgroup_observations
from ecoscope.distributed.tasks.preprocessing import _process_relocations, _relocations_to_trajectory


graph = {
    _get_subjectgroup_observations: {},
    _process_relocations: {_get_subjectgroup_observations},
    _relocations_to_trajectory: {_process_relocations}
}
ts = graphlib.TopologicalSorter(graph=graph)
static_order = tuple(ts.static_order())

# TODO: check I/0 type compatibility of static order (how to mark an arg as coming from an upstream task?)
# TODO: dump to jsonschema
