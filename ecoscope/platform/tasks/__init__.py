"""Task library for ecoscope platform workflows.

Re-exports all task subpackages (analysis, config, filter, groupby, io,
preprocessing, results, skip, test, transformation, warning) so they can be
imported as ``from ecoscope.platform.tasks import <subpackage>``.
"""

from . import analysis as analysis
from . import config as config
from . import filter as filter
from . import groupby as groupby
from . import io as io
from . import preprocessing as preprocessing
from . import results as results
from . import skip as skip
from . import test as test
from . import transformation as transformation
from . import warning as warning
