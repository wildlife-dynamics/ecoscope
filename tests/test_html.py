import sys
import os
from ecoscope.mapping.map import EcoMap

sys.path.append("../ecoscope")

m = EcoMap(draw_control=False)

relative_path = os.path.join("tests/test_output/new.html")
m.to_html(relative_path)
