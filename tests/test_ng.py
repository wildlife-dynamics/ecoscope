import sys
import os

from ecoscope.mapping.map import EcoMap

sys.path.append("../ecoscope")


m = EcoMap(draw_control=False)
m.add_north_arrow(position="topleft", angle=0, scale=1.0)

relative_path = os.path.join("tests/test_output/new.html")
m.to_html(relative_path)
