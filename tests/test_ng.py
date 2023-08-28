import sys
import os
sys.path.append('../ecoscope')
from ecoscope.mapping.map import EcoMap

m=EcoMap(draw_control=False)
m.add_north_arrow(position="topleft",angle=0,scale=1.0)

relative_path=os.path.join('tests/test_output/new.html')
m.to_html(relative_path)