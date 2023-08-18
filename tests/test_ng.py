import sys
sys.path.append('../ecoscope')
from ecoscope.mapping.map import EcoMap

m=EcoMap(draw_control=False)
m.add_north_arrow(position="topleft",angle=0,scale=1.0)
m.to_png(r'C:\Users\DELL\Documents\EcoMap\new.png')