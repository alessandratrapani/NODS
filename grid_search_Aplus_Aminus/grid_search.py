import sys
import os
import nest
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from simulateEBCC import SimulateEBCC
import numpy as np
import gc

n_sim = 10

noise_rate = 4.0

i = 4
j = 1

for k in range(0,n_sim):
    os.system(f'python /home/nomodel/code/NODS/grid_search_Aplus_Aminus/simulation_grid.py {noise_rate} {k} {i} {j}')