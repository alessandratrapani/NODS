import sys
import os
import nest
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from simulateEBCC import SimulateEBCC
import numpy as np
import gc

dim_min = 3
dim_plus = 3
n_sim = 10

noise_rate = 0.0

for k in range(3,n_sim):
    os.system(f'python /home/nomodel/code/NODS/grid_search_Aplus_Aminus/simulation_grid.py {noise_rate} {k}')