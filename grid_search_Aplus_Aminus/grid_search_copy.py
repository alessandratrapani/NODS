import sys
import os
import nest
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from simulateEBCC import SimulateEBCC
import numpy as np
import gc

dim_min = 3
dim_plus = 3
n_sim = 4

min = 1
max = 1
j = 69

for i in range(0,1):
    #for j in range(0,dim_plus):
        for k in range(1,n_sim):
            os.system(f'python /home/nomodel/code/NODS/grid_search_Aplus_Aminus/simulation_grid.py {i} {j} {k} {min} {max}')
