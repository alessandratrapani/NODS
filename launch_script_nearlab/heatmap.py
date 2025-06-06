import sys
import os
sys.path.append(os.path.join(os.getcwd(), '..'))
from utils import get_spike_activity, sdf, sdf_mean, sdf_maf
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.stats as st
import pandas as pd
import pickle

result_path = "/home/csartor1/code/NODS/results/grid_search/"
with open("/home/csartor1/code/NODS/network_configuration.json", "r") as json_file:
    net_config = json.load(json_file)

CS_burst_dur = net_config["devices"]["CS"]["parameters"]["burst_dur"]
CS_start_first = float(net_config["devices"]["CS"]["parameters"]["start_first"])
between_start = net_config["devices"]["CS"]["parameters"]["between_start"]
n_trials = net_config["devices"]["CS"]["parameters"]["n_trials"]
US_start_first = float(net_config["devices"]["US"]["parameters"]["start_first"])
cell = "pc_spikes"

plus = ['10_25', '10_5', '10_75']
minus = ['15', '16', '17']

grid_search = np.nan*np.ones((len(minus),len(plus)))
grid_frequency = np.zeros((len(minus),len(plus)))
sim_baseline = np.zeros((len(minus),len(plus),100))
sim_cr = np.zeros((len(minus),len(plus),100))
sim_frq = np.zeros((len(minus),len(plus),10))

for i,m in enumerate(minus):
    for j,p in enumerate(plus):
        folder_path = result_path + f"plus{p}_minus{m}/"
        for k in range (0,5):
            file_path = folder_path+ f"sim_{k}/"

            spk = get_spike_activity(cell_name=cell, path=file_path)
            print(f'sim {k}, min {m}, plus {p}')
            sdf_mean_over_trials = []
            
            sdf_baseline = np.zeros((n_trials))
            sdf_cr = np.zeros((n_trials))
            for trial in range(n_trials):
                start = trial * between_start
                stop = CS_start_first + CS_burst_dur + trial * between_start

                sdf_cells = sdf(start=start, stop=stop, spk=spk, step=5)
                sdf_mean_trial = sdf_mean(sdf_cells)
                sdf_mean_over_trials.append(sdf_mean_trial)
                sdf_baseline[trial] = np.mean(sdf_mean_trial[150:200])
                sdf_cr[trial] = np.mean(sdf_mean_trial[250:300])

            sdf_change_baseline = sdf_baseline[1:] - sdf_baseline[1]
            sdf_change_cr = sdf_cr[1:] - sdf_cr[1]
        
            sim_baseline[i,j,k*10:(k+1)*10] = sdf_change_baseline[-10:]
            sim_cr[i,j,k*10:(k+1)*10] = sdf_change_cr[-10:]
            sim_frq[i,j,k] = sdf_cr[1] - sdf_cr[-1]

        # learning in respect to the baseline
        grid_search[i,j] = (np.median(sim_baseline[i,j]) - np.median(sim_cr[i,j]))
        # pc frequency
        grid_frequency[i,j] = np.median(sim_frq[i,j])

with open (result_path+'grid_search.pkl', 'wb') as f:
    pickle.dump(grid_search, f)

with open (result_path+'grid_frequency.pkl', 'wb') as f:
    pickle.dump(grid_frequency, f)