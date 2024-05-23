#%%
import sys
import os
sys.path.append(os.path.join(os.getcwd(), '..'))
from utils import get_spike_activity, sdf, sdf_mean, sdf_maf
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

root_path = "/home/nomodel/code/NODS/results/grid_search/ord_of_mag/"
with open("/home/nomodel/code/NODS/network_configuration.json", "r") as json_file:
    net_config = json.load(json_file)

CS_burst_dur = net_config["devices"]["CS"]["parameters"]["burst_dur"]
CS_start_first = float(net_config["devices"]["CS"]["parameters"]["start_first"])
between_start = net_config["devices"]["CS"]["parameters"]["between_start"]
n_trials = net_config["devices"]["CS"]["parameters"]["n_trials"]
US_start_first = float(net_config["devices"]["US"]["parameters"]["start_first"])

#%%

fig = True

cell = "pc_spikes"

grid_search = np.zeros((6,6))
grid_frequency = np.zeros((6,6))
grid_learning = np.zeros((6,6))
grid_baseline = np.zeros((6,6))

for i in range(1,7):
    for j in range(1,7):

        results_path = root_path + f"min{i}_plus{j}/"

        spk = get_spike_activity(cell_name=cell, path=results_path)
        sdf_mean_over_trials = []
        sdf_baseline = np.zeros((n_trials))
        sdf_cr = np.zeros((n_trials))
        for trial in range(n_trials):

            start = trial * between_start
            stop = CS_start_first + CS_burst_dur + trial * between_start

            sdf_cells = sdf(start=start, stop=stop, spk=spk, step=5)
            sdf_mean_trial = sdf_mean(sdf_cells)
            sdf_mean_over_trials.append(sdf_mean_trial)
            sdf_baseline[trial] = np.mean(sdf_mean_trial[100:150])
            sdf_cr[trial] = np.mean(sdf_mean_trial[250:300])

        sdf_change_baseline = sdf_baseline[1:] - sdf_baseline[1]
        sdf_change_cr = sdf_cr[1:] - sdf_cr[1]
    
        # learning in respect to the baseline
        grid_search[i-1,j-1] = (np.median(sdf_change_baseline[-10:]) - np.median(sdf_change_cr[-10:]))
        # pc frequency
        grid_frequency[i-1,j-1] = sdf_cr[-1]
        # percentage change baseline
        grid_baseline[i-1,j-1]  = (np.mean(sdf_baseline[1:10])-np.mean(sdf_baseline[-10:]))/np.mean(sdf_baseline[1:10])*100
        # percentage learning CR
        grid_learning[i-1,j-1]  = (np.mean(sdf_cr[1:10])-np.mean(sdf_cr[-10:]))/np.mean(sdf_cr[1:10])*100

        if fig:

            palette = list(reversed(sns.color_palette("viridis", n_trials).as_hex()))
            sm = plt.cm.ScalarMappable(
            cmap="viridis_r", norm=plt.Normalize(vmin=0, vmax=n_trials)
            )
            sdf_mean_cell = []
            sdf_maf_cell = []
            step = 5
            for trial in range(n_trials):
                start = trial * between_start
                stop = CS_start_first + CS_burst_dur + trial * between_start
                sdf_cell = sdf(start=start, stop=stop, spk=spk, step=step)
                sdf_mean_cell.append(sdf_mean(sdf_cell))
                sdf_maf_cell.append(sdf_maf(sdf_cell))

            fig = plt.figure()
            for trial in range(n_trials):
                plt.plot(sdf_mean_cell[trial], palette[trial])
            plt.title(cell)
            plt.xlabel("Time [ms]")
            plt.ylabel("SDF [Hz]")
            plt.axvline(CS_start_first, label="CS start", c="grey")
            plt.axvline(US_start_first - between_start, label="US start", c="black")
            plt.axvline(CS_start_first + CS_burst_dur, label="CS & US end ", c="red")
            plt.legend()
            plt.colorbar(sm, label="Trial")
            fig.savefig(f"aa_sdf_{cell}_min{i}_plus{j}.png")

#%%

ticks = -1*np.arange(1,7)

#%%
cmap = plt.cm.get_cmap('coolwarm')
sns.heatmap(grid_search, cmap = cmap, xticklabels=ticks, yticklabels=ticks)
plt.xlabel('A_plus')
plt.ylabel('A_minus')
plt.title('CR window learning to baseline')

#%%
cmap = plt.cm.get_cmap('coolwarm')
sns.heatmap(grid_search[2:,2:], cmap = cmap, xticklabels=ticks[2:], yticklabels=ticks[2:])
plt.xlabel('A_plus')
plt.ylabel('A_minus')
plt.title('CR window learning to baseline')

#%%
cmap = plt.cm.get_cmap('hot')
sns.heatmap(grid_frequency, cmap = cmap, vmin = 100, vmax = 200, xticklabels=ticks, yticklabels=ticks)
plt.xlabel('A_plus')
plt.ylabel('A_minus')
plt.title('PC sdf in CR window')

#%%
cmap = plt.cm.get_cmap('summer')
sns.heatmap(grid_learning[2:,2:], cmap = cmap, xticklabels=ticks[2:], yticklabels=ticks[2:], vmin = -20, vmax = 20)
plt.xlabel('A_plus')
plt.ylabel('A_minus')
plt.title('%LTD in CR window')

cmap = plt.cm.get_cmap('summer')
sns.heatmap(grid_baseline[2:,2:], cmap = cmap, xticklabels=ticks[2:], yticklabels=ticks[2:], vmin = -20, vmax = 20)
plt.xlabel('A_plus')
plt.ylabel('A_minus')
plt.title('%LTD in baseline')