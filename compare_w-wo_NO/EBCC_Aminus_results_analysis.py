# %%
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import get_spike_activity, sdf, sdf_mean
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
A_minus_variations=[80,50,20,10]
with open("./demo_cerebellum.json", "r") as json_file:
    net_config = json.load(json_file)

CS_burst_dur = net_config["devices"]["CS"]["parameters"]["burst_dur"]
CS_start_first = float(net_config["devices"]["CS"]["parameters"]["start_first"])
between_start = net_config["devices"]["CS"]["parameters"]["between_start"]
n_trials = net_config["devices"]["CS"]["parameters"]["n_trials"]
US_start_first = float(net_config["devices"]["US"]["parameters"]["start_first"])
cell = "pc_spikes"
for A_minus in A_minus_variations:

    results_path = f"./results/complete_EBCC_withoutNO/EBCC_{noise_rate}Hz/"
    spk = get_spike_activity(cell_name=cell, path = results_path)
    sdf_mean_cell = []
    mean_sdf_intervals = np.zeros((n_trials,2))
    for trial in range(n_trials):
        start = trial * between_start
        stop = CS_start_first + CS_burst_dur + trial * between_start
        sdf_cells = sdf(start=start, stop=stop, spk=spk, step = 5)
        sdf_mean_trial = sdf_mean(sdf_cells)
        sdf_mean_cell.append(sdf_mean_trial)
        mean_sdf_intervals[trial,0] = np.mean(sdf_mean_trial[150:200])
        mean_sdf_intervals[trial,1] = np.mean(sdf_mean_trial[250:300])
    fig = plt.figure()
    plt.plot(mean_sdf_intervals[1:,0]-mean_sdf_intervals[1,0],'-*', label="150-200 ms")
    plt.plot(mean_sdf_intervals[1:,1]-mean_sdf_intervals[1,1],'-*', label="250-300 ms")
    plt.legend() 
    plt.ylim(-35,10)
    plt.title("Mean SDF")
    plt.xlabel("Trials")
    plt.ylabel("Difference in mean SDF [Hz]")
    # plt.show()
    fig.savefig(results_path+"aa_mean_diff_sdf_interval_over_trials.png")

    results_path = f"./results/complete_EBCC_withNO/EBCC_NO_{noise_rate}Hz/"
    spk = get_spike_activity(cell_name=cell, path = results_path)
    sdf_mean_cell = []
    mean_sdf_intervals_NO = np.zeros((n_trials,2))
    for trial in range(n_trials):
        start = trial * between_start
        stop = CS_start_first + CS_burst_dur + trial * between_start
        sdf_cells = sdf(start=start, stop=stop, spk=spk, step = 5)
        sdf_mean_trial = sdf_mean(sdf_cells)
        sdf_mean_cell.append(sdf_mean_trial)
        mean_sdf_intervals_NO[trial,0] = np.mean(sdf_mean_trial[150:200])
        mean_sdf_intervals_NO[trial,1] = np.mean(sdf_mean_trial[250:300])

    fig = plt.figure()
    plt.plot(mean_sdf_intervals_NO[1:,0]-mean_sdf_intervals_NO[1,0],'-*', label="150-200 ms")
    plt.plot(mean_sdf_intervals_NO[1:,1]-mean_sdf_intervals_NO[1,1],'-*', label="250-300 ms")
    plt.legend() 
    plt.ylim(-35,10)
    plt.title("Mean SDF")
    plt.xlabel("Trials")
    plt.ylabel("Difference in mean SDF [Hz]")
    # plt.show()
    fig.savefig(results_path+"aa_mean_diff_sdf_interval_over_trials.png")

    fig1 = plt.figure()
    K = mean_sdf_intervals[1:,0]-mean_sdf_intervals[1,0]
    plt.plot((mean_sdf_intervals[1:,1]-mean_sdf_intervals[1,1])/K,'-*', label="Without NO")
    K_NO = mean_sdf_intervals_NO[1:,0]-mean_sdf_intervals_NO[1,0]
    plt.plot((mean_sdf_intervals_NO[1:,1]-mean_sdf_intervals_NO[1,1])/K_NO,'-*', label="With NO")
    plt.legend() 
    plt.title("Mean SDF")
    plt.xlabel("Trials")
    plt.ylabel("Mean SDF [Hz]")
    # plt.show()
    fig1.savefig(f"aa_ratio_sdf_interval_over_trials_{noise_rate}Hz.png")