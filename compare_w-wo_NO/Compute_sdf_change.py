import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import get_spike_activity, sdf, sdf_mean, sdf_change
import json
import numpy as np
import matplotlib.pyplot as plt

noise_rates=[0,4,8]
with open("./demo_cerebellum.json", "r") as json_file:
    net_config = json.load(json_file)

CS_burst_dur = net_config["devices"]["CS"]["parameters"]["burst_dur"]
CS_start_first = float(net_config["devices"]["CS"]["parameters"]["start_first"])
between_start = net_config["devices"]["CS"]["parameters"]["between_start"]
n_trials = net_config["devices"]["CS"]["parameters"]["n_trials"]
US_start_first = float(net_config["devices"]["US"]["parameters"]["start_first"])
cell = "pc_spikes"

for noise_rate in noise_rates:
    results_path = f"./results/complete_EBCC_withoutNO/EBCC_{noise_rate}Hz/"
    spk = get_spike_activity(cell_name=cell, path = results_path)
    sdf_mean_cell = []
    sdf_change_alltrials = []

    results_path_NO = f"./results/complete_EBCC_withNO/EBCC_NO_{noise_rate}Hz/"
    spk_NO = get_spike_activity(cell_name=cell, path = results_path_NO)
    sdf_mean_cell_NO = []
    sdf_change_alltrials_NO = []

    for trial in range(20,30):
        start = trial * between_start
        stop = CS_start_first + CS_burst_dur + trial * between_start

        sdf_cells = sdf(start=start, stop=stop, spk=spk, step = 5)
        sdf_change_trial = sdf_change(sdf_cells)
        sdf_change_alltrials.append(np.array(sdf_change_trial))
        sdf_mean_trial = sdf_mean(sdf_cells)
        sdf_mean_cell.append(sdf_mean_trial)

        sdf_cells_NO = sdf(start=start, stop=stop, spk=spk_NO, step = 5)
        sdf_change_trial_NO = sdf_change(sdf_cells_NO)
        sdf_change_alltrials_NO.append(np.array(sdf_change_trial_NO))
        sdf_mean_trial_NO = sdf_mean(sdf_cells_NO)
        sdf_mean_cell_NO.append(sdf_mean_trial_NO)

    fig = plt.figure()
    plt.boxplot([np.mean(np.array(sdf_change_alltrials), axis=1), np.mean(np.array(sdf_change_alltrials_NO), axis=1)])
    plt.legend() 
    # plt.ylim(-35,10)
    plt.title("SDF change")
    plt.xlabel("Trials")
    plt.ylabel("SDF change [Hz]")
    plt.show()
    fig.savefig(f"aa_SDF_change_{noise_rate}Hz.png")
