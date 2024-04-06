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

noise_rates = [0, 4, 8]
with open("network_configuration.json", "r") as json_file:
    net_config = json.load(json_file)

CS_burst_dur = net_config["devices"]["CS"]["parameters"]["burst_dur"]
CS_start_first = float(net_config["devices"]["CS"]["parameters"]["start_first"])
between_start = net_config["devices"]["CS"]["parameters"]["between_start"]
n_trials = net_config["devices"]["CS"]["parameters"]["n_trials"]
US_start_first = float(net_config["devices"]["US"]["parameters"]["start_first"])

cell_color = net_config["cell_types"]["purkinje_cell"]["color"][0]
CS_color = net_config["colors"]["CS"]
US_color = net_config["colors"]["US"]
with_NO_color = net_config["devices"]["nNOS"]["color"][0]
without_NO_color = "#000000"

cell = "pc_spikes"

for i, noise_rate in enumerate(noise_rates):
   
    results_path = f"/media/amtra/Samsung_T5/RISULTATI_TESI/complete_EBCC/EBCC_{noise_rate}Hz/"
    spk = get_spike_activity(cell_name=cell, path=results_path)
    sdf_mean_over_trials = []
    sdf_baseline = np.zeros((n_trials))
    sdf_cr = np.zeros((n_trials))

    results_path_NO = f"/media/amtra/Samsung_T5/RISULTATI_TESI/complete_EBCC/EBCC_NO_{noise_rate}Hz/"
    spk_NO = get_spike_activity(cell_name=cell, path=results_path_NO)
    sdf_mean_over_trials_NO = []
    sdf_baseline_NO = np.zeros((n_trials))
    sdf_cr_NO = np.zeros((n_trials))

    for trial in range(n_trials):

        start = trial * between_start
        stop = CS_start_first + CS_burst_dur + trial * between_start

        sdf_cells = sdf(start=start, stop=stop, spk=spk, step=5)
        sdf_mean_trial = sdf_mean(sdf_cells)
        sdf_mean_over_trials.append(sdf_mean_trial)
        sdf_baseline[trial] = np.mean(sdf_mean_trial[100:150])
        sdf_cr[trial] = np.mean(sdf_mean_trial[250:300])

        sdf_cells_NO = sdf(start=start, stop=stop, spk=spk_NO, step=5)
        sdf_mean_trial_NO = sdf_mean(sdf_cells_NO)
        sdf_mean_over_trials_NO.append(sdf_mean_trial_NO)
        sdf_baseline_NO[trial] = np.mean(sdf_mean_trial_NO[100:150])
        sdf_cr_NO[trial] = np.mean(sdf_mean_trial_NO[250:300])

    sdf_change_baseline = sdf_baseline[1:] - sdf_baseline[1]
    sdf_change_cr = sdf_cr[1:] - sdf_cr[1]
    sdf_change_baseline_NO = sdf_baseline_NO[1:] - sdf_baseline_NO[1]
    sdf_change_cr_NO = sdf_cr_NO[1:] - sdf_cr_NO[1]

    fig,axs = plt.subplots(1,2,sharey=True)
    axs[0].plot(sdf_change_baseline,"--o",markersize=3,color=without_NO_color,label="Baseline")
    axs[0].plot(sdf_change_cr, "-o", markersize=3, color=without_NO_color, label="CR window")
    axs[1].plot(sdf_change_baseline_NO,'--o',markersize=3,color=with_NO_color,label="Baseline")
    axs[1].plot(sdf_change_cr_NO, "-o", markersize=3, color=with_NO_color, label="CR window")
    axs[0].set_ylim(-30,10)
    axs[0].set_xlabel("Trials")
    axs[1].set_xlabel("Trials")
    axs[0].set_ylabel("SDF change [Hz]")
    axs[0].set_title("standard STDP")
    axs[1].set_title("NO-dependent STDP")
    axs[0].legend()
    axs[1].legend()
    plt.show()
    plt.tight_layout()
    fig.suptitle("SDF change over trials", fontsize=16)
    fig.savefig(f"sdf_change_{noise_rate}Hz.png")
    fig.savefig(f"sdf_change_{noise_rate}Hz.svg")