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

color_standard_STPD = "#000000"
color_NO_STDP = "#82B366"
noise_rates = [0, 4, 8]
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
    spk = get_spike_activity(cell_name=cell, path=results_path)
    sdf_mean_over_trials = []
    sdf_baseline = np.zeros((n_trials))
    sdf_cr = np.zeros((n_trials))

    results_path_NO = f"./results/complete_EBCC_withNO/EBCC_NO_{noise_rate}Hz/"
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

    # fig = plt.figure()
    # plt.plot(sdf_change_baseline,"--o",markersize=3,color=color_standard_STPD,label="Baseline mean sdf without NO")
    # plt.plot(sdf_change_cr, "-o", markersize=3, color=color_standard_STPD, label="CR window sdf without NO")
    # plt.plot(sdf_change_baseline_NO,'--o',markersize=3,color=color_NO_STDP,label="Baseline mean sdf withNO")
    # plt.plot(sdf_change_cr_NO, "-o", markersize=3, color=color_NO_STDP, label="CR window sdf with NO")
    # plt.ylim(-30,10)
    # plt.xlabel("Trials")
    # plt.ylabel("SDF mean value [Hz]")
    # # plt.show()
    # fig.savefig(f"aa_sdf_change_{noise_rate}Hz.png")


    # fig1 = plt.figure()
    # for i in range(5):
    #     plt.plot(sdf_mean_over_trials[-1*i]-sdf_mean_over_trials[1], color=color_standard_STPD,alpha=0.5)
    #     plt.plot(sdf_mean_over_trials_NO[-1*i]-sdf_mean_over_trials_NO[1], color=color_NO_STDP,alpha=0.5)
    # plt.ylim(-35,10)
    # plt.xlabel("Time [s]")
    # plt.ylabel("SDF mean value [Hz]")
    # plt.show()
    # fig1.savefig(f"ab_sdf_diff_{noise_rate}Hz.png")

    boxes = [sdf_change_baseline[-10:],sdf_change_baseline_NO[-10:],sdf_change_cr[-10:],sdf_change_cr_NO[-10:]]
    colors = [color_standard_STPD, color_NO_STDP, color_standard_STPD, color_NO_STDP]
    medianprops = dict(linewidth=2, color='red')
    positions = [1,2,4,5]
    fig2, ax = plt.subplots(figsize=(5,10))

    bp1 = ax.boxplot(boxes, patch_artist=True, medianprops=medianprops, positions=positions)
    for patch, color  in zip(bp1['boxes'],colors):
        patch.set_facecolor(color)

    ax.set_xticks(positions)
    ax.set_xticklabels(["Baseline SDF\nstd STDP", "Baseline SDF\nNO-dep. STDP", "CR Window SDF\nstd STDP", "CR Window SDF\nNO-dep. STDP"], rotation=45)
    ax.set_ylim(-30,0)
    # plt.show()
    # fig2.savefig(f"aa_sdf_boxplot_{noise_rate}Hz.png")