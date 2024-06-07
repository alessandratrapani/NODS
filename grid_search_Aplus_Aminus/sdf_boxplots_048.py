# %%
import sys
import os
sys.path.append(os.path.join(os.getcwd(), '..'))
from utils import get_spike_activity, sdf, sdf_mean
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.stats as st

noise_rates = [0, 4]
rooth_path = "/home/nomodel/code/NODS/results/"
with open("/home/nomodel/code/NODS/network_configuration.json", "r") as json_file:
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

fig, axs = plt.subplots(1,2,figsize=(8,4))
colors = [without_NO_color, with_NO_color, without_NO_color, with_NO_color]
medianprops = dict(linewidth=1.5, color='white')
positions = [1,2,4,5]

for i, noise_rate in enumerate(noise_rates):
    sdf_mean_trials_simulations = []
    for k in range(0,10):
        results_path = rooth_path + f"grid_search/{noise_rate}Hz/min4_plus8/{k}/"
        spk = get_spike_activity(cell_name=cell, path=results_path)

        sdf_mean_over_trials = []
        step = 5
        for trial in range(n_trials):
            start = trial * between_start
            stop = CS_start_first + CS_burst_dur + trial * between_start
            sdf_cells = sdf(start=start, stop=stop, spk=spk, step=step)
            sdf_mean_trial = sdf_mean(sdf_cells)
            sdf_mean_over_trials.append(sdf_mean_trial)

        sdf_mean_trials_simulations.append(sdf_mean_over_trials)

    sdf_mean_trials_simulations = np.array(sdf_mean_trials_simulations)
    t_sim = sdf_mean_trials_simulations.shape[2]
    sdf_concatenated = np.zeros(t_sim)

    sdf_change_bs = np.zeros(n_trials)
    sdf_change_cr = np.zeros(n_trials)

    for trial in range(n_trials):
        concatenated_time = []
        for t in range(t_sim):
            concatenated_time = sdf_mean_trials_simulations[:, trial, t].tolist()
            sdf_concatenated[t] = np.median(concatenated_time)
        sdf_change_bs[trial] = np.median(sdf_concatenated[150:200])
        sdf_change_cr[trial] = np.median(sdf_concatenated[250:300])

    sdf_change_bs = sdf_change_bs[1:] - sdf_change_bs[1]
    sdf_change_cr = sdf_change_cr[1:] - sdf_change_cr[1]

    results_path_NO = rooth_path + f"grid_search/grid_NO/{noise_rate}Hz/0/"
    spk_NO = get_spike_activity(cell_name=cell, path=results_path_NO)
    sdf_mean_over_trials_NO = []
    sdf_baseline_NO = np.zeros((n_trials))
    sdf_cr_NO = np.zeros((n_trials))

    for trial in range(n_trials):

        start = trial * between_start
        stop = CS_start_first + CS_burst_dur + trial * between_start

        sdf_cells_NO = sdf(start=start, stop=stop, spk=spk_NO, step=5)
        sdf_mean_trial_NO = sdf_mean(sdf_cells_NO)
        sdf_mean_over_trials_NO.append(sdf_mean_trial_NO)
        sdf_baseline_NO[trial] = np.mean(sdf_mean_trial_NO[150:200])
        sdf_cr_NO[trial] = np.mean(sdf_mean_trial_NO[250:300])

    sdf_change_baseline_NO = sdf_baseline_NO[1:] - sdf_baseline_NO[1]
    sdf_change_cr_NO = sdf_cr_NO[1:] - sdf_cr_NO[1]

    boxes = [sdf_change_bs[-10:],sdf_change_baseline_NO[-10:],sdf_change_cr[-10:],sdf_change_cr_NO[-10:]]
    print(f"Noise level: {noise_rate}")
    stat, p= st.wilcoxon(x=sdf_change_bs[-10:],y=sdf_change_baseline_NO[-10:])
    print(f"baselines : {p}")
    stat, p = st.wilcoxon(x=sdf_change_bs[-10:],y=sdf_change_cr[-10:])
    print(f"baseline vs cr : {p}")
    stat, p = st.wilcoxon(x=sdf_change_cr_NO[-10:],y=sdf_change_baseline_NO[-10:])
    print(f"baselines vs cr wNO : {p}")
    stat, p = st.wilcoxon(x=sdf_change_cr[-10:],y=sdf_change_cr_NO[-10:])
    print(f"cr : {p}")
    bp1 = axs[i].boxplot(boxes, patch_artist=True, medianprops=medianprops, positions=positions)

    for patch, color  in zip(bp1['boxes'],colors):
        patch.set_facecolor(color)
    axs[i].axvline(3,linewidth=1, color='black')
    axs[i].set_xticks([])
    axs[i].set_ylim(-30,0)
    axs[i].set_title(f'CS: 40Hz, BkG noise: {noise_rate}Hz')  

plt.tight_layout()
plt.show()
fig.savefig(f"sdf_boxplots_04.png")
#fig.savefig(f"sdf_boxplots_04.svg")
# %%
