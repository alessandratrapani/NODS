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

noise_rate = [0,4]
rooth_path = "/home/nomodel/code/NODS/results/grid_search/"
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

trial_taken = 5
n_sim = 10
delta = []
delta_NO = []
for i,noise in enumerate(noise_rate):

    #result_path = os.path.join(rooth_path,'0Hz/')
    #folder_path = result_path + f"min4_plus8/"

    sdf_mean_trials_simulations = []
    sdf_mean_trials_simulations_NO = []
    sim_baseline = np.zeros(trial_taken*n_sim)
    sim_cr = np.zeros(trial_taken*n_sim)

    sim_baseline_NO = np.zeros(trial_taken*n_sim)
    sim_cr_NO = np.zeros(trial_taken*n_sim)

    for k in range(0,n_sim):
        results_path = rooth_path + f"{noise}Hz/min4_plus8/{k}/"
        spk = get_spike_activity(cell_name=cell, path=results_path)

        results_path_NO = rooth_path + f"grid_NO/{noise}Hz/{k}/"
        spk_NO = get_spike_activity(cell_name=cell, path=results_path_NO)

        sdf_mean_over_trials = []
        sdf_mean_over_trials_NO = []
        sdf_baseline = np.zeros((n_trials))
        sdf_cr = np.zeros((n_trials))
        sdf_baseline_NO = np.zeros((n_trials))
        sdf_cr_NO = np.zeros((n_trials))
        step = 5
        for trial in range(n_trials):
            start = trial * between_start
            stop = CS_start_first + CS_burst_dur + trial * between_start

            sdf_cells = sdf(start=start, stop=stop, spk=spk, step=step)
            sdf_mean_trial = sdf_mean(sdf_cells)
            sdf_mean_over_trials.append(sdf_mean_trial)
            sdf_baseline[trial] = np.mean(sdf_mean_trial[150:200])
            sdf_cr[trial] = np.mean(sdf_mean_trial[250:300])

            sdf_cells_NO = sdf(start=start, stop=stop, spk=spk_NO, step=step)
            sdf_mean_trial_NO = sdf_mean(sdf_cells_NO)
            sdf_mean_over_trials_NO.append(sdf_mean_trial_NO)
            sdf_baseline_NO[trial] = np.mean(sdf_mean_trial_NO[150:200])
            sdf_cr_NO[trial] = np.mean(sdf_mean_trial_NO[250:300])

        sdf_change_baseline = sdf_baseline[1:] - sdf_baseline[1]
        sdf_change_cr = sdf_cr[1:] - sdf_cr[1]
        sim_baseline[k*trial_taken:(k+1)*trial_taken] = sdf_change_baseline[-trial_taken:]
        sim_cr[k*trial_taken:(k+1)*trial_taken] = sdf_change_cr[-trial_taken:]

        sdf_change_baseline_NO = sdf_baseline_NO[1:] - sdf_baseline_NO[1]
        sdf_change_cr_NO = sdf_cr_NO[1:] - sdf_cr_NO[1]
        sim_baseline_NO[k*trial_taken:(k+1)*trial_taken] = sdf_change_baseline_NO[-trial_taken:]
        sim_cr_NO[k*trial_taken:(k+1)*trial_taken] = sdf_change_cr_NO[-trial_taken:]

    delta.append(np.median(sim_baseline)-np.median(sim_cr))
    delta_NO.append(np.median(sim_baseline_NO)-np.median(sim_cr_NO))

    boxes = [sim_baseline,sim_baseline_NO,sim_cr,sim_cr_NO]
    print(f"Noise level: {noise}")
    stat, p= st.wilcoxon(x=sim_baseline,y=sim_baseline_NO)
    print(f"baselines : {p}")
    stat, p = st.wilcoxon(x=sim_baseline,y=sim_cr)
    print(f"baseline vs cr : {p}")
    stat, p = st.wilcoxon(x=sim_cr_NO,y=sim_baseline_NO)
    print(f"baselines vs cr wNO : {p}")
    stat, p = st.wilcoxon(x=sim_cr,y=sim_cr_NO)
    print(f"cr : {p}")
    bp1 = axs[i].boxplot(boxes, patch_artist=True, medianprops=medianprops, positions=positions)

    for patch, color  in zip(bp1['boxes'],colors):
        patch.set_facecolor(color)
    axs[i].axvline(3,linewidth=1, color='black')
    axs[i].set_xticks([])
    axs[i].set_ylim(-30,0)
    axs[i].set_title(f'CS: 40Hz, BkG noise: {noise}Hz')  

plt.tight_layout()
plt.show()
fig.savefig(rooth_path + f"sdf_boxplots_stat_04.png")
#fig.savefig(f"sdf_boxplots_04.svg")
# %%
