# %%
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import get_spike_activity, sdf, sdf_mean
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

A = "A_minus"
variation_in_percentage = ["0", "+10", "-10"]

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


fig, ax = plt.subplots(figsize=(5, 4))
colors = [[without_NO_color, with_NO_color],["blue","blue"],["red","red"]]#, without_NO_color, with_NO_color]
medianprops = dict(linewidth=1.5, color="white")
positions = [[1, 3],[1.5, 3.5],[2, 4]]

for i, variation in enumerate(variation_in_percentage):
    if variation == "0":
        results_path = f"/media/amtra/Samsung_T5/EBCC_4Hz/"
        results_path_NO = f"/media/amtra/Samsung_T5/EBCC_NO_4Hz/"
    else:
        results_path = f"/media/amtra/Samsung_T5/{A}{variation}/"
        results_path_NO = f"/media/amtra/Samsung_T5/{A}{variation}_NO/"
        
    spk = get_spike_activity(cell_name=cell, path=results_path)
    sdf_mean_over_trials = []
    # sdf_baseline = np.zeros((n_trials))
    sdf_cr = np.zeros((n_trials))

    spk_NO = get_spike_activity(cell_name=cell, path=results_path_NO)
    sdf_mean_over_trials_NO = []
    # sdf_baseline_NO = np.zeros((n_trials))
    sdf_cr_NO = np.zeros((n_trials))

    for trial in range(n_trials):

        start = trial * between_start
        stop = CS_start_first + CS_burst_dur + trial * between_start

        sdf_cells = sdf(start=start, stop=stop, spk=spk, step=5)
        sdf_mean_trial = sdf_mean(sdf_cells)
        sdf_mean_over_trials.append(sdf_mean_trial)
        # sdf_baseline[trial] = np.mean(sdf_mean_trial[100:150])
        sdf_cr[trial] = np.mean(sdf_mean_trial[250:300])

        sdf_cells_NO = sdf(start=start, stop=stop, spk=spk_NO, step=5)
        sdf_mean_trial_NO = sdf_mean(sdf_cells_NO)
        sdf_mean_over_trials_NO.append(sdf_mean_trial_NO)
        # sdf_baseline_NO[trial] = np.mean(sdf_mean_trial_NO[100:150])
        sdf_cr_NO[trial] = np.mean(sdf_mean_trial_NO[250:300])

    # sdf_change_baseline = sdf_baseline[1:] - sdf_baseline[1]
    sdf_change_cr = sdf_cr[1:] - sdf_cr[1]
    # sdf_change_baseline_NO = sdf_baseline_NO[1:] - sdf_baseline_NO[1]
    sdf_change_cr_NO = sdf_cr_NO[1:] - sdf_cr_NO[1]

    boxes = [
        # sdf_change_baseline[-10:],
        # sdf_change_baseline_NO[-10:],
        sdf_change_cr[-10:],
        sdf_change_cr_NO[-10:],
    ]

    bp1 = ax.boxplot(
        boxes, patch_artist=True, medianprops=medianprops, positions=positions[i]
    )
    for patch, color in zip(bp1["boxes"], colors[i]):
        patch.set_facecolor(color)

ax.axvline(2.5, linewidth=1, color="black")
ax.set_xticks([])
ax.set_ylim(-30, 0)

fig.suptitle(f"{A} variation", fontsize=16)
plt.tight_layout()
plt.show()
fig.savefig(f"sdf_boxplots_{A}.png")
fig.savefig(f"sdf_boxplots_{A}.svg")
