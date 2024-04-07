import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

rooth_path = "/media/amtra/Samsung_T5/results/"
with open("network_configuration.json", "r") as json_file:
    net_config = json.load(json_file)

with_NO_color = net_config["devices"]["nNOS"]["color"][0]
without_NO_color = "#000000"

noise_rates = [0, 4, 8]
grouped_sums = []
group_labels = []
for i, noise_rate in enumerate(noise_rates):
    results_path = rooth_path+f"only_background/BG_{noise_rate}Hz.csv"
    results_path_NO = rooth_path+f"only_background/BG_{noise_rate}Hz_NO.csv"   
    # results_path = f"/media/amtra/Samsung_T5/RISULTATI_TESI/only_background/BG_8Hz.csv"
    # results_path_NO = f"/media/amtra/Samsung_T5/RISULTATI_TESI/only_background/BG_8Hz.csv"  
    df = pd.read_csv(results_path)
    df_NO = pd.read_csv(results_path_NO)
    array = df.iloc[:, 0].values 
    array_NO = df_NO.iloc[:, 0].values 
    # grouped_sums.append((array.sum(), array_NO.sum()))
    grouped_sums.append((np.mean(array), np.mean(array_NO)))
    # grouped_sums.append((np.count_nonzero(array), np.count_nonzero(array_NO)))
    group_labels.append(f"BkG at {noise_rate}Hz")
print(grouped_sums)

# data_arrays = [np.random.rand(100) for _ in range(6)]
# sums = [data_arrays[i].sum() for i in range(6)]
# grouped_sums = [(sums[0], sums[1]), (sums[2], sums[3]), (sums[4], sums[5])]
# group_labels = ['BkG at 0Hz', 'BkG at 4Hz', 'BkG at 8Hz']

x = np.arange(len(group_labels))
width = 0.25

fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, [group[0] for group in grouped_sums], width, label='Without NO', color=without_NO_color)
bars2 = ax.bar(x + width/2, [group[1] for group in grouped_sums], width, label='With NO', color=with_NO_color)

ax.set_ylabel('Mean pf-PC updates')
# ax.set_title('pf-PC synapses updates with only background noise')
ax.set_xticks(x)
ax.set_xticklabels(group_labels)

plt.tight_layout()
plt.show()
fig.savefig(rooth_path+"figures/bg_048_weight_changes_mean.png")