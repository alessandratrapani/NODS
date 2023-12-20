# %%-------------------------------------------IMPORT-----------------------------
import json
from nods.utils import *

# **************CONFIG*****************
with open("./demo_cerebellum.json", "r") as json_file:
    net_config = json.load(json_file)
# **************PLOTS********************
import matplotlib.pyplot as plt
import seaborn as sns

with open("demo_cerebellum.json", "r") as read_file:
    net_config = json.load(read_file)
pc_color = net_config["cell_types"]["purkinje_cell"]["color"][0]
grc_color = net_config["cell_types"]["granule_cell"]["color"][0]
nos_color = "#82B366"

between_start = net_config["devices"]["CS"]["parameters"]["between_start"]
n_trials = net_config["devices"]["CS"]["parameters"]["n_trials"]
CS_start_first = float(net_config["devices"]["CS"]["parameters"]["start_first"])
CS_burst_dur = net_config["devices"]["CS"]["parameters"]["burst_dur"]
US_start_first = net_config["devices"]["US"]["parameters"]["start_first"]

#results_path = "./results/20231206_171158/" #multiple_vt
results_path = "./" #single vt


palette = list(reversed(sns.color_palette("viridis", n_trials).as_hex()))
sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=plt.Normalize(vmin=0, vmax=n_trials))
cell = "pc_spikes"
step = 50
sdf_mean_cell = []
sdf_maf_cell = []
for trial in range(n_trials):
    start = trial * between_start
    stop = CS_start_first + CS_burst_dur + trial * between_start
    spk = get_spike_activity(cell_name=cell, path=results_path)
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

# plt.xticks(np.arange(0,351,50), np.arange(50,401,50))
plt.legend()
plt.colorbar(sm, label="Trial")
plt.show()