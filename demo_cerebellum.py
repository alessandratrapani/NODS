# %%-------------------------------------------IMPORT-----------------------------
import numpy as np
import json
import os
import dill
import time

# **************CONFIG*****************
with open("./demo_cerebellum.json", "r") as json_file:
    net_config = json.load(json_file)
data_path = "./data/"
hdf5_file = "cerebellum_330x_200z.hdf5"
network_geom_file = data_path + "geom_" + hdf5_file
network_connectivity_file = data_path + "conn_" + hdf5_file
neuronal_populations = dill.load(open(network_geom_file, "rb"))
connectivity = dill.load(open(network_connectivity_file, "rb"))

# **************NEST********************
import nest

nest.Install("cerebmodule")
RESOLUTION = 1.0
CORES = 24
nest.ResetKernel()

msd = 1000 # master seed
msdrange1 = range(msd, msd+CORES )
pyrngs = [np.random.RandomState(s) for s in msdrange1]
msdrange2=range(msd+CORES+1, msd+1+2*CORES)
nest.SetKernelStatus(
    {
        "overwrite_files": True,
        "resolution": RESOLUTION,
        'grng_seed': msd+CORES,
        'rng_seeds': msdrange2,
        "local_num_threads": CORES,
        "total_num_virtual_procs": CORES,
    }
)
nest.set_verbosity("M_ERROR")  # reduce plotted info

# **************NODS********************
from nods.core import NODS
from nods.utils import *
from nods.plot import plot_cell_activity

params_filename = "./nods/model_parameters.json"
with open(os.path.join(params_filename), "r") as read_file_param:
    params = json.load(read_file_param)

# **************PLOTS********************
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import seaborn as sns

with open("demo_cerebellum.json", "r") as read_file:
    net_config = json.load(read_file)
pc_color = net_config["cell_types"]["purkinje_cell"]["color"][0]
grc_color = net_config["cell_types"]["granule_cell"]["color"][0]
nos_color = "#82B366"
# %%**************SIMULATION DESCRIPTION*****
description = "one_vt_per_pf-PC_syn"
# %%**************NO DEPENDENCY**************
NO_dependency = False
plot = False
# %%-------------------------------------------CREATE NETWORK---------------------
for cell_name in list(neuronal_populations.keys()):
    print(f"Creating {cell_name} population")
    nest.CopyModel(net_config["cell_types"][cell_name]["neuron_model"], cell_name)
    neuronal_populations[cell_name]["cell_ids"] = nest.Create(
        cell_name, neuronal_populations[cell_name]["numerosity"]
    )

# create one volume transmitter for each pf-PC synapses
num_syn = len(connectivity["parallel_fiber_to_purkinje"]["id_pre"])
vt = nest.Create("volume_transmitter_alberto", num_syn)
connectivity["parallel_fiber_to_purkinje"]["id_vt"] = vt

# %% if "io_to_vt" is not already in connectivity, it creates io_to_vt dict
def create_io_to_vt_dict(connectivity, network_connectivity_file):
    ids_PC_post = connectivity["parallel_fiber_to_purkinje"]["id_post"]

    vt_parameters = []
    num_syn = len(ids_PC_post)
    for n in range(num_syn):
        vt_parameters.append({"vt_num": n})
    connectivity["parallel_fiber_to_purkinje"]["vt_num"] = vt_parameters

    connectivity["io_to_vt"] = {
        "id_pre": np.zeros((num_syn)),
        "id_post": np.array(connectivity["parallel_fiber_to_purkinje"]["id_vt"]),
    }
    for n, id_pc in enumerate(ids_PC_post):
        io_pc = connectivity["io_to_purkinje"]["id_pre"][
            np.where(connectivity["io_to_purkinje"]["id_post"] == id_pc)[0]
        ]
        connectivity["io_to_vt"]["id_pre"][n] = io_pc[0]
    connectivity["io_to_vt"]["id_pre"] = connectivity["io_to_vt"]["id_pre"].astype(int)
    dill.dump(connectivity, open(network_connectivity_file, "wb"))
    print("create io_to_vt dict")


if "io_to_vt" not in connectivity.keys():
    create_io_to_vt_dict(connectivity, network_connectivity_file)
# %%-------------------------------------------CONNECT NETWORK---------------------
connection_models = list(net_config["connection_models"].keys())
if NO_dependency:
    meta_l_set = np.zeros((num_syn))
else:
    meta_l_set = np.ones((num_syn))

for conn_model in connection_models:
    pre = net_config["connection_models"][conn_model]["pre"]
    post = net_config["connection_models"][conn_model]["post"]
    print("Connecting ", pre, " to ", post, "(", conn_model, ")")
    if conn_model == "parallel_fiber_to_purkinje":
        t0 = time.time()
        # Connect io and volume transmitter
        print("Connect io and volume trasmitter")

        nest.SetStatus(
            vt, connectivity[conn_model]["vt_num"]
        )  # TODO check that this is correct

        t = time.time() - t0
        print("volume transmitter created in: ", t, " sec")

        # Create weight recorder
        recdict2 = {
            "to_memory": False,
            "to_file": True,
            "label": "pf-PC_",
            "senders": neuronal_populations["granule_cell"]["cell_ids"],
            "targets": neuronal_populations["purkinje_cell"]["cell_ids"],
        }
        WeightPFPC = nest.Create("weight_recorder", params=recdict2)

        # Connect pf and PC
        print("Set connectivity parameters for pf-PC stdp synapse model")
        nest.SetDefaults(
            net_config["connection_models"][conn_model]["synapse_model"],
            {
                "A_minus": net_config["connection_models"][conn_model]["parameters"][
                    "A_minus"
                ],
                "A_plus": net_config["connection_models"][conn_model]["parameters"][
                    "A_plus"
                ],
                "Wmin": net_config["connection_models"][conn_model]["parameters"][
                    "Wmin"
                ],
                "Wmax": net_config["connection_models"][conn_model]["parameters"][
                    "Wmax"
                ],
                "vt": vt[0],
                "weight_recorder": WeightPFPC[0],
            },
        )
        syn_param = {
            "model": net_config["connection_models"][conn_model]["synapse_model"],
            "weight": net_config["connection_models"][conn_model]["weight"]
            * np.ones(num_syn),
            "delay": net_config["connection_models"][conn_model]["delay"]
            * np.ones(num_syn),
            "receptor_type": net_config["cell_types"]["purkinje_cell"]["receptors"][
                "granule_cell"
            ],
            "vt_num": np.arange(num_syn),
            "meta_l": meta_l_set,
        }

        granule_ids = connectivity[conn_model]["id_pre"]
        purkinje_ids = connectivity[conn_model]["id_post"]
        nest.Connect(
            granule_ids,
            purkinje_ids,
            {"rule": "one_to_one"},
            syn_param,
        )

        # Connect io and vt
        syn_param = {
            "model": "static_synapse",
            "weight": 1.0,
            "delay": 1.0,
        }
        io_ids = connectivity["io_to_vt"]["id_pre"]
        vt_ids = connectivity["io_to_vt"]["id_post"]
        nest.Connect(io_ids, vt_ids, {"rule": "one_to_one"}, syn_param)

        # pfs = nest.GetConnections(neuronal_populations["granule_cell"]["cell_ids"],neuronal_populations["purkinje_cell"]["cell_ids"])

    else:
        syn_param = {
            "model": "static_synapse",
            "weight": net_config["connection_models"][conn_model]["weight"],
            "delay": net_config["connection_models"][conn_model]["delay"],
            "receptor_type": net_config["cell_types"][post]["receptors"][pre],
        }
        id_pre = connectivity[conn_model]["id_pre"]
        id_post = connectivity[conn_model]["id_post"]
        nest.Connect(
            id_pre,
            id_post,
            {"rule": "one_to_one"},
            syn_param,
        )
"""
init_weight = [[] for i in range(len(neuronal_populations['purkinje_cell']['cell_ids']))]
for i,PC_id in enumerate(neuronal_populations['purkinje_cell']['cell_ids']):
    for j in range(len(pfs)):
        if (pfs[j][1]==PC_id):
            init_weight[i].append(nest.GetStatus([pfs[j]], {'weight'})[0][0])
init_weight = np.array(init_weight)
#"""
# %%-------------------------------------------STIMULUS GEOMETRY---------------------
# Stimulus geometry
print("stimulus geometry")
import plotly.graph_objects as go

fig = go.Figure()

radius = net_config["devices"]["CS"]["radius"]
x = net_config["devices"]["CS"]["x"]
z = net_config["devices"]["CS"]["z"]
origin = np.array((x, z))

ps = neuronal_populations["glomerulus"]["cell_pos"]
in_range_mask = np.sum((ps[:, [0, 2]] - origin) ** 2, axis=1) < radius**2
index = np.array(neuronal_populations["glomerulus"]["cell_ids"])
id_map_glom = list(index[in_range_mask])

# %%-------------------------------------------PLOT STIMULUS GEOMETRY---------------------
if plot:
    # Plot stimulus geometry
    xpos = ps[:, 0]
    ypos = ps[:, 2]
    zpos = ps[:, 1]
    xpos_stim = ps[in_range_mask, 0]
    ypos_stim = ps[in_range_mask, 2]
    zpos_stim = ps[in_range_mask, 1]
    fig.add_trace(
        go.Scatter3d(
            x=xpos_stim,
            y=ypos_stim,
            z=zpos_stim,
            mode="markers",
            marker=dict(size=4, color="yellow"),
        )
    )

    glom_ids_post = connectivity["glomerulus_to_granule"]["id_pre"]
    granule_ids_pre = connectivity["glomerulus_to_granule"]["id_post"]
    granule_ids_pre = np.array(granule_ids_pre)
    id_map_grc = granule_ids_pre[
        np.in1d(np.array(glom_ids_post), np.unique(id_map_glom))
    ]
    ps = neuronal_populations["granule_cell"]["cell_pos"]
    xpos = ps[:, 0]
    ypos = ps[:, 2]
    zpos = ps[:, 1]
    xpos_stim = []
    ypos_stim = []
    zpos_stim = []
    for i in id_map_grc:
        ps = neuronal_populations["granule_cell"]["cell_pos"][
            neuronal_populations["granule_cell"]["cell_ids"] == i
        ]
        xpos_stim.append(ps[0, 0])
        ypos_stim.append(ps[0, 2])
        zpos_stim.append(ps[0, 1])
    fig.add_trace(
        go.Scatter3d(
            x=xpos_stim,
            y=ypos_stim,
            z=zpos_stim,
            mode="markers",
            marker=dict(size=2, color="red"),
        )
    )

    ps = neuronal_populations["purkinje_cell"]["cell_pos"]
    xpos = ps[:, 0]
    ypos = ps[:, 2]
    zpos = ps[:, 1]
    fig.add_trace(
        go.Scatter3d(
            x=xpos, y=ypos, z=zpos, mode="markers", marker=dict(size=6, color="black")
        )
    )
    fig.show()
# %%-------------------------------------------DEFINE CS STIMULI---------------------
print("CS stimulus")
CS_burst_dur = net_config["devices"]["CS"]["parameters"]["burst_dur"]
CS_start_first = float(net_config["devices"]["CS"]["parameters"]["start_first"])
CS_f_rate = net_config["devices"]["CS"]["parameters"]["rate"]
CS_n_spikes = int(
    net_config["devices"]["CS"]["parameters"]["rate"] * CS_burst_dur / 1000
)
between_start = net_config["devices"]["CS"]["parameters"]["between_start"]
n_trials = net_config["devices"]["CS"]["parameters"]["n_trials"]
CS_isi = int(CS_burst_dur / CS_n_spikes)

CS_matrix_start_pre = np.round((np.linspace(100.0, 228.0, 11)))
CS_matrix_start_post = np.round((np.linspace(240.0, 368.0, 11)))
CS_matrix_first_pre = np.concatenate(
    [CS_matrix_start_pre + between_start * t for t in range(n_trials)]
)
CS_matrix_first_post = np.concatenate(
    [CS_matrix_start_post + between_start * t for t in range(n_trials)]
)

CS_matrix = []
for i in range(int(len(id_map_glom) / 2)):
    CS_matrix.append(CS_matrix_first_pre + i)
    CS_matrix.append(CS_matrix_first_post + i)

CS_device = nest.Create(net_config["devices"]["CS"]["device"], len(id_map_glom))

for sg in range(len(CS_device) - 1):
    nest.SetStatus(
        CS_device[sg : sg + 1], params={"spike_times": CS_matrix[sg].tolist()}
    )
nest.Connect(CS_device, id_map_glom, "all_to_all")

# %%-------------------------------------------DEFINE US STIMULUS --------------------
print("US stimulus")
US_burst_dur = net_config["devices"]["US"]["parameters"]["burst_dur"]
US_start_first = net_config["devices"]["US"]["parameters"]["start_first"]
US_isi = 1000 / net_config["devices"]["US"]["parameters"]["rate"]
between_start = net_config["devices"]["US"]["parameters"]["between_start"]
n_trials = net_config["devices"]["US"]["parameters"]["n_trials"]
US_matrix = np.concatenate(
    [
        np.arange(US_start_first, US_start_first + US_burst_dur + US_isi, US_isi)
        + between_start * t
        for t in range(n_trials)
    ]
)
US_device = nest.Create(
    net_config["devices"]["US"]["device"], params={"spike_times": US_matrix}
)
conn_param = {"model": "static_synapse", "weight": 1, "delay": 1, "receptor_type": 1}
nest.Connect(
    US_device,
    neuronal_populations["io_cell"]["cell_ids"],
    {"rule": "all_to_all"},
    conn_param,
)

# %%-------------------------------------------DEFINE NOISE---------------------
print("background noise")
noise_device = nest.Create(net_config["devices"]["background_noise"]["device"], 1)
nest.Connect(noise_device, neuronal_populations["glomerulus"]["cell_ids"], "all_to_all")
nest.SetStatus(
    noise_device,
    params={
        "rate": net_config["devices"]["background_noise"]["parameters"]["rate"],
        "start": net_config["devices"]["background_noise"]["parameters"]["start"],
        "stop": 1.0 * between_start * n_trials,
    },
)

# %%-------------------------------------------DEFINE RECORDERS---------------------
devices = list(net_config["devices"].keys())
spikedetectors = {}
for device_name in devices:
    if "record" in device_name:
        cell_name = net_config["devices"][device_name]["cell_types"]
        spikedetectors[cell_name] = nest.Create(
            net_config["devices"][device_name]["device"],
            params=net_config["devices"][device_name]["parameters"],
        )
        nest.Connect(
            neuronal_populations[cell_name]["cell_ids"], spikedetectors[cell_name]
        )

# %%-------------------------------------------INITIALIZE NODS---------------------
"""Initialize NODS"""
if NO_dependency:
    t0 = time.time()
    init_new_sim = True
    simulation_file = "NO_simulation"
    sim = NODS(params)
    if init_new_sim:
        sim.init_geometry(
            nNOS_coordinates=nNOS_coordinates,
            ev_point_coordinates=ev_point_coordinates,
            source_ids=source_ids,
            ev_point_ids=ev_point_ids,
            nos_ids=nos_ids,
            cluster_ev_point_ids=cluster_ev_point_ids,
            cluster_nos_ids=cluster_nos_ids,
        )
    else:
        sim = sim.load_simulation(simulation_file=simulation_file)
    sim.time = sim_time_steps
    sim.init_simulation(
        simulation_file, store_sim=True
    )  # If you want to save sim inizialization change store_sim=True
    t = time.time() - t0
    print("time {}".format(t))
# %%-------------------------------------------SIMULATE NETWORK---------------------
# Simulate Network
print("simulate")
print("Single trial length: ", between_start)
for trial in range(n_trials + 1):
    t0 = time.time()
    print("Trial ", trial + 1, "over ", n_trials)
    nest.Simulate(between_start)
    t = time.time() - t0
    print("Time: ", t)

# %%-------------------------------------------PLOT PC SDF MEAN OVER TRIALS-------
"""Plot PC sdf mean"""
palette = list(reversed(sns.color_palette("viridis", n_trials).as_hex()))
sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=plt.Normalize(vmin=0, vmax=n_trials))
cell = "pc_spikes"
step = 50
sdf_mean_cell = []
sdf_maf_cell = []
for trial in range(n_trials):
    start = trial * between_start
    stop = CS_start_first + CS_burst_dur + trial * between_start
    spk = get_spike_activity(cell)
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
fig.savefig(description+'.png')
# %%-------------------------------------------SAVE SIMULATION DESCRIPTION--------
from datetime import datetime
# Generate datetime string for the README
current_datetime = datetime.now().strftime("%Y-%m-%d")

# Define the README content
readme_content = f"""# Simulation Parameters

                Date: {current_datetime}

                ## Parameters
                - n_trials: {net_config["devices"]["CS"]["parameters"]["n_trials"]}
                - CS_rate: {net_config["devices"]["CS"]["parameters"]["rate"]}
                - US_rate: {net_config["devices"]["US"]["parameters"]["rate"]}
                - noise_rate:{net_config["devices"]["background_noise"]["parameters"]["rate"]}
                - A_minus: {net_config["connection_models"]["parallel_fiber_to_purkinje"]["parameters"]["A_minus"]}
                - A_plus: {net_config["connection_models"]["parallel_fiber_to_purkinje"]["parameters"]["A_plus"]}
                - Wmin: {net_config["connection_models"]["parallel_fiber_to_purkinje"]["parameters"]["Wmin"]}
                - Wmax: {net_config["connection_models"]["parallel_fiber_to_purkinje"]["parameters"]["Wmax"]}

                ## Description
                {description}
                """

# Write the README content to a file
with open('./sim_description.md', "w") as readme_file:
    readme_file.write(readme_content)