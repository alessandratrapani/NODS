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

msd = 1000  # master seed
msdrange1 = range(msd, msd + CORES)
pyrngs = [np.random.RandomState(s) for s in msdrange1]
msdrange2 = range(msd + CORES + 1, msd + 1 + 2 * CORES)
nest.SetKernelStatus(
    {
        "overwrite_files": True,
        "resolution": RESOLUTION,
        "grng_seed": msd + CORES,
        "rng_seeds": msdrange2,
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
# %%-------------------------------------------CREATE NETWORK---------------------
# the order of creation of the
for cell_name in list(neuronal_populations.keys()):
    print(f"Creating {cell_name} population")
    nest.CopyModel(net_config["cell_types"][cell_name]["neuron_model"], cell_name)
    neuronal_populations[cell_name]["cell_ids"] = nest.Create(
        cell_name, neuronal_populations[cell_name]["numerosity"]
    )

multiple_vt = False
# %%-------------------------------------------MULTIPLE VT-------------
if multiple_vt:
    # create one volume transmitter for each pf-PC synapses
    num_syn = len(connectivity["parallel_fiber_to_purkinje"]["id_pre"])
    vt = nest.Create("volume_transmitter_alberto", num_syn)
    connectivity["parallel_fiber_to_purkinje"]["id_vt"] = vt

    connection_models = list(net_config["connection_models"].keys())
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
                    "A_minus": net_config["connection_models"][conn_model][
                        "parameters"
                    ]["A_minus"],
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

            granule_ids_pre = connectivity[conn_model]["id_pre"]
            purkinje_ids_post = connectivity[conn_model]["id_post"]
            nest.Connect(
                granule_ids_pre,
                purkinje_ids_post,
                {"rule": "one_to_one"},
                syn_param,
            )

            # Connect io and vt
            syn_param = {
                "model": "static_synapse",
                "weight": 1.0,
                "delay": 1.0,
            }
            io_ids_pre = connectivity["io_to_vt"]["id_pre"]
            vt_ids_post = connectivity["io_to_vt"]["id_post"]
            nest.Connect(io_ids_pre, vt_ids_post, {"rule": "one_to_one"}, syn_param)

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
        
    # check:
    for vt_num, vt_id in enumerate(vt):
        # assert that vt_id is the same as listed in connectivity["parallel_fiber_to_purkinje"]
        assert vt_id == connectivity["parallel_fiber_to_purkinje"]["id_vt"][vt_num]
        # extract the io connected to the current vt_id 
        id_io_pre_cf = io_ids_pre[
            np.where(connectivity["io_to_vt"]["id_post"] == vt_id)[0]
        ]
        # extract the pc connected to the io connected to the current vt_id 
        id_pc_post_cf = connectivity["io_to_purkinje"]["id_post"][
            np.where(connectivity["io_to_purkinje"]["id_pre"] == id_io_pre_cf)[0]
        ]
        # assert that the current pc_id is the same as listed in connectivity["parallel_fiber_to_purkinje"]["id_post"]
        assert connectivity["parallel_fiber_to_purkinje"]["id_post"][vt_num] in id_pc_post_cf
# %%-------------------------------------------SINGLE VT-------------
if not multiple_vt:
    print("One vt per PC")
    num_syn = neuronal_populations["purkinje_cell"]["numerosity"]
    vt = nest.Create("volume_transmitter_alberto", num_syn)

    connection_models = list(net_config["connection_models"].keys())
    meta_l_set = 1.1

    for conn_model in connection_models:
        pre = net_config["connection_models"][conn_model]["pre"]
        post = net_config["connection_models"][conn_model]["post"]
        print("Connecting ", pre, " to ", post, "(", conn_model, ")")
        if conn_model == "parallel_fiber_to_purkinje":
            t0 = time.time()
            # Connect io and volume transmitter
            print("Connect io and volume trasmitter")

            for n, vti in enumerate(vt):
                nest.SetStatus([vti], {"vt_num": n})

            t = time.time() - t0
            print("volume transmitter created in: ", t, " sec")

            # Create weight recorder
            recdict2 = {
                "to_memory": False,
                "to_file": True,
                "label": "pf-PC_",
                "senders": neuronal_populations[pre]["cell_ids"],
                "targets": neuronal_populations[post]["cell_ids"],
            }
            WeightPFPC = nest.Create("weight_recorder", params=recdict2)

            # Connect pf and PC
            print("Set connectivity parameters for pf-PC stdp synapse model")
            nest.SetDefaults(
                net_config["connection_models"][conn_model]["synapse_model"],
                {
                    "A_minus": net_config["connection_models"][conn_model][
                        "parameters"
                    ]["A_minus"],
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
                "weight": net_config["connection_models"][conn_model]["weight"],
                "delay": net_config["connection_models"][conn_model]["delay"],
                "receptor_type": net_config["cell_types"]["purkinje_cell"]["receptors"][
                    "granule_cell"
                ],
            }
            ids_GrC_pre = connectivity[conn_model]["id_pre"]
            ids_PC_post = connectivity[conn_model]["id_post"]
            for n, id_PC in enumerate(neuronal_populations["purkinje_cell"]["cell_ids"]):
                syn_param["vt_num"] = float(n)
                syn_param["meta_l"] = meta_l_set
                indexes = np.where(ids_PC_post == id_PC)[0]
                pre_neurons = np.array(ids_GrC_pre)[indexes]
                post_neurons = np.array(ids_PC_post)[indexes]
                nest.Connect(pre_neurons,post_neurons, {"rule": "one_to_one"}, syn_param)

            # Connect io and vt
            syn_param = {
                "model": "static_synapse",
                "weight": 1.0,
                "delay": 1.0,
            }
            for n, id_PC in enumerate(
                neuronal_populations["purkinje_cell"]["cell_ids"]
            ):
                io_pc = connectivity["io_to_purkinje"]["id_pre"][
                    np.where(connectivity["io_to_purkinje"]["id_post"] == id_PC)[0]
                ]
                nest.Connect([io_pc[0]], [vt[n]], {"rule": "one_to_one"}, syn_param)
            cfs = nest.GetConnections(neuronal_populations["io_cell"]["cell_ids"])

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
   
    # check:    
    for n, syn in enumerate(cfs):
        id_io_pre_cf = syn[0]
        id_vt = syn[1]
        if id_vt in vt:
            # extract the pc connected to the io connected to the current vt_id 
            id_pc_post_cf = connectivity["io_to_purkinje"]["id_post"][
                np.where(connectivity["io_to_purkinje"]["id_pre"] == id_io_pre_cf)[0]
            ]
            
            vt_num = np.where(np.array(vt)==id_vt)[0]
            # assert that the current pc_id is the same as listed in neuronal_populations["purkinje_cell"]["cell_ids"]
            assert neuronal_populations["purkinje_cell"]["cell_ids"][vt_num[0]] in id_pc_post_cf
# %%
