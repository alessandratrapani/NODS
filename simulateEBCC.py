import numpy as np
import json
import os
import dill
import time
import nest
import random
from nods.core import NODS
from nods.utils import *
import pickle


class SimulateEBCC:
    def __init__(self, data_path="./data/") -> None:
        self.data_path = data_path
        params_filename = "model_parameters.json"
        root_path = "./nods/"
        with open(os.path.join(root_path, params_filename), "r") as read_file:
            self.params = json.load(read_file)
        pass

    def set_network_configuration(self) -> None:
        """configure network geometry, self.connectivity, and models"""
        with open("./demo_cerebellum.json", "r") as json_file:
            self.net_config = json.load(json_file)
        hdf5_file = "cerebellum_300x_200z.hdf5"
        network_geom_file = self.data_path + "geom_" + hdf5_file
        network_connectivity_file = self.data_path + "conn_" + hdf5_file
        self.neuronal_populations = dill.load(open(network_geom_file, "rb"))
        self.connectivity = dill.load(open(network_connectivity_file, "rb"))
        self.n_trials = self.net_config["devices"]["CS"]["parameters"]["n_trials"]
        self.between_start = self.net_config["devices"]["CS"]["parameters"][
            "between_start"
        ]

    def set_nest_kernel(self) -> None:
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

    def create_network(self) -> None:
        for cell_name in list(self.neuronal_populations.keys()):
            print(f"Creating {cell_name} population")
            nest.CopyModel(
                self.net_config["cell_types"][cell_name]["neuron_model"], cell_name
            )
            nest.SetDefaults(
                cell_name, self.net_config["cell_types"][cell_name]["parameters"]
            )
            self.neuronal_populations[cell_name]["cell_ids"] = nest.Create(
                cell_name, self.neuronal_populations[cell_name]["numerosity"]
            )

    def create_vt(self, vt_modality) -> None:
        if vt_modality == "1_vt_PC":
            # create one volume transmitter for each PC
            self.num_syn = self.neuronal_populations["purkinje_cell"]["numerosity"]
            self.vt = nest.Create("volume_transmitter_alberto", self.num_syn)
        elif vt_modality == "1_vt_pf-PC":
            # create one volume transmitter for each pf-PC synapses
            self.num_syn = len(
                self.connectivity["parallel_fiber_to_purkinje"]["id_pre"]
            )
            self.vt = nest.Create("volume_transmitter_alberto", self.num_syn)
            self.connectivity["parallel_fiber_to_purkinje"]["id_vt"] = self.vt
        else:
            print("vt_modality must be either <1_vt_PC> or <1_vt_pf-PC>")
        return

    def connect_network_plastic_syn(self, vt_modality) -> None:
        connection_models = list(self.net_config["connection_models"].keys())

        for conn_model in connection_models:
            pre = self.net_config["connection_models"][conn_model]["pre"]
            post = self.net_config["connection_models"][conn_model]["post"]
            print("Connecting ", pre, " to ", post, "(", conn_model, ")")
            if conn_model == "parallel_fiber_to_purkinje":
                # Connect io and volume transmitter
                print("Connect io and volume trasmitter")
                for n, vti in enumerate(self.vt):
                    nest.SetStatus([vti], {"vt_num": n})
                # Create weight recorder
                recdict2 = {
                    "to_memory": False,
                    "to_file": True,
                    "label": "pf-PC_",
                    "senders": self.neuronal_populations[pre]["cell_ids"],
                    "targets": self.neuronal_populations[post]["cell_ids"],
                }
                WeightPFPC = nest.Create("weight_recorder", params=recdict2)

                if vt_modality == "1_vt_PC":
                    nest.SetDefaults(
                        self.net_config["connection_models"][conn_model][
                            "synapse_model"
                        ],
                        {
                            "A_minus": self.net_config["connection_models"][conn_model][
                                "parameters"
                            ]["A_minus"],
                            "A_plus": self.net_config["connection_models"][conn_model][
                                "parameters"
                            ]["A_plus"],
                            "Wmin": self.net_config["connection_models"][conn_model][
                                "parameters"
                            ]["Wmin"],
                            "Wmax": self.net_config["connection_models"][conn_model][
                                "parameters"
                            ]["Wmax"],
                            "vt": self.vt[0],
                            "weight_recorder": WeightPFPC[0],
                        },
                    )
                    syn_param = {
                        "model": self.net_config["connection_models"][conn_model][
                            "synapse_model"
                        ],
                        "weight": self.net_config["connection_models"][conn_model][
                            "weight"
                        ],
                        "delay": self.net_config["connection_models"][conn_model][
                            "delay"
                        ],
                        "receptor_type": self.net_config["cell_types"]["purkinje_cell"][
                            "receptors"
                        ]["granule_cell"],
                    }
                    ids_GrC_pre = self.connectivity[conn_model]["id_pre"]
                    ids_PC_post = self.connectivity[conn_model]["id_post"]
                    for n, id_PC in enumerate(
                        self.neuronal_populations["purkinje_cell"]["cell_ids"]
                    ):
                        syn_param["vt_num"] = float(n)
                        syn_param["meta_l"] = 1.0
                        indexes = np.where(ids_PC_post == id_PC)[0]
                        pre_neurons = np.array(ids_GrC_pre)[indexes]
                        post_neurons = np.array(ids_PC_post)[indexes]
                        nest.Connect(
                            pre_neurons, post_neurons, {"rule": "one_to_one"}, syn_param
                        )

                    # Connect io and vt
                    syn_param = {
                        "model": "static_synapse",
                        "weight": 1.0,
                        "delay": 1.0,
                    }

                    for n, id_PC in enumerate(
                        self.neuronal_populations["purkinje_cell"]["cell_ids"]
                    ):
                        io_pc = self.connectivity["io_to_purkinje"]["id_pre"][
                            np.where(
                                self.connectivity["io_to_purkinje"]["id_post"] == id_PC
                            )[0]
                        ]
                        nest.Connect(
                            [io_pc[0]], [self.vt[n]], {"rule": "one_to_one"}, syn_param
                        )

                elif vt_modality == "1_vt_pf-PC":
                    print("Set connectivity parameters for pf-PC stdp synapse model")
                    nest.SetDefaults(
                        self.net_config["connection_models"][conn_model][
                            "synapse_model"
                        ],
                        {
                            "A_minus": self.net_config["connection_models"][conn_model][
                                "parameters"
                            ]["A_minus"],
                            "A_plus": self.net_config["connection_models"][conn_model][
                                "parameters"
                            ]["A_plus"],
                            "Wmin": self.net_config["connection_models"][conn_model][
                                "parameters"
                            ]["Wmin"],
                            "Wmax": self.net_config["connection_models"][conn_model][
                                "parameters"
                            ]["Wmax"],
                            "vt": self.vt[0],
                            "weight_recorder": WeightPFPC[0],
                        },
                    )

                    syn_param = {
                        "model": self.net_config["connection_models"][conn_model][
                            "synapse_model"
                        ],
                        "weight": self.net_config["connection_models"][conn_model][
                            "weight"
                        ]
                        * np.ones(self.num_syn),
                        "delay": self.net_config["connection_models"][conn_model][
                            "delay"
                        ]
                        * np.ones(self.num_syn),
                        "receptor_type": self.net_config["cell_types"]["purkinje_cell"][
                            "receptors"
                        ]["granule_cell"],
                        "vt_num": np.arange(self.num_syn),
                        "meta_l": np.zeros((self.num_syn)),
                    }

                    granule_ids = self.connectivity[conn_model]["id_pre"]
                    purkinje_ids = self.connectivity[conn_model]["id_post"]
                    nest.Connect(
                        granule_ids,
                        purkinje_ids,
                        {"rule": "one_to_one"},
                        syn_param,
                    )
                    # t0 = time.time()
                    # print("save pf-Pc connections")
                    # pfs = nest.GetConnections(
                    #     self.neuronal_populations["granule_cell"]["cell_ids"],
                    #     self.neuronal_populations["purkinje_cell"]["cell_ids"],
                    # )
                    # with open('pfs-PC.pkl', 'wb') as file:
                    #     pickle.dump(pfs, file)
                    # t = time.time() - t0
                    # print("Time to get the pf-Pc connections: ", t)
                    # Connect io and vt
                    syn_param = {
                        "model": "static_synapse",
                        "weight": 1.0,
                        "delay": 1.0,
                    }
                    io_ids = self.connectivity["io_to_vt"]["id_pre"]
                    vt_ids = self.connectivity["io_to_vt"]["id_post"]
                    nest.Connect(io_ids, vt_ids, {"rule": "one_to_one"}, syn_param)

            elif conn_model == "mossy_to_glomerulus":
                syn_param = {
                    "model": "static_synapse",
                    "weight": self.net_config["connection_models"][conn_model][
                        "weight"
                    ],
                    "delay": self.net_config["connection_models"][conn_model]["delay"],
                }
                id_pre = self.connectivity[conn_model]["id_pre"]
                id_post = self.connectivity[conn_model]["id_post"]
                nest.Connect(
                    id_pre,
                    id_post,
                    {"rule": "one_to_one"},
                    syn_param,
                )
            else:
                syn_param = {
                    "model": "static_synapse",
                    "weight": self.net_config["connection_models"][conn_model][
                        "weight"
                    ],
                    "delay": self.net_config["connection_models"][conn_model]["delay"],
                    "receptor_type": self.net_config["cell_types"][post]["receptors"][
                        pre
                    ],
                }
                id_pre = self.connectivity[conn_model]["id_pre"]
                id_post = self.connectivity[conn_model]["id_post"]
                nest.Connect(
                    id_pre,
                    id_post,
                    {"rule": "one_to_one"},
                    syn_param,
                )

    def stimulus_geometry(self, plot) -> None:
        import plotly.graph_objects as go

        with open("demo_cerebellum.json", "r") as read_file:
            self.net_config = json.load(read_file)
        pc_color = self.net_config["cell_types"]["purkinje_cell"]["color"][0]
        grc_color = self.net_config["cell_types"]["granule_cell"]["color"][0]
        nos_color = "#82B366"

        # Stimulus geometry
        print("stimulus geometry")
        import plotly.graph_objects as go

        fig = go.Figure()

        radius = self.net_config["devices"]["CS"]["radius"]
        x = self.net_config["devices"]["CS"]["x"]
        z = self.net_config["devices"]["CS"]["z"]
        origin = np.array((x, z))

        ps = self.neuronal_populations["glomerulus"]["cell_pos"]
        in_range_mask = np.sum((ps[:, [0, 2]] - origin) ** 2, axis=1) < radius**2
        index = np.array(self.neuronal_populations["glomerulus"]["cell_ids"])
        self.id_map_glom = list(index[in_range_mask])
        glom_ids_post = self.connectivity["mossy_to_glomerulus"]["id_post"]
        mossy_ids_pre = self.connectivity["mossy_to_glomerulus"]["id_pre"]
        self.id_map_mf = mossy_ids_pre[
            np.in1d(np.array(glom_ids_post), np.unique(self.id_map_glom))
        ]
        self.id_map_mf = np.unique(self.id_map_mf)
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

            glom_ids_pre = self.connectivity["glomerulus_to_granule"]["id_pre"]
            granule_ids_post = self.connectivity["glomerulus_to_granule"]["id_post"]
            granule_ids_post = np.array(granule_ids_post)
            id_map_grc = granule_ids_post[
                np.in1d(np.array(glom_ids_pre), np.unique(self.id_map_glom))
            ]
            ps = self.neuronal_populations["granule_cell"]["cell_pos"]
            xpos = ps[:, 0]
            ypos = ps[:, 2]
            zpos = ps[:, 1]
            xpos_stim = []
            ypos_stim = []
            zpos_stim = []
            for i in id_map_grc:
                ps = self.neuronal_populations["granule_cell"]["cell_pos"][
                    self.neuronal_populations["granule_cell"]["cell_ids"] == i
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
                    marker=dict(size=2, color=grc_color),
                )
            )

            ps = self.neuronal_populations["purkinje_cell"]["cell_pos"]
            xpos = ps[:, 0]
            ypos = ps[:, 2]
            zpos = ps[:, 1]
            fig.add_trace(
                go.Scatter3d(
                    x=xpos,
                    y=ypos,
                    z=zpos,
                    mode="markers",
                    marker=dict(size=6, color=pc_color),
                )
            )
            fig.show()

    def define_CS_stimuli(self) -> None:
        print("CS stimulus")
        CS_burst_dur = self.net_config["devices"]["CS"]["parameters"]["burst_dur"]
        CS_start_first = float(
            self.net_config["devices"]["CS"]["parameters"]["start_first"]
        )
        CS_f_rate = self.net_config["devices"]["CS"]["parameters"]["rate"]
        CS_n_spikes = int(CS_f_rate * CS_burst_dur / 1000)
        n_CS_device = len(self.id_map_mf)
        t0 = CS_start_first

        tf = CS_start_first + CS_burst_dur

        CS_device = nest.Create(self.net_config["devices"]["CS"]["device"], n_CS_device)
        np.random.seed(42)
        for sg in range(n_CS_device - 1):
            random_spikes = np.random.uniform(low=t0, high=tf, size=CS_n_spikes)
            CS_matrix_start = np.round(np.sort(random_spikes))
            CS_matrix = np.concatenate(
                [CS_matrix_start + self.between_start * t for t in range(self.n_trials)]
            )
            nest.SetStatus(
                CS_device[sg : sg + 1], params={"spike_times": CS_matrix.tolist()}
            )

        nest.Connect(CS_device, self.id_map_mf.tolist(), "one_to_one")

    def define_US_stimuli(self) -> None:
        print("US stimulus")
        US_burst_dur = self.net_config["devices"]["US"]["parameters"]["burst_dur"]
        US_start_first = self.net_config["devices"]["US"]["parameters"]["start_first"]
        US_isi = 1000 / self.net_config["devices"]["US"]["parameters"]["rate"]
        US_between_start = self.net_config["devices"]["US"]["parameters"][
            "between_start"
        ]
        US_trials = self.net_config["devices"]["US"]["parameters"]["n_trials"]
        US_matrix = np.concatenate(
            [
                np.arange(
                    US_start_first, US_start_first + US_burst_dur + US_isi, US_isi
                )
                + US_between_start * t
                for t in range(US_trials)
            ]
        )
        US_device = nest.Create(
            self.net_config["devices"]["US"]["device"],
            params={"spike_times": US_matrix},
        )
        nest.Connect(
            US_device,
            self.neuronal_populations["io_cell"]["cell_ids"],
            self.net_config["devices"]["US"]["connection"],
            self.net_config["devices"]["US"]["synapse"],
        )

    def define_bg_noise(self, rate: float = None) -> None:
        print("background noise")

        noise_device = nest.Create(
            self.net_config["devices"]["background_noise"]["device"], 1
        )
        nest.Connect(
            noise_device,
            self.neuronal_populations["glomerulus"]["cell_ids"],
            "all_to_all",
        )
        rate = (
            rate or self.net_config["devices"]["background_noise"]["parameters"]["rate"]
        )
        nest.SetStatus(
            noise_device,
            params={
                "rate": rate,
                "start": self.net_config["devices"]["background_noise"]["parameters"][
                    "start"
                ],
                "stop": 1.0 * self.between_start * self.n_trials,
            },
        )

    def define_recorders(self) -> None:
        devices = list(self.net_config["devices"].keys())
        spikedetectors = {}
        for device_name in devices:
            if "record" in device_name:
                cell_name = self.net_config["devices"][device_name]["cell_types"]
                spikedetectors[cell_name] = nest.Create(
                    self.net_config["devices"][device_name]["device"],
                    params=self.net_config["devices"][device_name]["parameters"],
                )
                nest.Connect(
                    self.neuronal_populations[cell_name]["cell_ids"],
                    spikedetectors[cell_name],
                )
        self.spikedetector_granule_cell = spikedetectors["granule_cell"]

    def NO_sources_geometry(self):
        print("nNOS placement")
        pc_soma = 20.0
        nNOS_coordinates = np.zeros((len(self.vt), 3))
        i = 0
        for grc_id, pc_id in zip(
            self.connectivity["parallel_fiber_to_purkinje"]["id_pre"],
            self.connectivity["parallel_fiber_to_purkinje"]["id_post"],
        ):
            nNOS_x = self.neuronal_populations["granule_cell"]["cell_pos"][
                self.neuronal_populations["granule_cell"]["cell_ids"] == grc_id
            ][0][0]
            nNOS_y = self.neuronal_populations["purkinje_cell"]["cell_pos"][
                self.neuronal_populations["purkinje_cell"]["cell_ids"] == pc_id
            ][0][2]
            y_ml = 150
            proportion = (
                self.neuronal_populations["granule_cell"]["cell_pos"][
                    self.neuronal_populations["granule_cell"]["cell_ids"] == grc_id
                ][0][2]
                / y_ml
            ) * y_ml
            nNOS_z = (
                pc_soma
                + proportion
                + self.neuronal_populations["purkinje_cell"]["cell_pos"][
                    self.neuronal_populations["purkinje_cell"]["cell_ids"] == pc_id
                ][0][1]
            )
            nNOS_coordinates[i, 0] = nNOS_x
            nNOS_coordinates[i, 1] = nNOS_y
            nNOS_coordinates[i, 2] = nNOS_z
            i += 1
        return nNOS_coordinates

    def initialize_nods(self):
        t0 = time.time()
        simulation_file = "NO_simulation.p"
        nods_sim = NODS(self.params)

        nNOS_coordinates = self.NO_sources_geometry()
        print("Initialize nods")
        nods_sim.init_geometry(
            nNOS_coordinates=nNOS_coordinates,
            ev_point_coordinates=nNOS_coordinates,
            source_ids=self.connectivity["parallel_fiber_to_purkinje"]["id_pre"],
            nos_ids=self.vt,
            cluster_ev_point_ids=self.connectivity["parallel_fiber_to_purkinje"][
                "id_post"
            ],
            cluster_nos_ids=self.connectivity["parallel_fiber_to_purkinje"]["id_post"],
        )
        nods_sim.time = np.arange(0, self.between_start * self.n_trials, 1.0)
        nods_sim.init_simulation(
            simulation_file, store_sim=False, number_of_evaluation_points=len(self.vt)
        )  # If you want to save sim inizialization change store_sim=True
        t = time.time() - t0
        print("time {}".format(t))
        return nods_sim

    def simulate_network(self) -> None:
        print("simulate")
        print("Single trial length: ", self.between_start)
        for trial in range(self.n_trials):
            t0 = time.time()
            print("Trial ", trial, "over ", self.n_trials)
            nest.Simulate(self.between_start)
            t = time.time() - t0
            print("Time: ", t)

    def simulate_network_with_NO(self, nods_sim) -> None:
        print("simulate with NO")
        print("Single trial length: ", self.between_start)
        with open(self.data_path+"pfs-PC.pkl", "rb") as file:
            pfs = pickle.load(file)
        processed = 0

        for t in range(self.n_trials * self.between_start):
            nest.Simulate(1.0)
            time.sleep(0.01)
            ID_cell = nest.GetStatus(self.spikedetector_granule_cell, "events")[0][
                "senders"
            ]
            active_sources = ID_cell[processed:]
            processed += len(active_sources)
            nods_sim.evaluate_diffusion(active_sources, t)
            list_dict = []
            for i in range(len(pfs)):
                list_dict.append(
                    {"meta_l": float(sig(x=nods_sim.NO_in_ev_points[i], A=1, B=130))}
                )
            nest.SetStatus(pfs, list_dict)

    def plot_cell_activity_over_trials(self, cell, step):
        import matplotlib.pyplot as plt
        import seaborn as sns

        CS_burst_dur = self.net_config["devices"]["CS"]["parameters"]["burst_dur"]
        CS_start_first = float(
            self.net_config["devices"]["CS"]["parameters"]["start_first"]
        )
        US_start_first = self.net_config["devices"]["US"]["parameters"]["start_first"]

        spk = get_spike_activity(cell)

        palette = list(reversed(sns.color_palette("viridis", self.n_trials).as_hex()))
        sm = plt.cm.ScalarMappable(
            cmap="viridis_r", norm=plt.Normalize(vmin=0, vmax=self.n_trials)
        )
        sdf_mean_cell = []
        sdf_maf_cell = []
        for trial in range(self.n_trials):
            start = trial * self.between_start
            stop = CS_start_first + CS_burst_dur + trial * self.between_start
            sdf_cell = sdf(start=start, stop=stop, spk=spk, step=step)
            sdf_mean_cell.append(sdf_mean(sdf_cell))
            sdf_maf_cell.append(sdf_maf(sdf_cell))

        fig = plt.figure()
        for trial in range(self.n_trials):
            plt.plot(sdf_mean_cell[trial], palette[trial])
        plt.title(cell)
        plt.xlabel("Time [ms]")
        plt.ylabel("SDF [Hz]")
        plt.axvline(CS_start_first, label="CS start", c="grey")
        plt.axvline(US_start_first - self.between_start, label="US start", c="black")
        plt.axvline(CS_start_first + CS_burst_dur, label="CS & US end ", c="red")
        plt.legend()
        plt.colorbar(sm, label="Trial")
        fig.savefig(f"aa_sdf_{cell}.png")

    def plot_cell_raster(self, cell):
        import matplotlib.pyplot as plt
        import seaborn as sns

        CS_burst_dur = self.net_config["devices"]["CS"]["parameters"]["burst_dur"]
        CS_start_first = float(
            self.net_config["devices"]["CS"]["parameters"]["start_first"]
        )
        US_start_first = self.net_config["devices"]["US"]["parameters"]["start_first"]

        spk = get_spike_activity(cell)
        evs_cell = spk[:, 0]
        n_cells = len(np.unique(evs_cell))
        ts_cell = spk[:, 1]
        title_plot = "Raster plot " + cell
        y_min = np.min(evs_cell)
        y = [i - y_min for i in evs_cell]
        fig = plt.figure(figsize=(20, 10))
        plt.scatter(ts_cell, y, marker=".", s=3)
        plt.vlines(
            np.arange(0, self.n_trials * self.between_start, self.between_start)
            + CS_start_first,
            0,
            n_cells,
            colors="grey",
        )
        plt.vlines(
            np.arange(0, (self.n_trials - 1) * self.between_start, self.between_start)
            + US_start_first,
            0,
            n_cells,
            colors="black",
        )
        plt.vlines(
            np.arange(0, self.n_trials * self.between_start, self.between_start)
            + CS_start_first
            + CS_burst_dur,
            0,
            n_cells,
            colors="red",
        )
        plt.title(title_plot)
        plt.xticks(ticks=np.linspace(0, self.n_trials * self.between_start, 4))
        plt.yticks(ticks=np.linspace(0, n_cells, 10))
        plt.xlabel("Time [ms]")
        plt.ylabel("Neuron ID")
        fig.savefig(f"aa_raster_{cell}.png")


