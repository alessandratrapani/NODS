# %%
import numpy as np

# %%
import pandas as pd
import math as m
import json
import os
import sys
import random
import time
import nest

nest.Install("cerebmodule")
from nods.core import NODS
from nods.utils import *
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from move_files import move_files_to_folder


class Simulation:
    def __init__(self, network_configuration, NO_model_parameters) -> None:
        self.net_config = network_configuration
        self.NO_model_parameters = NO_model_parameters

    def create_and_simulate_network(self, NO_dependency):
        nest.CopyModel("iaf_cond_exp", "granular_neuron")
        nest.CopyModel("iaf_cond_exp", "purkinje_neuron")
        nest.CopyModel("iaf_cond_exp", "olivary_neuron")
        nest.SetDefaults(
            "granular_neuron",
            {
                "t_ref": 1.0,
                "C_m": 2.0,
                "V_th": -40.0,
                "V_reset": -70.0,
                "g_L": 0.2,
                "tau_syn_ex": 0.5,
                "tau_syn_in": 10.0,
            },
        )
        nest.SetDefaults(
            "purkinje_neuron",
            {
                "t_ref": 2.0,
                "C_m": 400.0,
                "V_th": -52.0,
                "V_reset": -70.0,
                "g_L": 16.0,
                "tau_syn_ex": 0.5,
                "tau_syn_in": 1.6,
            },
        )
        nest.SetDefaults(
            "olivary_neuron",
            {
                "t_ref": 1.0,
                "C_m": 2.0,
                "V_th": -40.0,
                "V_reset": -70.0,
                "g_L": 0.2,
                "tau_syn_ex": 0.5,
                "tau_syn_in": 10.0,
            },
        )
        MF_num = self.net_config["cell_num"]["MF_num"]
        GR_num = self.net_config["cell_num"]["GR_num"]
        PC_num = self.net_config["cell_num"]["PC_num"]
        IO_num = self.net_config["cell_num"]["IO_num"]
        MF = nest.Create("parrot_neuron", MF_num)
        GR = nest.Create("granular_neuron", GR_num)
        num_subpop = self.net_config["geometry"]["num_subpop"]
        MF_subpop = [[] for i in range(num_subpop)]
        for n in range(num_subpop):
            for i in range(
                int(MF_num / num_subpop) * n, int(MF_num / num_subpop) * (n + 1)
            ):
                MF_subpop[n].append(MF[i])
        GR_subpop = [[] for i in range(num_subpop)]

        PC = nest.Create("purkinje_neuron", PC_num)
        IO = nest.Create("olivary_neuron", IO_num)

        Init_PFPC = {"distribution": "uniform", "low": 1.0, "high": 3.0}
        recdict2 = {
            "to_memory": False,
            "to_file": True,
            "label": "PFPC_",
            "senders": GR,
            "targets": PC,
        }

        # -----pf-PC connection------
        WeightPFPC = nest.Create("weight_recorder", params=recdict2)
        vt = nest.Create("volume_transmitter_alberto", int(PC_num * (GR_num * 0.8)))
        for n, vti in enumerate(vt):
            nest.SetStatus([vti], {"vt_num": n})
        nest.SetDefaults(
            "stdp_synapse_sinexp",
            {
                "A_minus": self.net_config["connections"]["A_minus"],
                "A_plus": self.net_config["connections"]["A_plus"],
                "Wmin": self.net_config["connections"]["Wmin"],
                "Wmax": self.net_config["connections"]["Wmax"],
                "vt": vt[0],
                "weight_recorder": WeightPFPC[0],
            },
        )
        PFPC_conn_param = {
            "model": "stdp_synapse_sinexp",
            "weight": Init_PFPC,
            "delay": 1.0,
        }
        vt_num = 0
        PC_vt_dict = {}
        source_ids = []
        PC_ids = []
        nos_ids = []
        for i, PCi in enumerate(PC):
            nest.Connect(
                GR,
                [PCi],
                {
                    "rule": "fixed_indegree",
                    "indegree": int(0.8 * GR_num),
                    "multapses": False,
                },
                PFPC_conn_param,
            )
            C = nest.GetConnections(GR, [PCi])
            for n in range(len(C)):
                nest.SetStatus([C[n]], {"vt_num": float(vt_num)})
                if not NO_dependency:
                    nest.SetStatus([C[n]], {"meta_l": float(1.0)})
                source_ids.append(C[n][0])
                PC_ids.append(C[n][1])
                nos_ids.append(int(vt_num))
                vt_num += 1
            PC_vt_dict[PCi] = np.array(nest.GetStatus(C, {"vt_num"}), dtype=int).T[0]

            # -----cf-PC connection------
            vt_tmp = [vt[n] for n in PC_vt_dict[PCi]]
            nest.Connect(
                [IO[i]],
                vt_tmp,
                {"rule": "all_to_all"},
                {"model": "static_synapse", "weight": 1.0, "delay": 1.0},
            )

        pfs = nest.GetConnections(GR, PC)

        init_weight = [[] for i in range(len(PC))]
        for i, PC_id in enumerate(PC):
            for j in range(len(pfs)):
                if pfs[j][1] == PC_id:
                    init_weight[i].append(nest.GetStatus([pfs[j]], {"weight"})[0][0])
        init_weight = np.array(init_weight)

        spikedetector_PC = nest.Create(
            "spike_detector",
            params={
                "withgid": True,
                "withtime": True,
                "to_file": True,
                "label": "purkinje",
            },
        )
        spikedetector_GR = nest.Create(
            "spike_detector",
            params={
                "withgid": True,
                "withtime": True,
                "to_file": True,
                "label": "granule",
            },
        )

        nest.Connect(PC, spikedetector_PC)
        nest.Connect(GR, spikedetector_GR)

        ev_point_ids = nos_ids  # [pfs[i][4] for i in range(len(pfs))]
        cluster_ev_point_ids = PC_ids  # [pfs[i][1] for i in range(len(pfs))]
        cluster_nos_ids = PC_ids  # [pfs[i][1] for i in range(len(pfs))]

        # def geometry constants
        y_ml = self.net_config["geometry"]["y_ml"]  # height of molecular layer
        y_gl = self.net_config["geometry"]["y_gl"]  # height of granular layer
        granule_density = self.net_config["geometry"]["granule_density"]
        nos_n = int(
            len(pfs) / len(PC)
        )  # calculate number of PF-PC synapses for each PC
        nos_density = self.net_config["geometry"]["nos_density"]
        A = nos_n / nos_density  # Compute area given the number of synapses (nos_n)
        x_scaffold = A / y_ml  # width of the scaffold (PC tree width)
        z_scaffold = (len(GR) / granule_density) / (
            x_scaffold * y_gl
        )  # thickness of the scaffold
        dim_ml = np.array([x_scaffold, y_ml, z_scaffold])
        dim_gl = np.array([x_scaffold, y_gl, z_scaffold])
        # placing Granule Cells
        GR_coord = pd.DataFrame(
            {
                "GR_id": GR,
                "GR_x": np.random.uniform(0, x_scaffold, len(GR)),
                "GR_y": np.random.uniform(0, y_gl, len(GR)),
                "GR_z": np.random.uniform(0, z_scaffold, len(GR)),
            }
        )
        # placing Purkinje Cells
        PC_dist = self.net_config["geometry"]["PC_dist"]
        PC_coord = pd.DataFrame(
            {
                "PC_id": PC,
                "PC_x": np.random.normal(x_scaffold / 2, x_scaffold / 4, len(PC)),
                "PC_y": y_gl + PC_dist / 2,
                "PC_z": np.linspace(0, PC_dist * (len(PC) - 1), len(PC)),
            }
        )
        x_nos_variation = 1
        z_nos_variation = 1
        y_nos_variation = 1
        pc_soma = 20.0
        nNOS_coordinates = np.zeros((len(pfs), 3))
        for i in range(len(pfs)):
            GR_x = GR_coord["GR_x"][GR_coord["GR_id"] == source_ids[i]].to_numpy()[0]
            GR_y = GR_coord["GR_y"][GR_coord["GR_id"] == source_ids[i]].to_numpy()[0]
            proportion = (GR_y / y_ml) * y_ml
            PC_y = PC_coord["PC_y"][PC_coord["PC_id"] == PC_ids[i]].to_numpy()[0]
            PC_z = PC_coord["PC_z"][PC_coord["PC_id"] == PC_ids[i]].to_numpy()[0]
            nNOS_coordinates[i, 0] = random.uniform(
                GR_x - x_nos_variation, GR_x + x_nos_variation
            )
            nNOS_coordinates[i, 1] = random.uniform(
                PC_y + pc_soma + proportion - y_nos_variation,
                PC_y + pc_soma + proportion + y_nos_variation,
            )
            nNOS_coordinates[i, 2] = random.uniform(
                PC_z - z_nos_variation / 2, PC_z + z_nos_variation / 2
            )
        ev_point_coordinates = nNOS_coordinates

        GR_subpop = [[] for i in range(num_subpop)]
        nos_coord_subpop = [[] for i in range(num_subpop)]
        x_left = np.min(nNOS_coordinates[:, 0])
        x_right = np.max(nNOS_coordinates[:, 0])
        subpop_boarder = np.linspace(x_left, x_right, num_subpop + 1)
        for n in range(num_subpop):
            nos_coord_subpop[n] = nNOS_coordinates[
                np.where(
                    (nNOS_coordinates[:, 0] > subpop_boarder[n])
                    & (nNOS_coordinates[:, 0] < subpop_boarder[n + 1])
                )[0]
            ]
            GR_subpop[n] = list(
                np.unique(
                    np.array(
                        [
                            source_ids[i]
                            for i in np.where(
                                (nNOS_coordinates[:, 0] > subpop_boarder[n])
                                & (nNOS_coordinates[:, 0] < subpop_boarder[n + 1])
                            )[0]
                        ]
                    )
                )
            )

        MFGR_conn_param = {
            "model": "static_synapse",
            "weight": {"distribution": "uniform", "low": 2.0, "high": 3.0},
            "delay": 1.0,
        }
        for n in range(num_subpop):
            nest.Connect(
                MF_subpop[n],
                GR_subpop[n],
                {"rule": "fixed_indegree", "indegree": 4, "multapses": False},
                MFGR_conn_param,
            )

        conn_param = {"model": "static_synapse", "weight": 1.0, "delay": 1.0}

        PG_input = nest.Create("spike_generator", MF_num)
        nest.Connect(PG_input, MF, "one_to_one", conn_param)
        if num_subpop != 1:
            PG_active = []
            for i in [0]:
                for mf_id in MF_subpop[i]:
                    A = nest.GetConnections(PG_input, [mf_id])
                    PG_active.append(A[0][0])
            PG_input = PG_active

        PG_error = nest.Create("spike_generator", IO_num)
        nest.Connect(PG_error, IO, "one_to_one", conn_param)

        PG_noise = nest.Create("poisson_generator", 1)
        nest.Connect(PG_noise, MF, "all_to_all", conn_param)
        nest.SetStatus(
            PG_noise,
            params={
                "rate": self.net_config["protocol"]["noise_rate"],
                "start": 0.0,
                "stop": 1.0
                * self.net_config["protocol"]["trial_len"]
                * self.net_config["protocol"]["n_trial"],
            },
        )
        total_sim_len = (
            self.net_config["protocol"]["trial_len"]
            * self.net_config["protocol"]["n_trial"]
        )
        sim_time_steps = np.arange(0, total_sim_len, 1.0)  # [ms]

        bin_size = 1 / 1000
        # CS spiking pattern
        CS_pattern = []
        CS_id = []
        for index, mf in enumerate(PG_input):
            CS_spikes = homogeneous_poisson_for_nest(
                self.net_config["protocol"]["input_rate"],
                (
                    self.net_config["protocol"]["stop_CS"]
                    - self.net_config["protocol"]["start_CS"]
                )
                / 1000,
                bin_size,
            )
            t = np.arange(len(CS_spikes)) * bin_size
            CS_time_stamps = (
                t[CS_spikes] * 1000 + self.net_config["protocol"]["start_CS"]
            )
            CS_id = np.append(CS_id, np.ones(len(CS_time_stamps)) * mf)
            CS_pattern = np.append(CS_pattern, CS_time_stamps)
        CS_stimulus = np.zeros((len(CS_pattern), 2))
        CS_stimulus[:, 0] = CS_id
        CS_stimulus[:, 1] = CS_pattern
        # US spiking pattern
        US_pattern = []
        US_id = []
        US_time_stamps = np.arange(
            self.net_config["protocol"]["start_US"],
            self.net_config["protocol"]["stop_CS"],
            1000 / self.net_config["protocol"]["error_rate"],
        )
        for index, cf in enumerate(PG_error):
            US_id = np.append(US_id, np.ones(len(US_time_stamps)) * cf)
            US_pattern = np.append(US_pattern, US_time_stamps)
        US_stimulus = np.zeros((len(US_pattern), 2))
        US_stimulus[:, 0] = US_id
        US_stimulus[:, 1] = US_pattern

        if NO_dependency:
            t0 = time.time()
            simulation_file = "NO_simulation"
            sim = NODS(self.NO_model_parameters)
            sim.init_geometry(
                nNOS_coordinates=nNOS_coordinates,
                ev_point_coordinates=ev_point_coordinates,
                source_ids=source_ids,
                ev_point_ids=ev_point_ids,
                nos_ids=nos_ids,
                cluster_ev_point_ids=cluster_ev_point_ids,
                cluster_nos_ids=cluster_nos_ids,
            )
            sim.time = sim_time_steps
            sim.init_simulation(
                simulation_file, store_sim=False
            )  # If you want to save sim inizialization change store_sim=True
            t = time.time() - t0
            print("time {}".format(t))

        processed = 0
        trial_CS_stimulus = np.copy(CS_stimulus)
        trial_US_stimulus = np.copy(US_stimulus)
        t0 = time.time()
        for trial in range(self.net_config["protocol"]["n_trial"]):
            print("Simulating trial: " + str(trial))
            trial_CS_stimulus[:, 1] = CS_stimulus[:, 1] + (
                trial * self.net_config["protocol"]["trial_len"]
            )
            trial_US_stimulus[:, 1] = US_stimulus[:, 1] + (
                trial * self.net_config["protocol"]["trial_len"]
            )
            for k, id_input in enumerate(PG_input):
                nest.SetStatus(
                    [PG_input[k]],
                    params={
                        "spike_times": trial_CS_stimulus[
                            trial_CS_stimulus[:, 0] == id_input, 1
                        ],
                        "allow_offgrid_times": True,
                    },
                )
            for k, id_error in enumerate(PG_error):
                nest.SetStatus(
                    [PG_error[k]],
                    params={
                        "spike_times": trial_US_stimulus[
                            trial_US_stimulus[:, 0] == id_error, 1
                        ],
                        "allow_offgrid_times": True,
                    },
                )
            for t in range(
                trial * self.net_config["protocol"]["trial_len"],
                (trial + 1) * self.net_config["protocol"]["trial_len"],
            ):
                nest.Simulate(1.0)
                time.sleep(0.01)
                if NO_dependency:
                    ID_cell = nest.GetStatus(spikedetector_GR, "events")[0]["senders"]
                    active_sources = ID_cell[processed:]
                    processed += len(active_sources)
                    sim.evaluate_diffusion(active_sources, t)
                    list_dict = []
                    for i in range(len(pfs)):
                        list_dict.append(
                            {
                                "meta_l": float(
                                    sig(
                                        x=sim.NO_in_ev_points[t, i],
                                        A=1,
                                        B=self.net_config["protocol"]["NO_threshold"],
                                    )
                                )
                            }
                        )
                    nest.SetStatus(pfs, list_dict)
        t = time.time() - t0
        print("time {}".format(t))

    def plot_cell_sdf(self, cell, step, filename):
        palette = list(
            reversed(
                sns.color_palette(
                    "viridis", self.net_config["protocol"]["n_trial"]
                ).as_hex()
            )
        )
        sm = plt.cm.ScalarMappable(
            cmap="viridis_r",
            norm=plt.Normalize(vmin=0, vmax=self.net_config["protocol"]["n_trial"]),
        )
        sdf_mean_cell = []
        sdf_maf_cell = []
        for trial in range(self.net_config["protocol"]["n_trial"]):
            start = trial * self.net_config["protocol"]["trial_len"]
            stop = (
                self.net_config["protocol"]["stop_CS"]
                + trial * self.net_config["protocol"]["trial_len"]
            )
            spk = get_spike_activity(cell)
            sdf_cell = sdf(start=start, stop=stop, spk=spk, step=step)
            sdf_mean_cell.append(sdf_mean(sdf_cell))
            sdf_maf_cell.append(sdf_maf(sdf_cell))

        fig = plt.figure()
        for trial in range(self.net_config["protocol"]["n_trial"]):
            plt.plot(sdf_mean_cell[trial], palette[trial])
        plt.title(cell)
        plt.xlabel("Time [ms]")
        plt.ylabel("SDF [Hz]")
        plt.axvline(self.net_config["protocol"]["start_CS"], label="CS start", c="grey")
        plt.axvline(
            self.net_config["protocol"]["start_US"], label="US start", c="black"
        )
        plt.axvline(
            self.net_config["protocol"]["stop_CS"], label="CS & US end ", c="red"
        )

        # plt.xticks(np.arange(0,351,50), np.arange(50,401,50))
        plt.legend()
        plt.colorbar(sm, label="Trial")
        fig.savefig(filename + ".png")

    def generate_sim_description(self, description):
        current_datetime = datetime.now().strftime("%Y-%m-%d")

        # Define the README content
        readme_content = f"""# Simulation Parameters
Date: {current_datetime}
## Parameters
- n_trials: {self.net_config["protocol"]["n_trial"]}
- CS_rate: {self.net_config["protocol"]["input_rate"]}
- US_rate: {self.net_config["protocol"]["error_rate"]}
- noise_rate:{self.net_config["protocol"]["noise_rate"]}
- A_minus: {self.net_config["connections"]["A_minus"]}
- A_plus: {self.net_config["connections"]["A_plus"]}
- Wmin: {self.net_config["connections"]["Wmin"]}
- Wmax: {self.net_config["connections"]["Wmax"]}
## Description
{description}
"""

        # Write the README content to a file
        with open("./sim_description.md", "w") as readme_file:
            readme_file.write(readme_content)


if __name__ == "__main__":
    A_plus_values = [
        0.008,
        0.008,
        0.008,
        0.008,
        0.008 + 0.0005,
        0.008 + 0.001,
        0.008 - 0.0005,
        0.008 - 0.001,
    ]
    A_minus_values = [
        -0.02 + 0.0005,
        -0.02 + 0.001,
        -0.02 - 0.0005,
        -0.02 - 0.001,
        -0.02,
        -0.02,
        -0.02,
        -0.02,
    ]
    # NO_dependency_list = np.append(np.zeros((len(A_plus_values)/2)), np.ones((len(A_plus_values)/2)))
    NO_dependency_list = np.ones((len(A_plus_values)))

    params_filename = "model_parameters.json"
    root_path = "./nods/"
    with open(os.path.join(root_path, params_filename), "r") as read_file:
        NO_model_par = json.load(read_file)

    with open("demo_single_pc_tree.json", "r") as read_file:
        net_config = json.load(read_file)
    NO_threshold = net_config["protocol"]["NO_threshold"]

    for i in range(len(NO_dependency_list)):
        nest.ResetKernel()
        nest.SetKernelStatus({"overwrite_files": True, "resolution": 1.0})
        random.seed()
        seed_g = random.randint(10, 10000)
        seed_r = random.randint(10, 10000)
        nest.SetKernelStatus({"grng_seed": seed_g})
        nest.SetKernelStatus({"rng_seeds": [seed_r]})
        nest.set_verbosity("M_ERROR")

        net_config["connections"]["A_plus"] = A_plus_values[i]
        net_config["connections"]["A_minus"] = A_minus_values[i]
        NO_dependency = NO_dependency_list[i]

        description = f"Test robustness over learning rate initialization (A_plus/A_minus parameters)\nsimulation number:{i}"

        if NO_dependency:
            description = description + "\n NO diffusion present"
            net_config["protocol"]["NO_threshold"] = NO_threshold
        else:
            net_config["protocol"]["NO_threshold"] = None

        simulation = Simulation(
            network_configuration=net_config, NO_model_parameters=NO_model_par
        )
        simulation.create_and_simulate_network(NO_dependency=NO_dependency)
        simulation.plot_cell_sdf(cell="purkinje", step=50, filename="purkinje_sdf")
        simulation.plot_cell_sdf(cell="granule", step=5, filename="granule_sdf")
        simulation.generate_sim_description(description=description)

        source_folder = "./"
        destination_folder = "./results/test_intrinsic_rob"
        file_prefixes = [
            "granule",
            "purkinje",
            "sim_description",
            "PFPC",
            "NO_simulation",
        ]
        move_files_to_folder(source_folder, destination_folder, file_prefixes)
