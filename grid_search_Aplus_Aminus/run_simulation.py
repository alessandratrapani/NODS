import sys
import os
import nest
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from simulateEBCC import SimulateEBCC
import numpy as np
import gc


data_path = "./data/"
condition = "without_NO"
folder_grid = f"paper/"
file_rel_dist = os.path.join(data_path,'relative_dist.csv')
A_minus = -0.0007
A_plus = 0.00008
noise_rate = 8.0
source_folder = "./"
destination_folder = "./results"
file_prefixes = [
    "glom_spikes",
    "pc_spikes",
    "io_spikes",
    "golgi_spikes",
    "basket_spikes",
    "stellate_spikes",
    "granule_spikes",
    "pf-PC",
    "aa_",
]
destination_folder = os.path.join(destination_folder, folder_grid)
os.makedirs(destination_folder, exist_ok=True)
nest.Install("cerebmodule")

simulation_description = f"EBCC with A_minus, A_plus= {A_minus},{A_plus}, {condition}"
print(simulation_description)

if condition == "without_NO":

    vt_modality = "1_vt_PC" 
    simulation = SimulateEBCC(data_path=data_path)
    simulation.set_network_configuration()
    simulation.set_nest_kernel()
    simulation.create_network()
    simulation.create_vt(vt_modality=vt_modality)
    simulation.connect_network_plastic_syn(vt_modality=vt_modality,A_minus=A_minus, A_plus=A_plus)
    simulation.stimulus_geometry(plot=False)
    simulation.define_CS_stimuli()
    simulation.define_US_stimuli()
    simulation.define_bg_noise(rate=noise_rate)
    simulation.define_recorders()
    simulation.simulate_network()

elif condition == "with_NO":

    vt_modality = "1_vt_pf-PC" 
    simulation = SimulateEBCC(data_path=data_path)
    simulation.set_network_configuration()
    simulation.set_nest_kernel()
    simulation.create_network()
    simulation.create_vt(vt_modality=vt_modality)
    simulation.connect_network_plastic_syn(vt_modality=vt_modality,A_minus=A_minus, A_plus=A_plus)
    simulation.stimulus_geometry(plot=False)
    simulation.define_CS_stimuli()
    simulation.define_US_stimuli()
    simulation.define_bg_noise(rate=noise_rate)
    simulation.define_recorders()
    nods_sim = simulation.initialize_nods(file_rel_dist)
    simulation.simulate_network_with_NO(nods_sim)

simulation.plot_cell_activity_over_trials(cell="pc_spikes", step=5)

from datetime import datetime
# Generate datetime string for the README
current_datetime = datetime.now().strftime("%Y-%m-%d")
# Define the README content
readme_content = f"""# Simulation Parameters

                Date: {current_datetime}

                ## Parameters
                - n_trials: {simulation.net_config["devices"]["CS"]["parameters"]["n_trials"]}
                - CS_rate: {simulation.net_config["devices"]["CS"]["parameters"]["rate"]}
                - US_rate: {simulation.net_config["devices"]["US"]["parameters"]["rate"]}
                - noise_rate:{noise_rate}
                - A_minus: {A_minus}
                - A_plus: {A_plus}
                - Wmin: {simulation.net_config["connection_models"]["parallel_fiber_to_purkinje"]["parameters"]["Wmin"]}
                - Wmax: {simulation.net_config["connection_models"]["parallel_fiber_to_purkinje"]["parameters"]["Wmax"]}
                - CS_radius: {simulation.net_config["devices"]["CS"]["radius"]}

                ## Description
                {simulation_description}
                {vt_modality}
                """
# Write the README content to a file
with open("./aa_sim_description.md", "w") as readme_file:
    readme_file.write(readme_content)
readme_file.close()

folder_sim = f'{condition}_{int(noise_rate)}Hz'
move_folder = os.path.join(os.path.join(destination_folder, folder_sim))
#os.makedirs(os.path.join(destination_folder, folder_sim), exist_ok=True)
from move_files import move_files_to_folder
move_files_to_folder(source_folder, move_folder, file_prefixes)

nest.ResetKernel()