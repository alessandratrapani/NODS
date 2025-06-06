import sys
import os
import nest
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from simulateEBCC import SimulateEBCC
import numpy as np
import gc

plus_str = sys.argv[1]
plus = float(plus_str.replace('_','.'))
minus = int(sys.argv[2])
A_plus = plus*10**-5
A_minus = -minus*10**-4

data_path = "/home/csartor1/code/NODS/data/"
condition = "w_NO"
file_rel_dist = os.path.join(data_path,'relative_dist.csv')
file_ev_points = os.path.join(data_path, "reshaped_ev_points.csv")
file_nNOS = os.path.join(data_path, "nNOS_dict.csv")
noise_rate = 0.0
source_folder = "/home/csartor1/code/NODS/"
destination_folder = "/results"
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

#os.makedirs(destination_folder, exist_ok=True)
nest.Install("cerebmodule")

simulation_description = f"EBCC with A_minus, A_plus= {A_minus},{A_plus}, {condition}"
print(simulation_description)

vt_modality = "1_vt_pf-PC" 
simulation = SimulateEBCC(data_path=os.path.join(source_folder, data_path))
simulation.set_network_configuration()
simulation.set_nest_kernel()
simulation.create_network()
simulation.create_vt(vt_modality=vt_modality)
simulation.connect_network_plastic_syn(vt_modality=vt_modality,A_minus=A_minus, A_plus=A_plus)
simulation.stimulus_geometry(plot=False)
simulation.define_recurrent_CS_stimuli()
simulation.define_US_stimuli()
simulation.define_bg_noise(rate=noise_rate)
simulation.define_recorders()
nods_sim = simulation.initialize_nods(file_rel_dist)
simulation.simulate_network_with_NO(nods_sim)

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

nest.ResetKernel()