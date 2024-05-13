import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from simulateEBCC import SimulateEBCC

data_path = "./data/"
condition = "without NO"
folder = "grid_search"
#A_minus = -0.0005
#A_plus = 0.0000225
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
destination_folder = os.path.join(destination_folder, folder)
os.makedirs(destination_folder, exist_ok=True)

for i in range(1,7):
    for j in range(1,7):
        
        A_minus = -10**(-i)
        A_plus = 10**(-j)

        simulation_description = f"EBCC with A_minus, A_plus= {A_minus},{A_minus}, {condition}"
        print(simulation_description)



        vt_modality = "1_vt_PC" 
        simulation = SimulateEBCC(data_path=data_path)
        simulation.set_network_configuration()
        simulation.set_nest_kernel()
        simulation.create_network()
        simulation.create_vt(vt_modality=vt_modality)
        simulation.connect_network_plastic_syn(vt_modality=vt_modality,A_minus=A_minus)
        simulation.stimulus_geometry(plot=False)
        simulation.define_CS_stimuli()
        simulation.define_US_stimuli()
        simulation.define_bg_noise()
        simulation.define_recorders()
        simulation.simulate_network()

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
                        - noise_rate:{simulation.net_config["devices"]["background_noise"]["parameters"]["rate"]}
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

        from move_files import move_files_to_folder
        dest_folder = os.path.join(destination_folder, str(i)+"_"+str(j))
        move_files_to_folder(source_folder, dest_folder, file_prefixes)
        del simulation