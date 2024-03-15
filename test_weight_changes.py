from simulateEBCC import SimulateEBCC
import os
data_path = "./data/"
conditions = ["without NO", "with NO"]
noise_rates = [0.0, 4.0, 8.0]
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
        
for condition, noise_rate in zip(conditions, noise_rates):
    simulation_description = f"Only background noise at {noise_rate}, {condition}"
    print(simulation_description)
    destination_folder = os.path.join(destination_folder, condition.replace(" ",""))
    os.makedirs(destination_folder, exist_ok=True)
    if condition == "with NO":
        vt_modality = "1_vt_pf-PC"  # 1_vt_PC
        simulation = SimulateEBCC(data_path=data_path)
        simulation.set_network_configuration(vt_modality=vt_modality)
        simulation.set_nest_kernel()
        simulation.create_network()
        simulation.create_vt(vt_modality=vt_modality)
        simulation.connect_network_plastic_syn(vt_modality=vt_modality)
        simulation.stimulus_geometry(plot=False)
        simulation.define_bg_noise()
        simulation.define_recorders()
        nods_sim = simulation.initialize_nods()
        simulation.simulate_network_with_NO(nods_sim)

        step = 5
        for cell in ["pc_spikes", "stellate_spikes", "granule_spikes"]:
            simulation.plot_cell_activity_over_trials(cell=cell, step=step)
            simulation.plot_cell_raster(cell=cell)

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
                        - A_minus: {simulation.net_config["connection_models"]["parallel_fiber_to_purkinje"]["parameters"]["A_minus"]}
                        - A_plus: {simulation.net_config["connection_models"]["parallel_fiber_to_purkinje"]["parameters"]["A_plus"]}
                        - Wmin: {simulation.net_config["connection_models"]["parallel_fiber_to_purkinje"]["parameters"]["Wmin"]}
                        - Wmax: {simulation.net_config["connection_models"]["parallel_fiber_to_purkinje"]["parameters"]["Wmax"]}
                        - CS_radius: {simulation.net_config["devices"]["CS"]["radius"]}

                        ## Description
                        {simulation.simulation_description}
                        {vt_modality}
                        """
        # Write the README content to a file
        with open("./aa_sim_description.md", "w") as readme_file:
            readme_file.write(readme_content)
        
        from move_files import move_files_to_folder
        move_files_to_folder(source_folder, destination_folder, file_prefixes)

    if condition == "without NO":
        vt_modality = "1_vt_PC" 
        simulation = SimulateEBCC(data_path=data_path)
        simulation.set_network_configuration(vt_modality=vt_modality)
        simulation.set_nest_kernel()
        simulation.create_network()
        simulation.create_vt(vt_modality=vt_modality)
        simulation.connect_network_plastic_syn(vt_modality=vt_modality)
        simulation.stimulus_geometry(plot=False)
        simulation.define_bg_noise()
        simulation.define_recorders()
        simulation.simulate_network()

        step = 5
        for cell in ["pc_spikes", "stellate_spikes", "granule_spikes"]:
            simulation.plot_cell_activity_over_trials(cell=cell, step=step)
            simulation.plot_cell_raster(cell=cell)

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
                        - A_minus: {simulation.net_config["connection_models"]["parallel_fiber_to_purkinje"]["parameters"]["A_minus"]}
                        - A_plus: {simulation.net_config["connection_models"]["parallel_fiber_to_purkinje"]["parameters"]["A_plus"]}
                        - Wmin: {simulation.net_config["connection_models"]["parallel_fiber_to_purkinje"]["parameters"]["Wmin"]}
                        - Wmax: {simulation.net_config["connection_models"]["parallel_fiber_to_purkinje"]["parameters"]["Wmax"]}
                        - CS_radius: {simulation.net_config["devices"]["CS"]["radius"]}

                        ## Description
                        {simulation.simulation_description}
                        {vt_modality}
                        """
        # Write the README content to a file
        with open("./aa_sim_description.md", "w") as readme_file:
            readme_file.write(readme_content)
        
        from move_files import move_files_to_folder
        move_files_to_folder(source_folder, destination_folder, file_prefixes)