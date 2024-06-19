import os
import shutil
from datetime import datetime

def move_files_to_folder(source_folder, destination_folder, file_prefixes):
    # Create a datetime string for the folder name
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create the destination folder with the datetime string
    #destination_folder_with_datetime = os.path.join(destination_folder, current_datetime)
    
    os.makedirs(destination_folder, exist_ok=True)

    # Iterate over files in the source folder
    for filename in os.listdir(source_folder):
        for file_prefix in file_prefixes:
            if filename.startswith(file_prefix):
                # Construct the source and destination paths
                source_path = os.path.join(source_folder, filename)
                destination_path = os.path.join(destination_folder, filename)

                # Move the file to the destination folder
                shutil.move(source_path, destination_path)
                print(f"Moved {filename} to {destination_folder}")

if __name__ == "__main__":
    # Example usage
    source_folder = "./"  
    destination_folder = "./results"  
    file_prefixes = ["glom_spikes","pc_spikes","io_spikes","golgi_spikes","basket_spikes","stellate_spikes","granule_spikes","pf-PC","sim_description","1_"]

    move_files_to_folder(source_folder, destination_folder, file_prefixes)
