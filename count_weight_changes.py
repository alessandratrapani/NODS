from weight_changes import extract_weight_changes
import numpy as np
import os

noise_rate = [0,4,8,12]

condition = ["w_NO","wo_NO"]

for cond in condition:
    folder_results = f"/home/csartor1/code/NODS/results/noise/{cond}/"
    for noise in noise_rate:
        noise_folder = folder_results+f"{noise}Hz/sim_2/"
        files = [f for f in os.listdir(noise_folder) if os.path.isfile(os.path.join(noise_folder, f))]
        for f in files:
            if f.startswith('pf'):
                file_pf_PC = noise_folder+f

        output_file = os.path.join(noise_folder, "weight_changes.txt")
        extract_weight_changes(file_pf_PC, output_file)