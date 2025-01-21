import numpy as np
import os
import sys
import subprocess
import time

type = input("Type of simulation: (test) (paper) (plus_area) (recurrent_CS) (grid_search): ")

if type == 'test':
    num_of_sim = int(input("Number of single simulation: "))
    condition = input("Condition of simulation: ")
    noise = int(input("Noise rate: "))
    # Create a unique results directory for each execution
    results_dir = f"/home/csartor1/code/NODS/results/{type}/{condition}/{noise}Hz/sim_{num_of_sim}"
    os.makedirs(results_dir, exist_ok=True)

    # Define the output file within the results directory
    output_file = os.path.join(results_dir, "output.txt")

    # Run the shell script, passing the execution number as an argument
    with open(output_file, "w") as f:
        subprocess.run(["bash", "run_script.sh", str(num_of_sim), condition, str(noise), type], stdout=f, stderr=f)

elif type == 'paper':
    num_of_sim = int(input("Number of simulations: "))
    condition = input("Condition of simulation: ")
    noise = int(input("Noise rate: "))
    for i in range(0, num_of_sim):
        # Create a unique results directory for each execution
        results_dir = f"/home/csartor1/code/NODS/results/{type}/{condition}/{noise}Hz/sim_{i}"
        os.makedirs(results_dir, exist_ok=True)

        # Define the output file within the results directory
        output_file = os.path.join(results_dir, "output.txt")

        # Run the shell script, passing the execution number as an argument
        with open(output_file, "w") as f:
            subprocess.run(["bash", "run_script.sh", str(i), condition, str(noise), type], stdout=f, stderr=f)

elif type == 'plus_area':
    
    num_of_sim = int(input("Number of single simulation: "))
    condition = input("Condition of simulation: ")
    noise = int(input("Noise rate: "))
    ratio = int(input("Ratio of CS: "))

    # Create a unique results directory for each execution
    results_dir = f"/home/csartor1/code/NODS/results/{type}/{condition}/{noise}Hz/sim_{num_of_sim}"
    os.makedirs(results_dir, exist_ok=True)

    # Define the output file within the results directory
    output_file = os.path.join(results_dir, "output.txt")

    # Run the shell script, passing the execution number as an argument
    with open(output_file, "w") as f:
        subprocess.run(["bash", "run_script.sh", str(num_of_sim), condition, str(noise), type], stdout=f, stderr=f)

elif type == 'recurrent_CS':

    num_of_sim = int(input("Number of single simulation: "))
    condition = input("Condition of simulation: ")
    noise = int(input("Noise rate: "))

    # Create a unique results directory for each execution
    results_dir = f"/home/csartor1/code/NODS/results/{type}/{condition}/{noise}Hz/sim_{num_of_sim}"
    os.makedirs(results_dir, exist_ok=True)

    # Define the output file within the results directory
    output_file = os.path.join(results_dir, "output.txt")

    # Run the shell script, passing the execution number as an argument
    with open(output_file, "w") as f:
        subprocess.run(["bash", "run_script.sh", str(num_of_sim), condition, str(noise), type], stdout=f, stderr=f)

elif type == 'grid_search':
    num_of_sim = int(input("Number of simulations: "))
    condition = 'wo_NO'
    noise = 0
    A_plus_max = int(input("A plus max: "))
    A_plus_min = int(input("A plus min: "))
    A_min_max = int(input("A minus max: "))
    A_min_min = int(input("A minus min: "))

    A_plus = np.arange(A_plus_min, A_plus_max)
    A_minus = np.arange(A_min_min, A_min_max)

    for plus in A_plus:
        for minus in A_minus:
            for i in range(num_of_sim):
                # Create a unique results directory for each execution
                results_dir = f"/home/csartor1/code/NODS/results/{type}/plus{plus}_minus{minus}/sim_{i}"
                os.makedirs(results_dir, exist_ok=True)

                # Define the output file within the results directory
                output_file = os.path.join(results_dir, "output.txt")

                # Run the shell script, passing the execution number as an argument
                with open(output_file, "w") as f:
                    subprocess.run(["bash", "run_grid_search.sh", type, str(plus), str(minus), str(i)], stdout=f, stderr=f)

elif type == 'dcn':
    num_of_sim = input("Number of single simulation: ")
    condition = input("Condition of simulation: ")
    noise = int(input("Noise rate: "))
    # Create a unique results directory for each execution
    results_dir = f"/home/csartor1/code/NODS/results/{type}/{condition}/{noise}Hz/sim_{num_of_sim}"
    os.makedirs(results_dir, exist_ok=True)

    # Define the output file within the results directory
    output_file = os.path.join(results_dir, "output.txt")

    # Run the shell script, passing the execution number as an argument
    with open(output_file, "w") as f:
        subprocess.run(["bash", "run_script.sh", num_of_sim, condition, str(noise), type], stdout=f, stderr=f)