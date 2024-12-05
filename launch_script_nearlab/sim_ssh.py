import numpy as np
import os
import sys
import subprocess

num_of_sim = int(input("Number of simulations: "))
condition = input("Condition of simulation: ")
noise = int(input("Noise rate: "))

#for i in range(1, num_of_sim + 1):
# Create a unique results directory for each execution
results_dir = f"/home/csartor1/code/NODS/results/{condition}/{noise}Hz/sim_{num_of_sim}"
os.makedirs(results_dir, exist_ok=True)

# Define the output file within the results directory
output_file = os.path.join(results_dir, "output.txt")

# Run the shell script, passing the execution number as an argument
with open(output_file, "w") as f:
    subprocess.run(["bash", "run_script.sh", str(num_of_sim), condition, str(noise)], stdout=f, stderr=f)
