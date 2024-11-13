import numpy as np
import os
import sys
import subprocess

num_of_sim = int(sys.argv[1])

for i in range(1, num_of_sim + 1):
        # Create a unique results directory for each execution
        results_dir = f"/home/csartor1/code/NODS/results/test/sim_{i}"
        os.makedirs(results_dir, exist_ok=True)

        # Define the output file within the results directory
        output_file = os.path.join(results_dir, "output.txt")

        # Run the shell script, passing the execution number as an argument
        with open(output_file, "w") as f:
            subprocess.run(["bash", "run_script.sh", str(i)], stdout=f, stderr=f)
