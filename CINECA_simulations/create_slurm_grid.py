import os
import subprocess
import time

def create_slurm_script(noise, simulation, condition, minus, plus):
    slurm_script_content = f"""#!/bin/bash
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24

#SBATCH --partition=g100_usr_prod
#SBATCH --account=EIRI_E_POLIMI
#SBATCH --output="job-%j.log"
#SBATCH --error="job-%j.log"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load python/3.8.12--gcc--10.2.0
module load spack
module load gsl
spack load libtool@2.4.6%gcc@10.2.0 arch=linux-centos8-cascadelake
spack load py-cython@0.29.24%gcc@10.2.0 arch=linux-centos8-cascadelake
source /g100_work/EIRI_E_POLIMI/no_paper/NO_env/bin/activate
source /g100_work/EIRI_E_POLIMI/no_paper/nest-install/bin/nest_vars.sh

cd /g100_work/EIRI_E_POLIMI/no_paper/NODS/results/grid_search_new

mkdir minus{int(minus)}_plus{int(plus)}

cd minus{int(minus)}_plus{int(plus)}

mkdir sim{simulation}

cd sim{simulation}

srun python /g100_work/EIRI_E_POLIMI/no_paper/NODS/grid_search_Aplus_Aminus/simulation_grid.py {noise} {simulation} {condition} {minus} {plus} 
"""
        
    slurm_script_path = "run_simulation_grid.slurm"
    
    with open(slurm_script_path, "w") as slurm_file:
        slurm_file.write(slurm_script_content)

    return slurm_script_path

def submit_slurm_script(script_path):
    try:
        result = subprocess.run(["sbatch", script_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("SLURM job submitted successfully.")
        print(result.stdout.decode())
    except subprocess.CalledProcessError as e:
        print("Failed to submit SLURM job.")
        print(e.stderr.decode())

if __name__ == "__main__":

    noise = 0
    num_simulation = int(input('Enter total number of simulation: '))
    condition = "without_NO"
    minus = [5]
    plus = [4,3]

    for m in minus:
        for p in plus:
            for simulation in range(0,num_simulation):
                slurm_script_path = create_slurm_script(noise, simulation, condition,m,p)
                submit_slurm_script(slurm_script_path)
            time.sleep(300)

