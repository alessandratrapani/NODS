import os
import subprocess

def create_slurm_script(noise, simulation):
    slurm_script_content = f"""#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48

#SBATCH --partition=g100_usr_prod
#SBATCH --account=EIRI_E_POLIMI
#SBATCH --output="job-%j.log"
#SBATCH --error="job-%j.log"
#SBATCH --mail-type=END
#SBATCH --mail-user=carloandrea.sartori@polimi.it

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=48

module load python/3.8.12--gcc--10.2.0
module load spack
module load gsl
spack load libtool@2.4.6%gcc@10.2.0 arch=linux-centos8-cascadelake
spack load py-cython@0.29.24%gcc@10.2.0 arch=linux-centos8-cascadelake
source /g100_work/EIRI_E_POLIMI/no_plasticity/NO_env/bin/activate
source /g100_work/EIRI_E_POLIMI/no_plasticity/nest-install/bin/nest_vars.sh

cd $WORK/no_plasticity/NODS/CINECA_simulations/results

mkdir simulation_{noise}Hz_sim{simulation}

cp /g100_work/EIRI_E_POLIMI/no_plasticity/NODS/grid_search_Aplus_Aminus/simulation.py simulation_{noise}Hz_sim{simulation}/

cd simulation_{noise}Hz_sim{simulation}

python simulation.py {noise} {simulation}
"""


    slurm_script_path = "run_simulation.slurm"
    
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

    #noise_rate = [0,4,8]
    noise = input("Enter noise: ")
    simulation = input("Enter simulation: ")

    slurm_script_path = create_slurm_script(noise, simulation)
    submit_slurm_script(slurm_script_path)
