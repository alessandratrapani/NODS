#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=g100_usr_prod
#SBATCH --account=EIRI_E_POLIMI
#SBATCH --output="job-%j.log"
#SBATCH --error="job-%j.log"
#SBATCH --partition=g100_usr_prod
#SBATCH --account=EIRI_E_POLIMI

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load python/3.8.12--gcc--10.2.0
module load spack
module load gsl
spack load libtool@2.4.6%gcc@10.2.0 arch=linux-centos8-cascadelake
spack load py-cython@0.29.24%gcc@10.2.0 arch=linux-centos8-cascadelake
source /g100_work/EIRI_E_POLIMI/no_paper/NO_env/bin/activate
source /g100_work/EIRI_E_POLIMI/no_paper/nest-install/bin/nest_vars.sh


srun python /g100_work/EIRI_E_POLIMI/no_paper/NODS/grid_search_Aplus_Aminus/heatmap_spike.py
