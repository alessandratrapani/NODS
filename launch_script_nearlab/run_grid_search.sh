
source /home/csartor1/.bashrc

screen -dmS $1_plus$2_minus$3_sim_$4

screen -S $1_plus$2_minus$3_sim_$4 -X stuff "source /home/csartor1/.bashrc \n"

screen -S $1_plus$2_minus$3_sim_$4 -X stuff "conda activate nest_2_18_bis \n"

screen -S $1_plus$2_minus$3_sim_$4 -X stuff "cd /home/csartor1/code/NODS/results/$1/plus$2_minus$3/sim_$4 \n"

screen -S $1_plus$2_minus$3_sim_$4 -X stuff "python3.8 /home/csartor1/code/NODS/launch_script_nearlab/run_grid_search.py $2 $3\n"

