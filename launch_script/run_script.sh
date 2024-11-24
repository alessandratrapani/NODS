
source /home/csartor1/.bashrc

screen -dmS $2_$3Hz_sim_$1

screen -S $2_$3Hz_sim_$1 -X stuff "source /home/csartor1/.bashrc \n"

screen -S $2_$3Hz_sim_$1 -X stuff "conda activate nest_2_18_bis \n"

screen -S $2_$3Hz_sim_$1 -X stuff "cd /home/csartor1/code/NODS/results/$2/$3Hz/sim_$1 \n"

screen -S $2_$3Hz_sim_$1 -X stuff "python3.8 /home/csartor1/code/NODS/launch_script/run_sim_server.py $2 $3\n"

