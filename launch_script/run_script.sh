
source /home/csartor1/.bashrc

screen -dmS run_$1

screen -S run_$1 -X stuff "source /home/csartor1/.bashrc \n"

screen -S run_$1 -X stuff "conda activate nest_2_18_bis \n"

screen -S run_$1 -X stuff "cd /home/csartor1/code/NODS/results/test/sim_$1 \n"

screen -S run_$1 -X stuff "python3.8 /home/csartor1/code/NODS/launch_script/run_sim_server.py \n"

