import sys
import os
sys.path.append(os.path.join(os.getcwd(), '..'))
from utils import get_spike_activity, sdf, sdf_mean, sdf_maf, count_spikes_across_trials, new_get_spike_activity
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

result_path = "/g100_work/EIRI_E_POLIMI/no_paper/NODS/results/grid_search_new/"
with open("/g100_work/EIRI_E_POLIMI/no_paper/NODS/network_configuration.json", "r") as json_file:
    net_config = json.load(json_file)
    
CS_burst_dur = net_config["devices"]["CS"]["parameters"]["burst_dur"]
CS_start_first = float(net_config["devices"]["CS"]["parameters"]["start_first"])
between_start = net_config["devices"]["CS"]["parameters"]["between_start"]
n_trials = net_config["devices"]["CS"]["parameters"]["n_trials"]
US_start_first = float(net_config["devices"]["US"]["parameters"]["start_first"])
num_sim = 100
trial_length = between_start

minus = [12,11,10,9,8,7,6,5,4,3]
plus = [10,9,8,7,6,5,4,3]


for i,m in enumerate(minus):
    for j,p in enumerate(plus):
        print(f'minus {m} plus {p}',flush=True)
        folder_path = result_path + f"minus{m}_plus{p}/"
        spk_bs_simulations =[]
        spk_cr_simulations =[]
        for k in range(0,num_sim):
            file_path = folder_path+ f"sim{k}/"

            spk = get_spike_activity('pc_spikes', file_path)

            bs_spk = np.array(count_spikes_across_trials(spk, n_trials, trial_length, (150,200)))
            cr_spk = np.array(count_spikes_across_trials(spk, n_trials, trial_length, (250,300)))
            
            spk_bs_simulations.append(bs_spk)
            spk_cr_simulations.append(cr_spk)

        stack_spk_bs = np.stack(spk_bs_simulations, axis=0)
        stack_spk_cr = np.stack(spk_cr_simulations, axis=0)   

        bs_median_spk = np.median(stack_spk_bs, axis=0)
        cr_median_spk = np.median(stack_spk_cr, axis=0)

        df_bs_spk = pd.DataFrame(bs_median_spk)
        df_cr_spk = pd.DataFrame(cr_median_spk)
        
        df_spk_median = pd.concat([df_bs_spk,df_cr_spk], axis = 1)
        df_spk_median.to_csv(folder_path+f'spk_median_minus{m}_plus{p}.csv', index = False)

