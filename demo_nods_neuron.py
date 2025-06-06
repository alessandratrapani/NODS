import nest
import numpy as np
import os

nest.Install("cerebmodule")
dest_folder = '/data/Neuron_comparison/'
nest.SetKernelStatus({"overwrite_files": True})
granule_cell = nest.Create('hh_psc_alpha',1)
"""nest.SetStatus(granule_cell, {
                "t_ref": 1.5,
                "C_m": 7.0,
                "V_th": -41.0,
                "V_reset": -70.0,
                "E_L": -62.0,
                "Vmin": -150.0,
                "Vinit": -62.0,
                "lambda_0": 1.0,
                "tau_V": 0.3,
                "tau_m": 24.15,
                "I_e": -0.888,
                "kadap": 0.022,
                "k1": 0.311,
                "k2": 0.041,
                "A1": 0.01,
                "A2": -0.94,
                "tau_syn1": 1.9,
                "tau_syn2": 4.5,
                "E_rev1": 0.0,
                "E_rev2": -80.0,
                "E_rev3": 0.0
            })"""


spikes = ([250.0,280.0,310.0,340.0,370.0])
spike_times = []
for i in range(5):
    spike_times.append([spikes[j]+(400*i) for j in range(len(spikes))])
spike_times = np.concatenate(spike_times).tolist()

spike_gen = nest.Create('spike_generator', params = {'spike_times': spike_times, 'spike_weights':[9999.0]*5*5})
voltmeter = nest.Create('voltmeter', params = {"withgid": True, "withtime": True, "to_file": True})
spike_recorder = nest.Create('spike_detector', params = {"withgid": True, "withtime": True, "to_file": True, "label": "granule_spikes"})

nest.Connect(spike_gen,granule_cell, 'one_to_one')
nest.Connect(voltmeter,granule_cell)
nest.Connect(granule_cell,spike_recorder)

nest.Simulate(2000)

