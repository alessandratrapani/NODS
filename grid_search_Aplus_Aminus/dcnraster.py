root_path = f"/home/csartor1/code/NODS/results/dcn/wo_NO/0Hz/"
cell = "dcn_spikes"

cond = "sim_0_70"
cond_1 = "sim_0_75"
results_path = root_path + cond
results_path_mf = root_path + cond_1

sdf_mean_cell_glut = []
sdf_mean_cell_gaba = []
sdf_mean_sim_glut = []
sdf_mean_sim_gaba = []
t_bs = 200
cr_window = 300 - t_bs
thr = 4
for i in range(5):
    sim_path_glut = f"{results_path}_{i}/"
    sim_path_gaba = f"{results_path_mf}_{i}/"
    spk_glut = get_spike_activity(cell_name=cell, path=sim_path_glut)
    spk_gaba = get_spike_activity(cell_name=cell, path=sim_path_gaba) 
    step = 5
    for trial in range(n_trials):
        start = trial * between_start
        stop = CS_start_first + CS_burst_dur + trial * between_start
        sdf_cell_glut = sdf(start=start, stop=stop, spk=spk_glut, step=step)
        sdf_mean_cell_glut.append(sdf_mean(sdf_cell_glut))
        sdf_cell_gaba = sdf(start=start, stop=stop, spk=spk_gaba, step=step)
        sdf_mean_cell_gaba.append(sdf_mean(sdf_cell_gaba))

    sdf_mean_sim_glut.append(sdf_mean_cell_glut)
    sdf_mean_sim_gaba.append(sdf_mean_cell_gaba)

sdf_median_glut = np.stack(sdf_mean_sim_glut, axis=0)
sdf_median_glut = np.median(sdf_median_glut, axis=0)
sdf_median_gaba = np.stack(sdf_mean_sim_gaba, axis=0)
sdf_median_gaba = np.median(sdf_median_gaba, axis=0)
sdf_bs = []
sdf_bs_mf = []

for i in range(1,4):
    sdf_bs.append(sdf_mean_cell[i][t_bs])
    sdf_bs_mf.append(sdf_mean_cell_mf[i][t_bs])

sdf_bs_glut_median = np.median(sdf_bs)
sdf_bs_gaba_median = np.median(sdf_bs_mf)

cr_sdf_glut = sdf_median_glut[:,t_bs:] - sdf_bs_glut_median
cr_sdf_gaba = sdf_median_gaba[:,t_bs:] - sdf_bs_gaba_median