import matplotlib.pyplot as plt
import numpy as np
import os

units = ''' 
    mean_tau_t, mean_phase_t,\
        csi_i_mean_mean, csi_i_var_mean, csi_i_range_mean,\
        csi_i_mean_var, csi_i_k_mean,\
        fft_i_avg_max_index, fft_i_avg_peak_height, fft_i_avg_num_peaks,\
        cir_mean_peak, cir_var_peak, cir_mean_var, n_factor,\
        cir_mean_eps_h, cir_var_eps_h,\
        cir_mean_skew, cir_var_skew, cir_mean_kurtosis, cir_var_kurtosis,\
        cir_mean_fall_time, cir_mean_var_idp, cir_var_var_idp,\
        cir_mean_se,\
        cir_mean_t_med, cir_var_t_med, cir_mean_t_rms, cir_var_t_rms,\
        cir_mean_cc, cir_var_cc,\
        mean_rssi, var_rssi
'''

units = units.replace('\n', '').replace(' ', '').split(',')
print(units)
print(f'Total # of units: {len(units)}')

print('Loading processed_data...')
# data_dict = np.load('saves/processed_data/NN_lltf_feature_extraction_data.npy', allow_pickle=True).item()
# data_dict = np.load('saves/processed_data/NN_htltf_feature_extraction_data.npy', allow_pickle=True).item()
data_dict = np.load('saves/processed_data/NN_stbchtltf_feature_extraction_data.npy', allow_pickle=True).item()

dataset_folder_paths = ['data/NLOS_data/', 'data/LOS_data/']
dataset_filepaths0 = [f for f in os.listdir(dataset_folder_paths[0])]
dataset_filepaths1 = [f for f in os.listdir(dataset_folder_paths[1])]

data_0 = []
data_1 = []
for df in dataset_filepaths0:
    data_0 += data_dict[dataset_folder_paths[0] + df]
for df in dataset_filepaths1:
    data_1 += data_dict[dataset_folder_paths[1] + df]

print(f'Class 0 Data Size: {len(data_0)}')
print(f'Class 1 Data Size: {len(data_1)}')

data_0 = np.array(data_0).T
data_1 = np.array(data_1).T

for file in os.listdir('plots/'):
    try:
        os.remove('plots/' + file)
    except:
        continue


def fe_histogram(data_0, data_1, label_0, label_1, unit, bar_count, unitname):
    
    bins = np.linspace(min(min(data_0), min(data_1)) - 0.5, max(max(data_0), max(data_1)) + 0.5, bar_count)
    plt.hist(data_0, bins=bins, label=label_0, align='mid', rwidth=0.8, alpha=0.4, edgecolor='black', hatch='////', density=True)
    plt.hist(data_1, bins=bins, label=label_1, align='mid', rwidth=0.8, alpha=0.4, edgecolor='black', hatch='\\\\', density=True)
    plt.xlabel(unit)
    plt.ylabel('Frequency')
    plt.title(f'{unit} Distribution')
    plt.legend()
    plt.savefig(f'plots/var_{unitname}.png')
    plt.clf()


for u in range(len(units)):

    fe_histogram(data_0[u], data_1[u], 'NLOS', 'LOS', units[u], 50, str(u) + '_' + units[u])
    