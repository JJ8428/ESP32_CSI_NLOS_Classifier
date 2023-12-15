import csv
import matplotlib.pyplot as plt 
import numpy as np
import os

# Extract column indices
raw_col_header = 'type,id,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,secondary_channel,local_timestamp,ant,sig_len,rx_state,len,first_word,data'
raw_col_header += ',class'
cols = raw_col_header.split(',')
col_count = len(cols)
rssi_index = -1
first_word_invalid_index = -1
data_index = -1
for a in range(0, len(cols)):
    if cols[a] == 'rssi':
        rssi_index = a
    elif cols[a] == 'first_word':
        first_word_invalid_index = a
    elif cols[a] == 'data':
        data_index = a

'''
Adapted from:
https://github.com/espressif/esp-csi/blob/master/examples/get-started/tools/csi_data_read_parse.py
'''
# LLTF
lltf_subcarrier_index = []
lltf_subcarrier_index += [i for i in range(6, 32)]
lltf_subcarrier_index += [i for i in range(33, 59)]
lltf_subcarrier_len = len(lltf_subcarrier_index)

# HT-LTF
htltf_subcarrier_index = []
htltf_subcarrier_index += [i for i in range(66, 94)]
htltf_subcarrier_index += [i for i in range(95, 123)]
htltf_subcarrier_len = len(htltf_subcarrier_index)

# STBC-HT-LTF
stbchtltf_subcarrier_index = []
stbchtltf_subcarrier_index += [i for i in range(134, 162)]
stbchtltf_subcarrier_index += [i for i in range(163, 191)]
stbchtltf_subcarrier_len = len(stbchtltf_subcarrier_index)

# dataset_path = 'csi_data.csv'
dataset_path = '../NLOS_classification/data/NLOS_data/session75.csv'

# Read data from files
window_size = 50
rssi_array = np.zeros(window_size)
lltf_csi_array = np.zeros([window_size, lltf_subcarrier_len], dtype=np.complex64)
htltf_csi_array = np.zeros([window_size, htltf_subcarrier_len], dtype=np.complex64)
stbchtltf_csi_array = np.zeros([window_size, stbchtltf_subcarrier_len], dtype=np.complex64)
with open(dataset_path) as csv_file:
    reader = csv.reader(csv_file)
    init = True
    for esp_data in reader:
        if init:
            init = False
            continue
        if int(esp_data[first_word_invalid_index]) == 1:
            continue
        try:
            rssi_data = int(esp_data[rssi_index])
            raw_csi_data = [int(x) for x in esp_data[data_index].replace('[', '').replace(']', '').split(',')]
        except:
            print('Skipping line with Task/Watchdog error found in ' + dataset_path)
            continue
        rssi_array[:-1] = rssi_array[1:]
        lltf_csi_array[:-1] = lltf_csi_array[1:]
        htltf_csi_array[:-1] = htltf_csi_array[1:]
        stbchtltf_csi_array[:-1] = stbchtltf_csi_array[1:]

        rssi_array[-1] = rssi_data
        for i in range(lltf_subcarrier_len):
            lltf_csi_array[-1][i] = complex(raw_csi_data[lltf_subcarrier_index[i] * 2 + 1], raw_csi_data[lltf_subcarrier_index[i] * 2])
        for i in range(htltf_subcarrier_len):
            htltf_csi_array[-1][i] = complex(raw_csi_data[htltf_subcarrier_index[i] * 2 + 1], raw_csi_data[htltf_subcarrier_index[i] * 2])
        for i in range(stbchtltf_subcarrier_len):
            stbchtltf_csi_array[-1][i] = complex(raw_csi_data[stbchtltf_subcarrier_index[i] * 2 + 1], raw_csi_data[stbchtltf_subcarrier_index[i] * 2])

    
# per timestep
def phase_sanitisation(csi_matrix):

    R = np.abs(csi_matrix)
    phase_matrix = np.unwrap(np.angle(csi_matrix), axis=1)
    fit_X = np.arange(0, phase_matrix.shape[1])
    tau_array = []
    phase_mean_array = []
    for m in range(phase_matrix.shape[0]):
        fit_Y = phase_matrix[m]
        tau_t = np.polyfit(fit_X, fit_Y, 1)[0]
        phase_mean_t = np.mean(phase_matrix[m])
        tau_array.append(tau_t)
        phase_mean_array.append(phase_mean_t)
        for n in range(phase_matrix.shape[1]):
            phase_matrix[m][n] -= (((n + 1) * tau_t) + 0)
    csi_matrix = R * np.exp(1j * phase_matrix)

    return csi_matrix, phase_matrix, np.mean(tau_array), np.mean(phase_mean_array)

adjusted_lltf_phase = phase_sanitisation(lltf_csi_array)[1]
adjusted_htltf_phase = phase_sanitisation(htltf_csi_array)[1]
adjusted_stbchtltf_phase = phase_sanitisation(stbchtltf_csi_array)[1]

for file in os.listdir('plots/'):
    try:
        os.remove('plots/' + file)
    except:
        continue

frames = 20
for time_stamp in range(frames): # B
    x1 = [x for x in range(lltf_csi_array.shape[1])]
    y1_1 = np.unwrap(np.angle(lltf_csi_array[time_stamp].tolist()))
    x2 = [x for x in range(htltf_csi_array.shape[1])]
    y2_1 = np.unwrap(np.angle(htltf_csi_array[time_stamp].tolist()))
    x3 = [x for x in range(stbchtltf_csi_array.shape[1])]
    y3_1 = np.unwrap(np.angle(stbchtltf_csi_array[time_stamp].tolist()))
    y1_2 = adjusted_lltf_phase[time_stamp].tolist()
    y2_2 = adjusted_htltf_phase[time_stamp].tolist()
    y3_2 = adjusted_stbchtltf_phase[time_stamp].tolist()
    plt.plot(x1, y1_1, label=f'raw LLTF ({lltf_subcarrier_len})')
    plt.plot(x2, y2_1, label=f'raw HTLTF ({htltf_subcarrier_len})')
    plt.plot(x3, y3_1, label=f'raw STBC-HT-LTF ({stbchtltf_subcarrier_len})')
    plt.plot(x1, y1_2, label=f'LLTF adjusted ({lltf_subcarrier_len})') 
    plt.plot(x2, y2_2, label=f'HTLTF adjsuted ({htltf_subcarrier_len})')
    plt.plot(x3, y3_2, label=f'STBC-HT-LTF adjusted ({stbchtltf_subcarrier_len})') 
    plt.title(f'Phase at time \'t\' = {time_stamp}')
    plt.xlabel('Subcarrier Index') 
    plt.ylabel('Unwrapped Phase') 
    plt.legend()
   
    plt.savefig(f'plots/PS_exp2_t{time_stamp}.png')
    plt.clf()

for carrier_index in range(lltf_csi_array.shape[1]): # A
    x = [x for x in range(lltf_csi_array.shape[0])]
    y1_1 = np.angle(lltf_csi_array[:, carrier_index])
    y2_1 = np.angle(htltf_csi_array[:, carrier_index])
    y3_1 = np.angle(stbchtltf_csi_array[:, carrier_index])
    y1_2 = adjusted_lltf_phase[:, carrier_index]
    y2_2 = adjusted_htltf_phase[:, carrier_index]
    y3_2 = adjusted_stbchtltf_phase[:, carrier_index]
    # plt.plot(x, y1_1, label=f'LLTF ({lltf_subcarrier_len})')
    # plt.plot(x, y2_1, label=f'HTLTF ({htltf_subcarrier_len})')
    # plt.plot(x, y3_1, label=f'STBC-HT-LTF ({stbchtltf_subcarrier_len})')
    plt.plot(x, y1_2, label=f'LLTF adjusted ({lltf_subcarrier_len})')
    plt.plot(x, y2_2, label=f'HTLTF adjusted ({htltf_subcarrier_len})')
    plt.plot(x, y3_2, label=f'STBC-HT-LTF adjusted ({stbchtltf_subcarrier_len})')
    plt.title(f'Unwrapped Phase at Carrier Index {carrier_index}')
    plt.xlabel('Timestamp')
    plt.ylabel('Unwrapped Phase')
    plt.legend()
    plt.savefig(f'plots/PS_exp3_ci{carrier_index}.png')
    plt.clf()