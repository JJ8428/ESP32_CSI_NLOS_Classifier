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
dataset_path = '../NLOS_classification/data/NLOS_data/session03.csv'

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

cir = np.fft.ifft(stbchtltf_csi_array, axis=1)
flattened_cir = np.ravel(cir[-15:][::-1])

x = [x for x in range(flattened_cir.shape[0])]
y = np.abs(flattened_cir.tolist())
plt.plot(x, y, label=f'STBC-HT-LTF CIR)')
plt.show()

for i in range(25):
    x = [x for x in range(len(cir[i]))]
    y = np.abs(cir[i].tolist())
    plt.plot(x, y, label=f'STBC-HT-LTF CIR)')
plt.show()