import csv
import math
import matplotlib.pyplot as plt 
import numpy as np
from scipy.stats import entropy

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

rssi_x = []
lltf_x = []
htltf_x = []
stbchtltf_x = []
y = []

line_count = 100
df = '../NLOS_classification/data/NLOS_data/session77.csv'
print('Parsing ' + df)
with open(df) as csv_file:
    reader = csv.reader(csv_file)
    init1 = True
    for esp_data in reader:
        if init1:
            init1 = False
            continue
        if int(esp_data[first_word_invalid_index]) == 1:
            continue
        try:
            rssi_data = int(esp_data[rssi_index])
            raw_csi_data = [int(x) for x in esp_data[data_index].replace('[', '').replace(']', '').split(',')]
            class_data = a
        except:
            print('Skipping line with Task/Watchdog error found in ' + df)
            continue
        lltf_data = []
        htltf_data = []
        stbchtltf_data = []
        for i in range(lltf_subcarrier_len):
            lltf_data.append(complex(raw_csi_data[lltf_subcarrier_index[i] * 2 + 1], raw_csi_data[lltf_subcarrier_index[i] * 2]))
        for i in range(htltf_subcarrier_len):
            htltf_data.append(complex(raw_csi_data[htltf_subcarrier_index[i] * 2 + 1], raw_csi_data[htltf_subcarrier_index[i] * 2]))
        for i in range(stbchtltf_subcarrier_len):
            stbchtltf_data.append(complex(raw_csi_data[stbchtltf_subcarrier_index[i] * 2 + 1], raw_csi_data[stbchtltf_subcarrier_index[i] * 2]))
        lltf_x.append(htltf_data)
        htltf_x.append(htltf_data)
        stbchtltf_x.append(stbchtltf_data)
        y.append(a)
        line_count -= 1
        if line_count == 0:
            break

lltf_x = np.array(lltf_x)
htltf_x = np.array(htltf_x)
stbchtltf_x = np.array(stbchtltf_x)

column_padding = int(100/8)
lltf_x = np.array([np.pad(column, (column_padding, 0), mode='constant', constant_values=0) for column in lltf_x.T]).T
htltf_x = np.array([np.pad(column, (column_padding, 0), mode='constant', constant_values=0) for column in htltf_x.T]).T
stbchtltf_x = np.array([np.pad(column, (column_padding, 0), mode='constant', constant_values=0) for column in stbchtltf_x.T]).T

modded_nyquist_frequency = 50 + int(column_padding/2)
lltf_CFR = np.fft.fft(np.abs(lltf_x), axis=0)[:modded_nyquist_frequency, :]
htltf_CFR = np.fft.fft(np.abs(htltf_x), axis=0)[:modded_nyquist_frequency, :]
stbchtltf_CFR = np.fft.fft(np.abs(stbchtltf_x), axis=0)[:modded_nyquist_frequency, :]

lltf_CFR2 = np.fft.fft(np.abs(lltf_CFR), axis=0)
htltf_CFR2 = np.fft.fft(np.abs(htltf_CFR), axis=0)
stbchtltf_CFR2 = np.fft.fft(np.abs(stbchtltf_CFR), axis=0)


def CFR_entropy(cfr_mag_matrix):
    
    entropy_values = []
    for Tcol in cfr_mag_matrix.T:
        probabilities = Tcol / np.sum(Tcol)
        spectral_entropy = entropy(probabilities, base=2)
        entropy_values.append(spectral_entropy)

    return np.mean(entropy_values)


def CFR_spectral_bandwidth(cfr_mag_matrix):

    bandwidth_values = []
    for Tcol in cfr_mag_matrix.T:
        centroid = np.sum(Tcol * np.arange(len(Tcol))) / np.sum(Tcol)
        bandwidth = np.sqrt(np.sum(Tcol * (np.arange(len(Tcol)) - centroid)**2) / np.sum(Tcol))
        bandwidth_values.append(bandwidth)

    return np.mean(bandwidth_values)

def smoothness_score_matrix(matrix):

    # Compute the first derivative along the rows (axis=0)
    first_derivative = np.diff(matrix, axis=0)

    # Compute the standard deviation along the rows (axis=0) and then calculate the mean
    mean_smoothness_score = np.mean(np.std(first_derivative, axis=0))

    return mean_smoothness_score


print(smoothness_score_matrix(stbchtltf_CFR2))
print(smoothness_score_matrix(stbchtltf_CFR))
print()
print(CFR_entropy(np.abs(stbchtltf_CFR2)))
print(CFR_entropy(np.abs(stbchtltf_CFR)))
print()
print(CFR_spectral_bandwidth(np.abs(stbchtltf_CFR2)))
print(CFR_spectral_bandwidth(np.abs(stbchtltf_CFR)))


if True:
    for i in range(30):
        # plt.plot(np.abs(lltf_CFR[10:, i]), label=f'LLTF CFR CI {i+1}')
        # plt.plot(np.abs(htltf_CFR[10:, i]), label=f'HTLTF CFR CI {i+1}')
        plt.plot(np.log(np.abs(stbchtltf_CFR[:, i]) + 1), label=f'STBCHTLTF CFR CI {i+1}')
        # plt.plot(np.abs(lltf_CFR2[10:27, i]), label=f'LLTF CFR2 CI {i+1}')
        # plt.plot(np.abs(htltf_CFR2[10:27, i]), label=f'HTLTF CFR2 CI {i+1}')
        # plt.plot(np.abs(stbchtltf_CFR2[5:27, i]), label=f'STBCHTLTF CFR2 CI {i+1}')
    plt.show()
    plt.clf()

#  np.log(np.abs(CFR_data)[trim:] + 1)


trimmed_CIR_index = column_padding
# lltf_CIR = np.fft.ifft(np.abs(lltf_CFR), axis=0)
# htltf_CIR = np.fft.ifft(np.abs(htltf_CFR), axis=0)
# stbchtltf_CIR = np.fft.ifft(np.abs(stbchtltf_CFR), axis=0)
lltf_CIR = np.fft.ifft((lltf_CFR), axis=0)[trimmed_CIR_index:]
htltf_CIR = np.fft.ifft((htltf_CFR), axis=0)[trimmed_CIR_index:]
stbchtltf_CIR = np.fft.ifft((stbchtltf_CFR), axis=0)[trimmed_CIR_index:]

if True:
    for i in range(30):
        # plt.plot(np.abs(lltf_CIR[:, i]), label=f'LLTF CIR CI {i+1}')
        # plt.plot(np.abs(htltf_CIR[:, i]), label=f'HTLTF CIR CI {i+1}')
        plt.plot(np.abs(stbchtltf_CIR[:, i]), label=f'STBCHTLTF CIR CI {i+1}')
        # plt.plot(np.abs(lltf_x[:, i]), label='lltf_x')
        # plt.plot(np.abs(htltf_x[:, i]), label='htltf_x')
        # plt.plot(np.abs(stbchtltf_x[:, i]), label='stbchtltf_x')
    plt.legend()
    plt.show()
    plt.clf()
