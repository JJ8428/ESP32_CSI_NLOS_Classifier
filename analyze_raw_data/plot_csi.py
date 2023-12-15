import matplotlib.pyplot as plt 
import numpy as np
import os
from scipy.signal import argrelextrema


# Function to calculate rolling average
def rolling_average(row, window_size):

    return np.convolve(row, np.ones(window_size) / window_size, mode='valid')


def count_peaks_minima(row):
    # Find the indices of relative maxima and minima
    peaks = argrelextrema(row, np.greater)[0]
    minima = argrelextrema(row, np.less)[0]
    
    # Return the counts
    return len(peaks), len(minima)


raw_col_header = 'type,id,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,secondary_channel,local_timestamp,ant,sig_len,rx_state,len,first_word,data'
cols = raw_col_header.split(',')
col_count = len(cols)
rssi_index = -1
first_word_invalid_index = -1
for a in range(0, len(cols)):
    if cols[a] == 'rssi':
        rssi_index = a
    elif cols[a] == 'first_index':
        first_word_invalid_index = a
    if rssi_index != -1 and first_word_invalid_index != -1:
        break

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

CSI_DATA_INDEX = 100  # buffer size

lltf_csi_array = np.zeros([CSI_DATA_INDEX, lltf_subcarrier_len], dtype=np.complex64)
htltf_csi_array = np.zeros([CSI_DATA_INDEX, htltf_subcarrier_len], dtype=np.complex64)
stbchtltf_csi_array = np.zeros([CSI_DATA_INDEX, stbchtltf_subcarrier_len], dtype=np.complex64)

r = open('../NLOS_classification/data/LOS_data/session93.csv')
r.readline()
lines = CSI_DATA_INDEX
for esp_data in r.readlines():
    lines -= 1
    data_index = -1
    for a in range(len(esp_data)):
        if esp_data[a] == '"':
            data_index = a
            break        
    if data_index == -1 or esp_data[:data_index].count(',') != col_count - 1:
        continue
    conn_data = esp_data[:data_index].split(',')
    if conn_data[first_word_invalid_index] == 1:
        continue
    rssi_data_str = str(int(conn_data[rssi_index]))

    try:
        raw_csi_data = [int(x) for x in esp_data[data_index:].replace('"[', '').replace(']"', '').split(',')]
    except:
        continue

    lltf_csi_array[:-1] = lltf_csi_array[1:]
    htltf_csi_array[:-1] = htltf_csi_array[1:]
    stbchtltf_csi_array[:-1] = stbchtltf_csi_array[1:]

    for i in range(lltf_subcarrier_len):
        lltf_csi_array[-1][i] = complex(raw_csi_data[lltf_subcarrier_index[i] * 2 + 1], raw_csi_data[lltf_subcarrier_index[i] * 2])
    for i in range(htltf_subcarrier_len):
        htltf_csi_array[-1][i] = complex(raw_csi_data[htltf_subcarrier_index[i] * 2 + 1], raw_csi_data[htltf_subcarrier_index[i] * 2])
    for i in range(stbchtltf_subcarrier_len):
        stbchtltf_csi_array[-1][i] = complex(raw_csi_data[stbchtltf_subcarrier_index[i] * 2 + 1], raw_csi_data[stbchtltf_subcarrier_index[i] * 2])
    if lines == 0:
        break

for file in os.listdir('plots/'):
    try:
        os.remove('plots/' + file)
    except:
        continue

window_size_t = 8
window_size_i = 15
# _t: for a specific time 't'
# _i: for each carrier index 'i'
smooth_htltf_t = np.apply_along_axis(rolling_average, axis=1, arr=np.abs(htltf_csi_array), window_size=window_size_t)
smooth_stbchtltf_t = np.apply_along_axis(rolling_average, axis=1, arr=np.abs(stbchtltf_csi_array), window_size=window_size_t)
htltf_counts_t = np.apply_along_axis(count_peaks_minima, axis=1, arr=smooth_htltf_t)
stbchtltf_counts_t = np.apply_along_axis(count_peaks_minima, axis=1, arr=smooth_stbchtltf_t)
smooth_htltf_i = np.apply_along_axis(rolling_average, axis=0, arr=np.abs(htltf_csi_array), window_size=window_size_i)
smooth_stbchtltf_i = np.apply_along_axis(rolling_average, axis=0, arr=np.abs(stbchtltf_csi_array), window_size=window_size_i)

print("Number of Peaks per Row:")
print(np.mean(htltf_counts_t[:, 0]))
print(np.mean(stbchtltf_counts_t[:, 0]))

print()

print("Number of Minima per Row:")
print(np.mean(htltf_counts_t[:, 1]))
print(np.mean(stbchtltf_counts_t[:, 1]))

for ci in range(lltf_csi_array.shape[1]):
    x1 = [x for x in range(lltf_csi_array.shape[0])]
    y1 = np.abs(lltf_csi_array[:, ci].tolist())
    x2 = [x for x in range(htltf_csi_array.shape[0])]
    y2 = np.abs(htltf_csi_array[:, ci].tolist())
    x3 = [x for x in range(stbchtltf_csi_array.shape[0])]
    y3 = np.abs(stbchtltf_csi_array[:, ci].tolist())
    x4 = [x * htltf_csi_array.shape[0]/smooth_htltf_i.shape[0] for x in range(smooth_htltf_i.shape[0])]
    y4 = np.abs(smooth_htltf_i[:, ci].tolist())
    x5 = [x * stbchtltf_csi_array.shape[0]/smooth_stbchtltf_i.shape[0] for x in range(smooth_stbchtltf_i.shape[0])]
    y5 = np.abs(smooth_stbchtltf_i[:, ci].tolist())
    plt.plot(x1, y1, label=f'LLTF ({lltf_subcarrier_len})') 
    plt.plot(x2, y2, label=f'HTLTF ({htltf_subcarrier_len})')
    plt.plot(x4, y4, label=f'smooth HTLTF ({htltf_subcarrier_len})')
    plt.plot(x3, y3, label=f'STBC-HT-LTF ({stbchtltf_subcarrier_len})')
    plt.plot(x5, y5, label=f'smooth STBC-HT-LTF ({stbchtltf_subcarrier_len})')
    plt.title(f'Amplitudes at index \'t\' = {ci}')
    plt.xlabel('Packet #') 
    plt.ylabel('Amplitude') 
    plt.legend()
    plt.savefig(f'plots/CSI_carriers_ci{ci}.png')
    plt.clf()

frames = 50
for time_stamp in range(frames):
    x1 = [x for x in range(lltf_csi_array.shape[1])]
    y1 = np.abs(lltf_csi_array[time_stamp].tolist())
    x2 = [x for x in range(htltf_csi_array.shape[1])]
    y2 = np.abs(htltf_csi_array[time_stamp].tolist())
    x3 = [x for x in range(stbchtltf_csi_array.shape[1])]
    y3 = np.abs(stbchtltf_csi_array[time_stamp].tolist())
    x4 = [x * htltf_csi_array.shape[1]/smooth_htltf_t.shape[1] for x in range(smooth_htltf_t.shape[1])]
    y4 = smooth_htltf_t[time_stamp]
    x5 = [x * stbchtltf_csi_array.shape[1]/smooth_stbchtltf_t.shape[1] for x in range(smooth_stbchtltf_t.shape[1])]
    y5 = smooth_stbchtltf_t[time_stamp]
    plt.subplot(121)
    plt.plot(x1, y1, label=f'LLTF ({lltf_subcarrier_len})') 
    plt.plot(x2, y2, label=f'HTLTF ({htltf_subcarrier_len})')
    plt.plot(x4, y4, label=f'smooth HTLTF ({htltf_subcarrier_len})')
    plt.plot(x3, y3, label=f'STBC-HT-LTF ({stbchtltf_subcarrier_len})')
    plt.plot(x5, y5, label=f'smooth STBC-HT-LTF ({stbchtltf_subcarrier_len})')
    plt.title(f'Amplitudes at time \'t\' = {time_stamp}')
    plt.xlabel('Subcarrier Index') 
    plt.ylabel('Amplitude') 
    plt.legend()

    sample_rate = 100 
    fft_result_lltf = np.fft.fft(lltf_csi_array[time_stamp].tolist())
    frequencies_lltf = np.fft.fftfreq(len(fft_result_lltf), 1 / sample_rate)
    freq_lltf_sorted_indices = np.argsort(frequencies_lltf)
    frequencies_lltf = frequencies_lltf[freq_lltf_sorted_indices]
    fft_result_lltf = fft_result_lltf[freq_lltf_sorted_indices]
    fft_result_htltf = np.fft.fft(htltf_csi_array[time_stamp].tolist())
    frequencies_htltf = np.fft.fftfreq(len(fft_result_htltf), 1 / sample_rate)
    freq_htltf_sorted_indices = np.argsort(frequencies_htltf)
    frequencies_htltf = frequencies_htltf[freq_htltf_sorted_indices]
    fft_result_htltf = fft_result_htltf[freq_htltf_sorted_indices]
    fft_result_stbchtltf = np.fft.fft(stbchtltf_csi_array[time_stamp].tolist())
    frequencies_stbchtltf = np.fft.fftfreq(len(fft_result_stbchtltf), 1 / sample_rate)
    freq_stbchtltf_sorted_indices = np.argsort(frequencies_stbchtltf)
    frequencies_stbchtltf = frequencies_stbchtltf[freq_stbchtltf_sorted_indices]
    fft_result_stbchtltf = fft_result_stbchtltf[freq_stbchtltf_sorted_indices]
    plt.subplot(122)
    plt.plot(frequencies_lltf, np.abs(fft_result_lltf), label='LLTF')
    plt.plot(frequencies_htltf, np.abs(fft_result_htltf), label='HTLTF', linestyle='--')
    plt.plot(frequencies_stbchtltf, np.abs(fft_result_stbchtltf), label='STBC-HT-LTF', linestyle='--')
    plt.title('Frequency Domain Signal')
    plt.xlabel('Subcarrier Index') 
    plt.ylabel('Amplitude') 
    plt.legend()

    plt.savefig(f'plots/CSI_carriers_FFT_t{time_stamp}.png')
    plt.clf()
