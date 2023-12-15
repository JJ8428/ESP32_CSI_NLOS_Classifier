import csv
import matplotlib.pyplot as plt 
import numpy as np

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
df = '../NLOS_classification/data/LOS_data/session36.csv'
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

column_padding = 10
lltf_x = np.array([np.pad(column, (column_padding, 0), mode='constant', constant_values=0) for column in lltf_x.T]).T
htltf_x = np.array([np.pad(column, (column_padding, 0), mode='constant', constant_values=0) for column in htltf_x.T]).T
stbchtltf_x = np.array([np.pad(column, (column_padding, 0), mode='constant', constant_values=0) for column in stbchtltf_x.T]).T

modded_nyquist_frequency = 50 + int(column_padding/2)
lltf_CFR = np.fft.fft(np.abs(lltf_x), axis=0)[:modded_nyquist_frequency, :]
htltf_CFR = np.fft.fft(np.abs(htltf_x), axis=0)[:modded_nyquist_frequency, :]
stbchtltf_CFR = np.fft.fft(np.abs(stbchtltf_x), axis=0)[:modded_nyquist_frequency, :]

from scipy.stats import entropy


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

window_size = 6
lltf_cfr_ra = np.apply_along_axis(lambda col: np.convolve(col, np.ones(window_size)/window_size, mode='valid'), axis=0, arr=lltf_CFR)
htltf_cfr_ra = np.apply_along_axis(lambda col: np.convolve(col, np.ones(window_size)/window_size, mode='valid'), axis=0, arr=htltf_CFR)
stbchtltf_cfr_ra = np.apply_along_axis(lambda col: np.convolve(col, np.ones(window_size)/window_size, mode='valid'), axis=0, arr=stbchtltf_CFR)

# Plot the 2D matrix as a heatmap
plt.imshow(np.abs(stbchtltf_cfr_ra[10:, :]), cmap='hot', interpolation='nearest')
plt.colorbar(label='Elevation')
plt.title('Elevation Map')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Show the plot
plt.show()

lltf_CFR2 = np.fft.fft(np.abs(lltf_cfr_ra), axis=0)[:modded_nyquist_frequency, :]
htltf_CFR2 = np.fft.fft(np.abs(htltf_cfr_ra), axis=0)[:modded_nyquist_frequency, :]
stbchtltf_CFR2 = np.fft.fft(np.abs(stbchtltf_cfr_ra), axis=0)[:modded_nyquist_frequency, :]

cutoff = int(len(stbchtltf_CFR2[:, 0])/2)

for i in range(20):
    # plt.plot(np.abs(lltf_CFR2[5:cutoff, i]), label=f'LLTF CFR2 CI {i+1}')
    # plt.plot(np.abs(htltf_CFR2[5:cutoff, i]), label=f'HTLTF CFR2 CI {i+1}')
    plt.plot(np.abs(stbchtltf_CFR2[5:cutoff, i]), label=f'STBCHTLTF CFR2 CI {i+1}')
plt.show()
plt.clf()
