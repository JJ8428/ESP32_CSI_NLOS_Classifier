import joblib
import numpy as np
import serial
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import threading
import time

from feature_extraction import *

serial_port_esp = '/dev/ttyACM0'
debug = True
duration = 20

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

'''
Adapted from:
https://github.com/espressif/esp-csi/blob/master/examples/get-started/tools/csi_data_read_parse.py
'''
# STBC-HT-LTF
stbchtltf_subcarrier_index = []
stbchtltf_subcarrier_index += [i for i in range(134, 162)]
stbchtltf_subcarrier_index += [i for i in range(163, 191)]
stbchtltf_subcarrier_len = len(stbchtltf_subcarrier_index)

hampel_conf_thres = 15
window_size = 50
hampel_window_size = 20
zero_padding = 7 
sparseness = 6 # TUNE (Adjust until time to run <= 10s with plenty of leeway. Recommended to be near 7s)

esp_data = ''
cont_th = True

stbchtltf_csi_array = np.zeros([window_size, stbchtltf_subcarrier_len], dtype=np.complex64)
rssi_array = np.zeros(window_size)
init1 = window_size + hampel_window_size
init2 = True
cooldown = sparseness
curr_pred = -1

scaler_fp = 'saves/scalers/'
stbchtltf_scaler = joblib.load(scaler_fp + 'stbchtltf_scaler.pkl')


def custom_accuracy(y_true, y_pred):

    y_pred_binary = K.round(y_pred)
    accuracy = K.mean(K.equal(y_true, y_pred_binary))

    return accuracy


stbchtltf_model = load_model('saves/models/stbchtltf_model.keras', custom_objects={'custom_accuracy': custom_accuracy})


def parse_esp_input(esp_input):

    index_find1 = esp_input.find('"')
    index_find2 = esp_input[index_find1+1:].find('"')
    data1 = esp_input[:index_find1-1].split(',')
    first_word = int(data1[first_word_invalid_index])
    rssi_data = int(data1[rssi_index])
    data2 = esp_input[index_find1 + 2:index_find1 + index_find2].split(',')
    raw_csi_data = [int(x) for x in data2]
    
    return first_word, rssi_data, raw_csi_data


def debug_clf(ser_esp):

    global esp_data
    global cont_th
    global init2
    global rssi_array
    global stbchtltf_csi_array
    global init1
    global hampel_window_size
    global hampel_conf_thres
    global cooldown
    global zero_padding
    global stbchtltf_scaler
    global stbchtltf_model
    global curr_pred

    esp_data = ser_esp # .readline().decode('utf-8').strip()
    if init2:
        init2 = False
        return
    try:
        first_word, rssi_data, raw_csi_data = parse_esp_input(esp_data)
        if first_word == 1:
            return
    except:
        print('Skipping line with Task/Watchdog error found in live feed')
        return
    rssi_array[:-1] = rssi_array[1:]
    stbchtltf_csi_array[:-1] = stbchtltf_csi_array[1:]
    rssi_array[-1] = rssi_data
    for i in range(stbchtltf_subcarrier_len):
        stbchtltf_csi_array[-1][i] = complex(raw_csi_data[stbchtltf_subcarrier_index[i] * 2 + 1], raw_csi_data[stbchtltf_subcarrier_index[i] * 2])
    if init1 > 0:
        init1 -= 1
    if init1 == 0:
        stbchtltf_csi_array = hampel_filter_correction(stbchtltf_csi_array, hampel_window_size, hampel_conf_thres)
        cooldown -= 1
        if cooldown == 0:
            cooldown = sparseness
            stbchtltf_x = feature_extraction(stbchtltf_csi_array, zero_padding, rssi_array)
            stbchtltf_x_transformed = stbchtltf_scaler.transform([stbchtltf_x])
            curr_pred = round(stbchtltf_model.predict(stbchtltf_x_transformed, verbose=0)[0][0])


def realtime_clf(ser_esp):

    global esp_data
    global cont_th
    global init2
    global rssi_array
    global stbchtltf_csi_array
    global init1
    global hampel_window_size
    global hampel_conf_thres
    global cooldown
    global zero_padding
    global stbchtltf_scaler
    global stbchtltf_model
    global curr_pred

    while cont_th:
        esp_data = ser_esp.readline().decode('utf-8').strip()
        if init2:
            init2 = False
            continue
        try:
            first_word, rssi_data, raw_csi_data = parse_esp_input(esp_data)
            if first_word == 1:
                continue
            new_csi_row = []
            for i in range(stbchtltf_subcarrier_len):
                new_csi_row.append(complex(raw_csi_data[stbchtltf_subcarrier_index[i] * 2 + 1], raw_csi_data[stbchtltf_subcarrier_index[i] * 2]))
        except:
            # print('Skipping line with Task/Watchdog error found in live feed')
            continue
        rssi_array[:-1] = rssi_array[1:]
        stbchtltf_csi_array[:-1] = stbchtltf_csi_array[1:]
        rssi_array[-1] = rssi_data
        stbchtltf_csi_array[-1] = new_csi_row
        if init1 > 0:
            init1 -= 1
        if init1 == 0:
            stbchtltf_csi_array = hampel_filter_correction(stbchtltf_csi_array, hampel_window_size, hampel_conf_thres)
            cooldown -= 1
            if cooldown == 0:
                cooldown = sparseness
                stbchtltf_x = feature_extraction(stbchtltf_csi_array, zero_padding, rssi_array)
                stbchtltf_x_transformed = stbchtltf_scaler.transform([stbchtltf_x])
                curr_pred = round(stbchtltf_model.predict(stbchtltf_x_transformed, verbose=0)[0][0])
                print(f'curr_pred: {curr_pred}')

if __name__ == '__main__':

    if False:
        start_time = time.time()    
        r = open('data/LOS_data/session52.csv', 'r')
        line_index = 0
        for line in r.readlines():
            debug_clf(line)
            print(f'{line_index}:\t{curr_pred}')
            line_index += 1
        r.close()
        fin_time = time.time() - start_time
        if fin_time <= 7.5:
            status = ':)'
        else: 
            status = ':('
        print(f'Time elapsed: {fin_time}, Status: {status}')
    else:
        ser_esp = serial.Serial(serial_port_esp, 460800)
        esp_thread = threading.Thread(target=realtime_clf, args=(ser_esp,))
        esp_thread.daemon = True
        esp_thread.start()

        for a in range(duration, 0, -1):
            time.sleep(1)
            print(a)

        cont_th = False 
        esp_thread.join()