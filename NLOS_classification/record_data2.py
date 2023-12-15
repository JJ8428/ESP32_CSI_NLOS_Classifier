import joblib
import json
import numpy as np
import os
import serial
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import threading
import time

from feature_extraction import *

csi_serial_port_esp = '/dev/ttyACM0'
ftm_serial_port_esp = '/dev/ttyUSB0'
duration = 10

ftm_dist = -1

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
sparseness = 8 # TUNE (Adjust until time to run <= 10s with plenty of leeway. Recommended to be near 7s)

ftm_esp_data = ''
csi_esp_data = ''
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


def realtime_clf(ser_esp):

    global csi_esp_data
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
        csi_esp_data = ser_esp.readline().decode('utf-8').strip()
        if init2:
            init2 = False
            continue
        try:
            first_word, rssi_data, raw_csi_data = parse_esp_input(csi_esp_data)
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
                curr_pred = stbchtltf_model.predict(stbchtltf_x_transformed, verbose=0)[0][0]


def get_esp_data(ser_esp):
    
    global ftm_esp_data
    global cont_th
    global ftm_dist

    while cont_th:
        ftm_esp_data = ser_esp.readline().decode('utf-8').strip()
        start_index = -1
        end_index = -1
        for a in range(len(ftm_esp_data)):
            if ftm_esp_data[a] == '{':
                start_index = a
            elif ftm_esp_data[a] == '}':
                end_index = a
                break
        if start_index != -1 and end_index != -1:
            ftm_json = json.loads(ftm_esp_data[start_index : end_index + 1])
            ftm_dist = ftm_json['est_dist']
            if ftm_json['ftm_fail']:
                print('FTM failure')


if __name__ == '__main__':

    # Modify this as necessary
    mode = 1 # 0: FTM, 1: CSI
    truth_dist = 17 / 3.28
    true_class = 1
    if true_class == 0:
        folder_path = 'data/NLOS_analysis_'
    else:
        folder_path = 'data/LOS_analysis_'

    if mode == 0:

        ftm_ser_esp = serial.Serial(ftm_serial_port_esp, 115200)
        ftm_esp_thread = threading.Thread(target=get_esp_data, args=(ftm_ser_esp,))
        ftm_esp_thread.daemon = True
        ftm_esp_thread.start()

        folder_path += 'FTM/'
        file_count = len(os.listdir(folder_path)) + 1
        file_name = f'session{file_count}.csv'
        folder_path += file_name
        start_time = time.time()
        w = open(folder_path, 'w')
        while True:
            elapsed_t = time.time() - start_time
            if elapsed_t >= 10:
                break
            time.sleep(.25)
            print(f'{ftm_dist}, {truth_dist}')
            if curr_pred != -1:
                w.write(f'{ftm_dist}, {truth_dist}\n')
        w.close()
        print('Saving data to ' + folder_path)
        cont_th = False
        ftm_esp_thread.join()

    else:

        csi_ser_esp = serial.Serial(csi_serial_port_esp, 460800)
        csi_esp_thread = threading.Thread(target=realtime_clf, args=(csi_ser_esp,))
        csi_esp_thread.daemon = True
        csi_esp_thread.start()

        folder_path += 'CSI/'
        file_count = len(os.listdir(folder_path)) + 1
        file_name = f'session{file_count}.csv'
        folder_path += file_name
        start_time = time.time()
        w = open(folder_path, 'w')
        while True:
            elapsed_t = time.time() - start_time
            if elapsed_t >= 10:
                break
            time.sleep(.25)
            print(f'{round(curr_pred)}, {curr_pred}, {true_class}')
            if curr_pred != -1:
                w.write(f'{curr_pred}, {true_class}\n')
        w.close()
        print('Saving data to ' + folder_path)
        cont_th = False
        csi_esp_thread.join()
