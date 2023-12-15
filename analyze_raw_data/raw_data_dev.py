import serial
import threading
import time

serial_port_esp = 'COM9'
duration = 30

folder_path = ''
file_name = 'csi_data.csv'
folder_path += file_name

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

esp_data = ''
cont_th = True
w = open(folder_path, 'w')
w.write(raw_col_header + '\n')


def get_esp_data(ser_esp):
    
    global esp_data
    global cont_th

    while cont_th:
        esp_data = ser_esp.readline().decode('utf-8').strip()

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
        
        w.write(esp_data + '\n')


if __name__ ==  '__main__':
    
    ser_esp = serial.Serial(serial_port_esp, 460800)
    esp_thread = threading.Thread(target=get_esp_data, args=(ser_esp,))
    esp_thread.daemon = True
    esp_thread.start()

    for a in range(duration, 0, -1):
        time.sleep(1)
        print(a)
    
    cont_th = False 
    esp_thread.join()

    w.close()
