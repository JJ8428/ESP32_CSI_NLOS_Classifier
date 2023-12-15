import csv
import joblib
import numpy as np
import os
import random
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, PowerTransformer
from tensorflow.keras import backend as K
from tensorflow.keras import models, layers, callbacks
from tensorflow.keras.regularizers import l1, l2, l1_l2

import time

from k_fold_fp import *
from feature_extraction import *

# Extract column indices
raw_col_header = 'type,id,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,secondary_channel,local_timestamp,ant,sig_len,rx_state,len,first_word,data'
raw_col_header += ',class'
cols = raw_col_header.split(',')
col_count = len(cols)
rssi_index = -1
first_word_invalid_index = -1
data_index = -1
class_index = -1
for a in range(0, len(cols)):
    if cols[a] == 'rssi':
        rssi_index = a
    elif cols[a] == 'first_word':
        first_word_invalid_index = a
    elif cols[a] == 'data':
        data_index = a
    elif cols[a] == 'class':
        class_index = a

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

# Save processed data into dictionary with filepaths as keys
fp_index_limit = 0 # TUNE
dataset_folder_paths = ['data/NLOS_data/', 'data/LOS_data/']
# lltf_dict = {}
htltf_dict = {}
stbchtltf_dict = {}
y_dict = {}

hampel_conf_thres = 15 # TUNE
window_size = 50 # TUNE
hampel_window_size = 20 # TUNE
zero_padding = 7 # TUNE
sparseness = 2 # TUNE

process_data = False # TUNE
if process_data:
    filepaths_array = []
    for a in range(len(dataset_folder_paths)):
        dataset_files = [f for f in os.listdir(dataset_folder_paths[a])][fp_index_limit:]
        for df in dataset_files:
            init1 = window_size + hampel_window_size
            # lltf_x = []
            # htltf_x = []
            stbchtltf_x = []
            y = []
            rssi_array = np.zeros(window_size)
            # lltf_csi_array = np.zeros([window_size, lltf_subcarrier_len], dtype=np.complex64)
            # htltf_csi_array = np.zeros([window_size, htltf_subcarrier_len], dtype=np.complex64)
            stbchtltf_csi_array = np.zeros([window_size, stbchtltf_subcarrier_len], dtype=np.complex64)
            print(f'Begin Parsing {dataset_folder_paths[a] + df}')
            start_time = time.time()
            cooldown = sparseness
            with open(dataset_folder_paths[a] + df) as csv_file:
                reader = csv.reader(csv_file)
                init2 = True
                for esp_data in reader:
                    if init2:
                        init2 = False
                        continue
                    if int(esp_data[first_word_invalid_index]) == 1:
                        continue
                    try:
                        rssi_data = int(esp_data[rssi_index])
                        raw_csi_data = [int(x) for x in esp_data[data_index].replace('[', '').replace(']', '').split(',')]
                        class_data = int(esp_data[class_index])
                    except:
                        print('Skipping line with Task/Watchdog error found in ' + dataset_folder_paths[a] + df)
                        continue
                    rssi_array[:-1] = rssi_array[1:]
                    # lltf_csi_array[:-1] = lltf_csi_array[1:]
                    # htltf_csi_array[:-1] = htltf_csi_array[1:]
                    stbchtltf_csi_array[:-1] = stbchtltf_csi_array[1:]
                    rssi_array[-1] = rssi_data
                    '''
                    for i in range(lltf_subcarrier_len):
                        lltf_csi_array[-1][i] = complex(raw_csi_data[lltf_subcarrier_index[i] * 2 + 1], raw_csi_data[lltf_subcarrier_index[i] * 2])
                    for i in range(htltf_subcarrier_len):
                        htltf_csi_array[-1][i] = complex(raw_csi_data[htltf_subcarrier_index[i] * 2 + 1], raw_csi_data[htltf_subcarrier_index[i] * 2])
                    '''
                    for i in range(stbchtltf_subcarrier_len):
                        stbchtltf_csi_array[-1][i] = complex(raw_csi_data[stbchtltf_subcarrier_index[i] * 2 + 1], raw_csi_data[stbchtltf_subcarrier_index[i] * 2])
                    if init1 > 0:
                        init1 -= 1            
                    if init1 == 0:
                        # lltf_csi_array = hampel_filter_correction(lltf_csi_array, hampel_window_size, hampel_conf_thres)
                        # htltf_csi_array = hampel_filter_correction(htltf_csi_array, hampel_window_size, hampel_conf_thres)
                        stbchtltf_csi_array = hampel_filter_correction(stbchtltf_csi_array, hampel_window_size, hampel_conf_thres)
                        cooldown -= 1
                        if cooldown == 0:
                            cooldown = sparseness
                            # lltf_x.append(feature_extraction(lltf_csi_array, zero_padding, rssi_array))
                            # htltf_x.append(feature_extraction(htltf_csi_array, zero_padding, rssi_array))
                            stbchtltf_x.append(feature_extraction(stbchtltf_csi_array, zero_padding, rssi_array))
                            # Debug with this!!! (TUNE)
                            # lltf_x.append([a, a, a, a, a, a])
                            # htltf_x.append([a, a, a, a, a, a])
                            # stbchtltf_x.append([a, a, a, a, a, a])
                            y.append(a)
            # lltf_dict[dataset_folder_paths[a] + df] = lltf_x
            # htltf_dict[dataset_folder_paths[a] + df] = htltf_x
            stbchtltf_dict[dataset_folder_paths[a] + df] = stbchtltf_x
            y_dict[dataset_folder_paths[a] + df] = y
            iteration_time = time.time() - start_time
            print(f'End Parsing {iteration_time} s') # Future Goal: Reduce time to less than 10 s for real time application

    # Save the dictionary to a numpy files
    print('Saving processed data...')
    # np.save('saves/processed_data/NN_lltf_feature_extraction_data.npy', lltf_dict)
    # np.save('saves/processed_data/NN_htltf_feature_extraction_data.npy', htltf_dict)
    np.save('saves/processed_data/NN_stbchtltf_feature_extraction_data.npy', stbchtltf_dict)
    np.save('saves/processed_data/NN_y.npy', y_dict)
else:
    print('Loading processed data...')
    # lltf_dict = np.load('saves/processed_data/NN_lltf_feature_extraction_data.npy', allow_pickle=True).item()
    # htltf_dict = np.load('saves/processed_data/NN_htltf_feature_extraction_data.npy', allow_pickle=True).item()
    stbchtltf_dict = np.load('saves/processed_data/NN_stbchtltf_feature_extraction_data.npy', allow_pickle=True).item()
    y_dict = np.load('saves/processed_data/NN_y.npy', allow_pickle=True).item()

'''
def drop_nth_column_from_dict(dictionary, n):
    new_dictionary = {}
    for key, matrix in dictionary.items():
        new_matrix = [row[:n] + row[n+1:] for row in matrix]
        new_dictionary[key] = new_matrix

    return new_dictionary

# Worth dropping: var_rssi, cir_var_cc, *t_rms (2), cir_var_t_med, cir_var_var_idp, cir_var_kurtosis, cir_mean_kurtosis, cir_mean_eps_h
'''

# lltf_accuracy = []
# lltf_accuracy_class_1 = []
# lltf_accuracy_class_0 = []
# tltf_accuracy = []
# htltf_accuracy_class_1 = []
# htltf_accuracy_class_0 = []
stbchtltf_accuracy = []
stbchtltf_accuracy_class_1 = []
stbchtltf_accuracy_class_0 = []

random.seed(314)

# Get filepaths and shuffle
dataset_filepaths0 = [f for f in os.listdir(dataset_folder_paths[0])][fp_index_limit:]
dataset_filepaths0 = random.sample(dataset_filepaths0, len(dataset_filepaths0))
dataset_filepaths1 = [f for f in os.listdir(dataset_folder_paths[1])][fp_index_limit:]
dataset_filepaths1 = random.sample(dataset_filepaths1, len(dataset_filepaths1))

# Train our scalers
# lltf_x_all = []
# htltf_x_all = []
stbchtltf_x_all = []
for df in dataset_filepaths0:
    # lltf_x_all += lltf_dict[dataset_folder_paths[0] + df]
    # htltf_x_all += htltf_dict[dataset_folder_paths[0] + df]
    stbchtltf_x_all += stbchtltf_dict[dataset_folder_paths[0] + df]
for df in dataset_filepaths1:
    # lltf_x_all += lltf_dict[dataset_folder_paths[1] + df]
    # htltf_x_all += htltf_dict[dataset_folder_paths[1] + df]
    stbchtltf_x_all += stbchtltf_dict[dataset_folder_paths[1] + df]

scale_type = 0 # TUNE
if scale_type == 0:
    # lltf_scaler = StandardScaler()
    # htltf_scaler = StandardScaler()
    stbchtltf_scaler = StandardScaler()
else:
    # lltf_scaler = PowerTransformer()
    # htltf_scaler = PowerTransformer()
    stbchtltf_scaler = PowerTransformer()

# lltf_scaler.fit(lltf_x_all)
# htltf_scaler.fit(htltf_x_all)
stbchtltf_scaler.fit(stbchtltf_x_all)

scaler_fp = 'saves/scalers/'
# joblib.dump(lltf_scaler, scaler_fp + 'lltf_scaler.pkl')
# joblib.dump(htltf_scaler, scaler_fp + 'htltf_scaler.pkl')
joblib.dump(stbchtltf_scaler, scaler_fp + 'stbchtltf_scaler.pkl')

# K-fold cross validation
k = 8 # TUNE
for n in range(k):
    
    # Validation set come from separate recordings
    train0, val0 = k_fold_cv_assist(dataset_filepaths0, k, n)
    train1, val1 = k_fold_cv_assist(dataset_filepaths1, k, n)

    # lltf_x_train = []
    # htltf_x_train = []
    stbchtltf_x_train = []
    y_train = []
    for df in train0:
        # lltf_x_train += lltf_dict[dataset_folder_paths[0] + df]
        # htltf_x_train += htltf_dict[dataset_folder_paths[0] + df]
        stbchtltf_x_train += stbchtltf_dict[dataset_folder_paths[0] + df]
        y_train += y_dict[dataset_folder_paths[0] + df]
    for df in train1:
        # lltf_x_train += lltf_dict[dataset_folder_paths[1] + df]
        # htltf_x_train += htltf_dict[dataset_folder_paths[1] + df]
        stbchtltf_x_train += stbchtltf_dict[dataset_folder_paths[1] + df]
        y_train += y_dict[dataset_folder_paths[1] + df]
    # lltf_x_train = np.array(lltf_x_train)
    # htltf_x_train = np.array(htltf_x_train)
    stbchtltf_x_train = np.array(stbchtltf_x_train)
    y_train = np.array(y_train)

    # lltf_x_val = []
    # htltf_x_val = []
    stbchtltf_x_val = []
    y_val = []
    for df in val0:
        # lltf_x_val += lltf_dict[dataset_folder_paths[0] + df]
        # htltf_x_val += htltf_dict[dataset_folder_paths[0] + df]
        stbchtltf_x_val += stbchtltf_dict[dataset_folder_paths[0] + df]
        y_val += y_dict[dataset_folder_paths[0] + df]
    for df in val1:
        # lltf_x_val += lltf_dict[dataset_folder_paths[1] + df]
        # htltf_x_val += htltf_dict[dataset_folder_paths[1] + df]
        stbchtltf_x_val += stbchtltf_dict[dataset_folder_paths[1] + df]
        y_val += y_dict[dataset_folder_paths[1] + df]
    # lltf_x_val= np.array(lltf_x_val)
    # htltf_x_val= np.array(htltf_x_val)
    stbchtltf_x_val= np.array(stbchtltf_x_val)
    y_val = np.array(y_val)

    # Scale the datasets
    # lltf_x_train = lltf_scaler.transform(lltf_x_train)
    # lltf_x_val = lltf_scaler.transform(lltf_x_val)
    # htltf_x_train = htltf_scaler.transform(htltf_x_train)
    # htltf_x_val = htltf_scaler.transform(htltf_x_val)
    stbchtltf_x_train = stbchtltf_scaler.transform(stbchtltf_x_train)
    stbchtltf_x_val = stbchtltf_scaler.transform(stbchtltf_x_val)


    def custom_accuracy(y_true, y_pred):

        y_pred_binary = K.round(y_pred)
        accuracy = K.mean(K.equal(y_true, y_pred_binary))

        return accuracy
    

    # Define callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=8, # TUNE
        restore_best_weights=True
    )
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    # Common hyperparameters
    input_length = len(stbchtltf_dict[dataset_folder_paths[0] + 'session52.csv'][0])
    layer_length = int(((1 + len(stbchtltf_dict[dataset_folder_paths[0] + 'session52.csv'][0]))**.5 + 5)) # TUNE
    dropout_percent = 0.35 # TUNE
    l1_strength = .0001 # TUNE
    l2_strength = .005 # TUNE
    epoch = 100 # TUNE
    batch = 500 # TUNE
    class_weight_dict = {0: .55, 1: .45} # TUNE

    # Build, compile, and train the NN models
    stbchtltf_clf = models.Sequential()
    stbchtltf_clf.add(layers.Input(shape=(input_length,)))
    stbchtltf_clf.add(layers.Dense(layer_length, activation=layers.LeakyReLU(alpha=0.1), kernel_regularizer=l1_l2(l1_strength, l2_strength)))
    stbchtltf_clf.add(layers.Dropout(dropout_percent))
    stbchtltf_clf.add(layers.Dense(layer_length, activation=layers.LeakyReLU(alpha=0.1), kernel_regularizer=l1_l2(l1_strength, l2_strength)))
    stbchtltf_clf.add(layers.Dropout(dropout_percent))  
    stbchtltf_clf.add(layers.Dense(layer_length, activation=layers.LeakyReLU(alpha=0.1), kernel_regularizer=l1_l2(l1_strength, l2_strength)))
    stbchtltf_clf.add(layers.Dropout(dropout_percent))
    stbchtltf_clf.add(layers.Dense(layer_length, activation=layers.LeakyReLU(alpha=0.1), kernel_regularizer=l1_l2(l1_strength, l2_strength)))
    stbchtltf_clf.add(layers.Dropout(dropout_percent))
    stbchtltf_clf.add(layers.Dense(layer_length, activation=layers.LeakyReLU(alpha=0.1), kernel_regularizer=l1_l2(l1_strength, l2_strength)))
    stbchtltf_clf.add(layers.Dropout(dropout_percent))
    stbchtltf_clf.add(layers.Dense(1, activation='sigmoid'))
    stbchtltf_clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=[custom_accuracy])

    '''
    print(f'Training LLTF model... k={n}')
    lltf_x_train = np.array(lltf_x_train)
    lltf_x_val = np.array(lltf_x_val)
    lltf_clf.fit(lltf_x_train, y_train, epochs=epoch, batch_size=batch, validation_data=(lltf_x_val, y_val), callbacks=[early_stopping, reduce_lr], class_weight=class_weight_dict)

    print(f'Training HTLTF model... k={n}')
    htltf_x_train = np.array(htltf_x_train)
    htltf_x_val = np.array(htltf_x_val)
    htltf_clf.fit(htltf_x_train, y_train, epochs=epoch, batch_size=batch, validation_data=(htltf_x_val, y_val), callbacks=[early_stopping, reduce_lr], class_weight=class_weight_dict)
    '''

    print(f'Training STBCHTLTF model... k={n}')
    stbchtltf_x_train = np.array(stbchtltf_x_train)
    stbchtltf_x_val = np.array(stbchtltf_x_val)
    stbchtltf_clf.fit(stbchtltf_x_train, y_train, epochs=epoch, batch_size=batch, validation_data=(stbchtltf_x_val, y_val), callbacks=[early_stopping, reduce_lr], class_weight=class_weight_dict)

    # lltf_y_pred = lltf_clf.predict(lltf_x_val)
    # htltf_y_pred = htltf_clf.predict(htltf_x_val)
    stbchtltf_y_pred = stbchtltf_clf.predict(stbchtltf_x_val)

    class_thres = .5
    # lltf_y_pred = (lltf_y_pred > class_thres).astype(int)
    # htltf_y_pred = (htltf_y_pred > class_thres).astype(int)
    stbchtltf_y_pred = (stbchtltf_y_pred > class_thres).astype(int)

    # Evaluate models
    '''
    lltf_accuracy.append(accuracy_score(y_val, lltf_y_pred))
    lltf_conf_matrix = confusion_matrix(y_val, lltf_y_pred)
    lltf_tp, lltf_tn, lltf_fp, lltf_fn = lltf_conf_matrix[1, 1], lltf_conf_matrix[0, 0], lltf_conf_matrix[0, 1], lltf_conf_matrix[1, 0]
    lltf_accuracy_class_1.append(lltf_tp / (lltf_tp + lltf_fn))
    lltf_accuracy_class_0.append(lltf_tn / (lltf_tn + lltf_fp))
    htltf_accuracy.append(accuracy_score(y_val, htltf_y_pred))
    htltf_conf_matrix = confusion_matrix(y_val, htltf_y_pred)
    htltf_tp, htltf_tn, htltf_fp, htltf_fn = htltf_conf_matrix[1, 1], htltf_conf_matrix[0, 0], htltf_conf_matrix[0, 1], htltf_conf_matrix[1, 0]
    htltf_accuracy_class_1.append(htltf_tp / (htltf_tp + htltf_fn))
    htltf_accuracy_class_0.append(htltf_tn / (htltf_tn + htltf_fp))
    '''
    stbchtltf_accuracy.append(accuracy_score(y_val, stbchtltf_y_pred))
    stbchtltf_conf_matrix = confusion_matrix(y_val, stbchtltf_y_pred)
    stbchtltf_tp, stbchtltf_tn, stbchtltf_fp, stbchtltf_fn = stbchtltf_conf_matrix[1, 1], stbchtltf_conf_matrix[0, 0], stbchtltf_conf_matrix[0, 1], stbchtltf_conf_matrix[1, 0]
    stbchtltf_accuracy_class_1.append(stbchtltf_tp / (stbchtltf_tp + stbchtltf_fn))
    stbchtltf_accuracy_class_0.append(stbchtltf_tn / (stbchtltf_tn + stbchtltf_fp))

# Present the statistics across k models compiled with an average
'''
print()
print(lltf_accuracy)
print(f'L NN Accuracy: \t\t{np.mean(lltf_accuracy)}')
print(lltf_accuracy_class_1)
print(f'L NN LOS Accuracy: \t\t{np.mean(lltf_accuracy_class_1)}')
print(lltf_accuracy_class_0)
print(f'L NN NLOS Accuracy: \t\t{np.mean(lltf_accuracy_class_0)}')
print()
print(htltf_accuracy)
print(f'HT NN Accuracy: \t\t{np.mean(htltf_accuracy)}')
print(htltf_accuracy_class_1)
print(f'HT NN LOS Accuracy: \t\t{np.mean(htltf_accuracy_class_1)}')
print(htltf_accuracy_class_0)
print(f'HT NN NLOS Accuracy: \t\t{np.mean(htltf_accuracy_class_0)}')
'''
print()
print(stbchtltf_accuracy)
print(f'STBCHT NN Accuracy: \t\t{np.mean(stbchtltf_accuracy)}')
print(stbchtltf_accuracy_class_1)
print(f'STBCHT NN LOS Accuracy: \t{np.mean(stbchtltf_accuracy_class_1)}')
print(stbchtltf_accuracy_class_0)
print(f'STBCHT NN NLOS Accuracy: \t{np.mean(stbchtltf_accuracy_class_0)}')
print()

'''
[0.8686522880292138, 0.8077925019821044, 0.8851112378779236, 0.9242941712204007, 0.8831595648232095, 0.868154389160879, 0.7968466424682396, 0.813531887342182]
STBCHT NN Accuracy:             0.8559428353630191
[0.9919651056014692, 0.926779197080292, 0.99632521819017, 0.9441495499653819, 0.8426453819840365, 0.8512985520569983, 0.8317395264116576, 0.8265625]
STBCHT NN LOS Accuracy:         0.9014331289112507
[0.7467665078284548, 0.6904386951631046, 0.775334391294491, 0.9049651763648618, 0.9231808965983329, 0.884702166064982, 0.7622061482820977, 0.8013369542510967]
STBCHT NN NLOS Accuracy:        0.8111163669809277
'''
