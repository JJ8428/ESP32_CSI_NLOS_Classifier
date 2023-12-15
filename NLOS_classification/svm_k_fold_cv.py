import joblib
import numpy as np
import os
import random
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.svm import SVC
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


dataset_folder_paths = ['data/NLOS_data/', 'data/LOS_data/']
# lltf_dict = {}
htltf_dict = {}
stbchtltf_dict = {}
y_dict = {}

print('Loading processed data...')
# lltf_dict = np.load('saves/processed_data/NN_lltf_feature_extraction_data.npy', allow_pickle=True).item()
htltf_dict = np.load('saves/processed_data/NN_htltf_feature_extraction_data.npy', allow_pickle=True).item()
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
htltf_accuracy = []
htltf_accuracy_class_1 = []
htltf_accuracy_class_0 = []
stbchtltf_accuracy = []
stbchtltf_accuracy_class_1 = []
stbchtltf_accuracy_class_0 = []

random.seed(314)

# Get filepaths and shuffle
dataset_filepaths0 = [f for f in os.listdir(dataset_folder_paths[0])]
dataset_filepaths0 = random.sample(dataset_filepaths0, len(dataset_filepaths0))
dataset_filepaths1 = [f for f in os.listdir(dataset_folder_paths[1])]
dataset_filepaths1 = random.sample(dataset_filepaths1, len(dataset_filepaths1))

# Train our scalers
# lltf_x_all = []
htltf_x_all = []
stbchtltf_x_all = []
for df in dataset_filepaths0:
    # lltf_x_all += lltf_dict[dataset_folder_paths[0] + df]
    htltf_x_all += htltf_dict[dataset_folder_paths[0] + df]
    stbchtltf_x_all += stbchtltf_dict[dataset_folder_paths[0] + df]
for df in dataset_filepaths1:
    # lltf_x_all += lltf_dict[dataset_folder_paths[1] + df]
    htltf_x_all += htltf_dict[dataset_folder_paths[1] + df]
    stbchtltf_x_all += stbchtltf_dict[dataset_folder_paths[1] + df]

scale_type = 0 # TUNE
if scale_type == 0:
    # lltf_scaler = StandardScaler()
    htltf_scaler = StandardScaler()
    stbchtltf_scaler = StandardScaler()
else:
    # lltf_scaler = PowerTransformer()
    htltf_scaler = PowerTransformer()
    stbchtltf_scaler = PowerTransformer()

# lltf_scaler.fit(lltf_x_all)
htltf_scaler.fit(htltf_x_all)
stbchtltf_scaler.fit(stbchtltf_x_all)

scaler_fp = 'saves/scalers/'
# joblib.dump(lltf_scaler, scaler_fp + 'lltf_scaler.pkl')
joblib.dump(htltf_scaler, scaler_fp + 'htltf_scaler.pkl')
joblib.dump(stbchtltf_scaler, scaler_fp + 'stbchtltf_scaler.pkl')

# K-fold cross validation
k = 8 # TUNE
for n in range(k):
    
    # Validation set come from separate recordings
    train0, val0 = k_fold_cv_assist(dataset_filepaths0, k, n)
    train1, val1 = k_fold_cv_assist(dataset_filepaths1, k, n)

    # lltf_x_train = []
    htltf_x_train = []
    stbchtltf_x_train = []
    y_train = []
    for df in train0:
        # lltf_x_train += lltf_dict[dataset_folder_paths[0] + df]
        htltf_x_train += htltf_dict[dataset_folder_paths[0] + df]
        stbchtltf_x_train += stbchtltf_dict[dataset_folder_paths[0] + df]
        y_train += y_dict[dataset_folder_paths[0] + df]
    for df in train1:
        # lltf_x_train += lltf_dict[dataset_folder_paths[1] + df]
        htltf_x_train += htltf_dict[dataset_folder_paths[1] + df]
        stbchtltf_x_train += stbchtltf_dict[dataset_folder_paths[1] + df]
        y_train += y_dict[dataset_folder_paths[1] + df]
    # lltf_x_train = np.array(lltf_x_train)
    htltf_x_train = np.array(htltf_x_train)
    stbchtltf_x_train = np.array(stbchtltf_x_train)
    y_train = np.array(y_train)

    # lltf_x_val = []
    htltf_x_val = []
    stbchtltf_x_val = []
    y_val = []
    for df in val0:
        # lltf_x_val += lltf_dict[dataset_folder_paths[0] + df]
        htltf_x_val += htltf_dict[dataset_folder_paths[0] + df]
        stbchtltf_x_val += stbchtltf_dict[dataset_folder_paths[0] + df]
        y_val += y_dict[dataset_folder_paths[0] + df]
    for df in val1:
        # lltf_x_val += lltf_dict[dataset_folder_paths[1] + df]
        htltf_x_val += htltf_dict[dataset_folder_paths[1] + df]
        stbchtltf_x_val += stbchtltf_dict[dataset_folder_paths[1] + df]
        y_val += y_dict[dataset_folder_paths[1] + df]
    # lltf_x_val= np.array(lltf_x_val)
    htltf_x_val= np.array(htltf_x_val)
    stbchtltf_x_val= np.array(stbchtltf_x_val)
    y_val = np.array(y_val)

    # Scale the datasets
    # lltf_x_train = lltf_scaler.transform(lltf_x_train)
    # lltf_x_val = lltf_scaler.transform(lltf_x_val)
    htltf_x_train = htltf_scaler.transform(htltf_x_train)
    htltf_x_val = htltf_scaler.transform(htltf_x_val)
    stbchtltf_x_train = stbchtltf_scaler.transform(stbchtltf_x_train)
    stbchtltf_x_val = stbchtltf_scaler.transform(stbchtltf_x_val)

    # Declare the SVM models
    # lltf_clf = SVC(kernel='rbf', C=1.0, gamma='scale')
    htltf_clf = SVC(kernel='rbf', C=1.0, gamma='scale')
    stbchtltf_clf = SVC(kernel='rbf', C=1.0, gamma='scale')

    '''
    print(f'Training LLTF model... k={n}')
    lltf_clf.fit(lltf_x_train, y_train)
    '''

    print(f'Training HTLTF model... k={n}')
    htltf_clf.fit(htltf_x_train, y_train)

    print(f'Training STBCHTLTF model... k={n}')
    stbchtltf_clf.fit(stbchtltf_x_train, y_train)
    # lltf_y_pred = lltf_clf.predict(lltf_x_val)
    htltf_y_pred = htltf_clf.predict(htltf_x_val)
    stbchtltf_y_pred = stbchtltf_clf.predict(stbchtltf_x_val)

    class_thres = .5
    # lltf_y_pred = (lltf_y_pred > class_thres).astype(int)
    htltf_y_pred = (htltf_y_pred > class_thres).astype(int)
    stbchtltf_y_pred = (stbchtltf_y_pred > class_thres).astype(int)

    # Evaluate models
    '''
    lltf_accuracy.append(accuracy_score(y_val, lltf_y_pred))
    lltf_conf_matrix = confusion_matrix(y_val, lltf_y_pred)
    lltf_tp, lltf_tn, lltf_fp, lltf_fn = lltf_conf_matrix[1, 1], lltf_conf_matrix[0, 0], lltf_conf_matrix[0, 1], lltf_conf_matrix[1, 0]
    lltf_accuracy_class_1.append(lltf_tp / (lltf_tp + lltf_fn))
    lltf_accuracy_class_0.append(lltf_tn / (lltf_tn + lltf_fp))
    '''
    htltf_accuracy.append(accuracy_score(y_val, htltf_y_pred))
    htltf_conf_matrix = confusion_matrix(y_val, htltf_y_pred)
    htltf_tp, htltf_tn, htltf_fp, htltf_fn = htltf_conf_matrix[1, 1], htltf_conf_matrix[0, 0], htltf_conf_matrix[0, 1], htltf_conf_matrix[1, 0]
    htltf_accuracy_class_1.append(htltf_tp / (htltf_tp + htltf_fn))
    htltf_accuracy_class_0.append(htltf_tn / (htltf_tn + htltf_fp))
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
'''
print()
print(htltf_accuracy)
print(f'HT SVM Accuracy: \t\t{np.mean(htltf_accuracy)}')
print(htltf_accuracy_class_1)
print(f'HT SVM LOS Accuracy: \t\t{np.mean(htltf_accuracy_class_1)}')
print(htltf_accuracy_class_0)
print(f'HT SVM NLOS Accuracy: \t\t{np.mean(htltf_accuracy_class_0)}')
print()
print(stbchtltf_accuracy)
print(f'STBCHT SVM Accuracy: \t\t{np.mean(stbchtltf_accuracy)}')
print(stbchtltf_accuracy_class_1)
print(f'STBCHT SVM LOS Accuracy: \t{np.mean(stbchtltf_accuracy_class_1)}')
print(stbchtltf_accuracy_class_0)
print(f'STBCHT SVM NLOS Accuracy: \t{np.mean(stbchtltf_accuracy_class_0)}')
print()

'''

'''
