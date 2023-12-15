import joblib
import numpy as np
import os
import random
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from tensorflow.keras import backend as K
from tensorflow.keras import models, layers, callbacks
from tensorflow.keras.regularizers import l1, l2, l1_l2

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
# htltf_dict = {}
stbchtltf_dict = {}
y_dict = {}

print('Loading processed_data...')
# lltf_dict = np.load('saves/processed_data/NN_lltf_feature_extraction_data.npy', allow_pickle=True).item()
# htltf_dict = np.load('saves/processed_data/NN_htltf_feature_extraction_data.npy', allow_pickle=True).item()
stbchtltf_dict = np.load('saves/processed_data/NN_stbchtltf_feature_extraction_data.npy', allow_pickle=True).item()
y_dict = np.load('saves/processed_data/NN_y.npy', allow_pickle=True).item()

random.seed(314)

# Get filepaths and shuffle
dataset_filepaths0 = [f for f in os.listdir(dataset_folder_paths[0])][fp_index_limit:]
dataset_filepaths0 = random.sample(dataset_filepaths0, len(dataset_filepaths0))
dataset_filepaths1 = [f for f in os.listdir(dataset_folder_paths[1])][fp_index_limit:]
dataset_filepaths1 = random.sample(dataset_filepaths1, len(dataset_filepaths1))

# lltf_x_all = []
# htltf_x_all = []
stbchtltf_x_all = []
y_all = []
for df in dataset_filepaths0:
    # lltf_x_all += lltf_dict[dataset_folder_paths[0] + df]
    # htltf_x_all += htltf_dict[dataset_folder_paths[0] + df]
    stbchtltf_x_all += stbchtltf_dict[dataset_folder_paths[0] + df]
    y_all += y_dict[dataset_folder_paths[0] + df]
for df in dataset_filepaths1:
    # lltf_x_all += lltf_dict[dataset_folder_paths[1] + df]
    # htltf_x_all += htltf_dict[dataset_folder_paths[1] + df]
    stbchtltf_x_all += stbchtltf_dict[dataset_folder_paths[1] + df]
    y_all += y_dict[dataset_folder_paths[1] + df]

scaler_fp = 'saves/scalers/'
# lltf_scaler = joblib.load(scaler_fp + 'lltf_scaler.pkl')
# htltf_scaler = joblib.load(scaler_fp + 'htltf_scaler.pkl')
stbchtltf_scaler = joblib.load(scaler_fp + 'stbchtltf_scaler.pkl')

y_all = np.array(y_all)

# htltf_x_train, htltf_x_val, _, _ = train_test_split(htltf_x_all, y_all, test_size=0.2, random_state=21)
stbchtltf_x_train, stbchtltf_x_val, y_train, y_val = train_test_split(stbchtltf_x_all, y_all, test_size=0.2, random_state=21)

# Scale the datasets
# lltf_x_train = lltf_scaler.transform(lltf_x_train)
# lltf_x_val = lltf_scaler.transform(lltf_x_val)
# htltf_x_train = htltf_scaler.transform(htltf_x_train)
# htltf_x_val = htltf_scaler.transform(htltf_x_val)
stbchtltf_x_train = stbchtltf_scaler.transform(stbchtltf_x_train)
stbchtltf_x_val = stbchtltf_scaler.transform(stbchtltf_x_val)

# Define callbacks
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=8, # TUNE
    restore_best_weights=True
)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, min_lr=1e-6) # Almost no effect w/ SGD optimizer

# Common hyperparameters
input_length = len(stbchtltf_dict[dataset_folder_paths[0] + 'session52.csv'][0])
layer_length = int(((1 + len(stbchtltf_dict[dataset_folder_paths[0] + 'session52.csv'][0]))**.5 + 5)) # TUNE
dropout_percent = 0.35 # TUNE
l1_strength = .0001 # TUNE
l2_strength = .005 # TUNE
epoch = 100 # TUNE
batch = 100 # TUNE
class_weight_dict = {0: .55, 1: .45} # TUNE


def custom_accuracy(y_true, y_pred):

    y_pred_binary = K.round(y_pred)
    accuracy = K.mean(K.equal(y_true, y_pred_binary))

    return accuracy


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

model_fp = 'saves/models/'

'''
print(f'Training final LLTF model...')
lltf_x_train = np.array(lltf_x_train)
lltf_x_val = np.array(lltf_x_val)
lltf_clf.fit(lltf_x_train, y_train, epochs=epoch, batch_size=batch, validation_data=(lltf_x_val, y_val), callbacks=[early_stopping, reduce_lr], class_weight=class_weight_dict) # OLD datasplit: y_... => lltf_y_...
lltf_clf.save(model_fp + 'lltf_model.h5')

print(f'Training final HTLTF model...')
htltf_x_train = np.array(htltf_x_train)
htltf_x_val = np.array(htltf_x_val)
htltf_clf.fit(htltf_x_train, y_train, epochs=epoch, batch_size=batch, validation_data=(htltf_x_val, y_val), callbacks=[early_stopping, reduce_lr], class_weight=class_weight_dict) # OLD datasplit: y_... => htltf_y_...
htltf_clf.save(model_fp + 'htltf_model.h5')
'''
    
print(f'Training final STBCHTLTF model...')
stbchtltf_x_train = np.array(stbchtltf_x_train)
stbchtltf_x_val = np.array(stbchtltf_x_val)
stbchtltf_clf.fit(stbchtltf_x_train, y_train, epochs=epoch, batch_size=batch, validation_data=(stbchtltf_x_val, y_val), callbacks=[early_stopping, reduce_lr], class_weight=class_weight_dict) # OLD datasplit: y_... => stbchtltf_y_...
stbchtltf_clf.save(model_fp + 'stbchtltf_model.keras')

# lltf_y_pred = lltf_clf.predict(lltf_x_val)
# htltf_y_pred = htltf_clf.predict(htltf_x_val)
stbchtltf_y_pred = stbchtltf_clf.predict(stbchtltf_x_val)

class_thres = .5
# lltf_y_pred = (lltf_y_pred > class_thres).astype(int)
# htltf_y_pred = (htltf_y_pred > class_thres).astype(int)
stbchtltf_y_pred = (stbchtltf_y_pred > class_thres).astype(int)

# Evaluate models
'''
lltf_accuracy = accuracy_score(y_val, lltf_y_pred)
lltf_conf_matrix = confusion_matrix(y_val, lltf_y_pred)
lltf_tp, lltf_tn, lltf_fp, lltf_fn = lltf_conf_matrix[1, 1], lltf_conf_matrix[0, 0], lltf_conf_matrix[0, 1], lltf_conf_matrix[1, 0]
lltf_accuracy_class_1 = lltf_tp / (lltf_tp + lltf_fn)
lltf_accuracy_class_0 = lltf_tn / (lltf_tn + lltf_fp)
htltf_accuracy = accuracy_score(y_val, htltf_y_pred)
htltf_conf_matrix = confusion_matrix(y_val, htltf_y_pred)
htltf_tp, htltf_tn, htltf_fp, htltf_fn = htltf_conf_matrix[1, 1], htltf_conf_matrix[0, 0], htltf_conf_matrix[0, 1], htltf_conf_matrix[1, 0]
htltf_accuracy_class_1 = htltf_tp / (htltf_tp + htltf_fn)
htltf_accuracy_class_0 = htltf_tn / (htltf_tn + htltf_fp)
'''
stbchtltf_accuracy = accuracy_score(y_val, stbchtltf_y_pred)
stbchtltf_conf_matrix = confusion_matrix(y_val, stbchtltf_y_pred)
stbchtltf_tp, stbchtltf_tn, stbchtltf_fp, stbchtltf_fn = stbchtltf_conf_matrix[1, 1], stbchtltf_conf_matrix[0, 0], stbchtltf_conf_matrix[0, 1], stbchtltf_conf_matrix[1, 0]
stbchtltf_accuracy_class_1 = stbchtltf_tp / (stbchtltf_tp + stbchtltf_fn)
stbchtltf_accuracy_class_0 = stbchtltf_tn / (stbchtltf_tn + stbchtltf_fp)

# Present the statistics across k models compiled with an average
'''
print()
print(lltf_accuracy)
print(f'L NN Accuracy: \t\t{lltf_accuracy}')
print(lltf_accuracy_class_1)
print(f'L NN LOS Accuracy: \t\t{lltf_accuracy_class_1}')
print(lltf_accuracy_class_0)
print(f'L NN NLOS Accuracy: \t\t{lltf_accuracy_class_0}')
print()
print(htltf_accuracy)
print(f'HT NN Accuracy: \t\t{htltf_accuracy}')
print(htltf_accuracy_class_1)
print(f'HT NN LOS Accuracy: \t\t{htltf_accuracy_class_1}')
print(htltf_accuracy_class_0)
print(f'HT NN NLOS Accuracy: \t\t{htltf_accuracy_class_0}')
print()
'''
print(stbchtltf_accuracy)
print(f'STBCHT NN Accuracy: \t\t{stbchtltf_accuracy}')
print(stbchtltf_accuracy_class_1)
print(f'STBCHT NN LOS Accuracy: \t{stbchtltf_accuracy_class_1}')
print(stbchtltf_accuracy_class_0)
print(f'STBCHT NN NLOS Accuracy: \t{stbchtltf_accuracy_class_0}')
print()

'''
0.9038610856215148
STBCHT NN Accuracy:             0.9038610856215148
0.9603367099443573
STBCHT NN LOS Accuracy:         0.9603367099443573
0.8485610505727857
STBCHT NN NLOS Accuracy:        0.8485610505727857
'''