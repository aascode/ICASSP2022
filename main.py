from transformers import Wav2Vec2Processor, TFHubertModel
from data_generator import DataGenerator

import librosa
import os
import numpy as np

import config

import pickle

from sklearn.utils.class_weight import compute_class_weight
import model as my_models

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import History
import matplotlib.pyplot as plt

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

dt_generator = DataGenerator(training_feature_type='HuBERT')

######### TEST generating clean data #######################
# dt_generator.generate_clean_data()
# dt_generator.generate_clean_data_corpus(multi_columns=True)

######## TEST generating feature from HuBERT ########################
processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
hubert_model = TFHubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
dt_generator.generate_training_features(hubert_processor=processor, hubert_model=hubert_model)

# dt_generator.generate_testing_corpus(training_corpus='hubert10000_train_corpus.csv')

# ########### TEST saving very huge pickle file ########################
# X, y = dt_generator.load_training_pickle(['hubert3000.p', 'hubert5000.p', 'hubert3000.p', 'hubert5000.p'])
#
# print('Write to hubert16000_001.p')
# with open(config.__cfg__.pickle_dir + '/hubert16000_001.p', 'wb') as f_handle:
#     pickle.dump((X[0:15001], y[0:15001]), f_handle, protocol=4)
#     f_handle.close()
#
# print('Write to hubert16000_002.p')
# with open(config.__cfg__.pickle_dir + '/hubert16000_002.p', 'wb') as f_handle:
#     pickle.dump((X[15000:30001], y[15000:30001]), f_handle, protocol=4)
#     f_handle.close()
#
# print('Write to hubert16000_003.p')
# with open(config.__cfg__.pickle_dir + '/hubert16000_003.p', 'wb') as f_handle:
#     pickle.dump((X[30000:45001], y[30000:45001]), f_handle, protocol=4)
#     f_handle.close()
#
# print('Write to hubert16000_004.p')
# with open(config.__cfg__.pickle_dir + '/hubert16000_004.p', 'wb') as f_handle:
#     pickle.dump((X[45000:], y[45000:]), f_handle, protocol=4)
#     f_handle.close()
#
# print('Done!')
