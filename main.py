from transformers import Wav2Vec2Processor, TFHubertModel
from data_generator import DataGenerator

import librosa
import os
import numpy as np
import pandas as pd

import config

import pickle

from sklearn.utils.class_weight import compute_class_weight
import model as my_models

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import History
import matplotlib.pyplot as plt

import tensorflow as tf

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

dt_generator = DataGenerator(training_feature_type='HuBERT')

#------------ TEST generating clean data --------------------------------
#DataGenerator.generate_clean_data_corpus(output_csv='./csv/clean_data_corpus_multicol.csv', multi_columns=True)

#------------ TEST generating feature from HuBERT -----------------------
# processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
# hubert_model = TFHubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
# #features = dt_generator.generate_training_features(hubert_processor=processor, hubert_model=hubert_model)
#
# wav_signal, sr = librosa.load('clean_audio/nlx-51d54050-a43a-11ea-ad75-47afc39b88d6.wav', sr=None)
# input_values = processor(wav_signal, return_tensors="tf", sampling_rate=sr).input_values
# features = hubert_model(input_values, output_hidden_states=True).hidden_states
# features = np.array(features)
# print('All hidden states shape:', features.shape)
# embedding = hubert_model(input_values, output_hidden_states=True).last_hidden_state
# embedding = np.array(embedding)
# print('Embedding shape = ', embedding.shape)
#
#
# wav_signal, sr = librosa.load('clean_audio/nlx-fff91c10-a4d0-11ea-bcc0-954f47b73b1e.wav', sr=None)
# input_values = processor(wav_signal, return_tensors="tf", sampling_rate=sr).input_values
# features = hubert_model(input_values, output_hidden_states=True).hidden_states
# features = np.array(features)
# print('All hidden states shape:', features.shape)
# embedding = hubert_model(input_values, output_hidden_states=True).last_hidden_state
# embedding = np.array(embedding)
# print('Embedding shape = ', embedding.shape)

# from pydub import AudioSegment
# from pydub.silence import detect_leading_silence
# wav_signal = AudioSegment.from_file('clean_audio/nlx-fff91c10-a4d0-11ea-bcc0-954f47b73b1e.wav', format='wav')
# mm = detect_leading_silence(wav_signal)
#
# start_trim = detect_leading_silence(wav_signal)
# end_trim = detect_leading_silence(wav_signal.reverse())
#
# duration = len(wav_signal)
# trimmed_sound = wav_signal[start_trim:duration-end_trim]
#
#
# print(len(wav_signal))
# print(len(trimmed_sound))
#
#


X_list = np.load('pickle/hubert-feature/nlx-3d7dee30-a873-11ea-9c5f-5fb704133341.npy')
print(X_list.shape)


X_list = np.load('pickle/hubert-feature/nlx-7dc92dc0-a219-11ea-bcc0-954f47b73b1e.npy')
print(X_list.shape)





X_list = np.load('pickle/hubert-feature/nlx-c26f7080-a145-11ea-a9bd-05f6eec0ad7f.npy')
print(X_list.shape)


#----------- TEST generating training corpus ----------------------------
# dt_generator.generate_testing_corpus(training_corpus='hubert10000_train_corpus.csv')





