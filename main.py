
from transformers import Wav2Vec2Processor, TFHubertModel
from data_generator import DataGenerator

import librosa
import os
import numpy as np

dt_generator = DataGenerator(raw_data_dir='/media/virtual3T/sessionsdata',
                             raw_data_corpus='/media/virtual3T/bang/SleepinessDetection/new2-v1.csv',
                             clean_data_dir='/media/virtual3T/bang/SleepinessDetection/clean',
                             clean_data_corpus='clean_data_corpus.csv',
                             pickle_dir='/media/virtual3T/bang/SleepinessDetection/pickle',
                             training_feature_type='hubert')

#dt_generator.generate_clean_data()
#dt_generator.generate_clean_data_corpus(multi_columns=True)

processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model = TFHubertModel.from_pretrained("facebook/hubert-large-ls960-ft", output_hidden_states=True)
dt_generator.generate_training_features(hubert_processor=processor, hubert_model=model)




#
# clean_files = os.listdir('../SleepinessDetection/clean/')
# X = []
# _max_rows_number = -float('inf')
# for f in clean_files[1:10]:
#     wav_signal, sr = librosa.load('../SleepinessDetection/clean/'+f, sr=None)
#     if len(wav_signal) == 0:
#         continue
#     input_values = processor(wav_signal, return_tensors="tf", sampling_rate=sr).input_values
#     feat = model(input_values).hidden_states # this returns a tuple of 25 features
#
#     # convert to a list of 25 elements; each element is an array of shape (1, N, 1024)
#     # --> reshape each element to (N, 1024)
#     feat = list(map(lambda arr: np.squeeze(arr), list(feat)))
#     #feat = np.array(feat) # convert to an array with shape of (25, N, 1024)
#     X.append(feat)
#
#     print(feat[0].shape)
#     _max_rows_number = max(_max_rows_number, feat[0].shape[0])
#
#
# # Do padding
# print('max number of rows:', _max_rows_number)
# for i, x in enumerate(X):
#     print('before:', x[0].shape)
#     r, c = x[0].shape
#     if r < _max_rows_number:
#         padded_r = _max_rows_number - r
#         padded_0 = np.zeros((padded_r, c))
#         x = list(map(lambda ft: np.vstack([ft, padded_0]), x))
#     X[i] = x
#     print('   --> after padding:', x[0].shape)
#













#36 seconds
# wav_signal, sr = librosa.load('../SleepinessDetection/clean/nlx-56f885b0-a1e2-11ea-a9bd-05f6eec0ad7f.wav', sr=None)
# input_values = processor(wav_signal, return_tensors="tf", sampling_rate=sr).input_values
# feat = model(input_values).hidden_states
# for ft in feat:
#     print(ft.shape)



