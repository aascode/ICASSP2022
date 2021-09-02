
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
                             training_feature_type='mfcc')

#dt_generator.generate_clean_data()
#dt_generator.generate_clean_data_corpus(multi_columns=True)

processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model = TFHubertModel.from_pretrained("facebook/hubert-large-ls960-ft", output_hidden_states=True)
# dt_generator.generate_training_features(hubert_processor=processor, hubert_model=model)

#dt_generator.generate_training_features()


clean_files = os.listdir('../SleepinessDetection/clean/')
features = []
for f in clean_files[100:120]:
    wav_signal, sr = librosa.load('../SleepinessDetection/clean/nlx-56f885b0-a1e2-11ea-a9bd-05f6eec0ad7f.wav', sr=None)
    input_values = processor(wav_signal, return_tensors="tf", sampling_rate=sr).input_values

    feat = model(input_values).hidden_states
    features.append(feat)
    print('({}, {})'.format(len(feat), feat[np.random.randint(25)].shape) )

features = np.array(features)

#36 seconds
# wav_signal, sr = librosa.load('../SleepinessDetection/clean/nlx-56f885b0-a1e2-11ea-a9bd-05f6eec0ad7f.wav', sr=None)
# input_values = processor(wav_signal, return_tensors="tf", sampling_rate=sr).input_values
# feat = model(input_values).hidden_states
# for ft in feat:
#     print(ft.shape)



