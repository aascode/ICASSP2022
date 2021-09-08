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

#------------ TEST generating clean data --------------------------------
# dt_generator.generate_clean_data()
# dt_generator.generate_clean_data_corpus(multi_columns=True)

#------------ TEST generating feature from HuBERT -----------------------
processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
hubert_model = TFHubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
dt_generator.generate_training_features(hubert_processor=processor, hubert_model=hubert_model)


#----------- TEST generating training corpus ----------------------------
# dt_generator.generate_testing_corpus(training_corpus='hubert10000_train_corpus.csv')

