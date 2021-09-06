import argparse

import numpy as np
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import History
import matplotlib.pyplot as plt

from data_generator import DataGenerator
import model as my_models
import config

# prevent run out memory of GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# load data from pickle files
dt_generator = DataGenerator(training_feature_type='HuBERT')
X, y, _min, _max = dt_generator.load_training_pickle(
    ['hubert10000_001.p', 'hubert10000_002.p', 'hubert10000_003.p', 'hubert10000_004.p', 'hubert10000_005.p'])

### use this values to normalize feature when running test process
print('Normalization: _min={}, _max={}'.format(_min, _max))

input_shape = (X.shape[1], X.shape[2], 1)
y_flat = np.argmax(y, axis=1)
class_weight = compute_class_weight('balanced', np.unique(y_flat), y_flat)
class_weight = {k: v for k, v in enumerate(class_weight)}

conv_model = my_models.get_model1(input_shape)
checkpoint = ModelCheckpoint(config.__cfg__.checkpoint_dir + '/model1-hubert10000.model',  # checkpoints path
                             monitor='val_loss',
                             verbose=0, mode='max',
                             save_best_only=True,
                             save_weights_only=False, save_freq=1)
hist = History()
hist = conv_model.fit(X, y,
                      epochs=20, batch_size=32, shuffle=True,
                      class_weight=class_weight,
                      validation_split=0.1,
                      callbacks=[checkpoint])
conv_model.save(config.__cfg__.checkpoint_dir+'/model1-hubert100.model')

#  Plot training Losses
fig, ax = plt.subplots(figsize=(12, 8))
plt.grid(True)
plt.plot(hist.history['acc'], label='Acc', alpha=0.5)
plt.plot(hist.history['loss'], label='Loss', alpha=0.5)
plt.plot(hist.history['val_acc'], label='Valid_Acc', alpha=0.5)
plt.plot(hist.history['val_loss'], label='Valid_Loss', alpha=0.5)
plt.title("Training Set: 21h 11min HuBERT Self-Supervised Learning Features From 10K Audio Files(21h")
plt.legend(['Accuracy', 'Loss', 'Validation Accuracy', 'Validation Loss'])
plt.savefig('training_loss.png')


