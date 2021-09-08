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


def run_train(args):
    # load data from pickle files
    dt_generator = DataGenerator(training_feature_type='HuBERT')
    X, y, _min, _max = dt_generator.load_training_pickle(args.p_file)

    ## use this values to normalize feature when running test process
    print('Normalization: _min={}, _max={}'.format(_min, _max))

    input_shape = (X.shape[1], X.shape[2], 1)
    y_flat = np.argmax(y, axis=1)
    class_weight = compute_class_weight('balanced', np.unique(y_flat), y_flat)
    class_weight = {k: v for k, v in enumerate(class_weight)}

    conv_model = my_models.get_model1(input_shape)
    checkpoint = ModelCheckpoint(config.__cfg__.checkpoint_dir + '/' + args.model_name,  # checkpoints path
                                 monitor='val_loss',
                                 verbose=0, mode='max',
                                 save_best_only=True,
                                 save_weights_only=False, save_freq=1)
    hist = History()
    hist = conv_model.fit(X, y,
                          epochs=args.epoch,
                          batch_size=args.batch_size,
                          shuffle=True,
                          class_weight=class_weight,
                          validation_split=0.1,
                          callbacks=[checkpoint])
    conv_model.save(config.__cfg__.checkpoint_dir + '/' + args.model_name)

    #  Plot training Losses
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.grid(True)
    plt.plot(hist.history['acc'], label='Accuracy', alpha=0.5)
    plt.plot(hist.history['loss'], label='Loss', alpha=0.5)
    plt.plot(hist.history['val_acc'], label='Validation Accuracy', alpha=0.5)
    #plt.plot(hist.history['val_loss'], label='Validation Loss', alpha=0.5)
    plt.title("Leaning Curve Of Training Model "+args.model_name)
    plt.legend()
    plt.xlabel('Epochs')
    plt.savefig('images/lc-'+args.model_name + '.png')  # learning curve


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Sleepiness Classification Saved Model')
    parser.add_argument('--model_name', type=str, help='Name of model')
    parser.add_argument('--epoch', type=int, default=20, help='Epochs of training')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--p_file', type=str, action='append', help='Pickle file of training data')
    args, _ = parser.parse_known_args()

    print(args)
    run_train(args)

'''
----- Command line to train the model ----

python3 train.py --model_name model1-hubert20k-20ep \
epoch 512 \
batch_size 32 \
--p_file hubert20000_001.p  --p_file hubert20000_002.p --p_file hubert20000_003.p --p_file hubert20000_004.p \
--p_file hubert20000_005.p  --p_file hubert20000_006.p --p_file hubert20000_007.p --p_file hubert20000_008.p \
--p_file hubert20000_009.p  --p_file hubert20000_0010.p

'''
