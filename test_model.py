import argparse

import numpy as np
import pandas as pd

from transformers import Wav2Vec2Processor, TFHubertModel

from tqdm import tqdm
import librosa

from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model

from data_generator import DataGenerator
import config

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def build_prediction(saved_model_path, dtframe, _min, _max):
    hubert_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
    hubert_model = TFHubertModel.from_pretrained("facebook/hubert-large-ls960-ft")

    print('\nTest pre-trained model: ' + saved_model_path)

    model = load_model(saved_model_path)  # load saved-trained model
    classes = list(np.unique(dtframe.sss))

    y_true = []
    y_pred = []
    predict_prob = []

    print('Test model ')
    for idx in tqdm(range(len(dtframe))):
        fname = dtframe.at[idx, 'file']
        label = dtframe.at[idx, 'sss']
        c = classes.index(label)
        y_prob = []

        wav_file = config.__cfg__.clean_data_dir + '/' + str(fname)
        wav_signal, sr = librosa.load(wav_file, sr=None)

        # adding 0 to the end of wav_signal make the last trunk has desired length.
        if len(wav_signal) % config.__cfg__.CHUNK_SIZE > 0:
            padding_size = config.__cfg__.CHUNK_SIZE - (len(wav_signal) % config.__cfg__.CHUNK_SIZE)
            wav_signal = np.pad(wav_signal, (0, padding_size))

        # chop the wav_signal into chunks
        n_chunks = len(wav_signal) // config.__cfg__.CHUNK_SIZE
        for i in range(0, n_chunks):
            idx = i * config.__cfg__.CHUNK_SIZE
            chunk_signal = wav_signal[idx: idx + config.__cfg__.CHUNK_SIZE]
            input_values = hubert_processor(chunk_signal, return_tensors="tf", sampling_rate=sr).input_values
            x = hubert_model(input_values).last_hidden_state
            # reshape (1, N, 1024) --> (1, N, 1024, 1)
            x = tf.reshape(x, [1, x.shape[1], x.shape[2], x.shape[0]])
            x = x.numpy()
            # normalize x based on training feature's _min/_max
            x = (x - _min) / (_max - _min)
            y_hat = model.predict(x)
            y_prob.append(y_hat)

        pred = np.mean(y_prob, axis=0).flatten()
        y_pred.append(np.argmax(pred))
        y_true.append(c)

        predict_prob.append(np.mean(y_prob, axis=0).flatten())

    return y_true, y_pred, predict_prob


def run_test(args):
    #----- these value are suitable for hubert1000.
    # _min = -7.5425848960876465
    # _max = 3.371579170227051

    df = pd.read_csv(args.test_corpus)

    y_true, y_pred, fn_prob = build_prediction(args.model_name, df, args.xmin, args.xmax)
    acc_score = accuracy_score(y_true=y_true, y_pred=y_pred)
    print('Accuracy score: {}'.format(acc_score))

    df['sss_pred'] = y_pred
    df['probability'] = fn_prob
    df.to_csv(args.output, index=False)

if __name__ == '__main__':
    # data_gen = DataGenerator()
    # data_gen.generate_testing_corpus(number_of_test=1000, training_corpus='hubert10000_train_corpus.csv',
    #                                       shuffle=True)

    parser = argparse.ArgumentParser(description='Run Testing Sleepiness Classification Saved Model')
    parser.add_argument('--model_name', type=str, default='./checkpoint/model1-hubert10000.model',
                        help='the name of model stored in ./checkpoint')
    parser.add_argument('--test_corpus', type=str, default='./csv/testing_corpus1000.csv', help='the testing corpus')
    parser.add_argument('--output', type=str, default='./csv/prediction.csv', help='name of csv prediction file ')
    parser.add_argument('--xmin', type=str, default='-7.5425848960876465', help='xmin value for regularization purpose')
    parser.add_argument('--xmax', type=str, default='3.371579170227051', help='xmin value for regularization purpose')
    args, _ = parser.parse_known_args()

    run_test(args)
