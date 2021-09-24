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


def build_prediction1(saved_model_path, dtframe, _min=0, _max=1):
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

        predict_prob.append(np.argmax(np.mean(y_prob, axis=0).flatten()))

    return y_true, y_pred, predict_prob


def build_prediction2(saved_model_path, test_df):
    print('\nTest pre-trained model: ' + saved_model_path)
    df = pd.read_csv(test_df)

    model = load_model(saved_model_path)  # load saved-trained model
    classes = list(np.unique(df.sss))

    y_true, y_pred, pred_prob = [], [], []

    print('Test model ')
    for idx in tqdm(range(len(df))):
        fname = df.at[idx, 'file']
        label = df.at[idx, 'sss']
        c = classes.index(label)
        y_hat = model.predict(np.load(fname))
        pred_prob.append(np.max(y_hat))
        y_pred.append(np.argmax(y_hat))
        y_true.append(c)
    return y_true, y_pred, pred_prob


def run_test(args):
    df = pd.read_csv(args.test_corpus)

    y_true, y_pred, pred_prob = build_prediction2(args.model_name, df)
    acc_score = accuracy_score(y_true=y_true, y_pred=y_pred)

    print('Accuracy score: {}'.format(acc_score))
    df['pred'] = y_pred
    df['pred_prob'] = pred_prob
    df.to_csv(args.output, index=False)


'''
Command line:

python3 test_model.py --model_name model1-hubert5000-20ep\
 --test_corpus testing_corpus1000.csv\
 --output test_result.csv\
 --new_test_corpus false \
 --test_files_num 5000\
 --train_corpus hubert2000_train_corpus.csv 
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Testing Sleepiness Classification Saved Model')
    parser.add_argument('--model_name', type=str, help='The name of model stored in ./checkpoint')
    parser.add_argument('--test_corpus', type=str, help='the testing corpus')
    parser.add_argument('--output', type=str, help='name of csv testing file ')
    # parser.add_argument('--new_test_corpus', type=str, default='false', help='Generate testing corpus')
    # parser.add_argument('--test_files_num', type=int, help='The number of audio file in testing corpus')
    # parser.add_argument('--train_corpus', type=str,
    #                    help='The training corpus that testing files shouldn\'t be selected ')
    args, _ = parser.parse_known_args()

    # if args.new_test_corpus.lower() == 'true':
    #     data_gen = DataGenerator()
    #     data_gen.generate_testing_corpus(number_of_test=args.test_files_num, training_corpus=args.train_corpus)

    run_test(args)
