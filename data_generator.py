import os
import re
import numpy as np
import pandas as pd
import json
import pickle
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

import librosa
from python_speech_features import mfcc

from utils import envelope, calc_spectrogram
import config


class DataGenerator:
    def __init__(self, training_feature_type='HuBert'):
        self.training_feature_type = training_feature_type  # MFCC | SPECTROGRAM | HuBERT




    '''
        This function generates cleaned wav files from original audio files.
        (down sampling to 16kHz and save to directory ./clean)
    '''

    @staticmethod
    def generate_clean_data(log_file='clean_data.log'):
        cols = [i for i in range(3, 53)]  # response samples
        cols.insert(0, 0)  # session id
        cols.append(143)  # sleepiness scale (label)
        df = pd.read_csv(config.__cfg__.raw_data_corpus, usecols=cols)

        col_names = ['response' + str(i) for i in range(1, 51)]  # response1,...,response51
        col_names.insert(0, 'session_id')
        col_names.append('sss')

        df.columns = col_names

        # extract audio file name from responses
        for c in col_names[1:-1]:
            sample_ids = df[c]
            str_pattern = "nlx-\w*-\w*-\w*-\w*-\w*"
            sample_ids = list(
                map(lambda x: (re.findall(str_pattern, x)[0] + '.wav') if type(x) is str else '__', sample_ids))
            df[c] = sample_ids

        # extract sleepiness level
        sss_levels = df['sss']
        str_pattern = "\d"
        sss_levels = list(map(lambda x: (re.findall(str_pattern, x)[0]), sss_levels))
        df['sss'] = sss_levels

        # Down sampling to 16kHz for every audio sample files
        print('Generating clean audio samples...')

        skipped_files = {}
        df.set_index('session_id', inplace=True)
        for sess in tqdm(df.index):
            # print('Cleaning for session_id:', sess)
            for resp in col_names[1:-1]:  # take response1 to response50
                f = df.loc[sess, resp]
                wav_file = config.__cfg__.raw_data_dir + '/' + str(sess) + '/' + f
                if not os.path.isfile(wav_file):
                    skipped_files[wav_file] = 'file not found'
                    continue

                if os.path.isfile(config.__cfg__.clean_data_dir + '/' + f):  # skip duplicated files
                    skipped_files[wav_file] = 'duplicated file'
                    continue

                if os.path.getsize(wav_file) == 0:  # skip empty files
                    skipped_files[wav_file] = 'file size = 0'
                    continue

                signal, rate = librosa.load(wav_file, sr=None)
                if len(signal) == 0:  # for some reasons, the wav file doesn't contain any signal
                    skipped_files[wav_file] = 'length of signal = 0'
                    continue

                # Down sampling the signal to 16kHz
                new_signal = librosa.resample(signal, rate, config.__cfg__.SAMPLE_RATE)
                mask = envelope(new_signal, config.__cfg__.SAMPLE_RATE)
                librosa.output.write_wav(filename=config.__cfg__.clean_data_dir + '/' + f,
                                         rate=config.__cfg__.SAMPLE_RATE,
                                         data=new_signal[mask])

        print('\n\tDONE! Check the {} for detail.\n'.format(log_file))
        with open(log_file, 'w') as fhandle:
            fhandle.write(json.dumps(skipped_files))

    ''' 
        build a new CSV file that contains session_id, wav_response_file, sleepiness_scale, total_audio_length 
        parameters:
            - multi_columns:
                    + True: the output CSV file has columns corresponding to responses
                    + False: the output CSV file has only 3 columns ['file', 'duration', 'sss'] 
    '''

    @staticmethod
    def generate_clean_data_corpus(multi_columns=False):
        cols = [i for i in range(3, 53)]  # response samples
        cols.insert(0, 0)  # session id
        cols.append(143)  # sleepiness scale (label)
        df = pd.read_csv(config.__cfg__.raw_data_corpus, usecols=cols)

        # change the name of column 1..50
        col_names = ['response' + str(i) for i in range(1, 51)]  # response1,...,response51
        col_names.insert(0, 'session_id')
        col_names.append('sss')
        df.columns = col_names

        # extract audio file name from responses
        for col in col_names[1:-1]:  # skip column 'session_id' and 'sss'
            responses = df[col]
            re_pattern = "nlx-\w*-\w*-\w*-\w*-\w*"
            wav_files = list(
                map(lambda x: (re.findall(re_pattern, x)[0] + '.wav') if type(x) is str else '__', responses)
            )
            df[col] = wav_files

        # extract level of sleepiness
        sss = df['sss']
        re_pattern = "\d"
        sss = list(map(lambda x: (re.findall(re_pattern, x)[0]), sss))
        df['sss'] = sss

        # generate data corpus with different columns corresponding to responses
        df.set_index('session_id', inplace=True)
        # generate data corpus with three columns 'file', 'duration', 'sss'
        if not multi_columns:
            wav_files = []  # filenames
            wav_durations = []  # length of wav files
            wav_sss = []  # sleepiness class
            for sess in tqdm(df.index):
                sss = df.loc[sess, 'sss']
                for resp in col_names[1:-1]:  # take response1 to response50
                    wav_file = config.__cfg__.clean_data_dir + '/' + df.loc[sess, resp]
                    if not os.path.isfile(wav_file):
                        continue

                    y, sr = librosa.load(wav_file, sr=None)
                    length = librosa.get_duration(y=y, sr=sr)

                    wav_files.append(df.loc[sess, resp])
                    wav_durations.append(length)
                    wav_sss.append(sss)

            data = {'file': wav_files,
                    'duration': wav_durations,
                    'sss': wav_sss}
            csv_df = pd.DataFrame(data)

            # drop rows that has = 0.0
            csv_df = csv_df[csv_df.duration > 0]

            # export csv file
            csv_df.to_csv(config.__cfg__.clean_data_corpus, index=False)

        elif multi_columns:
            for sess in tqdm(df.index):  # calculate total length of wav files
                length = 0
                for resp in col_names[1:-1]:  # take response1 to response50
                    wav_file = config.__cfg__.clean_data_dir + '/' + df.loc[sess, resp]
                    if not os.path.isfile(wav_file):
                        df.loc[sess, resp] = None
                        continue
                    y, sr = librosa.load(wav_file, sr=None)
                    length += librosa.get_duration(y=y, sr=sr)
                df.loc[sess, 'total_duration'] = length

            # drop rows that has total_duration = 0.0
            df = df[df.total_duration > 0]

            # export csv file
            df.to_csv(config.__cfg__.clean_data_corpus, index=False)

    '''
        Generate features with HuBERT model
        - Each audio file will be chopped down into 3 seconds chunks
        - the HuBERT model will be used to extract features of each chunk of wav signals
        - The selected audio files' name will be save into '.csv/<feature_type><number_of_training_files>.csv'
    '''

    def __generate_hubert_features(self, hubert_model, hubert_processor, pickle_file=''):
        print("\nGenerate {} features for {} audio files".format(self.training_feature_type.upper(),
                                                                 config.__cfg__.NUMBER_TRAINING_SAMPLES))
        _min, _max = float('inf'), -float('inf')  # for normalization purpose

        df = pd.read_csv(config.__cfg__.clean_data_corpus)  # load the csv file
        classes = list(np.unique(df.sss))
        X, y = [], []
        training_files, training_sss, training_durations = [], [], []

        # the clean_data_corpus is in multi-responses format
        #       response1 | response2 | ... | response50 | sss | total_length
        if 'response1' in df.columns:
            class_dist = df.groupby(['sss'])['total_duration'].mean()
            prob_dist = class_dist / class_dist.sum()  # probability distribution
            for _tqdm_index in tqdm(range(config.__cfg__.NUMBER_TRAINING_SAMPLES)):
                rand_class = np.random.choice(class_dist.index, p=prob_dist)  # pick a random class
                r_idx = np.random.choice(df[df.sss == rand_class].index)  # pick a random session (row index)
                c_idx = np.random.choice(df.columns[:-2])  # pick a random response (column index)
                fname = df.at[r_idx, c_idx]
                sss = df.at[r_idx, 'sss']  # get sleepiness level
                if not os.path.isfile(config.__cfg__.clean_data_dir + '/' + fname):
                    continue
                wav_signal, sr = librosa.load(config.__cfg__.clean_data_dir + '/' + fname, sr=None)
                if len(wav_signal) == 0:
                    continue
                duration = librosa.get_duration(wav_signal, sr=sr)

                # save selected audio file's name and its sleepy level
                training_files.append(fname)
                training_sss.append(sss)
                training_durations.append(duration)

                # adding 0 to the end of wav_signal, make the last chunk has desired length.
                if len(wav_signal) % config.__cfg__.CHUNK_SIZE > 0:
                    padding_size = config.__cfg__.CHUNK_SIZE - (len(wav_signal) % config.__cfg__.CHUNK_SIZE)
                    wav_signal = np.pad(wav_signal, (0, padding_size))

                # chop the wav_signal into chunks
                n_chunks = len(wav_signal) // config.__cfg__.CHUNK_SIZE
                for _i in range(0, n_chunks):
                    idx = _i * config.__cfg__.CHUNK_SIZE
                    chunk_signal = wav_signal[idx: idx + config.__cfg__.CHUNK_SIZE]
                    input_values = hubert_processor(chunk_signal, return_tensors="tf",
                                                    sampling_rate=sr).input_values
                    feat = hubert_model(input_values).last_hidden_state
                    _min = min(np.amin(feat), _min)
                    _max = max(np.amax(feat), _max)

                    # reform the shape of feature (1, N, 1024) --> (N, 1024, 1)
                    feat = tf.reshape(feat, [feat.shape[1], feat.shape[2], feat.shape[0]])
                    X.append(feat.numpy())
                    y.append(classes.index(sss))

                # avoid crash when dumping with pickle. Detail in config.py
                if _tqdm_index % config.__cfg__.PICKlE_FILES_THRESHOLD == 0:
                    config.__cfg__.PICKLE_OFFSET_INDEXES.append(len(y) - 1)
                    # print('file {} with length of X is {}'.format(_tqdm_index, len(y) - 1))

        else:  # the clean_data_corpus has 3 columns: file | duration | sss
            class_dist = df.groupby(['sss'])['duration'].mean()
            prob_dist = class_dist / class_dist.sum()  # probability distribution
            df.set_index('file', inplace=True)
            for _tqdm_index in tqdm(range(config.__cfg__.NUMBER_TRAINING_SAMPLES)):
                rand_class = np.random.choice(class_dist.index, p=prob_dist)  # pick randomly a file of random class
                fname = np.random.choice(df[df.sss == rand_class].index)
                sss = df.at[fname, 'sss']
                if not os.path.isfile(config.__cfg__.clean_data_dir + '/' + fname):
                    continue
                wav_signal, sr = librosa.load(config.__cfg__.clean_data_dir + '/' + fname, sr=None)
                if len(wav_signal) == 0:
                    continue
                duration = librosa.get_duration(wav_signal, sr=sr)

                # save selected audio file's name and its sleepy level
                training_files.append(fname)
                training_sss.append(sss)
                training_durations.append(duration)

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
                    feat = hubert_model(input_values).last_hidden_state
                    _min = min(np.amin(feat), _min)
                    _max = max(np.amax(feat), _max)

                    # reform the shape of feature (1, N, 1024) --> (N, 1024, 1)
                    feat = tf.reshape(feat, [feat.shape[1], feat.shape[2], feat.shape[0]])
                    X.append(feat.numpy())
                    y.append(classes.index(sss))

                # avoid crash when dumping with pickle. Detail in config.py
                if _tqdm_index % config.__cfg__.PICKlE_FILES_THRESHOLD == 0:
                    config.__cfg__.PICKLE_OFFSET_INDEXES.append(len(y) - 1)
                    # print('file {} with length of X is {}'.format(_tqdm_index, len(y)-1))

        # save selected audio files to training corpus
        training_corpus = pd.DataFrame(
            data={'file': training_files, 'duration': training_durations, 'sss': training_sss})
        training_corpus.to_csv(
            './csv/' + self.training_feature_type.lower() + str(
                config.__cfg__.NUMBER_TRAINING_SAMPLES) + '_train_corpus.csv',
            index=False)

        # convert features list to a numpy-array
        X, y = np.array(X), np.array(y)
        X = (X - _min) / (_max - _min)
        y = to_categorical(y, num_classes=len(classes))

        config.__cfg__.min_X = _min
        config.__cfg__.max_X = _max

        # save to pickle file.
        if len(pickle_file) > 0:
            print('Saving to pickle file at:', config.__cfg__.pickle_dir + '/' + pickle_file)

            # avoid crash when dumping with pickle. Detail in config.py
            offsets = config.__cfg__.PICKLE_OFFSET_INDEXES
            if len(offsets) > 1:
                name_extension = os.path.splitext(pickle_file)
                for i in range(len(offsets) - 1):
                    pickle_file_i = name_extension[0] + '_' + str(i + 1).rjust(3, '0') + name_extension[1]
                    _start, _end = offsets[i], offsets[i + 1]
                    print('\t- writing to {}; offset = {}->{}'.format(pickle_file_i, _start, _end - 1))
                    with open(config.__cfg__.pickle_dir + '/' + pickle_file_i, 'wb') as f_handle:
                        config.__cfg__.data = (X[_start:_end], y[_start:_end])
                        pickle.dump(config.__cfg__, f_handle, protocol=4)
                        f_handle.close()
                else:  # the last pickle file
                    pickle_file_i = name_extension[0] + '_' + str(len(offsets)).rjust(3, '0') + name_extension[1]
                    _start = offsets[-1]  # the last offset
                    config.__cfg__.data = (X[_start:], y[_start:])
                    print('\t- writing to {}; offset = {}->end'.format(pickle_file_i, offsets[-1]))
                    with open(config.__cfg__.pickle_dir + '/' + pickle_file_i, 'wb') as f_handle:
                        pickle.dump(config.__cfg__, f_handle, protocol=4)
                        f_handle.close()
            else:
                config.__cfg__.data = (X, y)
                with open(config.__cfg__.pickle_dir + '/' + pickle_file, 'wb') as f_handle:
                    pickle.dump(config.__cfg__, f_handle, protocol=4)
        else:
            pass

        print('Done.\n')
        # and return list of features as well
        return X, y

    '''
        Generate random mfcc features from wav files.
            - The number of training features is:       10*total_length_wav_length
            - We will pick random 4000 samples from a wav file. This such wav file will be chosen randomly according to
              probabilities of sleepiness class.
            - We calculate 13 mfcc coefficients with 26 filter banks and number of FFT = 512
    '''

    def __generate_mfcc_features(self, pickle_file=''):
        print("\nGenerate {} features for {} audio files".format(self.training_feature_type.upper(),
                                                                 config.__cfg__.NUMBER_TRAINING_SAMPLES))
        X, y = [], []
        training_files, training_sss, training_durations = [], [], []

        _min, _max = float('inf'), -float('inf')
        df = pd.read_csv(config.__cfg__.clean_data_corpus)  # load the csv file

        # data_corpus.csv was saved in multiple columns (response1 ... response50)
        # then convert it to single column format (file | duration | sss)
        if 'total_duration' in df.columns:
            dat = {'file': [], 'duration': [], 'sss': []}
            f_names = []
            f_durations = []
            f_sss = []
            col_names = ['response' + str(i) for i in range(1, 51)]  # response1,...,response50
            for i in df.index:
                for j in col_names:
                    wav_file = config.__cfg__.clean_data_dir + '/' + df.at[i, j]
                    if not os.path.isfile(wav_file):
                        continue
                    wav_signal, sr = librosa.load(wav_file, sr=None)
                    if len(wav_signal) == 0:
                        continue
                    duration = librosa.get_duration(y=wav_signal, sr=sr)
                    f_names.append(df.at[i, j])
                    f_durations.append(duration)
                    f_sss.append(df.at[i, 'sss'])
            dat = {'file': f_names, 'duration': f_durations, 'sss': f_sss}
            df = pd.DataFrame(dat)  # update dataframe
        else:
            pass
        df.set_index('file', inplace=True)
        classes = list(np.unique(df.sss))
        class_dist = df.groupby(['sss'])['duration'].mean()
        prob_dist = class_dist / class_dist.sum()  # probability distribution

        for _ in tqdm(range(config.__cfg__.NUMBER_TRAINING_SAMPLES)):
            rand_class = np.random.choice(class_dist.index, p=prob_dist)  # randomly choose class
            rand_file = np.random.choice(df[df.sss == rand_class].index)  # randomly choose a file in the class
            sss = df.at[rand_file, 'sss']

            wav_signal, rate = librosa.load(config.__cfg__.clean_data_dir + '/' + rand_file, sr=None)
            if len(wav_signal) == 0:
                continue
            duration = librosa.get_duration(wav_signal, sr=rate)

            # save selected audio file's name and its sleepy level
            training_files.append(rand_file)
            training_sss.append(sss)
            training_durations.append(duration)

            # adding 0 to the end of wav_signal, make the last chunk has desired length.
            if len(wav_signal) % config.__cfg__.MFCC_CHUNK_SIZE > 0:
                padding_size = config.__cfg__.MFCC_CHUNK_SIZE - (len(wav_signal) % config.__cfg__.MFCC_CHUNK_SIZE)
                wav_signal = np.pad(wav_signal, (0, padding_size))

            # chop the wav_signal into chunks
            n_chunks = len(wav_signal) // config.__cfg__.CHUNK_SIZE
            for _i in range(0, n_chunks):
                idx = _i * config.__cfg__.CHUNK_SIZE
                chunk_signal = wav_signal[idx: idx + config.__cfg__.CHUNK_SIZE]
                feat = mfcc(chunk_signal, rate, numcep=13, nfilt=26, nfft=512).T  # reshape to the form (512, 13)
                _min = min(np.amin(feat), _min)
                _max = max(np.amax(feat), _max)
                X.append(feat)
                y.append(classes.index(sss))

        # save selected audio files to training corpus
        training_corpus = pd.DataFrame(
            data={'file': training_files, 'duration': training_durations, 'sss': training_sss})
        training_corpus.to_csv('./csv/' + self.training_feature_type.lower() + str(
            config.__cfg__.NUMBER_TRAINING_SAMPLES) + '_train_corpus.csv',
                               index=False)

        config.__cfg__.min_X, config.__cfg__.max_X = _min, _max
        # normalize X
        X, y = np.array(X), np.array(y)
        X = (X - _min) / (_max - _min)

        # reshape the mfcc features into the form of (N, 512, 13)
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        y = to_categorical(y, num_classes=len(classes))

        # save to pickle file
        if len(pickle_file) > 0:
            print('Saving to pickle file at:', config.__cfg__.pickle_dir + '/' + pickle_file)
            config.__cfg__.data = (X, y)
            try:
                with open(config.__cfg__.pickle_dir + '/' + pickle_file, 'wb') as f:
                    pickle.dump(config.__cfg__, f, protocol=4)
            except RuntimeError as e:
                print(e)

        print('Done.\n')
        # return the features as well
        return X, y

    '''
        Generate random spectrogram features from wav files.
            - The number of training features is:       10*total_length_wav_length
            - We will pick random 4000 samples from a wav file. This such wav file will be chosen randomly according to
              probabilities of sleepiness class.
            - We calculate spectrogram feature thanks to Nam.S.Nguyen's function 
    '''

    def __generate_spectrogram_features(self, pickle_file=''):
        print("\nGenerate {} features for {} audio files".format(self.training_feature_type.upper(),
                                                                 config.__cfg__.NUMBER_TRAINING_SAMPLES))

        X, y = [], []
        training_files, training_sss, training_durations = [], [], []
        _min, _max = float('inf'), -float('inf')

        df = pd.read_csv(config.__cfg__.clean_data_corpus)  # load the csv file

        # If data_corpus.csv was saved in multiple columns (response1 ... response50) then convert it into the format
        #   (file | duration | sss)
        if 'total_duration' in df.columns:
            dat = {'file': [], 'duration': [], 'sss': []}
            f_names = []
            f_durations = []
            f_sss = []
            col_names = ['response' + str(i) for i in range(1, 51)]  # response1,...,response50
            for i in df.index:
                for j in col_names:
                    wav_file = config.__cfg__.clean_data_dir + '/' + df.at[i, j]
                    if not os.path.isfile(wav_file):
                        continue
                    wav_signal, sr = librosa.load(wav_file, sr=None)
                    if len(wav_signal) == 0:
                        continue
                    duration = librosa.get_duration(y=wav_signal, sr=sr)
                    f_names.append(df.at[i, j])
                    f_durations.append(duration)
                    f_sss.append(df.at[i, 'sss'])
            dat = {'file': f_names, 'duration': f_durations, 'sss': f_sss}
            df = pd.DataFrame(dat)
        else:
            pass

        df.set_index('file', inplace=True)
        classes = list(np.unique(df.sss))
        class_dist = df.groupby(['sss'])['duration'].mean()
        prob_dist = class_dist / class_dist.sum()  # probability distribution

        for _ in tqdm(range(config.__cfg__.NUMBER_TRAINING_SAMPLES)):
            rand_class = np.random.choice(class_dist.index, p=prob_dist)  # randomly choose class
            rand_file = np.random.choice(df[df.sss == rand_class].index)  # randomly choose a file in the class

            wav_signal, rate = librosa.load(config.__cfg__.clean_data_dir + '/' + rand_file, sr=None)
            sss = df.at[rand_file, 'sss']
            if len(wav_signal) == 0:
                continue
            duration = librosa.get_duration(wav_signal, sr=rate)

            # save selected audio file's name and its sleepy level
            training_files.append(rand_file)
            training_sss.append(sss)
            training_durations.append(duration)

            # adding 0 to the end of wav_signal, make the last chunk has desired length.
            if len(wav_signal) % config.__cfg__.MFCC_CHUNK_SIZE > 0:
                padding_size = config.__cfg__.MFCC_CHUNK_SIZE - (len(wav_signal) % config.__cfg__.MFCC_CHUNK_SIZE)
                wav_signal = np.pad(wav_signal, (0, padding_size))

            # chop the wav_signal into chunks
            n_chunks = len(wav_signal) // config.__cfg__.CHUNK_SIZE
            for _i in range(0, n_chunks):
                idx = _i * config.__cfg__.CHUNK_SIZE
                chunk_signal = wav_signal[idx: idx + config.__cfg__.CHUNK_SIZE]
                feat = calc_spectrogram(chunk_signal, rate)
                _min = min(np.amin(feat), _min)
                _max = max(np.amax(feat), _max)
                X.append(feat)
                y.append(classes.index(sss))

        # save selected audio files to training corpus
        training_corpus = pd.DataFrame(
            data={'file': training_files, 'duration': training_durations, 'sss': training_sss})
        training_corpus.to_csv('./csv/' + self.training_feature_type.lower() + str(
            config.__cfg__.NUMBER_TRAINING_SAMPLES) + '_train_corpus.csv',
                               index=False)

        # normalize X
        config.__cfg__.min_X, config.__cfg__.max_X = _min, _max
        X, y = np.array(X), np.array(y)
        X = (X - _min) / (_max - _min)
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        y = to_categorical(y, num_classes=len(classes))

        # save to pickle file
        if len(pickle_file) > 0:
            config.__cfg__.data = (X, y)
            print('Saving to pickle file at:', config.__cfg__.pickle_dir + '/' + pickle_file)
            try:
                with open(config.__cfg__.pickle_dir + '/' + pickle_file, 'wb') as f:
                    pickle.dump(config.__cfg__, f, protocol=4)
            except RuntimeError as e:
                print(e)

        print('Done.\n')
        # return list of features as well
        return X, y

    '''
        This function generates training features from audio files
    '''

    def generate_training_features(self, hubert_model=None, hubert_processor=None):
        if self.training_feature_type.lower() == 'hubert':
            if config.__cfg__.NUMBER_TRAINING_SAMPLES == -1:
                return self.__generate_hubert_features_corpus(hubert_model, hubert_processor,
                                                              pickle_file=self.training_feature_type.lower() + '.p')
            else:
                return self.__generate_hubert_features(hubert_model, hubert_processor,
                                                       pickle_file=self.training_feature_type.lower() + \
                                                                   str(config.__cfg__.NUMBER_TRAINING_SAMPLES) + '.p')
        elif self.training_feature_type.lower() == 'mfcc':
            return self.__generate_mfcc_features(pickle_file=self.training_feature_type.lower() + \
                                                             str(config.__cfg__.NUMBER_TRAINING_SAMPLES) + '.p')
        elif self.training_feature_type.lower() == 'spectrogram':
            return self.__generate_mfcc_features(pickle_file=self.training_feature_type.lower() + \
                                                             str(config.__cfg__.NUMBER_TRAINING_SAMPLES) + '.p')
        else:
            return None

    '''
        This function load training feature (X, y) from pickle file
    '''

    def load_training_pickle(self, pickle_files=None):
        if not pickle_files:
            pickle_files = [config.__cfg__.pickle_dir + '/' + \
                            self.training_feature_type + int(config.__cfg__.NUMBER_TRAINING_SAMPLES) + '.p']
        X, y = [], []

        for fname in pickle_files:
            fname = config.__cfg__.pickle_dir + '/' + fname
            if not os.path.isfile(fname):
                print('pickle file was not found:', fname)
                continue
            with open(fname, 'rb') as f_handle:
                feat = pickle.load(f_handle)

                print('Loaded from ' + fname + ' --> X={}, y={}'.format(feat.data[0].shape, feat.data[1].shape))

                X = X + list(feat.data[0])
                y = y + list(feat.data[1])
                f_handle.close()

        X, y = np.array(X), np.array(y)
        print('Total: X={}, y={}'.format(X.shape, y.shape))
        return X, y, feat.min_X, feat.max_X

    '''
        This function generating testing corpus. The function will select a number of audio files as testing dataset;
        these selected files does not occur in training dataset 
    '''

    def generate_testing_corpus(self, number_of_test, training_corpus=None, shuffle=True):
        print("\nGenerate testing corpus of {} audio files".format(config.__cfg__.NUMBER_TESTING_SAMPLES))
        if not training_corpus:
            fname = self.training_feature_type.lower() + config.__cfg__.NUMBER_TRAINING_SAMPLES + '.csv'
        else:
            fname = training_corpus

        if not number_of_test:
            number_of_test = config.__cfg__.NUMBER_TESTING_SAMPLES

        df_train = pd.read_csv('./csv/' + fname)  # training corpus is always in single column format
        df_train.set_index('file', inplace=True)

        df_clean_data = pd.read_csv(config.__cfg__.clean_data_corpus)  # load the csv file
        df_clean_data.set_index('file', inplace=True)

        testing_files, testing_duration, testing_sss = [], [], []

        # if the data corpus was saved in multiple column format
        if 'total_duration' in df_clean_data.columns:
            pass
        # if the data corpus was save in single column format
        else:
            class_dist = df_clean_data.groupby(['sss'])['duration'].mean()
            prob_dist = class_dist / class_dist.sum()  # probability distribution

            for _ in tqdm(range(number_of_test)):
                rand_class = np.random.choice(class_dist.index, p=prob_dist)  # pick a random class
                fname = np.random.choice(df_clean_data[df_clean_data.sss == rand_class].index)
                sss = df_clean_data.at[fname, 'sss']
                duration = df_clean_data.at[fname, 'duration']

                while (fname in df_train.index) or (not os.path.isfile(config.__cfg__.clean_data_dir + '/' + fname)):
                    fname = np.random.choice(df_clean_data[df_clean_data.sss == rand_class].index)
                    sss = df_clean_data.at[fname, 'sss']
                    duration = df_clean_data.at[fname, 'duration']

                testing_sss.append(sss)
                testing_duration.append(duration)
                testing_files.append(fname)

        testing_df = pd.DataFrame(data={'file': testing_files, 'duration': testing_duration, 'sss': testing_sss})
        testing_df.to_csv('./csv/testing_corpus' + str(number_of_test) + '.csv', index=False)

        print('Done!')
