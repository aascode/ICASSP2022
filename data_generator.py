import os
import re
import numpy as np
import pandas as pd
import json
import pickle

import librosa
from python_speech_features import mfcc

from tqdm import tqdm
from tensorflow.keras.utils import to_categorical

from utils import envelope, calc_spectro




class DataGenerator:
    def __init__(self,
                 raw_data_dir='./data', clean_data_dir='./clean',
                 raw_data_corpus='new2.csv', clean_data_corpus='clean_data.csv',
                 training_feature_type = 'hubert',
                 pickle_dir = './pickle'):
        self.raw_data_dir = raw_data_dir
        self.clean_data_dir = clean_data_dir
        self.raw_data_corpus = raw_data_corpus
        self.clean_data_corpus = clean_data_corpus

        self.CLEAN_SAMPLE_RATE = 16000          # sample rate = 16kHz for clean audio files
        self.training_feature_type = training_feature_type     # mfcc | spectrogram | hubert
        self.pickle_dir = pickle_dir



    '''
        This function generates cleaned wav files from original audio files.
        (down sampling to 16kHz and save to directory ./clean)
    '''
    def generate_clean_data(self, log_file='clean_data.log'):
        cols = [i for i in range(3, 53)]                # response samples
        cols.insert(0, 0)                               # session id
        cols.append(143)                                # sleepiness scale (label)
        df = pd.read_csv(self.raw_data_corpus, usecols=cols)

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
        sss_levels = list(map(lambda x: (re.findall(str_pattern, x)[0]) , sss_levels))
        df['sss'] = sss_levels


        # Down sampling to 16000 for every audio sample files
        print('Generating clean audio samples...')

        skipped_files = {}
        df.set_index('session_id', inplace=True)
        for sess in tqdm(df.index):
            #print('Cleaning for session_id:', sess)
            for resp in col_names[1:-1]:  # take response1 to response50
                f = df.loc[sess, resp]
                wav_file = self.raw_data_dir + '/' + str(sess) + '/' + f
                if not os.path.isfile(wav_file):
                    skipped_files[wav_file] = 'file not found'
                    continue

                if os.path.isfile(self.clean_data_dir + '/' + f):  # skip duplicated files
                    skipped_files[wav_file] = 'duplicated file'
                    continue

                if os.path.getsize(wav_file) == 0:  # skip empty files
                    skipped_files[wav_file] = 'file size = 0'
                    continue

                signal, rate = librosa.load(wav_file, sr=None)
                if len(signal) == 0:  # for some reasons, the wav file doesn't contain any signal
                    skipped_files[wav_file] = 'length of signal = 0'
                    continue

                # Downsampling the signal to 16kHz
                new_signal = librosa.resample(signal, rate, self.CLEAN_SAMPLE_RATE)
                mask = envelope(new_signal, self.CLEAN_SAMPLE_RATE)
                librosa.output.write_wav(filename=self.clean_data_dir + '/' + f,
                                         rate=self.CLEAN_SAMPLE_RATE,
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
    def generate_clean_data_corpus(self, multi_columns=False):
        cols = [i for i in range(3, 53)]  # response samples
        cols.insert(0, 0)  # session id
        cols.append(143)  # sleepiness scale (label)
        df = pd.read_csv(self.raw_data_corpus, usecols=cols)

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
        sss = list( map(lambda x: (re.findall(re_pattern, x)[0] ) , sss) )
        df['sss'] = sss

        # generate data corpus with different columns corresponding to responses
        df.set_index('session_id', inplace=True)
        # generate data corpus with three columns 'file', 'duration', 'sss'
        if not multi_columns:
            wav_files = []          # filenames
            wav_durations = []      # length of wav files
            wav_sss = []            # sleepiness class
            for sess in tqdm(df.index):
                sss = df.loc[sess, 'sss']
                for resp in col_names[1:-1]:  # take response1 to response50
                    wav_file = self.clean_data_dir + '/' + df.loc[sess, resp]
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
            csv_df.to_csv(self.clean_data_corpus, index=False)

        elif multi_columns:
            for sess in tqdm(df.index): # calculate total length of wav files
                length = 0
                for resp in col_names[1:-1]:  # take response1 to response50
                    wav_file = self.clean_data_dir + '/' + df.loc[sess, resp]
                    if not os.path.isfile(wav_file):
                        df.loc[sess, resp] = None
                        continue
                    y, sr = librosa.load(wav_file, sr=None)
                    length += librosa.get_duration(y=y, sr=sr)
                df.loc[sess, 'total_duration'] = length

            # drop rows that has total_duration = 0.0
            df = df[df.total_duration > 0]

            # export csv file
            df.to_csv(self.clean_data_corpus, index=False)


    '''
        Generate features with HuBERT model
    '''
    def __generate_hubert_features(self, hubert_model, hubert_processor, pickle_file=''):
        df = pd.read_csv(self.clean_data_corpus)    #load the csv file

        classes = list(np.unique(df.sss))
        X = []
        y = []
        _max_numOfRows = -float('inf')

        # the clean_data_corpus is in multi-responses format
        #       response1 | response2 | ... | response50 | sss | total_length
        if 'response1' in df.columns:
            for i in tqdm(df.index):
                sss = df.at[i, 'sss']        # the second column from the right
                for j in range(0, 51):      # column0...column50 contain wav-file-name
                    fname = df.iloc[i, j]
                    if not os.path.isfile(self.clean_data_dir + '/' + fname):
                        continue
                    wav_signal, sr = librosa.load(self.clean_data_dir + '/' + fname, sr=None)
                    input_values = hubert_processor(wav_signal, return_tensors="tf", sampling_rate=sr).input_values
                    feat = hubert_model(input_values).hidden_states

                    # 'feat' is a list of 25 elements; each element is an array of shape (1, N, 1024) --> We'll
                    # convert to (N, 1024) before store 'feat' into X
                    feat = list(map(lambda arr: np.squeeze(arr), list(feat)))
                    X.append(feat)
                    y.append(sss)
                    _max_numOfRows = max(_max_numOfRows, feat[0].shape[0])
        else:   # the clean_data_corpus has 3 columns: file | duration | sss
            df.set_index('file', inplace=True)
            for fname in tqdm(df.index):
                sss = df.loc[fname, 'sss']
                if not os.path.isfile(self.clean_data_dir + '/' + fname):
                    continue
                wav_signal, sr = librosa.load(self.clean_data_dir + '/' + fname, sr=None)
                input_values = hubert_processor(wav_signal, return_tensors="tf", sampling_rate=sr).input_values
                feat = hubert_model(input_values).hidden_states

                # 'feat' is a list of 25 elements; each element is an array of shape (1, N, 1024) --> We'll
                # convert to (N, 1024) before store 'feat' into X
                feat = list(map(lambda arr: np.squeeze(arr), list(feat)))
                X.append(feat)
                y.append(sss)
                _max_numOfRows = max(_max_numOfRows, feat[0].shape[0])

        y = to_categorical(y, num_classes=len(classes))
        # Do padding each feature in X
        for i, x in X:
            # print('before:', x[0].shape)
            r, c = x[0].shape
            if r < _max_numOfRows:
                padded_r = _max_numOfRows - r
                padded_0 = np.zeros((padded_r, c))
                x = list(map(lambda ft: np.vstack([ft, padded_0]), x))
                X[i] = x
            # print('   --> after padding:', x[0].shape)

        # save to pickle file
        if len(pickle_file) > 0:
            print('Saving to pickle file at:', self.pickle_dir + '/' + pickle_file)
            with open(self.pickle_dir + '/' + pickle_file, 'wb') as f:
                pickle.dump((X, y), f, protocol=4)

        # and return the features as well
        return X, y


    '''
        Generate random mfcc features from wav files.
            - The number of training features is:       10*total_length_wav_length
            - We will pick random 4000 samples from a wav file. This such wav file will be chosen randomly according to
              probabilities of sleepiness class.
            - We calculate 13 mfcc coefficients with 26 filter banks and number of FFT = 512
    '''
    def __generate_random_mfcc_features(self, pickle_file=''):

        _STEP = int(self.CLEAN_SAMPLE_RATE/4) # 0.25 second
        X = []
        y = []
        _min, _max = float('inf'), -float('inf')

        df = pd.read_csv(self.clean_data_corpus)  # load the csv file

        # data_corpus.csv was saved in multiple columns (response1 ... response50) then convert it to the format
        #   (file | duration | sss)
        if 'total_duration' in df.columns:
            dat = {'file':[], 'duration': [], 'sss': []}
            file_names = []
            file_durations = []
            file_sss = []
            col_names = ['response' + str(i) for i in range(1, 51)]  # response1,...,response50
            for i in df.index:
                for j in col_names:
                    wav_file = self.clean_data_dir + '/' + df.at[i, j]
                    if not os.path.isfile(wav_file):
                        continue
                    y, sr = librosa.load(wav_file, sr=None)
                    duration = librosa.get_duration(y=y, sr=sr)
                    file_names.append(df.at[i, j])
                    file_durations.append(duration)
                    file_sss.append(df.at[i, 'sss'])
            dat = {'file': file_names, 'duration': file_durations, 'sss': file_sss}
            df = pd.DataFrame(dat)
        else:
            pass
        df.set_index('file', inplace=True)

        classes = list(np.unique(df.sss))
        class_distrib = df.groupby(['sss'])['duration'].sum()
        prob_dist = class_distrib / class_distrib.sum()  # probability distribution
        #n_samples = 2 * int(df['duration'].sum()/0.1)  # number of samples we gonna take from wav files
        n_samples = 5000

        print("Total length of wav files %.2f (hours)" % df['duration'].sum()/3600)
        print("{} samples gonna be selected(each sample contains {} values)".format(n_samples, _STEP))
        for _ in tqdm(range(n_samples)):
            rand_class = np.random.choice(class_distrib.index, p=prob_dist)  # randomly choose class
            file = np.random.choice(df[df.sss == rand_class].index)  # randomly choose a file in the class

            wav, rate = librosa.load(self.clean_data_dir + '/' + file, sr=None)
            label = df.at[file, 'sss']

            # take random index of sample in wav file then calculate the feature on that
            if wav.shape[0] < _STEP:
                continue

            rand_index = np.random.randint(0, wav.shape[0] - _STEP )
            sample = wav[rand_index:rand_index + _STEP]

            X_sample =  mfcc(sample, rate, numcep=13, nfilt=26, nfft=512).T
            _min = min(np.amin(X_sample), _min)
            _max = max(np.amax(X_sample), _max)
            X.append(X_sample)
            y.append(classes.index(label))

        # normalize X
        X, y = np.array(X), np.array(y)
        X = (X - _min) / (_max - _min)
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

        y = to_categorical(y, num_classes=len(classes))

        # save to pickle file
        if len(pickle_file) > 0:
            print('Saving to pickle file at:', self.pickle_dir + '/' + pickle_file)
            with open(self.pickle_dir + '/' + pickle_file, 'wb') as f:
                pickle.dump( (X, y), f, protocol=4)

        return X, y


    '''
        Generate random spectrogram features from wav files.
            - The number of training features is:       10*total_length_wav_length
            - We will pick random 4000 samples from a wav file. This such wav file will be chosen randomly according to
              probabilities of sleepiness class.
            - We calculate spectrogram feature thanks to Nam.S.Nguyen's function 
    '''
    def __generate_random_spectrogram_features(self, pickle_file=''):
        _STEP = int(self.CLEAN_SAMPLE_RATE/4) # 0.25 second
        X = []
        y = []
        _min, _max = float('inf'), -float('inf')

        df = pd.read_csv(self.clean_data_corpus)  # load the csv file

        # If data_corpus.csv was saved in multiple columns (response1 ... response50) then convert it into the format
        #   (file | duration | sss)
        if 'total_duration' in df.columns:
            dat = {'file':[], 'duration': [], 'sss': []}
            file_names = []
            file_durations = []
            file_sss = []
            col_names = ['response' + str(i) for i in range(1, 51)]  # response1,...,response50
            for i in df.index:
                for j in col_names:
                    wav_file = self.clean_data_dir + '/' + df.at[i, j]
                    if not os.path.isfile(wav_file):
                        continue
                    y, sr = librosa.load(wav_file, sr=None)
                    duration = librosa.get_duration(y=y, sr=sr)
                    file_names.append(df.at[i, j])
                    file_durations.append(duration)
                    file_sss.append(df.at[i, 'sss'])
            dat = {'file': file_names, 'duration': file_durations, 'sss': file_sss}
            df = pd.DataFrame(dat)
        else:
            pass
        df.set_index('file', inplace=True)

        classes = list(np.unique(df.sss))
        class_distrib = df.groupby(['sss'])['duration'].sum()
        prob_dist = class_distrib / class_distrib.sum()  # probability distribution
        #n_samples = 2 * int(df['duration'].sum()/0.1)  # number of samples we gonna take from wav files
        n_samples = 5000

        print("Total length of wav files %.2f (hours)" % df['duration'].sum()/3600)
        print("{} training samples gonna be selected(each sample contains {} values)".format(n_samples, _STEP))
        for _ in tqdm(range(n_samples)):
            rand_class = np.random.choice(class_distrib.index, p=prob_dist)  # randomly choose class
            file = np.random.choice(df[df.sss == rand_class].index)  # randomly choose a file in the class

            wav, rate = librosa.load(self.clean_data_dir + '/' + file, sr=None)
            label = df.at[file, 'sss']

            # take random index of sample in wav file then calculate the feature on that
            if wav.shape[0] < _STEP:
                continue

            rand_index = np.random.randint(0, wav.shape[0] - _STEP )
            sample = wav[rand_index:rand_index + _STEP]
            X_sample =  calc_spectro(sample, rate)

            _min = min(np.amin(X_sample), _min)
            _max = max(np.amax(X_sample), _max)

            X.append(X_sample)
            y.append(classes.index(label))


        # normalize X
        X, y = np.array(X), np.array(y)
        X = (X - _min) / (_max - _min)
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

        y = to_categorical(y, num_classes=len(classes))

        # save to pickle file
        if len(pickle_file) > 0:
            print('Saving to pickle file at:', self.pickle_dir + '/' + pickle_file)
            with open(self.pickle_dir + '/' + pickle_file, 'wb') as f:
                pickle.dump( (X, y), f, protocol=4)

        return X, y


    '''
        This function generates training features from audio files
    '''
    def generate_training_features(self, hubert_model=None, hubert_processor=None):
        if self.training_feature_type == 'hubert':
            return self.__generate_hubert_features(hubert_model=hubert_model,
                                                   hubert_processor=hubert_processor,
                                                   pickle_file=self.training_feature_type+'.p')
        elif self.training_feature_type == 'mfcc':
            return self.__generate_random_mfcc_features(pickle_file=self.training_feature_type+'.p')

        elif self.training_feature_type == 'spectrogram':
            return self.__generate_random_mfcc_features(pickle_file=self.training_feature_type + '.p')

        else:
            return None