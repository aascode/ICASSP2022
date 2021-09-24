import argparse
import json

import librosa
from transformers import Wav2Vec2Processor, TFHubertModel

import pickle
import seaborn as sn

import numpy as np
import tensorflow as tf
import os

from tqdm import tqdm

import audiofile

from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import History
import matplotlib.pyplot as plt

from plot_confusion_matrix import pretty_plot_confusion_matrix
from data_generator import DataGenerator
import model as my_models
import config

import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, plot_confusion_matrix

# update font size for pyplot
SMALL_FONT_SIZE = 10
MEDIUM_FONT_SIZE = 13
BIGGER_FONT_SIZE = 16

plt.rc('font', size=SMALL_FONT_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_FONT_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_FONT_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_FONT_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_FONT_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_FONT_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_FONT_SIZE)  # fontsize of the figure title

# prevent run out memory of GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

'''
Train with group-X of responses; Binary classification
    Step1: throw wav file into HuBERT --> Receive output of last hidden layer as N vectors (N x 1024)
    Step2: doing global average to get an embedding vector of (1 x 1024)
    Step3: train the embedding vectors with new added layers.
'''


def run_train(args):
    '''
    Get HuBert embeddings for slected responses
    :param args:
        - sessions_list: the session id of selected one
        - response_set: list of the selected columns
    :return: A matrix (m x 1024 x n) where n = len(response_set), and m = len(sessions_list)
    '''

    def get_embeddings(selected_indexes, dataframe, response_set=['response1', 'response6']):
        X_embeddings = []
        y_labels = []

        # ucols = ['session_id', 'startTime']
        # for answer in response_set:
        #     ucols.append(answer)
        # ucols.append('sss')
        # df = pd.read_csv('csv/clean_data_corpus_multicol.csv', usecols=ucols)

        zeropad_count = 0
        total_count = 0

        error_files = []
        causes = []
        for idx in tqdm(selected_indexes):  # go through the selected sessions
            label = dataframe.at[idx, 'sss']
            X = [[] for _ in range(len(response_set))]  # initialize each selected answer-column as []

            numpy_files = ['pickle/hubert-embedding/' + str(dataframe.at[idx, answer]).replace('.wav', '.npy') for
                           answer in
                           response_set]

            for i in range(len(numpy_files)):  # load HuBERT embedding from numpy file of each response column.
                if os.path.isfile(numpy_files[i]):
                    X[i] = np.load(numpy_files[i])

                    # error even the numpy file existed
                    if np.isnan(X[i]).any():
                        X[i] = np.zeros((1, 1024))
                        zeropad_count += 1
                        error_files.append(numpy_files[i])
                        causes.append('contains NaN')
                    elif str(X[i].shape) == '()':
                        # print('I received:', X[i])
                        X[i] = np.zeros((1, 1024))
                        zeropad_count += 1
                        error_files.append(numpy_files[i])
                        causes.append('not a list')
                else:
                    X[i] = np.zeros((1, 1024))
                    zeropad_count += 1
                    error_files.append(numpy_files[i])
                    causes.append('File not found')

                # reshape the X[i]
                X[i] = np.array(X[i]).reshape(1024, 1)
                total_count += 1

            # convert x to an array (1024 x n)
            X = np.array(X).T
            X_embeddings.append(X)
            y_labels.append(label)

        # err_df = pd.DataFrame(data={'file': error_files, 'cause': causes})
        # err_df.to_csv('error_nyp_files.csv')

        # convert X_embeddings to a shape (m x 1024 x n) where m=len(sessions_list), and n=len(response_set)
        X_embeddings = np.array(X_embeddings)
        X_embeddings = np.squeeze(X_embeddings)

        # normalize the embedding
        _min, _max = np.min(X_embeddings), np.max(X_embeddings)
        # X_embeddings = (X_embeddings - _min) / (_max - _min)

        # train with 2 classes (1..4 means alert; 5..7 means sleepy)
        y_labels = list(map(lambda y: 1 if y >= config.__cfg__.sleepy_threshold else 0, y_labels))  # 2 classes
        y_labels = to_categorical(y_labels, 2)

        # train with 7 classes (0..6)
        # y_labels = list(map(lambda y: y-1, y_labels))  # 2 classes
        # y_labels = to_categorical(y_labels, 7)

        print('{} Zero padding out of {}'.format(zeropad_count, total_count))
        return X_embeddings, y_labels

    '''
    generate HuBERT embedding for Thorsten dataset
    '''

    def generate_thorsen_embedding():
        wav_dir = 'test_audio/thorsten-emotional_v02/sleepy'
        df = pd.read_csv('csv/thorsten_test.csv')

        hubert_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
        hubert_model = TFHubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
        for idx in tqdm(df.index):
            label = df.at[idx, 'sss']
            fn = df.at[idx, 'file']

            wav_signal, sr = librosa.load(wav_dir + '/' + fn, sr=16000)
            if len(wav_signal) == 0:
                continue
            input_values = hubert_processor(wav_signal, return_tensors="tf", sampling_rate=sr).input_values
            feat = hubert_model(input_values).last_hidden_state
            x_embedding = np.squeeze(feat)  # (1, N, 1024) --> (N, 1024)
            x_embedding = np.mean(x_embedding, axis=0).T  # (1024, )

            # save to numpy file
            name_extension = os.path.splitext(fn)
            np.save('test_audio/thorsten-emotional_v02/sleepy/' + name_extension[0] + '.npy', x_embedding)

    '''
    this function return HuBERT embedding for testing dataset Thorsten 
    '''

    def get_thorsten_embedding(response_set):
        print('\nGetting Thorsten test data')
        df = pd.read_csv('csv/thorsten_test.csv')
        filenames = df.file
        filenames = list(map(lambda f: 'test_audio/thorsten-emotional_v02/sleepy/' + os.path.splitext(f)[0] + '.npy', \
                             filenames))
        df.file = filenames

        X_embeddings = []
        y_labels = []
        for idx in tqdm(df.index):
            label = df.at[idx, 'sss']
            fn = df.at[idx, 'file']

            x_embedding = np.load(fn)
            if np.isnan(x_embedding).any():
                x_embedding = np.zeros((1, 1024))
            elif str(x_embedding.shape) == '()':
                x_embedding = np.zeros((1, 1024))

            X = [[] for _ in range(len(response_set))]  # initialize each selected answer-column as []
            for i in range(len(response_set)):
                X[i] = x_embedding
                X[i] = np.array(X[i]).reshape(1024, 1)

            # convert x to an array (1024 x n)
            X = np.array(X).T
            X_embeddings.append(X)
            y_labels.append(label)

        # convert X_embeddings to a shape (m x 1024 x n) where m=len(df), and n=len(response_set)
        X_embeddings = np.array(X_embeddings)
        X_embeddings = np.squeeze(X_embeddings)

        y_origin_labels = y_labels
        # train with 2 classes (1..4 means alert; 5..7 means sleepy)
        y_labels = list(map(lambda y: 1 if y >= config.__cfg__.sleepy_threshold else 0, y_labels))  # 2 classes
        y_labels = to_categorical(y_labels, 2)

        return X_embeddings, y_labels, y_origin_labels

    # -------------------------------------------------------------------------
    # generate_thorsen_embedding()
    print('Loading data...')
    dataset = pd.read_csv('csv/clean_data_corpus_multicol.csv')
    dataset = shuffle(dataset)
    # drop non-existed sessions
    mask = []
    for i in range(len(dataset)):
        s = config.__cfg__.raw_data_dir + '/' + dataset.at[i, 'session_id']
        if not os.path.isdir(s):
            mask.append(i)
    dataset = dataset.drop(mask)

    # group3 (None-semantic)
    resp_set3 = ['response7', 'response8', 'response9'] + ['response' + str(i) for i in range(35, 45)]
    # group4 (Picture description)
    resp_set4 = ['response3'] + ['response' + str(i) for i in range(10, 35)]
    responses_set_old = [['response1', 'response6'],  # group1 (reading )
                         ['response46', 'response48'],  # group2 (Hearing and repeat)
                         resp_set3,  # group3 (None-semantic)
                         resp_set4,  # group4 (Picture description)
                         ['response2', 'response49', 'response50'],  # group5 (personal memory)
                         ['response4', 'response5']  # group6 (semantic fluency)
                         ]

    responses_set_new = [
        ['response1', 'response6'],  # group1 (reading)
        ['response46', 'response48'],  # group2  (Hearing & repeat)
        ['response7', 'response8', 'response9'],  # group3  (speechless)
        ['response' + str(i) for i in range(35, 45)],  # group4  (nonsense words)
        ['response2', 'response49', 'response50'],  # group5  (Personal memory)
        ['response4', 'response5'],  # group6  (Semantic fluency)
        ['response3'],  # group7  (picture description)
        ['response' + str(i) for i in range(10, 35)]  # group8  (Object naming)
    ]

    dataset_female = dataset[dataset.gender == 'Female']
    dataset_male = dataset[dataset.gender == 'Male']

    selected_group = args.group
    selected_data = dataset

    resp_set = responses_set_new[selected_group - 1]  # select group of response
    _ndepth = len(resp_set)
    _nrows = 32
    _ncols = 32
    print(resp_set)

    kf = KFold(n_splits=5, random_state=True, shuffle=True)
    train_acc_scores = []
    test_acc_scores = []
    train_f1_scores = []
    test_f1_scores = []
    test_f1_scores_0 = []
    k = 0
    for train_index, test_index in kf.split(selected_data.index):
        k += 1
        train_labels, test_labels = dataset.sss[train_index], dataset.sss[test_index]

        # get embeddings
        X_train, y_train = get_embeddings(selected_indexes=train_index, dataframe=dataset, response_set=resp_set)
        X_train = X_train.reshape(X_train.shape[0], _nrows, _ncols, _ndepth)
        # X_test, y_test = get_embeddings(selected_indexes=test_index, dataframe=dataset, response_set=resp_set)

        X_test, y_test, test_labels = get_thorsten_embedding(response_set=resp_set)

        X_test = X_test.reshape(X_test.shape[0], _nrows, _ncols, _ndepth)

        print('\n**************** Fold {}*****************************'.format(k))
        # print('Train shape: ({}, 1024, {})'.format(len(train_index), len(resp_set)))  # (m, 1024, n)
        # print('Test shape: ({}, 1024, {})'.format(len(test_index), len(resp_set)))  # (m, 1024, n)
        print('Train shape: {})'.format(X_train.shape))  # (m, 1024, n)
        print('Test shape: {}'.format(X_test.shape))  # (m, 1024, n)

        # Avoid imbalance training set
        y_flat = np.argmax(y_train, axis=1)
        class_weight = compute_class_weight('balanced', np.unique(y_flat), y_flat)
        class_weight = {k: v for k, v in enumerate(class_weight)}
        print('Class weights:', class_weight)

        # checkpoint to save trained model
        modelName = args.model_name + '_(' + str(len(test_acc_scores) + 1) + ')'
        checkpoint = ModelCheckpoint(config.__cfg__.checkpoint_dir + '/' + modelName,
                                     monitor='val_loss', save_best_only=True, save_weights_only=False,
                                     save_freq=5, verbose=0, mode='max')

        # Get the model
        model = my_models.get_model4(input_shape=(_nrows, _ncols, _ndepth))

        # train & save the model
        hist = model.fit(X_train, y_train, epochs=args.epoch, batch_size=args.batch_size, shuffle=False,
                         validation_data=(X_test, y_test),
                         class_weight=class_weight,
                         callbacks=[checkpoint])
        model.save(config.__cfg__.checkpoint_dir + '/' + modelName)

        # Validating the model
        print('Validating the model')
        _, train_acc = model.evaluate(X_train, y_train, verbose=0)
        _, test_acc = model.evaluate(X_test, y_test, verbose=0)
        test_acc_scores.append(test_acc)
        train_acc_scores.append(train_acc)

        y_train_true = list(map(lambda l: 1 if l >= config.__cfg__.sleepy_threshold else 0, train_labels))
        y_train_predict = np.argmax(model.predict(X_train), axis=1)
        f1_train = f1_score(y_train_true, y_train_predict)
        train_f1_scores.append(f1_train)

        y_test_true = list(map(lambda l: 1 if l >= config.__cfg__.sleepy_threshold else 0, test_labels))

        y_test_predict = model.predict(X_test)
        # y_test_predict[:, 0] *= class_weight[0]
        # y_test_predict[:, 1] *= class_weight[1]
        y_test_predict = list(np.argmax(y_test_predict, axis=1))
        # y_test_predict = list(np.argmax(model.predict(X_test), axis=1))

        f1_test = f1_score(y_test_true, y_test_predict)
        test_f1_scores.append(f1_test)
        f1_test0 = f1_score(y_test_true, y_test_predict, pos_label=0)
        test_f1_scores_0.append(f1_test0)

        print('Train Acc: %.2f, Test Acc: %.2f' % (train_acc, test_acc))
        print('F1-score Train: %.2f, F1-core1 Test(1): %.2f, F1-score Test(0): %.2f' % (f1_train, f1_test, f1_test0))

        # Plot training history
        '''
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.grid(True)
        ax.plot(hist.history['acc'], label='Train Accuracy', alpha=0.5)
        ax.plot(hist.history['val_acc'], label='Test Accuracy', alpha=0.5)
        ax.plot(hist.history['loss'], label='Train Loss', alpha=0.5)
        # plt.plot(hist.history['val_loss'], label='Test Lost', alpha=0.5)
        plt.title('Learning Curve of Model: \'{}\''.format(str.upper(modelName)), fontdict=dict(fontsize=16))
        ax.legend()
        plt.savefig('images/lc_' + modelName + '.pdf')
        '''

        # plot confusion matrix
        '''
        conf_matrix = confusion_matrix(y_true=y_test_true, y_pred=y_test_predict)
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Actuals', fontsize=18)
        plt.title('Confusion Matrix Of'+modelName, fontsize=18)
        plt.savefig('images/cm_' + modelName + '.pdf')
        '''

    print('Train accuracies - {}, Avg = {}'.format(train_acc_scores, np.mean(train_acc_scores)))
    print('Train F1 scores - {}, Avg = {}'.format(train_f1_scores, np.mean(train_f1_scores)))
    print('Test accuracies - {}, {}'.format(test_acc_scores, np.mean(test_acc_scores)))
    print('Test F1 scores(1) - {}, Avg = {}'.format(test_f1_scores, np.mean(test_f1_scores)))
    print('Test F1 scores(0) - {}, Avg = {}'.format(test_f1_scores_0, np.mean(test_f1_scores_0)))

    # plot accuracy test -----
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.title('Accuracy of Model - \'{}\''.format(str.upper(args.model_name)), fontdict=dict(fontsize=16))
    x_axis = np.arange(len(test_acc_scores)) + 1
    ax.bar(x_axis - 0.1, train_acc_scores, width=0.2)
    ax.bar(x_axis + 0.1, test_acc_scores, width=0.2)
    for i, acc in enumerate(train_acc_scores):
        plt.text(x=i - .1 + 1, y=acc - .01, s=f'%.2f' % acc, fontdict=dict(fontsize=12), color='white', ha='center',
                 va='top')
    for i, acc in enumerate(test_acc_scores):
        plt.text(x=i + .1 + 1, y=acc - .01, s=f'%.2f' % acc, fontdict=dict(fontsize=12), color='blue', ha='center',
                 va='top')

    ax.text(1.5, np.min(test_acc_scores) - .1,
            f'Avg Accuracy (Train/Test): %.2f/%.2f' % (np.mean(train_acc_scores), np.mean(test_acc_scores)), \
            bbox={'facecolor': '#cb9e5a', 'alpha': 2, 'edgecolor': '#cb9e5a', 'pad': 2},
            fontdict=dict(fontsize=16),
            ha='left')
    ax.set_xlabel('i-th Fold')
    ax.set_ylabel('Accuracy')
    ax.set_ylim((0, 1))
    ax.legend(['Train', 'Test'])
    plt.savefig('images/acc_' + args.model_name + '.pdf')

    # plot F1 test ----
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.title('f1-scores of Model - \'{}\''.format(str.upper(args.model_name)), fontdict=dict(fontsize=16))
    x_axis = np.arange(len(train_f1_scores)) + 1
    ax.bar(x_axis - .2, test_f1_scores_0, width=0.2)
    ax.bar(x_axis, train_f1_scores, width=0.2)
    ax.bar(x_axis + .2, test_f1_scores, width=0.2)
    for i, acc in enumerate(test_f1_scores_0):
        plt.text(x=i - .2 + 1, y=acc - .01, s=f'%.2f' % acc, fontdict=dict(fontsize=12), color='white', ha='center',
                 va='top')
    for i, acc in enumerate(train_f1_scores):
        plt.text(x=i + 1, y=acc - .01, s=f'%.2f' % acc, fontdict=dict(fontsize=12), color='white', ha='center',
                 va='top')
    for i, acc in enumerate(test_f1_scores):
        plt.text(x=i + .2 + 1, y=acc - .01, s=f'%.2f' % acc, fontdict=dict(fontsize=12), color='blue', ha='center',
                 va='top')

    ax.text(1.5, np.max(test_f1_scores) + .02,
            f'Avg F1 scores (Train/Test[1]) %.2f/%.2f' % (np.mean(train_f1_scores), np.mean(test_f1_scores)), \
            bbox={'facecolor': '#cb9e5a', 'alpha': 2, 'edgecolor': '#cb9e5a', 'pad': 2},
            fontdict=dict(fontsize=16), ha='left')
    ax.set_xlabel('i-th Test')
    ax.set_ylabel('F1 scores')
    ax.set_ylim((0, 1))
    ax.legend(['Test[0]', 'Train[1]', 'Test[1]'])
    plt.savefig('images/f1_' + args.model_name + '.pdf')

    return None
    # --------------------------


'''
Training with all responses except the one in selected mask-group
In detail:
    - We have 48 responses in different 8 groups --> the input of the model will have shape (32, 32, 48)
    - then we replace all of responses in the selected group by 0
    
'''
def run_1group_masked_train(args):
    '''
    :return: HuBERT embedding for training or testing session
    '''

    def get_embeddings(selected_indexes, dataframe, masked_responses=[]):
        print('\tGet embeddings for {} samples...'.format(len(selected_indexes)))
        X_embeddings = []
        y_labels = []
        response_columns = ['response' + str(i) for i in range(1, 51)]
        response_columns.remove('response45')
        response_columns.remove('response47')

        for idx in tqdm(selected_indexes):  # go through the selected sessions
            label = dataframe.at[idx, 'sss']
            X = [[] for _ in range(len(response_columns))]  # initialize each selected answer-column as []

            numpy_files = []
            for response in response_columns:
                if response in masked_responses:
                    numpy_files.append('pickle/hubert-embedding/non_existed.npy')
                else:
                    numpy_files.append('pickle/hubert-embedding/' +
                                       str(dataframe.at[idx, response]).replace('.wav', '.npy'))

            for i in range(len(numpy_files)):  # load HuBERT embedding from numpy file of each response column.
                if not os.path.isfile(numpy_files[i]):
                    X[i] = np.zeros((1, 1024))
                else:
                    X[i] = np.load(numpy_files[i])

                    # sometime there is an error in existed numpy file
                    if np.isnan(X[i]).any():
                        X[i] = np.zeros((1, 1024))
                    elif str(X[i].shape) == '()':
                        # print('I received:', X[i])
                        X[i] = np.zeros((1, 1024))
                # reshape the X[i]
                X[i] = np.array(X[i]).reshape(1024, 1)

            # convert x to an array (1024 x n)
            X = np.array(X).T
            X_embeddings.append(X)
            y_labels.append(label)

        # convert X_embeddings to a shape (m x 1024 x n) where m=len(sessions_list), and n=len(response_set)
        X_embeddings = np.array(X_embeddings)
        X_embeddings = np.squeeze(X_embeddings)

        # train with 2 classes (1..3 means alert; 4..7 means sleepy)
        y_labels = list(map(lambda y: 1 if y >= config.__cfg__.sleepy_threshold else 0, y_labels))  # 2 classes
        y_labels = to_categorical(y_labels, 2)

        return X_embeddings, y_labels

    # -------------------------------------------------------------------------
    print('Loading data...')
    dataset = pd.read_csv('csv/clean_data_corpus_multicol.csv')
    dataset = shuffle(dataset)

    # drop non-existed sessions
    mask = []
    for i in range(len(dataset)):
        s = config.__cfg__.raw_data_dir + '/' + dataset.at[i, 'session_id']
        if not os.path.isdir(s):
            mask.append(i)
    dataset = dataset.drop(mask)

    response_columns = ['response' + str(i) for i in range(1, 51)]
    response_columns.remove('response45')
    response_columns.remove('response47')
    response_groups = [[],
                       ['response1', 'response6'],  # group1 (reading)
                       ['response46', 'response48'],  # group2  (Hearing & repeat)
                       ['response7', 'response8', 'response9'],  # group3  (speechless)
                       ['response' + str(i) for i in range(35, 45)],  # group4  (nonsense words)
                       ['response2', 'response49', 'response50'],  # group5  (Personal memory)
                       ['response4', 'response5'],  # group6  (Semantic fluency)
                       ['response3'],  # group7  (picture description)
                       ['response' + str(i) for i in range(10, 35)]  # group8  (Object naming)
                       ]

    _ndepth = len(response_columns)  # we have 48 responses in total
    _nrows = 32
    _ncols = 32

    # Get the model
    model = my_models.get_model4(input_shape=(_nrows, _ncols, _ndepth))

    # Using K-Fold training
    kf = KFold(n_splits=5, random_state=1, shuffle=True)

    k = 0
    training_results = {}
    for train_index, test_index in kf.split(dataset.index):
        k += 1
        print('\n**************************************************************************')
        print('*                      Training Round-{}                                 *'.format(k))
        print('**************************************************************************')

        train_labels, test_labels = dataset.sss[train_index], dataset.sss[test_index]

        # train and test the model with different masked_groups
        masked_train_results = {}

        # Get full embeddings for 48 audio answers
        X_train, y_train = get_embeddings(selected_indexes=train_index, dataframe=dataset)
        X_test, y_test = get_embeddings(selected_indexes=test_index, dataframe=dataset)
        print('\t+ Train shape: {})'.format(X_train.shape))  # (m, 1024, 48)
        print('\t+ Test shape: {}'.format(X_test.shape))  # (m, 1024, 48)

        # Avoid imbalance training set
        y_flat = np.argmax(y_train, axis=1)
        class_weight = compute_class_weight('balanced', np.unique(y_flat), y_flat)
        class_weight = {k: v for k, v in enumerate(class_weight)}
        print('\t+ Class weights:{}'.format(class_weight))

        # train & test the model by masking specific goup
        for masked_group in range(0, len(response_groups)):
            model.reset_states()
            model.reset_metrics()
            print('\n--------\nRound-{}; Mask group{}={}\n---------'.format(k, masked_group,
                                                                            response_groups[masked_group]))

            X_train_masked, X_test_masked = X_train, X_test
            # mask the embeddings belonging to masked_group
            for response in response_groups[masked_group]:
                col = response_columns.index(response)
                X_train_masked[:, :, col] = 0
                X_test_masked[:, :, col] = 0

            # reshape to (M, 32, 32, 48)
            X_train_masked = X_train_masked.reshape(X_train_masked.shape[0], _nrows, _ncols, _ndepth)
            X_test_masked = X_test_masked.reshape(X_test_masked.shape[0], _nrows, _ncols, _ndepth)

            # checkpoint to save trained model in the form of (kth-fold)_(masked-group)
            modelName = args.model_name + '_Train' + str(k) + '_' + 'Mask' + str(masked_group)
            checkpoint = ModelCheckpoint(config.__cfg__.checkpoint_dir + '/' + modelName,
                                         monitor='val_loss', save_best_only=True, save_weights_only=False,
                                         save_freq=10, verbose=0, mode='max')
            # train & save the model
            hist = model.fit(X_train_masked, y_train,
                             epochs=args.epoch, batch_size=args.batch_size, shuffle=False, class_weight=class_weight,
                             validation_data=(X_test_masked, y_test), callbacks=[checkpoint])
            # model.save(config.__cfg__.checkpoint_dir + '/' + modelName)

            # ----- Validating the model ----
            print('\t + Validating the model {}'.format(modelName))

            # accuracy
            _, train_acc = model.evaluate(X_train_masked, y_train, verbose=0)
            _, test_acc = model.evaluate(X_test_masked, y_test, verbose=0)

            # f1 scores of training set
            y_train_true = list(map(lambda l: 1 if l >= config.__cfg__.sleepy_threshold else 0, train_labels))
            y_train_predict = model.predict(X_train_masked)
            y_train_predict = list(np.argmax(y_train_predict, axis=1))
            f1_train0 = f1_score(y_train_true, y_train_predict, pos_label=0)
            f1_train1 = f1_score(y_train_true, y_train_predict)
            f1_train = np.mean([f1_train0, f1_train1])

            # f1 scores of testing set
            y_test_true = list(map(lambda l: 1 if l >= config.__cfg__.sleepy_threshold else 0, test_labels))
            y_test_predict = model.predict(X_test_masked)
            y_test_predict = list(np.argmax(y_test_predict, axis=1))
            f1_test0 = f1_score(y_test_true, y_test_predict, pos_label=0)
            f1_test1 = f1_score(y_test_true, y_test_predict)
            f1_test = np.mean([f1_test0, f1_test1])

            print('\n--------\nRound-{}; Mask group{}={}\n---------'.format(k, masked_group,
                                                                            response_groups[masked_group]))
            print(f'\tAccuracy (train, test)=(%.2f, %.2f)' % (train_acc * 100, test_acc * 100))
            print(f'\tTraining F1-scores(l0, l1, mean)=(%.2f, %.2f, .%2f)' % (f1_train0, f1_train1, f1_train))
            print(f'\tTesting F1-scores(l0, l1, mean)=(%.2f, %.2f, .%2f)' % (f1_test0, f1_test1, f1_test))

            outputs = {'masked_responses': response_groups[masked_group],
                       'Accuracy (train,test)': (train_acc * 100, test_acc * 100),
                       'Train f1-score(l0,l1,mean)': (f1_train0, f1_train1, f1_train),
                       'Test f1-score(l0,l1,mean)': (f1_test0, f1_test1, f1_test)}
            masked_train_results['masked_group'+str(masked_group)] = outputs

        training_results['Train-'+str(k)] = masked_train_results

    with open(args.model_name + '_results.json', 'w') as outfile:
        outfile.write(json.dumps(training_results, indent=4))


'''
Train will only 1 group of responses unmasked
'''
def run_1group_unmasked_train(args):
    '''
    :return: HuBERT embedding for training or testing session
    '''

    def get_embeddings(selected_indexes, dataframe, masked_responses=[]):
        print('\tGet embeddings for {} samples...'.format(len(selected_indexes)))
        X_embeddings = []
        y_labels = []
        response_columns = ['response' + str(i) for i in range(1, 51)]
        response_columns.remove('response45')
        response_columns.remove('response47')

        for idx in tqdm(selected_indexes):  # go through the selected sessions
            label = dataframe.at[idx, 'sss']
            X = [[] for _ in range(len(response_columns))]  # initialize each selected answer-column as []

            numpy_files = []
            for response in response_columns:
                if response in masked_responses:
                    numpy_files.append('pickle/hubert-embedding/non_existed.npy')
                else:
                    numpy_files.append('pickle/hubert-embedding/' +
                                       str(dataframe.at[idx, response]).replace('.wav', '.npy'))

            for i in range(len(numpy_files)):  # load HuBERT embedding from numpy file of each response column.
                if not os.path.isfile(numpy_files[i]):
                    X[i] = np.zeros((1, 1024))
                else:
                    X[i] = np.load(numpy_files[i])

                    # sometime there is an error in existed numpy file
                    if np.isnan(X[i]).any():
                        X[i] = np.zeros((1, 1024))
                    elif str(X[i].shape) == '()':
                        # print('I received:', X[i])
                        X[i] = np.zeros((1, 1024))
                # reshape the X[i]
                X[i] = np.array(X[i]).reshape(1024, 1)

            # convert x to an array (1024 x n)
            X = np.array(X).T
            X_embeddings.append(X)
            y_labels.append(label)

        # convert X_embeddings to a shape (m x 1024 x n) where m=len(sessions_list), and n=len(response_set)
        X_embeddings = np.array(X_embeddings)
        X_embeddings = np.squeeze(X_embeddings)

        # train with 2 classes (1..3 means alert; 4..7 means sleepy)
        y_labels = list(map(lambda y: 1 if y >= config.__cfg__.sleepy_threshold else 0, y_labels))  # 2 classes
        y_labels = to_categorical(y_labels, 2)

        return X_embeddings, y_labels

    # -------------------------------------------------------------------------
    print('Loading data...')
    dataset = pd.read_csv('csv/clean_data_corpus_multicol.csv')
    dataset = shuffle(dataset)

    # drop non-existed sessions
    mask = []
    for i in range(len(dataset)):
        s = config.__cfg__.raw_data_dir + '/' + dataset.at[i, 'session_id']
        if not os.path.isdir(s):
            mask.append(i)
    dataset = dataset.drop(mask)

    response_columns = ['response' + str(i) for i in range(1, 51)]
    response_columns.remove('response45')
    response_columns.remove('response47')
    response_groups = [[],
                       ['response1', 'response6'],  # group1 (reading)
                       ['response46', 'response48'],  # group2  (Hearing & repeat)
                       ['response7', 'response8', 'response9'],  # group3  (speechless)
                       ['response' + str(i) for i in range(35, 45)],  # group4  (nonsense words)
                       ['response2', 'response49', 'response50'],  # group5  (Personal memory)
                       ['response4', 'response5'],  # group6  (Semantic fluency)
                       ['response3'],  # group7  (picture description)
                       ['response' + str(i) for i in range(10, 35)]  # group8  (Object naming)
                       ]

    _ndepth = len(response_columns)  # we have 48 responses in total
    _nrows = 32
    _ncols = 32

    # Get the model
    model = my_models.get_model4(input_shape=(_nrows, _ncols, _ndepth))

    # Using K-Fold training
    kf = KFold(n_splits=5, random_state=1, shuffle=True)

    k = 0
    training_results = {}
    for train_index, test_index in kf.split(dataset.index):
        k += 1
        print('\n**************************************************************************')
        print('*                      Training Round-{}                                 *'.format(k))
        print('**************************************************************************')

        train_labels, test_labels = dataset.sss[train_index], dataset.sss[test_index]

        # train and test the model with different masked_groups
        masked_train_results = {}

        # Get full embeddings for 48 audio answers
        X_train, y_train = get_embeddings(selected_indexes=train_index, dataframe=dataset)
        X_test, y_test = get_embeddings(selected_indexes=test_index, dataframe=dataset)
        print('\t+ Train shape: {})'.format(X_train.shape))  # (m, 1024, 48)
        print('\t+ Test shape: {}'.format(X_test.shape))  # (m, 1024, 48)

        # Avoid imbalance training set
        y_flat = np.argmax(y_train, axis=1)
        class_weight = compute_class_weight('balanced', np.unique(y_flat), y_flat)
        class_weight = {k: v for k, v in enumerate(class_weight)}
        print('\t+ Class weights:{}'.format(class_weight))

        # train & test the model by masking all other groups except 1
        for umasked_group in range(1, len(response_groups)):
            model.reset_states()
            model.reset_metrics()
            print('\n--------\nRound-{}; Unmask group{}={}\n---------'.format(k, umasked_group,
                                                                            response_groups[umasked_group]))

            X_train_masked, X_test_masked = X_train, X_test

            # mask the embeddings not belonging umasked_group
            masked_columns = ['response' + str(i) for i in range(1, 51)]
            masked_columns.remove('response45')
            masked_columns.remove('response47')
            for r in response_groups[umasked_group]:
                masked_columns.remove(r)
            for c in masked_columns:
                col = response_columns.index(c)
                X_train_masked[:, :, col] = 0
                X_test_masked[:, :, col] = 0

            # reshape to (M, 32, 32, 48)
            X_train_masked = X_train_masked.reshape(X_train_masked.shape[0], _nrows, _ncols, _ndepth)
            X_test_masked = X_test_masked.reshape(X_test_masked.shape[0], _nrows, _ncols, _ndepth)

            # checkpoint to save trained model in the form of (kth-fold)_(masked-group)
            modelName = args.model_name + '_Train' + str(k) + '_' + 'Mask' + str(umasked_group)
            checkpoint = ModelCheckpoint(config.__cfg__.checkpoint_dir + '/' + modelName,
                                         monitor='val_loss', save_best_only=True, save_weights_only=False,
                                         save_freq=10, verbose=0, mode='max')
            # train & save the model
            hist = model.fit(X_train_masked, y_train,
                             epochs=args.epoch, batch_size=args.batch_size, shuffle=False, class_weight=class_weight,
                             validation_data=(X_test_masked, y_test), callbacks=[checkpoint])
            # model.save(config.__cfg__.checkpoint_dir + '/' + modelName)

            # ----- Validating the model ----
            print('\t + Validating the model {}'.format(modelName))

            # accuracy
            _, train_acc = model.evaluate(X_train_masked, y_train, verbose=0)
            _, test_acc = model.evaluate(X_test_masked, y_test, verbose=0)

            # f1 scores of training set
            y_train_true = list(map(lambda l: 1 if l >= config.__cfg__.sleepy_threshold else 0, train_labels))
            y_train_predict = model.predict(X_train_masked)
            y_train_predict = list(np.argmax(y_train_predict, axis=1))
            f1_train0 = f1_score(y_train_true, y_train_predict, pos_label=0)
            f1_train1 = f1_score(y_train_true, y_train_predict)
            f1_train = np.mean([f1_train0, f1_train1])

            # f1 scores of testing set
            y_test_true = list(map(lambda l: 1 if l >= config.__cfg__.sleepy_threshold else 0, test_labels))
            y_test_predict = model.predict(X_test_masked)
            y_test_predict = list(np.argmax(y_test_predict, axis=1))
            f1_test0 = f1_score(y_test_true, y_test_predict, pos_label=0)
            f1_test1 = f1_score(y_test_true, y_test_predict)
            f1_test = np.mean([f1_test0, f1_test1])

            print('\n--------\nRound-{}; Unmask group{}={}\n---------'.format(k, umasked_group,
                                                                            response_groups[umasked_group]))
            print(f'\tAccuracy (train, test)=(%.2f, %.2f)' % (train_acc * 100, test_acc * 100))
            print(f'\tTraining F1-scores(l0, l1, mean)=(%.2f, %.2f, .%2f)' % (f1_train0, f1_train1, f1_train))
            print(f'\tTesting F1-scores(l0, l1, mean)=(%.2f, %.2f, .%2f)' % (f1_test0, f1_test1, f1_test))

            outputs = {'unmasked_responses': response_groups[umasked_group],
                       'Accuracy (train,test)': (train_acc * 100, test_acc * 100),
                       'Train f1-score(l0,l1,mean)': (f1_train0, f1_train1, f1_train),
                       'Test f1-score(l0,l1,mean)': (f1_test0, f1_test1, f1_test)}
            masked_train_results['unmasked_group'+str(umasked_group)] = outputs

        training_results['Train-'+str(k)] = masked_train_results

    with open(args.model_name + '_results.json', 'w') as outfile:
        outfile.write(json.dumps(training_results, indent=4))

'''
*********************************************
*               MAIN PROGRAM                *
*********************************************
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Sleepiness Classification Saved Model')
    parser.add_argument('--model_name', type=str, help='Name of model')
    parser.add_argument('--epoch', type=int, default=20, help='Epochs of training')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--group', type=int, default=1, help='The group of responses used to train the model (1..8)')
    args, _ = parser.parse_known_args()

    # run_train(args)                   # train with single group of response
    # run_1group_masked_train(args)            # train with 1 group is masked
    run_1group_unmasked_train(args)       # train with 1 group is not masked


    '''
    -----Command line: ----
    
    python3 train.py --model_name=model1-gr1-200ep.model --epoch=200  --batch_size 32  
    
    '''
