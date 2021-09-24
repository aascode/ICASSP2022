import argparse
import pickle

import numpy as np
import tensorflow as tf
import os

from tqdm import tqdm

from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import History
import matplotlib.pyplot as plt

from data_generator import DataGenerator
import model as my_models
import config

import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score

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


def read_npy_file(filename, label):
    X = np.load(filename.numpy().decode())
    _min = np.min(X)
    _max = np.max(X)
    X = (X - _min) / (_max - _min)

    label = tf.one_hot(label, 7, dtype=tf.float32)
    return X.astype(np.float32), label


def set_shape_function(image, label):
    image.set_shape([None, ])
    label.set_shape([None])
    return image, label


def run_train1(args):
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
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.grid(True)
    plt.plot(hist.history['acc'], label='Acc', alpha=0.5)
    plt.plot(hist.history['loss'], label='Loss', alpha=0.5)
    plt.plot(hist.history['val_acc'], label='Valid_Acc', alpha=0.5)
    # plt.plot(hist.history['val_loss'], label='Valid_Loss', alpha=0.5)
    plt.title("Training result with generated feature from 20K audio files")
    plt.legend(['Accuracy', 'Loss', 'Validation Accuracy', 'Validation Loss'])
    plt.savefig(args.model_name + '.pdf')


def run_train2(args):
    df = pd.read_csv('csv/hubert_embedding_corpus.csv', nrows=10000)
    df = shuffle(df)  # shuffling the data
    filenames = list(map(lambda f: 'pickle/hubert-embedding/' + f, df.file))
    y_labels = df.label

    train_filenames, test_filenames, train_labels, test_labels = train_test_split(filenames, y_labels, test_size=0.2,
                                                                                  random_state=1)
    print('dimension of train data: ({}, 1024)'.format(len(train_filenames)))  # (?, 1024)
    print('dimension of test data: ({}, 1024)'.format(len(test_filenames)))  # (?, 1024)
    # save the testing filenames and testing labels for testing purpose.
    pd.DataFrame(data={'file': test_filenames, 'sss': test_labels}).to_csv(
        'csv/' + args.model_name + '_test_corpus.csv')

    train_dataset = tf.data.Dataset.from_tensor_slices((filenames, y_labels))
    train_dataset = train_dataset.map(
        lambda x, y: tf.py_function(read_npy_file, inp=[x, y], Tout=[tf.float32, tf.float32]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.map(set_shape_function, num_parallel_calls=tf.data.AUTOTUNE)
    data_batch_size = 1000
    train_dataset = train_dataset.batch(batch_size=data_batch_size)

    # save trained model
    checkpoint = ModelCheckpoint(config.__cfg__.checkpoint_dir + '/' + args.model_name,  # checkpoints path
                                 monitor='val_loss', save_best_only=True, save_weights_only=False, save_freq=1,
                                 verbose=0, mode='max')
    model = my_models.get_model0(input_shape=(1024,))
    hist = model.fit(train_dataset, epochs=args.epoch, batch_size=args.batch_size, shuffle=True, callbacks=[checkpoint])
    model.save(config.__cfg__.checkpoint_dir + '/' + args.model_name)

    #  Plot training Losses
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.grid(True)
    plt.plot(hist.history['acc'], label='Accuracy', alpha=0.5)
    plt.plot(hist.history['loss'], label='Loss', alpha=0.5)
    plt.title('Training result of the model \'{}\''.format(args.model_name))
    plt.legend(['Accuracy', 'Loss'])
    plt.savefig('images/' + args.model_name + '.pdf')


'''
    this function trains the model based on GROUP1_DATA (Reading data)
    - Group1 contains audio recordings from reading speech. The speaker read a sentence and a paragraph
        columns = [response1, response6]
    - Process:
        + Get HuBET embedding of recordings
        + Get time of session --> normalize the time   
'''


def set_shape_train3(image, label):
    image.set_shape([None, None, None])
    label.set_shape([None])
    return image, label


def read_embedding(session_id, label):
    sess = session_id.numpy().decode()
    df = pd.read_csv('csv/clean_data_corpus_multicol.csv',
                     usecols=['session_id', 'startTime', 'response1', 'response6', 'sss'])

    df.set_index('session_id')

    startTime = int(str(df.at[sess, 'startTime']).split(':')[0])
    session_time = np.full((1024, 1), startTime)
    npy_file1 = 'pickle/hubert-embedding/' + str(df.at[sess, 'response1']).replace('.wav', '.npy')
    npy_file6 = 'pickle/hubert-embedding/' + str(df.at[sess, 'response6']).replace('.wav', '.npy')

    if os.path.isfile(npy_file1):
        npy_resp1 = np.load(npy_file1)
    else:
        npy_resp1 = np.zeros((1024, 1))

    if os.path.isfile(npy_file6):
        npy_resp6 = np.load(npy_file6)
    else:
        npy_resp6 = np.zeros((1024, 1))
    X = np.hstack((npy_resp1, npy_resp6, session_time))

    # normalize
    _min = np.min(X)
    _max = np.max(X)
    X = (X - _min) / (_max - _min)

    label = tf.one_hot(label, 7, dtype=tf.float32)

    return X.astype(np.float32), label


'''
Train with group-x of responses; Binary classification

'''


def run_train3(args):
    '''
    Get HuBert embeddings for slected responses
    :param args:
        - sessions_list: the session id of selected one
        - response_set: list of the selected columns
    :return: A matrix (m x 1024 x n) where n = len(response_set), and m = len(sessions_list)
    '''

    def get_embeddings(selected_indexes, dataframe, response_set=['response1', 'response6'],):
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

            numpy_files = ['pickle/hubert-embedding/' + str(dataframe.at[idx, answer]).replace('.wav', '.npy') for answer in
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
        # y_labels = list(map(lambda y: 1 if y > 4 else 0, y_labels))  # 2 classes
        # y_labels = to_categorical(y_labels, 2)

        # train with 7 classes (0..6)
        y_labels = list(map(lambda y: y-1, y_labels))  # 2 classes
        y_labels = to_categorical(y_labels, 7)

        print('{} Zero padding out of {}'.format(zeropad_count, total_count))
        return X_embeddings, y_labels

    # -------------------------------------------------------------------------
    print('Loading data...')
    dataset = pd.read_csv('csv/clean_data_corpus_multicol.csv')
    dataset = shuffle(dataset)

    resp_set1 = ['response1', 'response6'] # group1 (reading )
    resp_set2 = ['response46', 'response48'] # group2 (Hearing and repeat)
    resp_set3 = ['response7', 'response8', 'response9'] # group3 (None-semantic)
    for i in range(35, 45):
        resp_set3.append('response' + str(i))
    resp_set4 = ['response' + str(i) for i in range(10, 35)] # group4 (Picture description)
    resp_set4.insert(0, 'response3')
    resp_set5 = ['response2', 'response49', 'response50'] # group5 (personal memory)
    resp_set6 = ['response4', 'response5'] # group6 (semantic fluency)

    # drop non-existed sessions
    mask = []
    for i in range(len(dataset)):
        s = config.__cfg__.raw_data_dir + '/' + dataset.at[i, 'session_id']
        if not os.path.isdir(s):
            mask.append(i)
    dataset = dataset.drop(mask)

    resp_set = resp_set1  # select group of response
    _ndepth = len(resp_set)
    _nrows = 32
    _ncols = 32

    kf = KFold(n_splits=5, random_state=True, shuffle=True)
    train_acc_scores = []
    test_acc_scores = []
    train_f1_scores = []
    test_f1_scores = []
    k = 0
    for train_index, test_index in kf.split(dataset.index):
        k+=1
        train_sessions, test_sessions = dataset.session_id[train_index], dataset.session_id[test_index]
        train_labels, test_labels = dataset.sss[train_index], dataset.sss[test_index]

        # get train embeddings
        X_train, y_train = get_embeddings(selected_indexes=train_index, dataframe=dataset, response_set=resp_set)
        X_train = X_train.reshape(X_train.shape[0], _nrows, _ncols, _ndepth)
        X_test, y_test = get_embeddings(selected_indexes=test_index, dataframe=dataset, response_set=resp_set)
        X_test = X_test.reshape(X_test.shape[0], _nrows, _ncols, _ndepth)

        print('Fold {}'.format(k))
        print('Train shape: ({}, 1024, {})'.format(len(train_index), len(resp_set)))  # (m, 1024, n)
        print('Test shape: ({}, 1024, {})'.format(len(test_index), len(resp_set)))  # (m, 1024, n)

        y_flat = np.argmax(y_train, axis=1)
        class_weight = compute_class_weight('balanced', np.unique(y_flat), y_flat)
        class_weight = {k: v for k, v in enumerate(class_weight)}
        print('Class weight:', class_weight)

        # checkpoint to save trained model
        modelName = args.model_name + '(' + str(len(test_acc_scores) + 1) + ')'
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
        #model.save(config.__cfg__.checkpoint_dir + '/' + modelName)

        # Validating the model
        print('Validating the model')
        _, train_acc = model.evaluate(X_train, y_train, verbose=0)
        _, test_acc = model.evaluate(X_test, y_test, verbose=0)
        test_acc_scores.append(test_acc)
        train_acc_scores.append(train_acc)

        y_train_true = list(map(lambda l: 1 if l > 4 else 0, train_labels))
        y_train_predict = np.argmax(model.predict(X_train), axis=1)
        f1_train = f1_score(y_train_true, y_train_predict)
        train_f1_scores.append(f1_train)

        y_test_true = list(map(lambda l: 1 if l > 4 else 0, test_labels))
        y_test_predict = np.argmax(model.predict(X_test), axis=1)
        f1_test = f1_score(y_test_true, y_test_predict)
        test_f1_scores.append(f1_test)

        print('Train Acc: %.2f, Test Acc: %.2f' % (train_acc, test_acc))
        #print('F1-score Train: %.2f, F1-cores Test: %.2f' % (f1_train, f1_test))

        #  Plot training history
        fig, ax = plt.subplots(figsize=(15, 8))
        plt.grid(True)
        plt.plot(hist.history['acc'], label='Train Accuracy', alpha=0.5)
        plt.plot(hist.history['val_acc'], label='Test Accuracy', alpha=0.5)
        plt.plot(hist.history['loss'], label='Train Loss', alpha=0.5)
        # plt.plot(hist.history['val_loss'], label='Test Lost', alpha=0.5)
        plt.title('Learning Curve of Model: \'{}\''.format(str.upper(modelName)))
        plt.legend()
        plt.savefig('images/' + modelName + '.pdf')

    print('Train accuracies - {}, Avg = {}'.format(train_acc_scores, np.mean(train_acc_scores)))
    print('Train F1 scores - {}, Avg = {}'.format(train_f1_scores, np.mean(train_f1_scores)))
    print('Test accuracies - {}, {}'.format(test_acc_scores, np.mean(test_acc_scores)))
    print('Test F1 scores - {}, Avg = {}'.format(test_f1_scores, np.mean(test_f1_scores)))

    # plot accuracy test -----
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.title('Accuracy of Model - \'{}\''.format(str.upper(args.model_name)), fontdict=dict(fontsize=16))
    x_axis = np.arange(len(test_acc_scores))+1
    ax.bar(x_axis-0.1, train_acc_scores, width=0.2)
    ax.bar(x_axis+0.1, test_acc_scores,  width=0.2)
    for i, acc in enumerate(train_acc_scores):
        plt.text(x=i-.1+1, y=acc-.01, s=f'%.2f' % acc, fontdict=dict(fontsize=12), color='white', ha='center', va='top')
    for i, acc in enumerate(test_acc_scores):
        plt.text(x=i+.1+1, y=acc-.01, s=f'%.2f' % acc, fontdict=dict(fontsize=12), color='blue', ha='center', va='top')

    ax.text(1.5, np.min(test_acc_scores)-.1, f'Avg Accuracy (Train/Test): %.2f/%.2f'%(np.mean(train_acc_scores), np.mean(test_acc_scores)), \
            bbox={'facecolor': '#cb9e5a', 'alpha': 2, 'edgecolor': '#cb9e5a', 'pad': 2},
            fontdict=dict(fontsize=16),
            ha='left')
    ax.set_xlabel('i-th Fold')
    ax.set_ylabel('Accuracy')
    ax.set_ylim((0, 1))
    ax.legend(['Train', 'Test'])
    plt.savefig('images/accuracy_' + args.model_name + '.pdf')

    # plot F1 test ----
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.title('f1-scores of Model - \'{}\''.format(str.upper(args.model_name)))
    x_axis =np.arange(len(train_f1_scores))+1
    ax.bar(x_axis-.1, train_f1_scores, width=0.2)
    ax.bar(x_axis+.1, test_f1_scores, width=0.2)
    for i, acc in enumerate(train_f1_scores):
        plt.text(x=i-.1+1, y=acc-.01, s=f'%.2f' % acc, fontdict=dict(fontsize=12), color='white', ha='center', va='top')
    for i, acc in enumerate(test_f1_scores):
        plt.text(x=i+.1+1, y=acc-.01, s=f'%.2f' % acc, fontdict=dict(fontsize=12), color='blue',ha='center', va='top')

    ax.text(1.5, np.max(test_f1_scores)+.02, f'Avg F1 scores (Train/Test) %.2f/%.2f'%(np.mean(train_f1_scores), np.mean(test_f1_scores)), \
            bbox={'facecolor': '#cb9e5a', 'alpha': 2, 'edgecolor': '#cb9e5a', 'pad': 2},
            fontdict=dict(fontsize=16), ha='left')
    ax.set_xlabel('i-th Test')
    ax.set_ylabel('F1 scores')
    ax.set_ylim((0, 1))
    ax.legend(['Train', 'Test'])
    plt.savefig('images/f1_' + args.model_name + '.pdf')

    return None
    # --------------------------


'''
    print('Loading data...')
    df = pd.read_csv('csv/clean_data_corpus_multicol.csv')
    sessions = df.session_id
    y_labels = df.sss

    # separate train/test part
    train_sessions, test_sessions, train_labels, test_labels = \
        train_test_split(sessions, y_labels, test_size=0.2, random_state=1)

    # group1 of answers
    resp_set = ['response1', 'response6']
    print('dimension of train data: ({}, 1024, {})'.format(len(train_sessions), len(resp_set)))  # (m, 1024, n)
    print('dimension of test data: ({}, 1024, {})'.format(len(test_sessions), len(resp_set)))  # (m, 1024, n)


    #save the testing filenames and testing labels for testing purpose.
    #pd.DataFrame(data={'file': test_filenames, 'sss': test_labels}).to_csv('csv/' + args.model_name + '_test_corpus.csv')
    #
    # train_dataset = tf.data.Dataset.from_tensor_slices((train_sessions, train_labels))
    # train_dataset = train_dataset.map(
    #     lambda x, y: tf.py_function(read_embedding, inp=[x, y], Tout=[tf.float32, tf.float32]),
    #     num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # train_dataset = train_dataset.map(set_shape_train3, num_parallel_calls=tf.data.AUTOTUNE)
    # data_batch_size = 100
    # train_dataset = train_dataset.batch(batch_size=data_batch_size)

    #---- Load all to memory
    X_train = []
    y_train = []
    _min, _max = float('inf'), -float('inf')

    for i, sess in tqdm(enumerate(train_sessions)):
        npy_file1 = 'pickle/hubert-embedding/' + str(df.at[i, 'response1']).replace('.wav', '.npy')
        npy_file6 = 'pickle/hubert-embedding/' + str(df.at[i, 'response6']).replace('.wav', '.npy')
        startTime = int(str(df.at[i, 'startTime']).split(':')[0])
        startTime = np.array([[startTime, startTime]])
        # session_time = np.full((1024, 1), startTime)
        label = int(df.at[i, 'sss']) - 1

        npy_resp1 = np.load(npy_file1) if os.path.isfile(npy_file1) else np.zeros((1024, 1))
        npy_resp6 = np.load(npy_file6) if os.path.isfile(npy_file6) else np.zeros((1024, 1))

        count_zero1 = 0
        if np.isnan(npy_resp1).any():
            print('error NaN at: (res1)', sess)
            npy_resp1 = np.zeros((1024, 1))
            count_zero1 += 1

        count_zero6 = 0
        if np.isnan(npy_resp6).any():
            print('error NaN at: (res6)', sess)
            npy_resp6 = np.zeros((1024, 1))
            count_zero6 += 1

        npy_resp1 = np.array(npy_resp1).reshape((1024, 1))
        npy_resp6 = np.array(npy_resp6).reshape((1024, 1))
        X = np.hstack((npy_resp1, npy_resp6))
        # X = np.concatenate((X, startTime)) # attack time at the end

        X_train.append(X)
        y_train.append(label)

    X_train = np.array(X_train)
    _min = min(_min, np.min(X_train))
    _max = max(_max, np.max(X_train))

    print('min, max = ', _min, _max)
    # print('Zeros:', count_zero1, count_zero6)

    X_train = (X_train - _min) / (_max - _min)  # normalize X_train
    X_train = X_train.reshape(X_train.shape[0], 32, 32, 2)
    y_train = to_categorical(y_train, 7)  # 7 classes

    #---- End Load all to memory

    #save trained model
    checkpoint = ModelCheckpoint(config.__cfg__.checkpoint_dir + '/' + args.model_name,  # checkpoints path
                                 monitor='val_loss', save_best_only=True, save_weights_only=False, save_freq=1,
                                 verbose=0, mode='max')

    model = my_models.get_model3(input_shape=(32, 32, 2))
    # hist = model.fit(train_dataset, epochs=args.epoch, batch_size=args.batch_size, shuffle=True, callbacks=[checkpoint])
    hist = model.fit(X_train, y_train, epochs=args.epoch, batch_size=args.batch_size, shuffle=True,
                     callbacks=[checkpoint])
    model.save(config.__cfg__.checkpoint_dir + '/' + args.model_name)

    ### Evaluating the model--------------------------
    print('Testing result:')
    X_test = []
    y_test = []
    for i, sess in tqdm(enumerate(test_sessions)):
        npy_file1 = 'pickle/hubert-embedding/' + str(df.at[i, 'response1']).replace('.wav', '.npy')
        npy_file6 = 'pickle/hubert-embedding/' + str(df.at[i, 'response6']).replace('.wav', '.npy')
        startTime = int(str(df.at[i, 'startTime']).split(':')[0])
        session_time = np.full((1024, 1), startTime)
        label = int(df.at[i, 'sss']) - 1

        npy_resp1 = np.load(npy_file1) if os.path.isfile(npy_file1) else np.zeros((1024, 1))
        npy_resp6 = np.load(npy_file6) if os.path.isfile(npy_file6) else np.zeros((1024, 1))

        if np.isnan(npy_resp1).any():
            print('error NaN at: (res1)', sess)

        if np.isnan(npy_resp1).any():
            print('error NaN at: (res6)', sess)
            npy_resp1 = np.zeros((1024, 1))

        if np.isnan(npy_resp6).any():
            print('error NaN at: (res6)', sess)
            npy_resp6 = np.zeros((1024, 1))

        npy_resp1 = np.array(npy_resp1).reshape((1024, 1))
        npy_resp6 = np.array(npy_resp6).reshape((1024, 1))
        X = np.hstack((npy_resp1, npy_resp6))

        X_test.append(X)
        y_test.append(label)
    X_test = np.array(X_test)
    _min = np.min(X_test)
    _max = np.max(X_test)
    X_test = (X_test - _min) / (_max - _min)

    X_test = X_test.reshape(X_train.shape[0], 32, 32, 2)
    y_test = to_categorical(y_test, 7)  # 7 classes
    score = model.evaluate(X_test, y_test, verbose=1)
    print('Loss/Accuracy:', score)

    #  Plot training Losses
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.grid(True)
    plt.plot(hist.history['acc'], label='Accuracy', alpha=0.5)
    plt.plot(hist.history['loss'], label='Loss', alpha=0.5)
    plt.title('Training result of the model \'{}\''.format(args.model_name))
    plt.legend(['Accuracy', 'Loss'])
    plt.savefig('images/' + args.model_name + '.pdf')
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Sleepiness Classification Saved Model')
    parser.add_argument('--model_name', type=str, help='Name of model')
    parser.add_argument('--epoch', type=int, default=20, help='Epochs of training')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--p_file', type=str, action='append', help='Pickle file of training data')
    args, _ = parser.parse_known_args()

    run_train3(args)

    '''
    -----Command line: ----
    
    python3 train.py --model_name model1-20ep.model --epoch 20  --batch_size 32  
    
    '''
