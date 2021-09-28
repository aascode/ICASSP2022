import argparse
import json
import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
from transformers import Wav2Vec2Processor, TFHubertModel

import config
import model as my_models

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

''''
Some globals parameters for every training fashion
'''
RESPONSE_COLUMNS = ['response' + str(i) for i in range(1, 51)]
RESPONSE_COLUMNS.remove('response45')
RESPONSE_COLUMNS.remove('response47')

# RESPONSE_GROUPS = [[],  # this means all of responses
#                    ['response1', 'response6'],  # group1 (reading)
#                    ['response46', 'response48'],  # group2  (Hearing & repeat)
#                    ['response7', 'response8', 'response9'],  # group3  (speechless)
#                    ['response' + str(i) for i in range(35, 45)],  # group4  (nonsense words)
#                    ['response2', 'response49', 'response50'],  # group5  (Personal memory)
#                    ['response4', 'response5'],  # group6  (Semantic fluency)
#                    ['response3'],  # group7  (picture description)
#                    ['response' + str(i) for i in range(10, 35)]  # group8  (Object naming)
#                    ]


# this fashion based on the utility of tasks
# RESPONSE_GROUPS = [[],  # this means all of responses
#                    ['response1', 'response6'],  # group1 (Mic test & attention - reading)
#                    ['response2', 'response3'],  # group2, task4 (open-end responses)
#                    ['response4', 'response5'],  # group3 (test memory of participants),
#                    ['response7'],               # group4 (Measures respiratory volume and various muscles that produce vocalizations)
#                    ['response8', 'response9'],  # group5 (Measures psychomotor symptoms)
#                    ['response' + str(i) for i in range(10, 44)],  # group6  (Measures memory in aging populations)
#                    ['response46', 'response48'],  # group6 (immediate recall ability)
#                    ['response49', 'response50']  # group7  (Self-reported information)
#                    ]

# this fashion based on name of tasks
# 1. Microphone test
# 2. Free speech
# 3. Picture description
# 4. Category naming
# 5. Phonemic fluency
# 6. Phonetically-balanced paragraph reading
# 7. Sustained phonation
# 8. Diadochokinetic tasks (pa-pa-pa, puh-tuh-kuh)
# 9. Confrontational naming
# 10. Non-word pronunciation
# 11. Memory recall (repeat a sentence)
# 12. Self report (medication information)

RESPONSE_GROUPS = [[],  # this means all of responses
                   ['response1'],  # task 1 (Mic test)
                   ['response2'],  # task 2 (Free speech)
                   ['response3'],  # task 3 (picture description)
                   ['response4'],  # task 4 (category naming - give name of animals/tools/fruits/household)
                   ['response5'],  # task 5 (list all words begin with letter ?)
                   ['response6'],  # task 6 (Phonetically-balanced paragraph reading)
                   ['response7'],  # task 7 (say /aaa/)
                   ['response8'],  # task 8 (pa-pa-pa)
                   ['response9'],  # task 9 (pa-ta-ka)
                   ['response' + str(i) for i in range(10, 34)], # task 10 (Confrontational naming)
                   ['response' + str(i) for i in range(35, 44)], # task 11 (non-word)
                   ['response46', 'response48'],                 # task 12 (sentence repeat)
                   ['response49', 'response50']                  # task 13 Self report (medication information)
                   ]



'''
Train with group-X of responses; Binary classification
    Step1: get embedding for all responses in a particular group by put it into HuBERT 
        + With 1 wav file, we get HuBERT's embedding as the output of last hidden layer, (T x 1024) where T is vary
          depend on the length of the wav file.
        + Calculate the mean (average) pooling for 1024 columns --> We get a vector (1024, 1) 
    Step2: Train the model with embedding vectors. 
        The input shape of data will be (M, 32, 32, N) where
            + M is number of sessions (samples) 
            + N is number of wav file in the selected group
'''


def run_train_by_1group(args):
    def get_embeddings(selected_indexes, dataframe, response_set=['response1', 'response6']):
        print('\tGet embedding for {} sessions'.format(len(selected_indexes)))

        X_embeddings, y_labels = [], []
        total_count, zeropad_count = 0, 0
        error_files, causes = [], []
        for idx in tqdm(selected_indexes):  # go through the selected sessions
            label = dataframe.at[idx, 'sss']
            X = [[] for _ in range(len(response_set))]  # initialize each selected answer-column as []

            numpy_files = ['pickle/hubert-embedding/' + str(dataframe.at[idx, answer]).replace('.wav', '.npy') for
                           answer in
                           response_set]

            for j in range(len(numpy_files)):  # load HuBERT embedding from numpy file of each response column.
                if os.path.isfile(numpy_files[j]):
                    X[j] = np.load(numpy_files[j])

                    # error even the numpy file existed
                    if np.isnan(X[j]).any():
                        X[j] = np.zeros((1, 1024))
                        zeropad_count += 1
                        error_files.append(numpy_files[j])
                        causes.append('contains NaN')
                    elif str(X[j].shape) == '()':
                        # print('I received:', X[j])
                        X[j] = np.zeros((1, 1024))
                        zeropad_count += 1
                        error_files.append(numpy_files[j])
                        causes.append('Not a list')
                else:
                    X[j] = np.zeros((1, 1024))
                    zeropad_count += 1
                    error_files.append(numpy_files[j])
                    causes.append('File not found')

                # reshape the X[j]
                X[j] = np.array(X[j]).reshape(1024, 1)
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
        # _min, _max = np.min(X_embeddings), np.max(X_embeddings)
        # X_embeddings = (X_embeddings - _min) / (_max - _min)

        # Map the labels to 2 classes (non-sleepy/sleepy)
        y_labels = list(map(lambda y: 1 if y >= config.__cfg__.sleepy_threshold else 0, y_labels))  # 2 classes
        y_labels = to_categorical(y_labels, 2)

        # train with 7 classes (0..6)
        # y_labels = list(map(lambda y: y-1, y_labels))  # 7 classes
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

    # For different gender training purpose
    female_dataset = dataset[dataset.gender == 'Female']
    male_dataset = dataset[dataset.gender == 'Male']

    selected_group = args.group
    selected_data = dataset

    response_set = RESPONSE_GROUPS[selected_group] if selected_group != 0 else RESPONSE_COLUMNS
    _nrows, _ncols, _ndepth = 32, 32, len(response_set)

    # Get the model
    the_model = my_models.get_model4(input_shape=(_nrows, _ncols, _ndepth))

    kf = KFold(n_splits=5, random_state=128, shuffle=True)
    train_accuracies, test_accuracies = [], []
    train_f1_scores, test_f1_scores = [], []
    train_f1_scores_l0, train_f1_scores_l1 = [], []
    test_f1_scores_l0, test_f1_scores_l1 = [], []
    k = 0
    for train_index, test_index in kf.split(selected_data.index):
        the_model.reset_states()
        the_model.reset_metrics()

        k += 1
        print('\n**************************************************************************')
        print('*                      Training Round-{}                                 *'.format(k))
        print('**************************************************************************')

        train_labels, test_labels = dataset.sss[train_index], dataset.sss[test_index]

        # get embeddings
        X_train, y_train = get_embeddings(selected_indexes=train_index, dataframe=dataset, response_set=response_set)
        X_train = X_train.reshape(X_train.shape[0], _nrows, _ncols, _ndepth)
        X_test, y_test = get_embeddings(selected_indexes=test_index, dataframe=dataset, response_set=response_set)
        # X_test, y_test, test_labels = get_thorsten_embedding(response_set=response_set)
        X_test = X_test.reshape(X_test.shape[0], _nrows, _ncols, _ndepth)

        print('\tTrain shape: {})'.format(X_train.shape))  # (m, 1024, n)
        print('\tTest shape: {}'.format(X_test.shape))  # (m, 1024, n)

        # calculate weight for each class --> dealing with  unbalance training dataset
        y_flat = np.argmax(y_train, axis=1)
        class_weight = compute_class_weight('balanced', np.unique(y_flat), y_flat)
        class_weight = {k: v for k, v in enumerate(class_weight)}
        print('\tClass weights:', class_weight)

        # checkpoint to save trained model
        model_name = args.model_name + '_train' + str(len(test_accuracies) + 1)
        checkpoint = ModelCheckpoint(config.__cfg__.checkpoint_dir + '/' + model_name,
                                     monitor='val_loss', save_best_only=True, save_weights_only=False,
                                     save_freq=1, verbose=0, mode='max')

        # train & save the model
        training_history = the_model.fit(X_train, y_train, class_weight=class_weight,
                                         epochs=args.epoch, batch_size=args.batch_size, shuffle=False,
                                         validation_data=(X_test, y_test), callbacks=[checkpoint])
        #the_model.save(config.__cfg__.checkpoint_dir + '/' + model_name)

        # Validating the model
        print('------------- Validating the model ----------------')
        _, train_acc = the_model.evaluate(X_train, y_train, verbose=0)
        _, test_acc = the_model.evaluate(X_test, y_test, verbose=0)
        test_accuracies.append(test_acc)
        train_accuracies.append(train_acc)

        # f1 scores of training
        y_train_true = list(map(lambda l: 1 if l >= config.__cfg__.sleepy_threshold else 0, train_labels))
        y_train_predict = the_model.predict(X_train)
        y_train_predict = list(np.argmax(y_train_predict, axis=1))
        f1_train0 = f1_score(y_train_true, y_train_predict, pos_label=0)
        f1_train1 = f1_score(y_train_true, y_train_predict)
        f1_train = np.mean([f1_train0, f1_train1])
        train_f1_scores.append(f1_train)
        train_f1_scores_l0.append(f1_train0)
        train_f1_scores_l1.append(f1_train1)

        # f1 scores of testing
        y_test_true = list(map(lambda l: 1 if l >= config.__cfg__.sleepy_threshold else 0, test_labels))
        y_test_predict = the_model.predict(X_test)
        y_test_predict = list(np.argmax(y_test_predict, axis=1))
        f1_test0 = f1_score(y_test_true, y_test_predict, pos_label=0)
        f1_test1 = f1_score(y_test_true, y_test_predict)
        f1_test = np.mean([f1_test0, f1_test1])
        test_f1_scores.append(f1_test)
        test_f1_scores_l0.append(f1_test0)
        test_f1_scores_l1.append(f1_test1)

        print('\n--------\nTRAINING ROUND-{}\n---------'.format(k))
        print(f'\tAccuracy (train, test)=(%.2f, %.2f)' % (train_acc * 100, test_acc * 100))
        print(f'\tTraining F1-scores(l0, l1, mean)=(%.2f, %.2f, .%2f)' % (f1_train0, f1_train1, f1_train))
        print(f'\tTesting F1-scores(l0, l1, mean)=(%.2f, %.2f, .%2f)' % (f1_test0, f1_test1, f1_test))

        # save training history to json file
        training_log = {'acc': training_history.history['acc'],
                        'val_acc': training_history.history['val_acc'],
                        'loss': training_history.history['loss'],
                        'val_loss': training_history.history['val_loss']}
        with open('images/hist_' + model_name + '.json', 'w') as outfile:
            outfile.write(json.dumps(training_log, indent=4))
        outfile.close()

        # Plot training history (Learning curve)
        '''
        print('------------- plotting training history ----------------')
        fig, ax = plt.subplots(figsize=(15, 8))
        plt.title('Learning Curve of Model: \'{}\''.format(str.upper(model_name)), fontdict=dict(fontsize=16))
        ax.plot(training_history.history['acc'], label='Train Accuracy', alpha=0.5)
        ax.plot(training_history.history['val_acc'], label='Test Accuracy', alpha=0.5)
        ax.plot(training_history.history['loss'], label='Train Loss', alpha=0.5)
        # plt.plot(training_history.history['val_loss'], label='Test Lost', alpha=0.5)
        ax.grid(True)
        ax.legend()
        plt.savefig('images/lc_' + model_name + '.pdf')
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


    # ---- Plot the result -------
    plt.figure(figsize=(15, 10))
    # plot accuracies after training -----
    ax1 = plt.subplot(211)
    plt.title('Accuracy of Model - {}'.format(args.model_name), fontdict=dict(fontsize=13))
    #plt.xlabel('Training round')
    plt.ylabel('Accuracy')
    plt.ylim((0, 1))
    x_axis = np.arange(len(test_accuracies)) + 1
    plt.bar(x_axis - 0.1, train_accuracies, width=0.2)
    plt.bar(x_axis + 0.1, test_accuracies, width=0.2)
    for i, acc in enumerate(train_accuracies):
        plt.text(x=i-.1 + 1, y=acc - .01, s=f'%.2f'%(acc* 100), color='white', ha='center', va='top', )

    for i, acc in enumerate(test_accuracies):
        plt.text(x=i+.1 + 1, y=acc - .01, s=f'%.2f'%(acc*100), color='blue', ha='center', va='top', )
        # fontdict = dict(fontsize=12),

    plt.text(1.5, np.min(test_accuracies) - .1,
             f'Avg. accuracy (Train/Test): %.2f/%.2f'%(
                 np.array(train_accuracies).mean() * 100, np.array(test_accuracies).mean()*100),
             bbox={'facecolor': '#cb9e5a', 'alpha': 2, 'edgecolor': '#cb9e5a', 'pad': 2}, )
            #fontdict=dict(fontsize=16), ha='left')
    plt.legend(['Train', 'Test'])

    # plot f1 score after training ----
    ax2 = plt.subplot(212)
    #plt.title('f1-scores of Model - {}'.format(args.model_name), fontdict=dict(fontsize=13))
    plt.xlabel('Training round')
    plt.ylabel('F1 scores')
    plt.ylim((0, 1))
    x_axis = np.arange(len(train_f1_scores)) + 1
    plt.bar(x_axis - .1, train_f1_scores, width=0.2)
    plt.bar(x_axis + .1, test_f1_scores, width=0.2)
    for i, score in enumerate(train_f1_scores):
        plt.text(x=i - .1 + 1, y=score - .01, s=f'%.2f' % score, color='white', ha='center', va='top')
    for i, score in enumerate(test_f1_scores):
        plt.text(x=i + .1 + 1, y=score - .01, s=f'%.2f' % score, color='blue', ha='center', va='top')

    plt.text(1.5, np.max(test_f1_scores) + .02,
             f'Avg. f1-score (Train/Test) %.2f/%.2f'%(
                 np.array(train_f1_scores).mean(), np.array(test_f1_scores).mean()), \
             bbox={'facecolor': '#cb9e5a', 'alpha': 2, 'edgecolor': '#cb9e5a', 'pad': 2}, ha='left')
    plt.legend(['Train', 'Test'])
    plt.savefig('images/acc_' + args.model_name + '.pdf')

    # ----- save the data to .json file
    log_data = {'train_accuracy': train_accuracies,
                'test_accuracy': test_accuracies,
                'train_f1_score': train_f1_scores,
                'test_f1_score': test_f1_scores,
                'train_f1_score_l0': train_f1_scores_l0,
                'test_f1_score_l0': test_f1_scores_l0,
                'train_f1_score_l1': train_f1_scores_l1,
                'test_f1_score_l1': test_f1_scores_l1,
                }
    with open('images/acc_' + args.model_name + '.json', 'w') as outfile:
        outfile.write(json.dumps(log_data, indent=4))
    outfile.close()

    return None
    # --------------------------


'''
Training with all responses except the one in selected mask-group
In detail:
    - We have 48 responses in different 8 groups --> the input of the model will have shape (32, 32, 48)
    - then we replace all of responses in the selected group by 0
    
'''


def run_train_with_1group_masked(args):
    '''
    :return: HuBERT embedding for training or testing session
    '''

    def get_embeddings(selected_indexes, dataframe, masked_responses=[]):
        print('\tGet embeddings for {} samples...'.format(len(selected_indexes)))
        X_embeddings = []
        y_labels = []

        for idx in tqdm(selected_indexes):  # go through the selected sessions
            label = dataframe.at[idx, 'sss']
            X = [[] for _ in range(len(RESPONSE_COLUMNS))]  # initialize each selected answer-column as []

            numpy_files = []
            for resp in RESPONSE_COLUMNS:
                if resp in masked_responses:
                    numpy_files.append('pickle/hubert-embedding/non_existed.npy')
                else:
                    numpy_files.append('pickle/hubert-embedding/' +
                                       str(dataframe.at[idx, resp]).replace('.wav', '.npy'))

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

    _ndepth = len(RESPONSE_COLUMNS)  # we have 48 responses in total
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
        for masked_group in range(0, len(RESPONSE_GROUPS)):
            model.reset_states()
            model.reset_metrics()
            print('\n--------\nRound-{}; Mask group{}={}\n---------'.format(k, masked_group,
                                                                            RESPONSE_GROUPS[masked_group]))

            X_train_masked, X_test_masked = X_train, X_test
            # mask the embeddings belonging to masked_group
            for response in RESPONSE_GROUPS[masked_group]:
                col = RESPONSE_COLUMNS.index(response)
                X_train_masked[:, :, col] = 0
                X_test_masked[:, :, col] = 0

            # reshape to (M, 32, 32, 48)
            X_train_masked = X_train_masked.reshape(X_train_masked.shape[0], _nrows, _ncols, _ndepth)
            X_test_masked = X_test_masked.reshape(X_test_masked.shape[0], _nrows, _ncols, _ndepth)

            # checkpoint to save trained model in the form of (kth-fold)_(masked-group)
            modelName = args.model_name + '_masked_gr' + str(masked_group) + '_train' + str(k)
            checkpoint = ModelCheckpoint(config.__cfg__.checkpoint_dir + '/' + modelName,
                                         monitor='val_loss', save_best_only=True, save_weights_only=False,
                                         save_freq=1, verbose=0, mode='max')
            # train & save the model
            hist = model.fit(X_train_masked, y_train,
                             epochs=args.epoch, batch_size=args.batch_size, shuffle=False, class_weight=class_weight,
                             validation_data=(X_test_masked, y_test), callbacks=[checkpoint])

            # model.save(config.__cfg__.checkpoint_dir + '/' + modelName)

            # save training history to json file
            training_log = {'acc': hist.history['acc'],
                            'val_acc': hist.history['val_acc'],
                            'loss': hist.history['loss'],
                            'val_loss': hist.history['val_loss']}
            with open('images/hist_' + modelName + '.json', 'w') as outfile:
                outfile.write(json.dumps(training_log, indent=4))
            outfile.close()

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
                                                                            RESPONSE_GROUPS[masked_group]))
            print(f'\tAccuracy (train, test)=(%.2f, %.2f)' % (train_acc * 100, test_acc * 100))
            print(f'\tTraining F1-scores(l0, l1, mean)=(%.2f, %.2f, .%2f)' % (f1_train0, f1_train1, f1_train))
            print(f'\tTesting F1-scores(l0, l1, mean)=(%.2f, %.2f, .%2f)' % (f1_test0, f1_test1, f1_test))

            outputs = {'masked_responses': RESPONSE_GROUPS[masked_group],
                       'Accuracy (train,test)': (train_acc * 100, test_acc * 100),
                       'Train f1-score(l0,l1,mean)': (f1_train0, f1_train1, f1_train),
                       'Test f1-score(l0,l1,mean)': (f1_test0, f1_test1, f1_test)}
            masked_train_results['masked_group' + str(masked_group)] = outputs

        training_results['Train-' + str(k)] = masked_train_results

    with open('images/acc_'+args.model_name + '_masked_1group.json', 'w') as outfile:
        outfile.write(json.dumps(training_results, indent=4))
    outfile.close()


'''
In this train, we consecutively select 1 group of responses to mask out. 
The training data will be augmented from 1462 to 1462 x 8 = 11696 samples 
'''
def run_train_with_1group_masked_augmentation(args):
    '''
        This function returns augmented masked-embeddings for selected samples
    '''
    def get_embeddings(selected_indexes, dataframe):
        print('\tGet embeddings for {} samples...'.format(len(selected_indexes)))
        X_embeddings = []
        y_labels = []

        for idx in tqdm(selected_indexes):  # go through the selected sessions
            label = dataframe.at[idx, 'sss']
            X = [[] for _ in range(len(RESPONSE_COLUMNS))]  # initialize each selected answer-column as []

            numpy_files = []
            for resp in RESPONSE_COLUMNS:
                    numpy_files.append('pickle/hubert-embedding/' +
                                       str(dataframe.at[idx, resp]).replace('.wav', '.npy'))

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

    _ndepth = len(RESPONSE_COLUMNS)  # we have 48 responses in total
    _nrows = 32
    _ncols = 32

    # Get the model
    model = my_models.get_model4(input_shape=(_nrows, _ncols, _ndepth))

    # Using K-Fold training
    kf = KFold(n_splits=5, random_state=1, shuffle=True)
    train_accuracies, test_accuracies = [], []
    train_f1_scores, test_f1_scores = [], []
    train_f1_scores_l0, train_f1_scores_l1 = [], []
    test_f1_scores_l0, test_f1_scores_l1 = [], []
    k = 0
    training_results = {}
    for train_index, test_index in kf.split(dataset.index):
        k += 1
        print('\n**************************************************************************')
        print('*                       Training Round-{}                                 *'.format(k))
        print('**************************************************************************')

        # reset model state to ensure the next train will be fair with previous one.
        model.reset_states()
        model.reset_metrics()

        train_labels, test_labels = dataset.sss[train_index], dataset.sss[test_index]

        # train and test the model with different masked_groups
        masked_train_results = {}

        # Get full embeddings of 48 audio responses
        X_train, y_train = get_embeddings(selected_indexes=train_index, dataframe=dataset)
        X_test, y_test = get_embeddings(selected_indexes=test_index, dataframe=dataset)

        # Generate masked data for training set
        X_train_augmented = []
        y_train_augmented = []
        for masked_group in range(1, len(RESPONSE_GROUPS)): # from GR1 to GR8
            X_train_masked = X_train
            for response in RESPONSE_GROUPS[masked_group]:
                col = RESPONSE_COLUMNS.index(response)
                X_train_masked[:, :, col] = 0
            X_train_augmented.append(X_train_masked)
            y_train_augmented.append(y_train)
        X_train_augmented = np.vstack(X_train_augmented)
        y_train_augmented = np.vstack(y_train_augmented)
        print('\t+ Train shape (augmented): {})'.format(X_train_augmented.shape))  # (m, 1024, 48)
        print('\t+ Test shape: {}'.format(X_test.shape))  # (m, 1024, 48)

        # reshape to (M, 32, 32, 48)
        X_train_augmented = X_train_augmented.reshape(X_train_augmented.shape[0], _nrows, _ncols, _ndepth)
        X_test = X_test.reshape(X_test.shape[0], _nrows, _ncols, _ndepth)

        # Avoid imbalance training set
        y_flat = np.argmax(y_train_augmented, axis=1)
        class_weight = compute_class_weight('balanced', np.unique(y_flat), y_flat)
        class_weight = {k: v for k, v in enumerate(class_weight)}
        print('\t+ Class weights:{}'.format(class_weight))

        # checkpoint to save trained model in the form of (kth-fold)_(masked-group)
        modelName = args.model_name + '_masked_augmentation_train' + str(k)
        checkpoint = ModelCheckpoint(config.__cfg__.checkpoint_dir + '/' + modelName,
                                     monitor='val_loss', save_best_only=True, save_weights_only=False,
                                     save_freq=1, verbose=0, mode='max')
        # train & save the model
        hist = model.fit(X_train_augmented, y_train_augmented,
                         epochs=args.epoch, batch_size=args.batch_size, shuffle=False, class_weight=class_weight,
                         validation_data=(X_test, y_test), callbacks=[checkpoint])
        # model.save(config.__cfg__.checkpoint_dir + '/' + modelName)

        # ----- Validating the model ----
        print('\t + Validating the model {}'.format(modelName))

        # accuracy
        _, train_acc = model.evaluate(X_train_augmented, y_train_augmented, verbose=0)
        _, test_acc = model.evaluate(X_test, y_test, verbose=0)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        # f1 scores of training set
        train_labels_augmented = []
        for _ in range(1, len(RESPONSE_GROUPS) ):
            train_labels_augmented += list(train_labels)

        y_train_true = list(map(lambda l: 1 if l >= config.__cfg__.sleepy_threshold else 0, train_labels_augmented))
        y_train_predict = model.predict(X_train_augmented)
        y_train_predict = list(np.argmax(y_train_predict, axis=1))
        f1_train0 = f1_score(y_train_true, y_train_predict, pos_label=0)
        f1_train1 = f1_score(y_train_true, y_train_predict)
        f1_train = np.mean([f1_train0, f1_train1])
        train_f1_scores.append(f1_train)
        train_f1_scores_l0.append(f1_train0)
        train_f1_scores_l1.append(f1_train1)

        # f1 scores of testing set
        y_test_true = list(map(lambda l: 1 if l >= config.__cfg__.sleepy_threshold else 0, test_labels))
        y_test_predict = model.predict(X_test)
        y_test_predict = list(np.argmax(y_test_predict, axis=1))
        f1_test0 = f1_score(y_test_true, y_test_predict, pos_label=0)
        f1_test1 = f1_score(y_test_true, y_test_predict)
        f1_test = np.mean([f1_test0, f1_test1])
        test_f1_scores.append(f1_test)
        test_f1_scores_l0.append(f1_test0)
        test_f1_scores_l1.append(f1_test1)

        print('\n--------\nRound-{}\n---------'.format(k))
        print(f'\tAccuracy (train, test)=(%.2f, %.2f)' % (train_acc * 100, test_acc * 100))
        print(f'\tTraining F1-scores(l0, l1, mean)=(%.2f, %.2f, .%2f)' % (f1_train0, f1_train1, f1_train))
        print(f'\tTesting F1-scores(l0, l1, mean)=(%.2f, %.2f, .%2f)' % (f1_test0, f1_test1, f1_test))

        # save training history to json file
        training_log = {'acc': hist.history['acc'],
                        'val_acc': hist.history['val_acc'],
                        'loss': hist.history['loss'],
                        'val_loss': hist.history['val_loss']}
        with open('images/hist_' + modelName + '.json', 'w') as outfile:
            outfile.write(json.dumps(training_log, indent=4))
        outfile.close()

    # ---- Plot the result -------
    plt.figure(figsize=(15, 10))
    # plot accuracies after training -----
    ax1 = plt.subplot(211)
    plt.title('Accuracy of Model - {}'.format(args.model_name), fontdict=dict(fontsize=13))
    # plt.xlabel('Training round')
    plt.ylabel('Accuracy')
    plt.ylim((0, 1))
    x_axis = np.arange(len(test_accuracies)) + 1
    plt.bar(x_axis - 0.1, train_accuracies, width=0.2)
    plt.bar(x_axis + 0.1, test_accuracies, width=0.2)
    for i, acc in enumerate(train_accuracies):
        plt.text(x=i - .1 + 1, y=acc - .01, s=f'%.2f' % (acc * 100), color='white', ha='center', va='top', )

    for i, acc in enumerate(test_accuracies):
        plt.text(x=i + .1 + 1, y=acc - .01, s=f'%.2f' % (acc * 100), color='blue', ha='center', va='top', )
        # fontdict = dict(fontsize=12),

    plt.text(1.5, np.min(test_accuracies) - .1,
             f'Avg. accuracy (Train/Test): %.2f/%.2f' % (
                 np.array(train_accuracies).mean() * 100, np.array(test_accuracies).mean() * 100),
             bbox={'facecolor': '#cb9e5a', 'alpha': 2, 'edgecolor': '#cb9e5a', 'pad': 2}, )
    # fontdict=dict(fontsize=16), ha='left')
    plt.legend(['Train', 'Test'])

    # plot f1 score after training ----
    ax2 = plt.subplot(212)
    # plt.title('f1-scores of Model - {}'.format(args.model_name), fontdict=dict(fontsize=13))
    plt.xlabel('Training round')
    plt.ylabel('F1 scores')
    plt.ylim((0, 1))
    x_axis = np.arange(len(train_f1_scores)) + 1
    plt.bar(x_axis - .1, train_f1_scores, width=0.2)
    plt.bar(x_axis + .1, test_f1_scores, width=0.2)
    for i, score in enumerate(train_f1_scores):
        plt.text(x=i - .1 + 1, y=score - .01, s=f'%.2f' % score, color='white', ha='center', va='top')
    for i, score in enumerate(test_f1_scores):
        plt.text(x=i + .1 + 1, y=score - .01, s=f'%.2f' % score, color='blue', ha='center', va='top')

    plt.text(1.5, np.max(test_f1_scores) + .02,
             f'Avg. f1-score (Train/Test) %.2f/%.2f' % (
                 np.array(train_f1_scores).mean(), np.array(test_f1_scores).mean()), \
             bbox={'facecolor': '#cb9e5a', 'alpha': 2, 'edgecolor': '#cb9e5a', 'pad': 2}, ha='left')
    plt.legend(['Train', 'Test'])
    plt.savefig('images/acc_' + args.model_name + '_augmentation.pdf')


    # ----- save training results to .json file
    log_data = {'train_accuracy': train_accuracies,
                'test_accuracy': test_accuracies,
                'train_f1_score': train_f1_scores,
                'test_f1_score': test_f1_scores,
                'train_f1_score_l0': train_f1_scores_l0,
                'test_f1_score_l0': test_f1_scores_l0,
                'train_f1_score_l1': train_f1_scores_l1,
                'test_f1_score_l1': test_f1_scores_l1,
                }

    with open('images/acc_'+args.model_name + '_masked_augmentation.json', 'w') as outfile:
        outfile.write(json.dumps(log_data, indent=4))
    outfile.close()

    return None
    # --------------------------


'''
*********************************************
*               MAIN PROGRAM                *
*********************************************
'''
if __name__ == '__main__':
    # ----------------------------------------------------------------------
    # Run this part to train the model by using audio responses in 1 group
    #   (single-group training)
    # ----------------------------------------------------------------------
    for gr in range(1, 13):
        mod_name = 'model4-train-by-task' + str(gr)
        print('---------------------------------------------')
        print('----------TRAIN THE MODEL {}-----------------'.format(mod_name))
        print('---------------------------------------------')
        custom_parser = argparse.ArgumentParser(description='Training Sleepiness Classification Model')
        custom_parser.add_argument('--model_name', type=str, default=mod_name)
        custom_parser.add_argument('--epoch', type=int, default=200)
        custom_parser.add_argument('--batch_size', type=int, default=64)
        custom_parser.add_argument('--group', type=int, default=gr)
        custom_args, _ = custom_parser.parse_known_args()
        run_train_by_1group(custom_args)  # train with single group of response


    # ----------------------------------------------------------------------
    # Run this part to train the model by using all responses except the one belonging to masked group
    #       (1-group masked training)
    # ----------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Training Sleepiness Classification Saved Model')
    parser.add_argument('--model_name', type=str, default='model4-task')
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    args, _ = parser.parse_known_args()
    run_train_with_1group_masked(args)

    # ----------------------------------------------------------------------
    # Run this part to the model by using augmented data.
    # The training dataset is generated by adding 1-group masked embedding
    #       (data-masked augmented training)
    # ----------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Training Sleepiness Classification Saved Model')
    parser.add_argument('--model_name', type=str, default='model4-task')
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    args, _ = parser.parse_known_args()
    run_train_with_1group_masked_augmentation(args)



'''
-----Command line: ----
python3 train.py --model_name=name-of-model --epoch=200  --batch_size=64  
'''
