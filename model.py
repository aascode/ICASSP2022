from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, BatchNormalization, LSTM
from tensorflow.keras.layers import Dropout, Dense, TimeDistributed, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, schedules
from tensorflow.keras.regularizers import l2
from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# ----------------------------------------------------------------------------

def get_model0(input_shape):  # Convolution model
    initial_lr = 0.1
    # lr_schedule = schedules.ExponentialDecay(
    #     initial_lr,
    #     decay_steps=100000,
    #     decay_rate=0.96,
    #     staircase=True)
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    opt = Adam(learning_rate=initial_lr)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
    model.build()
    model.summary()
    return model


def get_model1(input_shape):  # Convolution model
    initial_lr = 1e-6
    # lr_schedule = schedules.ExponentialDecay(
    #     initial_lr,
    #     decay_steps=100000,
    #     decay_rate=0.96,
    #     staircase=True)
    model = Sequential()
    model.add(Dense(1024, activation='relu', input_shape=input_shape))
    model.add()
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(7, activation='softmax'))
    opt = Adam(learning_rate=initial_lr)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
    model.build()
    model.summary()
    return model


def get_model2(input_shape):  # Convolution model
    initial_lr = 1e-6
    # lr_schedule = schedules.ExponentialDecay(
    #     initial_lr,
    #     decay_steps=100000,
    #     decay_rate=0.96,
    #     staircase=True)

    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', strides=(1, 1), padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(7, activation='softmax'))
    model.summary()
    opt = Adam(learning_rate=initial_lr)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
    model.build()
    return model


def get_model3(input_shape):  # Convolution model
    initial_lr = 1e-4
    lr_schedule = schedules.ExponentialDecay(
        initial_lr,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)

    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', strides=(1, 1), padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(7, activation='softmax'))
    model.summary()
    opt = Adam(learning_rate=lr_schedule)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
    model.build()
    return model


def get_model4(input_shape):  # Convolution model
    initial_lr = 1e-4
    lr_schedule = schedules.ExponentialDecay(
        initial_lr,
        decay_steps=10000,
        decay_rate=0.96,
        staircase=True)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', name='conv2d_relu_1', input_shape=input_shape ))


    model.add(MaxPool2D(pool_size=(2, 2), padding='same', name='max_pool_2d_1'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='conv2d_relu_2'))
    model.add(MaxPool2D(pool_size=(2, 2), padding='same', name='max_pool_2d_2'))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name='conv2d_relu_3'))
    model.add(MaxPool2D(pool_size=(2, 2), padding='same', name='max_pool_2d_3'))
    # model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='conv2d_relu_4'))
    # model.add(MaxPool2D(pool_size=(2, 2), padding='same', name='max_pool_2d_4'))

    model.add(Flatten(name='flatten'))

    #model.add(Dense(256, activation='relu', activity_regularizer=l2(0.0001), name='dense_1'))
    model.add(Dense(128, activation='relu', activity_regularizer=l2(0.0001), name='dense_2'))
    model.add(Dense(64, activation='relu', activity_regularizer=l2(0.0001), name='dense_3'))
    #model.add(Dropout(rate=0.1, name='dropout_1'))
    model.add(Dense(32, activation='relu', activity_regularizer=l2(0.0001), name='dense_4'))
    model.add(Dense(16, activation='relu', activity_regularizer=l2(0.0001), name='dense_5'))
    #model.add(Dropout(rate=0.1, name='dropout_2'))
    model.add(Dense(2, activation='softmax', activity_regularizer=l2(0.0001), name='dense_6'))

    opt = Adam(learning_rate=initial_lr)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['acc'])
                  # metrics = ['acc', f1_m, precision_m, recall_m])
    model.build()

    model.summary()
    return model
