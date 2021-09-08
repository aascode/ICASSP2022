from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from tensorflow.keras.layers import Dropout, Dense, TimeDistributed
from tensorflow.keras.optimizers import Adam, schedules

def get_model1(input_shape):        # Convolution model
    initial_lr = 1e-4
    lr_schedule = schedules.ExponentialDecay(
        initial_lr,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)

    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', strides=(1, 1), padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    model.add(MaxPool2D((2, 2)))
    #model.add(Dropout(0.5))        # do it later
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(7, activation='softmax'))
    model.summary()
    opt = Adam(learning_rate=lr_schedule)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
    model.build()
    return model


