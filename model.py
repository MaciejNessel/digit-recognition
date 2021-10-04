from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D


def create_model():
    no_of_filters = 60
    size_of_filter1 = (5, 5)
    size_of_filter2 = (3, 3)
    no_of_nodes = 500

    model = Sequential()
    model.add((Conv2D(no_of_filters, size_of_filter1, input_shape=(32, 32, 1), activation='relu')))
    model.add((Conv2D(no_of_filters, size_of_filter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add((Conv2D(no_of_filters//2, size_of_filter2, activation='relu')))
    model.add((Conv2D(no_of_filters//2, size_of_filter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(no_of_nodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


