"""
Train an MNIST image recognition model.
"""

import keras
import numpy as np
from keras.datasets import mnist
from keras.layers import (Conv2D, Dense, Dropout, Flatten, Input, MaxPooling1D,
                          MaxPooling2D, BatchNormalization)
from keras.models import Model, Sequential
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

# training data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train / np.max(x_train), -1)
x_test = np.expand_dims(x_test / np.max(x_test), -1)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# our convolutional model
input_shape = x_train[0].shape
num_classes = 10
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=8,
                    verbose=1,
                    validation_data=(x_test, y_test))

# save the model in an HDF5 file, built in to keras
model.save('var/data/mnist.h5')
