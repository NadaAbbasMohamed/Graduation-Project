import keras
from keras.models import Sequential
from keras.utils import to_categorical
#from keras.utils import sparse_categorical_crossentropy
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import  BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
import pickle
import cv2

x_train = pickle.load( open("x_train.pickle", "rb"))
#x_train = to_categorical(x_train)
#x_train = sparse_categorical_crossentropy(x_train)
y_train = pickle.load( open("y_train.pickle", "rb"))
#y_train = to_categorical(y_train)
#y_train = sparse_categorical_crossentropy(y_train)

x_test = pickle.load( open("x_test.pickle", "rb"))
#x_test = to_categorical(x_test)
#x_test = sparse_categorical_crossentropy(x_test)
y_test = pickle.load( open("y_test.pickle", "rb"))
#y_test = to_categorical(y_test)
#y_test = sparse_categorical_crossentropy(y_test)

np.random.seed(1000)

# (3) Create a sequential model
model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# Passing it to a Fully Connected layer
model.add(Flatten())
# 1st Fully Connected Layer
model.add(Dense(4096, input_shape=(224*224*3,)))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))

# 2nd Fully Connected Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))

# 3rd Fully Connected Layer
model.add(Dense(1000))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))

# Output Layer
model.add(Dense(1))
model.add(Activation('softmax'))

model.summary()

# Compile the model
model.compile(loss= 'binary_crossentropy', optimizer='adam', metrics=["accuracy"])
# (5) Train
model.fit(x_train, y_train, batch_size=64, epochs=1, verbose=1, validation_data= (x_test, y_test), shuffle=True)

fer_json = model.to_json()
with open("fer.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("Weights_AlexNet.h5")
#model.save("AlexNet.model")
