# import matplotlib.pyplot as plt
import numpy as np
# keras imports
from keras.datasets import mnist
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing import image
# import to access OS file system
import os.path as path
# file dialog box and selector imports
import tkinter as tk
from tkinter import filedialog

# initialization
K.set_image_dim_ordering('th')
# file name constant
h5FileName = "weights.h5"

# load data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
	
# setting a fixed random seed for reproducibility
seed = 7
np.random.seed(seed)

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
num_classes = Y_test.shape[1]

# define the model
def modelDefn():
	# create model
    model = Sequential()
    # layer definitions
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model

def prediction(imageURL):
    prediction = ""
    test_image = image.load_img(imageURL, target_size = (28, 28), color_mode="grayscale")
    test_image = np.array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    print(result)

# build the model
model = modelDefn()

# check to see if the weights file exists
if path.isfile(h5FileName):
    print("Loading pre-compiled Neural Net Weights")
    model.load_weights(h5FileName)
else:
    print("Creating Neural Net and saving Weights to file")
    # Fit the model
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=14, batch_size=200, verbose=2)
    model.save_weights(h5FileName)

# Final evaluation of the model
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

# loop until the program is exited
while True:
    # open a new file explorer window
    root = tk.Tk()
    root.withdraw()

    try:
        # store the url of the selected file
        file_path = filedialog.askopenfilename()
        # call the prediction method of our model
        prediction(file_path)
    except:
        print("No file selected. Exiting application!")
        break
