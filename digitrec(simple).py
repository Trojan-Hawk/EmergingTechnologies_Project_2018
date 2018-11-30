import matplotlib.pyplot as plt
import numpy as np
# keras imports
from keras.datasets import mnist
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.preprocessing import image
# import to access OS file system
import os.path as path
# file dialog box and selector imports
import tkinter as tk
from tkinter import filedialog

# load or downloaded the MNIST dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()        
# setting a fixed random seed for reproducibility
seed = 7
np.random.seed(seed)
# flatten the 28*28 images in the dataset to a 1D array of length 784
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
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
	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def prediction(imageURL):
    prediction = ""
    test_image = image.load_img(imageURL, target_size = (28, 28), color_mode="grayscale")
    test_image = np.array(test_image)
    print(test_image.shape)
    #test_image = np.expand_dims(test_image, axis = 0)
    test_image = np.reshape(test_image, (1, 784))
    print(test_image.shape)
    result = model.predict(test_image)
    # training_set.class_indices
    print(result)

# build the model
model = modelDefn()
# Fit the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

# loop until the program is exited
while True:
    # open a new file explorer window
    root = tk.Tk()
    root.withdraw()
    # store the url of the selected file
    file_path = filedialog.askopenfilename()
    # call the prediction method of our model
    prediction(file_path)