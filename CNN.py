from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

import imageio
from PIL import Image
import numpy as np

import os
from skimage import io

from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(2048, activation='relu'))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(250, activation='linear'))

model.summary()

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

tot_batch = 1000
batch_size = 200
tot_num = 10
train_num = 9
test_num = tot_num - train_num

for batch in range(0, tot_batch, tot_num):
  training_images = []
  testing_images = []
  training_labels = []
  testing_labels = []
  for i in range(0, train_num):
    coefficients = np.load('D:\\zernike\\zernike_coefficients\\zernike_coefficients_batch_' + str(batch + i) + '.npy')
    for index in range(batch_size):
      image = np.load('D:\\zernike\\captured_images_for_training\\image_' + str(batch + i) + '_' + str(index) + '.npy')
      training_images.append(image)
      training_labels.append(1000 * coefficients[index][0:250])
  for i in range(train_num, tot_num):
    coefficients = np.load('D:\\zernike\\zernike_coefficients\\zernike_coefficients_batch_' + str(batch + i) + '.npy')
    for index in range(batch_size):
      image = np.load('D:\\zernike\\captured_images_for_training\\image_' + str(batch + i) + '_' + str(index) + '.npy')
      testing_images.append(image)
      testing_labels.append(1000 * coefficients[index][0:250])

  train_images = np.array(training_images)
  test_images = np.array(testing_images)
  train_labels = np.array(training_labels)
  test_labels = np.array(testing_labels)
  X_train,X_test = train_images/255, test_images/255
  X_train = X_train.reshape(train_num * batch_size, 224, 224, 1)
  X_test = X_test.reshape(test_num * batch_size, 224, 224, 1)
  Y_train = np.array(train_labels)
  Y_test = np.array(test_labels)

  #with suppress_stdout():
  #  history = model.fit(X_train, Y_train, epochs=1, validation_data=(X_test, Y_test))
  #  test_loss, test_acc = model.evaluate(X_test,  Y_test, verbose=2)
  #print(test_acc)

model.save("D:\\zernike\\model.h5")