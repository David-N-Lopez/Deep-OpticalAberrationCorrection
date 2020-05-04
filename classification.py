from generator import DataGenerator
import numpy as np
import tensorflow as tf
import sys, os
from tensorflow import keras

# Building model

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(256, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))

model.summary()

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(2048, activation='relu'))
model.add(keras.layers.Dense(1024, activation='relu'))
model.add(keras.layers.Dense(399, activation='linear'))

model.summary()

model.compile(optimizer='adam',
             loss='mean_squared_error',
             metrics=['accuracy'])

# paths for scripts
path_for_labels = '/Users/davidlopez/Documents/ZernikeResearch/coefficients'
path_for_images = '/Users/davidlopez/Documents/ZernikeResearch/captures_images'
label_title = '/zernike_coefficients_batch_{}.npy'
subdir, dirs, filenames = os.walk(path_for_images).__next__()

# dividing training and validation sets
partition = {
              'train': filenames[:36000],
              'validation': filenames[36000:]
            }

# Parameters
params = {'dim': (224,224),
          'batch_size': 32,
          'shuffle': True}

training_generator = DataGenerator(partition['train'], **params)
validation_generator = DataGenerator(partition['validation'], **params)

# train model on dataset
res = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6)
print(res.history)