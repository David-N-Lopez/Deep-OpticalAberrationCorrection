import numpy as np
import re
import sys, os
import tensorflow as tf
from tensorflow import keras



# paths for scripts
path_for_labels = '/Users/davidlopez/Documents/ZernikeResearch/coefficients'
path_for_images = '/Users/davidlopez/Documents/ZernikeResearch/captures_images'
label_title = '/zernike_coefficients_batch_{}.npy'
subdir, dirs, filenames = os.walk(path_for_images).__next__()


partition = {
              'train': filenames[:36000],
              'validation': filenames[36000:]
            }

# Parameters
params = {'dim': (224,224),
          'batch_size': 32,
          'shuffle': True}


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(224,224),
                 shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def get_label(self, string):
        'getting the images parameters'
        id = list(string)
        count = 0
        mult = 1
        for char in id:
            if char.isdigit() and count == 0:
                batch_number = int(char)
                count+=1
            if char.isdigit() and count ==1:
                batch_index = int(char)
                count+=1
            if char.isdigit() and count == 2 and int(char) < 1:
                mult = 0.5
                count+=1
        return batch_number, batch_index, mult

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, 399), dtype='float64')

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load(path_for_images + '/' + ID)

            # Store class
            batch_number,batch_index, mult = self.get_label(ID)
            label_data_path = os.path.join(path_for_labels+ label_title.format(batch_number))
            label_data = np.load(label_data_path)[batch_index].dot(mult)
            y[i] = label_data


        X = X.reshape(self.batch_size,*self.dim,1)
        return X, y

training_generator = DataGenerator(partition['train'], **params)
validation_generator = DataGenerator(partition['validation'], **params)