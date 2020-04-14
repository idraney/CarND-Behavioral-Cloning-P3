# clone.py
# Udacity Self-Driving Car Engineer
# Behavioral Cloning Project
# Script to import data saved from self-driving car simulator
# and train in a Keras neural network 

import os
import csv
import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer

from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Dropout, Cropping2D
from keras.layers import Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model

import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split



# Generator function to save memory
def generator(samples, steering_correction, batch_size):
    num_samples = len(samples)

    # Loop forever so the generator never terminates
    while True: 
        sklearn.utils.shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            # Get Camera Images and Steering Points from CSV file
            # Augment dataset by flipping image horizontally and opposite angle.
            images = []                                 # Image paths read from CSV file
            measurements = []                           # Steering measurements from CSV file

            for batch_sample in batch_samples:
                images.append(cv2.imread(batch_sample[0]))                          # Center Image
                measurements.append(float(batch_sample[3]))
                images.append(cv2.imread(batch_sample[1]))                          # Left Image
                measurements.append(float(batch_sample[3]) + steering_correction)
                images.append(cv2.imread(batch_sample[2]))                          # Right Image
                measurements.append(float(batch_sample[3]) - steering_correction)

                images.append(np.fliplr(cv2.imread(batch_sample[0])))               # Center Image Flipped
                measurements.append(-1.0 * float(batch_sample[3]))
                images.append(np.fliplr(cv2.imread(batch_sample[1])))               # Left Image Flipped
                measurements.append((-1.0 * float(batch_sample[3])) - steering_correction)
                images.append(np.fliplr(cv2.imread(batch_sample[2])))               # Left Image Flipped
                measurements.append((-1.0 * float(batch_sample[3])) + steering_correction)

            X_train = np.array(images)
            y_train = np.array(measurements)

            yield sklearn.utils.shuffle(X_train, y_train)



lines = []                      # Lines read from CSV file
steering_correction = 0.065     # Steering correction factor from left/right cameras
batch_size = 128                # Batch size for training

# Read CSV file from local machine  
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Split 20% of training data set for test set
train_samples, validation_samples = train_test_split(lines, test_size = 0.2)

# Run generator function on training and validation datasets
train_generator = generator(train_samples, steering_correction = steering_correction, batch_size = batch_size)
validation_generator = generator(validation_samples, steering_correction = steering_correction, batch_size = batch_size)

# Model Neural Network
# Follows CNN architecture in NVIDIA's "End to End Learning for Self-Driving Cars"
# http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping = ((65, 20), (0, 0))))
model.add(Conv2D(24, kernel_size = (5, 5), strides = (2, 2), activation = 'relu'))
model.add(Conv2D(36, kernel_size = (5, 5), strides = (2, 2), activation = 'relu'))
model.add(Conv2D(48, kernel_size = (5, 5), strides = (2, 2), activation = 'relu'))
model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(500))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss = 'mse', optimizer = 'adam')
history_object = model.fit_generator(train_generator, 
                                    steps_per_epoch = np.ceil(len(train_samples)/batch_size), 
                                    validation_data = validation_generator, 
                                    validation_steps = np.ceil(len(validation_samples)/batch_size), 
                                    epochs = 5, 
                                    verbose = 1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('Mean Squared Error Loss vs. Training Epochs')
plt.ylabel('Mean Squared Error Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper right')
plt.show()

print("\nSteering Correction: %4.3f\n" % (steering_correction))
model.summary()

model.save('model.h5')
