#!/usr/bin/env python
#
# # USAGE
# python train_liveness.py --dataset dataset --model liveness.model --le le.pickle

# set the matplotlib backend so figures can be saved in the background

import matplotlib
matplotlib.use('Agg')

# import the necessary packages
from spoofing.livenessnet import LivenessNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import np_utils
from imutils import paths
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d',
                '--dataset',
                required=True,
                help='path to input dataset (input)')
ap.add_argument('-m',
                '--model',
                type=str,
                required=True,
                help='name for trained model/encoder (output)')
ap.add_argument('-e',
                '--epochs',
                type=int,
                default=10,
                help='path to output loss/accuracy plot (output)')
args = vars(ap.parse_args())

# initialize the initial learning rate, batch size, and number of
# epochs to train for
INIT_LR = 1e-4
BS = 64
epochs = args['epochs']

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print('[INFO] loading images...')
imagePaths = list(paths.list_images(args['dataset']))

data = []
labels = []

for imagePath in imagePaths:
    # extract the class label from the filename, load the image and
    # resize it to be a fixed N x N pixels, ignoring aspect ratio
    label = Path(imagePath).parents[1].stem
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (64, 64))

    # update the data and labels lists, respectively
    data.append(image)
    labels.append(label)

# convert the data into a NumPy array, then preprocess it by scaling
# all pixel intensities to the range [0, 1]
data = np.array(data, dtype='float') / 255.0

# encode the labels (which are currently strings) as integers and then
# one-hot encode them
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = np_utils.to_categorical(labels, 2)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels,
                                                  test_size=0.25,
                                                  random_state=42)

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20,
                         zoom_range=0.15,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         shear_range=0.15,
                         horizontal_flip=False,
                         fill_mode='nearest')

# initialize the optimizer and model
print('[INFO] compiling model...')
opt = Adam(lr=INIT_LR, decay=INIT_LR / epochs)
model = LivenessNet.build(width=64,
                          height=64,
                          depth=3,
                          classes=len(le.classes_))
model.summary()
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# train the network
print(f'[INFO] training network for {epochs} epochs...')
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
                        validation_data=(testX, testY),
                        steps_per_epoch=len(trainX) // BS,
                        epochs=epochs)

# evaluate the network
print('[INFO] evaluating network...')
predictions = model.predict(testX, batch_size=BS)
print(
    classification_report(testY.argmax(axis=1),
                          predictions.argmax(axis=1),
                          target_names=le.classes_))

# save the network to disk
model_name = args['model']
print(f'[INFO] serializing network to {model_name}\'.* files...')
model_def = 'spoofing/' + model_name + '.model'
label_enc = 'spoofing/' + model_name + '.pickle'
image_png = 'spoofing/' + model_name + '.png'
model.save(model_def)
# save the label encoder to disk
file = open(label_enc, 'wb')
pickle.dump(le, file)
file.close()

# plot the training loss and accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, epochs), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, epochs), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, epochs), H.history['accuracy'], label='train_acc')
plt.plot(np.arange(0, epochs), H.history['val_accuracy'], label='val_acc')
plt.title('Training Loss and Accuracy on Dataset')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='lower left')
plt.savefig(image_png)
