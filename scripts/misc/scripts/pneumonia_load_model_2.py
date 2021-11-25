#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 17:38:01 2019

@author: hippolyte
"""

import os
import h5py
import imgaug.augmenters as iaa
import numpy as np
import tensorflow as tf
import imgaug as aug
import datetime
import pickle

from keras.models import load_model

from matplotlib import pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

"""# Variables"""

# root path of the project
#ROOT_PATH = 'drive/My Drive/master1/medical_image_recognition/'
ROOT_PATH = '/home/hippolyte/Documents/universite/m1/TER/'

# name of the dataset
DATASET_NAME = 'chest_xray'

# define image extensions we accept
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg']

# dimensions for the images
HEIGHT, WIDTH, CHANNELS = 224, 224, 3

# output var to see infos
VERBOSE = True

# paths to each directory
directory_list = None #['test', 'train', 'val']

# labels for each directory
# if setup to None, detect according to subdirectories inside each directory
label_list = None #['NORMAL', 'PNEUMONIA']

# image augmentation sequence
# WE DON'T USE THAT YET
seq = iaa.OneOf([
    iaa.Fliplr(), # horizontal flips
    iaa.Affine(rotate=30)]) # rotation

# the next instructions are used to make results reproducible
seed = 1234
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(seed)
tf.set_random_seed(seed)
aug.seed(seed)

################################################################################
# DON'T TOUCH THE VARIABLES UNDER
################################################################################

# these are helpers to ease consistency
DATASET_PATH = ROOT_PATH + 'datasets/' + DATASET_NAME + '/'
MODEL_PATH = ROOT_PATH + 'models/' + DATASET_NAME + '/'
ARRAY_PATH = ROOT_PATH + 'arrays/' + DATASET_NAME + '/'

# create the directories to save arrays and models if they don't exist
os.system('mkdir -p {} {}'.format(MODEL_PATH, ARRAY_PATH))

# list directories according to the list defined on top
if directory_list is None:
    DIRECTORIES = sorted([d for d in os.listdir(DATASET_PATH)])
else:
    DIRECTORIES = directory_list

# get the labels inside the first directory
# of course, it should be the same in every directory
if label_list is None:
    LABELS = sorted(os.listdir(DATASET_PATH + DIRECTORIES[0]))
else:
    LABELS = label_list

NUM_LABELS = len(LABELS)

# get the paths
PATHS = dict()
for cur_dir in DIRECTORIES:
    PATHS[cur_dir] = DATASET_PATH + cur_dir + '/'

################################################################################
# DATA
################################################################################

def get_data(directory: str):
    """
    TODO
    """
    shape_str = '({},{},{})'.format(HEIGHT, WIDTH, CHANNELS)
    filename = ARRAY_PATH + directory + shape_str + '.h5'

    with h5py.File(filename, 'r') as file:
        start = datetime.datetime.now()

        # assuming the file contains same amount of image and labels
        num_images = np.ceil(len(file.keys()) / 2).astype(int)
        print('Reading {} images ({},{},{}) from {} ...'.format(num_images,
                                                                HEIGHT,
                                                                WIDTH,
                                                                CHANNELS,
                                                                directory))

        # define arrays holding data and labels
        data = np.zeros((num_images, HEIGHT, WIDTH, CHANNELS), dtype=np.float32)
        labels = np.zeros((num_images, NUM_LABELS), dtype=np.float32)

        # iterate over all images
        # format is x0 y0 for the first image and so on
        for image_index in range(num_images):
            image = file['x' + str(image_index)]
            label = file['y' + str(image_index)]

            data[image_index] = image
            labels[image_index] = label

    # shapes
    print('Shapes. Data: {} Labels: {}'.format(data.shape,labels.shape))
    #data = np.array(data).astype(np.float32)
    #labels = np.array(labels)
    #print('Final shapes. {}   {}'.format(data.shape, labels.shape))

    end = datetime.datetime.now()
    print('Done in {} seconds.'.format((end-start).seconds))

    return data, labels

################################################################################
# METRICS ON MODEL
################################################################################

def test_model(model, model_name, test_data, test_labels, batch_size=16):
    """
    TODO
    """
    test_loss, test_score = model.evaluate(test_data,
                                           test_labels,
                                           batch_size=16)

    print('Results for {} model.'.format(model_name))
    print('Loss : {}'.format(test_loss))
    print('Score : {}'.format(test_score))

    # predictions
    preds = model.predict(test_data, batch_size=16)
    preds = np.argmax(preds, axis=-1)

    # original labels
    orig_test_labels = np.argmax(test_labels, axis=-1)

    # confusion matrix
    cm  = confusion_matrix(orig_test_labels, preds)

    # metrics
    recall = np.diag(cm) / np.sum(cm, axis=1)
    precision = np.diag(cm) / np.sum(cm, axis=0)
    accuracy = np.diag(cm) / np.sum(cm)

    tn, fp, fn, tp = cm.ravel()
    precision_ = tp / (tp + fp)
    recall_ =    tp / (tp + fn)
    accuracy_ = (tp + tn) / (tn + fp + fn + tp)

    print("Recall of the model is {} - {:.5f}".format(recall, recall_))
    print("Precision of the model is {} - {:.5f}".format(precision, precision_))
    print("Accuracy of the model is {} - {:.5f}".format(accuracy, accuracy_))

    # plot confusion matrix
    plt.figure()
    plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
    plt.xticks(range(NUM_LABELS), LABELS, fontsize=16)
    plt.yticks(range(NUM_LABELS), LABELS, fontsize=16)
    plt.show()

    return preds


################################################################################
# JUST DO IT
################################################################################

data, labels = dict(), dict()

for directory in DIRECTORIES:
    # get data and labels
    data[directory], labels[directory] = get_data(directory)

    # plot an image with label
    if True:
        plt.imshow(data[directory][0])
        plt.show()
        print(labels[directory][0])


### RELOAD MODEL
#del model, history
MODEL_NAME = 'vgg19'
CNN_CONFIG_PATH = '6_vgg19_blocks_fc1/'

model = load_model(MODEL_PATH + CNN_CONFIG_PATH + 'model.hdf5')
model.summary()

### TEST MODEL

preds = test_model(model, MODEL_NAME, data['test'], labels['test'], batch_size=1)

### PLOT HISTORY

history_path = MODEL_PATH + CNN_CONFIG_PATH + 'history'
print('Loading history from {} ...'.format(history_path))
history = pickle.load(open(history_path, 'rb'))

fig, axes = plt.subplots(1, 2, figsize=(12, 4)) #, constrained_layout=True)

# accuracy
axes[0].plot(history['acc'])
axes[0].plot(history['val_acc'])
axes[0].set_title('Model Accuracy')
axes[0].set_ylabel('Accuracy')
axes[0].set_xlabel('Epochs')
axes[0].legend(['train', 'val'])

# loss
axes[1].plot(history['loss'])
axes[1].plot(history['val_loss'])
axes[1].set_title('Model Loss')
axes[1].set_ylabel('Loss')
axes[1].set_xlabel('Epochs')
axes[1].legend(['train', 'val'])
plt.show()

### CAM

preds = model.predict(data['test'])

from vis.visualization import visualize_cam
def compare_original_heatmap(i):

    if labels['test'][i][0] == 1:
        heat_map = visualize_cam(model, 19, 0, data['test'][i])
    else:
        heat_map = visualize_cam(model, 19, 1, data['test'][i])

    plt.subplot(1,2,1)
    plt.imshow(data['test'][i])
    plt.subplot(1,2,2)
    plt.imshow(heat_map)
    plt.show()

indexes = np.random.choice(data['test'].shape[0], 10, replace=False)

for i in indexes:
    print('True label {}'.format(labels['test'][i]))
    print('Predict label {}'.format(preds[i]))
    compare_original_heatmap(i)


#for i, layer in enumerate(model.layers):
#    print(i, layer.name)

"""
for i in range(NUM_LABELS):
    ind = np.where(labels['test'] == i)[0][0]

    plt.subplot(141)
    plt.imshow(data['test'][ind].reshape((28,28)))
    for j, modifier in enumerate([None, 'guided', 'relu']):
        heat_map = visualize_cam(model, 4, labels['test'][ind], data['test'][ind], backprop_modifier=modifier)
        plt.subplot(1,4,j+2)
        plt.imshow(heat_map)

    plt.show()
"""