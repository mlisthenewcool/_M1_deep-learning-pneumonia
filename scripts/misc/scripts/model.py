#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 19:35:11 2019

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

from keras.models import load_model, save_model

from matplotlib import pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

################################################################################
### MODIFIABLE VARIABLES
################################################################################
ROOT_PATH = '/home/hippolyte/Documents/universite/m1/TER/'
DATASET_NAME = 'chest_xray'
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg']
HEIGHT, WIDTH, CHANNELS = 224, 224, 3

# image augmentation sequence
SEQ = iaa.OneOf([
    iaa.Fliplr(), # horizontal flips
    iaa.Affine(rotate=30)]) # rotation

################################################################################
### DON'T CHANGE THESE VARIABLES
################################################################################
DATASET_PATH = ROOT_PATH + 'datasets/' + DATASET_NAME + '/'
ARRAY_PATH   = ROOT_PATH + 'arrays/'   + DATASET_NAME + '/'
MODEL_PATH   = ROOT_PATH + 'models/'   + DATASET_NAME + '/'

# create the directory to save arrays if it doesn't exist
os.system('mkdir -p {} {}'.format(ARRAY_PATH, MODEL_PATH))

# get directories and labels
DIRECTORIES = sorted([d for d in os.listdir(DATASET_PATH)])
LABELS = sorted(os.listdir(DATASET_PATH + DIRECTORIES[0]))

# helpers for labels
NUM_LABELS = len(LABELS)
LABEL_MAPPING = dict()
for code, label in enumerate(LABELS):
    LABEL_MAPPING[label] = code

# helper for paths
#PATHS = {key: value for key, value in DIRECTORIES}
PATHS = dict()
for cur_dir in DIRECTORIES:
    PATHS[cur_dir] = DATASET_PATH + cur_dir + '/'

# the next instructions are used to make results reproducible
seed = 1234
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(seed)
#tf.set_random_seed(seed)
#aug.seed(seed)

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
def return_model():
    ### RELOAD MODEL
    #del model, history
    CNN_CONFIG_PATH = '6_vgg19_blocks_fc1/'
    model = load_model(MODEL_PATH + CNN_CONFIG_PATH + 'model.hdf5')
    #model.summary()

    return model

if __name__ == "__main__":
    from data import get_data
    data_, labels_ = get_data()

    MODEL_NAME = 'vgg19'
    CNN_CONFIG_PATH = '6_vgg19_blocks_fc1/'
    model = return_model()

    if False:
        ### TEST MODEL
        preds = test_model(model, MODEL_NAME, data_['test'], labels_['test'], batch_size=1)

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