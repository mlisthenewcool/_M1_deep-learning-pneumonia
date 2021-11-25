#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 19:35:11 2019

@author: hippolyte
"""

import os
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
# VISUALIZE
################################################################################
def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer

def visualize_class_activation_map(model, img_path, output_path):
    from data import process_image
    import keras.backend as K

    original_img = process_image(img_path) #cv2.imread(img_path, 1)
    width, height, channels = original_img.shape

    #print('Shape orginal : {}'.format(original_img.shape))

    img = np.expand_dims(original_img, axis=0)
    #Reshape to the network input shape (3, w, h).
    #img = np.array([np.transpose(np.float32(original_img), (2, 0, 1))])

    #Get the 512 input weights to the softmax.
    class_weights = model.layers[-1].get_weights()[0]
    #print(class_weights)

    final_conv_layer = get_output_layer(model, "block5_conv4")
    #print(final_conv_layer)

    #print(model.layers[0].input)
    #print(final_conv_layer.output)
    #print(model.layers[-1].output)

    #merge_layer = concatenate([])

    get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
    #get_output = K.function([model.layers[0].input], [final_conv_layer.output])
    print('get_output defined')

    #print(get_output([img]))

    [conv_outputs, predictions] = get_output([img])
    #print('BEFORE {}'.format(conv_outputs))

    #conv_outputs = conv_outputs[0, :, :, :]
    print(conv_outputs)
    print(predictions)

    #Create the class activation map.
    cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[0:2])
    for i, w in enumerate(class_weights[:, 1]):
            cam += w * conv_outputs[:, :, i]

    print('cam done')

    #print("predictions", predictions)
    cam /= np.max(cam)
    cam = cv2.resize(cam, (height, width))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam < 0.2)] = 0
    img = heatmap * 0.5 + original_img
    cv2.imwrite(output_path, img)

if __name__ == "__main__":
    from data import get_data, process_image
    from model import return_model

    data, labels = get_data()
    model = return_model()

    img_path = DATASET_PATH + 'train/PNEUMONIA/person1_bacteria_1.jpeg'

    visualize_class_activation_map(model, img_path, ROOT_PATH + 'heatmap.jpg')
