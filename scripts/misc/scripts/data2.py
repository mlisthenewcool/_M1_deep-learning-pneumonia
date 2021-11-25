#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import h5py
import numpy as np
import cv2
import datetime
import argparse
import imgaug.augmenters as iaa
from matplotlib import pyplot as plt
from keras.utils import to_categorical
from glob import glob

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

def label_code_to_str(label_encoded: list):
    """
    TODO
    """
    label_code = np.argmax(label_encoded)

    for label, code in LABEL_MAPPING.items():
        if code == label_code:
            return label

    raise ValueError('Cannot find the label for {}'.format(label_encoded))

def process_image(image_path):
    """
    TODO
    """
    # read the image
    img = cv2.imread(image_path)

    # resize image to the shape we want
    img = cv2.resize(img, (HEIGHT, WIDTH))

    # if the image is in grayscale,
    # we should change its scale by adding each img to a channel
    # of (img, img, img) dimension
    # visit the excellent course listed bellow to understand why
    # http://me.umn.edu/courses/me5286/vision/Notes/2015/ME5286-Lecture3.pdf
    if img.shape[2] == 1:
        print(image_path)
        img = np.dstack([img, img, img])

    # convert the format used by default in cv2
    # to be consistent with tensorflow `preprocess_input` function
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # normalize image pixels
    img = img.astype(np.float32) / 255.

    return img

def _save_images_dir(directory: str):
    """
    TODO
    """
    print('Saving images ({},{},{}) for {} ...'.format(HEIGHT,
                                                       WIDTH,
                                                       CHANNELS,
                                                       directory))
    start = datetime.datetime.now()

    image_count = 0
    shape_str = '({},{},{})'.format(HEIGHT, WIDTH, CHANNELS)
    filename = ARRAY_PATH + directory + shape_str + '.h5'

    with h5py.File(filename, 'w') as file:
        for label in LABELS:
            # encode the label into one hot encoder
            label_code = LABEL_MAPPING[label]
            label_encoded = to_categorical(label_code,
                                               num_classes=NUM_LABELS)

            path = PATHS[directory] + label + '/'

            # get a list of list with all the images
            images = [glob(path + e) for e in IMAGE_EXTENSIONS]

            # make a flat list out of list of lists
            images = [item for sublist in images for item in sublist]

            if len(images) == 0:
                print('No image {} found in {}'.format(IMAGE_EXTENSIONS, path))
                continue

            # process all the images
            for image_index, image_path in enumerate(images):
                # process the image before to save it
                image = process_image(image_path)

                # image
                file.create_dataset(name='x' + str(image_count),
                                    data=image,
                                    shape=(HEIGHT, WIDTH, CHANNELS),
                                    maxshape=(HEIGHT, WIDTH, CHANNELS),
                                    compression='gzip',
                                    compression_opts=9)

                # label
                file.create_dataset(name='y' + str(image_count),
                                    data=label_encoded,
                                    shape=(NUM_LABELS,),
                                    maxshape=(None,),
                                    compression='gzip',
                                    compression_opts=9)

                image_count += 1

            print('Found {} images with {}({}) label.'.format(image_index+1,
                                                              label,
                                                              label_code))

    end = datetime.datetime.now()

    print('Saved {} images in {} seconds.'.format(image_count,
                                                  (end-start).seconds))
    return 1

def _get_data_dir(directory: str):
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
    print('Data: {}\nLabels: {}'.format(data.shape,labels.shape))
    #data = np.array(data).astype(np.float32)
    #labels = np.array(labels)
    #print('Final shapes. {}   {}'.format(data.shape, labels.shape))

    end = datetime.datetime.now()
    print('Done in {} seconds.'.format((end-start).seconds))

    return data, labels

def get_data():
    data, labels = dict(), dict()
    for directory in DIRECTORIES:
        data[directory], labels[directory] = _get_data_dir(directory)
    return data, labels

def save_images():
    for directory in DIRECTORIES:
        _save_images_dir(directory)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_images', type=bool, default=False, help='Save images according to image scale.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    if args.save_images:
        save_images()

    data, labels = get_data()

    for directory in DIRECTORIES:
        ind = np.random.randint(0, len(data[directory]))
        plt.imshow(data[directory][ind])
        plt.show()
        print(label_code_to_str(labels[directory][ind]))
