#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import h5py
import numpy as np
import cv2
import datetime
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
NUM_LABELS = len(LABELS)
MAPPING_LABELS = dict()
for code, label in LABELS:
    MAPPING_LABELS[label] = code
#PATHS = {key: value for key, value in DIRECTORIES}
PATHS = dict()
for cur_dir in DIRECTORIES:
    PATHS[cur_dir] = DATASET_PATH + cur_dir + '/'


class Dataset:
    def __init__(self, image_scale):
        self.HEIGHT   = image_scale[0]
        self.WIDTH    = image_scale[1]
        self.CHANNELS = image_scale[2]

        self.data = None
        self.labels = None
        self.label_maping = None

    def _process_image(self, image_path):
        """
        TODO
        """
        # read the image
        img = cv2.imread(image_path)

        # resize image to the shape we want
        img = cv2.resize(img, (self.HEIGHT, self.WIDTH))

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

    def save_images(self):
        """
        TODO
        """
        total_image_count = 0

        for directory in DIRECTORIES:
            print('Saving images ({},{},{}) for {} ...'.format(
                self.HEIGHT, self.WIDTH, self.CHANNELS, directory))

            start = datetime.datetime.now()

            directory_image_count = 0

            image_shape_str = '({},{},{})'.format(
                self.HEIGHT, self.WIDTH, self.CHANNELS)
            filename = ARRAY_PATH + directory + image_shape_str + '.h5'

            with h5py.File(filename, 'w') as file:
                for label in LABELS:
                    # encode the label into one hot encoder
                    label_code = self.label_maping[label]
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
                        image = self._process_image(image_path)

                        # image
                        file.create_dataset(name='x' + str(image_count),
                                            data=image,
                                            shape=(self.HEIGHT,
                                                   self.WIDTH,
                                                   self.CHANNELS),
                                            maxshape=(self.HEIGHT,
                                                      self.WIDTH,
                                                      self.CHANNELS),
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

                        print('Saved {} images with {}({}) label.'.format(
                            image_index + 1, label, label_code))

                    print('Saved {} {} images.'.format(
                        directory_image_count, directory))

            end = datetime.datetime.now()

        print('Saved {} images in {} seconds.'.format(
            total_image_count, (end-start).seconds))


    def get_data():
        """
        TODO
        """
        with h5py.File(ARRAY_PATH + directory + '.h5', 'r') as file:
            # assuming the file contains same amount of image and labels
            num_images = np.ceil(len(file.keys()) / 2).astype(int)

            print('Reading {} images from {} ...'.format(num_images, directory))

            # define arrays holding data and labels
            data = np.zeros((num_images, HEIGHT, WIDTH, CHANNELS), dtype=np.float32)
            labels = np.zeros((num_images, NUM_LABELS), dtype=np.float32)

            # iterate over all images
            for image_index in range(num_images):
                image = file['x' + str(image_index)]
                label = file['y' + str(image_index)]

                data[image_index] = image
                labels[image_index] = label

        # shapes
        print('Initial shapes. {}   {}'.format(data.shape,labels.shape))
        data = np.array(data).astype(np.float32)
        labels = np.array(labels)
        print('Final shapes. {}   {}'.format(data.shape, labels.shape))
        return data, labels

