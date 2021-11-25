#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Package description
"""

__author__    = "Hippolyte Debernardi"
__copyright__ = "Copyright 2018, Hippolyte Debernardi"
__license__   = "GPL"
__version__   = "0.0.1"
__email__     = "contact@hippolyte-debernardi.com"

import os
import h5py
import imgaug.augmenters as iaa
import numpy as np
import cv2
import datetime
from matplotlib import pyplot as plt
from keras.utils import to_categorical
from skimage import exposure
from glob import glob


class Data:
    def __init__(self,
                 dataset_name,
                 dataset_path,
                 array_path,
                 image_extensions=['*.jpg', '*.jpeg'],
                 height=224,
                 width=224,
                 channels=3,
                 histogram_equalization=True):
        """
        TODO
        """
        # data dictionnaries, x for data, y for labels, z for original paths
        self.x = {}
        self.y = {}
        self.z = {}

        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.array_path = array_path

        # image variables
        self.image_extensions = image_extensions
        self.height = height
        self.width = width
        self.channels = channels
        self.histogram_equalization = histogram_equalization

        # directory variables like train, test, val
        self.directories = sorted([d for d in os.listdir(self.dataset_path)])
        self.paths = {d: self.dataset_path + d + '/' for d in self.directories}

        # label variables
        labels = sorted(os.listdir(self.dataset_path + self.directories[0]))
        self.labels = {label: code for code, label in enumerate(labels)}
        self.num_labels = len(self.labels)

    def set_image_scale(self, height, width, channels):
        """
        """
        self.height = height
        self.width = width
        self.channels = channels

    def process_image(self, image_path: str) -> np.ndarray:
        """
        """
        image = cv2.imread(image_path)
        image = cv2.resize(image, (self.height, self.width))

        # in case the image is in grayscale, build an array with channels dims
        # http://me.umn.edu/courses/me5286/vision/Notes/2015/ME5286-Lecture3.pdf

        if image.shape[2] == 1:
            print('{} is in grayscale, changed dimensions.'.format(image_path))
            image = [image] * self.channels

        # normalize image pixels between 0 and 1
        image = image.astype(np.float32) / 255.

        # histogram equalization
        if self.histogram_equalization:
            image = exposure.equalize_adapthist(image)

        return image

    def _save_images_dir(self, directory: str):
        """
        """
        shape = '_eq' if self.histogram_equalization else ''
        shape += '({},{},{})'.format(self.height, self.width, self.channels)
        filename = self.array_path + directory + shape + '.h5'
        print('{}: saving images {} into {}'.format(directory, shape, filename))

        # some helper variables
        start = datetime.datetime.now()
        image_count = 0

        with h5py.File(filename, 'w') as file:
            for label in self.labels.keys():
                # encode the label into one hot encoder
                label_encoded = to_categorical(self.labels[label],
                                               num_classes=self.num_labels)

                # get the path of the images
                path = self.paths[directory] + label + '/'

                # get a list of lists with all the images per extensions
                images = [glob(path + e) for e in self.image_extensions]

                # make a flat list out of list of lists
                images = [item for sublist in images for item in sublist]

                if not images:
                    raise IndexError('There is no images with {} in {}'.format(
                        self.image_extensions, path))

                # process all the images with compression
                # here we choose the best method (9) at expense of speed
                # https://ss64.com/bash/gzip.html
                for image_index, image_path in enumerate(images):
                    # process the image before to save it
                    image = self.process_image(image_path)

                    # image
                    file.create_dataset(name='x' + str(image_count),
                                        data=image,
                                        shape=(self.height,
                                               self.width,
                                               self.channels),
                                        maxshape=(self.height,
                                                  self.width,
                                                  self.channels),
                                        compression='gzip',
                                        compression_opts=9)

                    # label
                    file.create_dataset(name='y' + str(image_count),
                                        data=label_encoded,
                                        shape=(self.num_labels,),
                                        maxshape=(None,),
                                        compression='gzip',
                                        compression_opts=9)
                    """
                    # original path
                    file.create_dataset(name='z' + str(image_count),
                                        data=image_path,
                                        shape=(300,),
                                        dtype=h5py.special_dtype(vlen=str),
                                        maxshape=(None,),
                                        compression='gzip',
                                        compression_opts=9)
                    """

                    image_count += 1

                print('{} images with {}({}) label.'.format(image_index+1,
                                                            label,
                                                            self.labels[label]))

        end = datetime.datetime.now()
        print('{} images in {} seconds.'.format(image_count, (end-start).seconds))

    def save_images(self):
        """
        TODO
        """
        for directory in self.directories:
            self._save_images_dir(directory)

    def _get_images_dir(self, directory: str):
        """
        TODO
        """
        shape = '_eq' if self.histogram_equalization else ''
        shape += '({},{},{})'.format(self.height, self.width, self.channels)
        filename = self.array_path + directory + shape + '.h5'
        print('{}: getting images from {}'.format(directory, filename))

        # some helper variables
        start = datetime.datetime.now()

        with h5py.File(filename, 'r') as file:
            # assuming the file contains same amount of image and labels
            num_images = np.ceil(len(file.keys()) / 2).astype(int)

            # define arrays holding data, labels and original paths
            x = np.zeros((num_images,
                          self.height,
                          self.width,
                          self.channels), dtype=np.float32)
            y = np.zeros((num_images, self.num_labels), dtype=np.float32)
            #z = np.zeros((num_images, 1))

            # iterate over all images
            # format is x0 y0 for the first image and so on
            for image_index in range(num_images):
                image = file['x' + str(image_index)]
                label = file['y' + str(image_index)]
                #original_path = file['z' + str(image_index)]

                x[image_index] = image
                y[image_index] = label
                #z[image_index] = original_path

        # shapes
        print('Data shape : {}\nLabels shape : {}'.format(x.shape, y.shape))

        end = datetime.datetime.now()
        print('Found {} images in {} seconds.'.format(x.shape[0],
                                                    (end-start).seconds))

        self.x[directory] = x
        self.y[directory] = y
        #self.z[directory] = z

    def get_images(self):
        """
        TODO
        """
        for directory in self.directories:
            self._get_images_dir(directory)

    def show_random_image(self):
        pass

"""
if __name__ == "__main__":
    ROOT_PATH = '/home/hippolyte/Documents/universite/m1/TER/'
    DATASET_NAME = 'cancer_cells_split'
    DATASET_PATH = ROOT_PATH + 'datasets/' + DATASET_NAME + '/'
    ARRAY_PATH = DATASET_PATH + 'arrays/' + DATASET_NAME + '/'
    MODEL_PATH = DATASET_PATH + 'models/' + DATASET_NAME + '/'

    DATA = Data(
        DATASET_NAME,
        DATASET_PATH,
        ARRAY_PATH,
        image_extensions=['*.jpg', '*.jpeg'],
        height=224,
        width=224,
        channels=3,
        histogram_equalization=True)

    DATA.save_images()
    DATA.get_images()

    for directory in DATA.directories:
        # get a random index
        ind = np.random.randint(DATA.x[directory].shape[0])

        # plot the image with label
        plt.imshow(DATA.x[directory][ind])
        img_label = np.argmax(DATA.y[directory][ind])
        img_label_str = label_code_to_str(img_label)
        print('{}({})'.format(img_label, img_label_str))
        plt.show()
"""