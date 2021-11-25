#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""## Libraries"""

import os
import h5py
import imgaug.augmenters as iaa
import numpy as np
import cv2
import datetime
from matplotlib import pyplot as plt
from keras.utils import to_categorical
from glob import glob

"""### Modifiable variables"""

# root path of the project
#root_path = Path('drive/My Drive/master1/medical_image_recognition')
ROOT_PATH = '/home/hippolyte/Documents/universite/m1/TER/'

# name of the dataset
DATASET_NAME = 'chest_xray'

# define image extensions we accept
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg']

# dimensions for the images
HEIGHT, WIDTH, CHANNELS = 299, 299, 3

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

"""### Don't touch to these variables"""

# these are helpers to ease consistency
DATASET_PATH = ROOT_PATH + 'datasets/' + DATASET_NAME + '/'
ARRAY_PATH = ROOT_PATH + 'arrays/' + DATASET_NAME + '/'

# create the directory to save arrays if it doesn't exist
os.system('mkdir -p {}'.format(ARRAY_PATH))

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

def label_str_to_int(label: str) -> int:
    """
    TODO
    """
    for i, label_ in enumerate(LABELS):
        if label_ == label:
            return i

    raise ValueError('Couldn\'t find {} label in labels'.format(label))

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

def save_images(directory: str):
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
            label_code = label_str_to_int(label)
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

if __name__ == '__main__':
    for directory in DIRECTORIES:
        save_images(directory)

        # show the first image
        shape_str = '({},{},{})'.format(HEIGHT, WIDTH, CHANNELS)
        with h5py.File(ARRAY_PATH + directory + shape_str + '.h5', 'r') as file:
            plt.imshow(file['x0'])
            plt.show()
            print(file['y0'].value)