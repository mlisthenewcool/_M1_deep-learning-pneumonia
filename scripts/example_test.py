#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

# custom
from gradcam import *

# system
import sys
import os
import glob
import re
import numpy as np
import cv2

# Keras
from keras import applications
from keras import optimizers
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Model
from keras.layers import (
    InputLayer, Dense, Flatten, Dropout, Input, Conv2D, SeparableConv2D,
    MaxPooling2D, BatchNormalization
)

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# global variables
model = None
guided_model = None

def load_model(image_shape,
               classes,
               num_layers_from_vgg,
               weights_path):

    global model
    global guided_model

    model_name = f'vgg19_depthwise_{num_layers_from_vgg}'

    # build vgg 19 model
    vgg = applications.vgg19.VGG19(input_shape=image_shape,
                                   weights='imagenet',
                                   include_top=False)

    print('Load VGG19 as base model.')

    # block 4
    x = SeparableConv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block4_sepconv1')(vgg.layers[num_layers_from_vgg].output)
    x = BatchNormalization(name='block4_conv1_bn')(x)
    x = SeparableConv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block4_sepconv2')(x)
    x = BatchNormalization(name='block4_conv2_bn')(x)
    x = SeparableConv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block4_sepconv3')(x)
    x = MaxPooling2D((2, 2), name='block4_pool')(x)

    # block 5
    x = SeparableConv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block5_sepconv1')(x)
    x = BatchNormalization(name='block5_conv1_bn')(x)
    x = SeparableConv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block5_sepconv2')(x)
    x = BatchNormalization(name='block5_conv2_bn')(x)
    x = SeparableConv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block5_sepconv3')(x)
    x = MaxPooling2D((2, 2), name='block5_pool')(x)

    print('Load blocks 4 and 5.')

    # adding classification block on top
    x = Flatten(input_shape=image_shape, name='flatten')(x)
    x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dropout(0.7, name='dropout1')(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dropout(0.5, name='dropout2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    # combine both
    model = Model(inputs=vgg.input, outputs=x, name=model_name)

    # freeze layers
    for i, layer in enumerate(model.layers):
        if i <= num_layers_from_vgg:
            layer.trainable = False
        else:
            layer.trainable = True
            print(f'Layer {i} {layer.name} is trainable')

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    print('Model compiled.')

    # loading weights
    model.load_weights(weights_path)
    print('Load saved weights into model.')

    print(model.summary())

    # construct guided one
    guided_model = build_guided_model(model)

def process_image(image_path):
    """
    """
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))

    # in case the image is in grayscale, build an array with channels dims
    # http://me.umn.edu/courses/me5286/vision/Notes/2015/ME5286-Lecture3.pdf

    if image.shape[2] == 1:
        print('{} is in grayscale, changed dimensions.'.format(image_path))
        image = [image] * 3

    # normalize image pixels between 0 and 1
    image = image.astype(np.float32) / 255.

    return image

def model_predict(img_path, model):
    #img = load_img(img_path, target_size=(224, 224))
    # Preprocessing the image
    #x = img_to_array(img)
    # x = np.true_divide(x, 255)

    x = process_image(img_path)

    x_extended = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x, mode='caffe')

    preds = model.predict(x_extended)
    pred_class = preds.argmax(axis=-1)

    # GradCAM
    layer_to_visualize = 'block5_pool'
    print(f'Custom CNN GradCAM for layer : {layer_to_visualize}')
    pred_str = 'NORMAL' if pred_class == 0 else 'PNEUMONIA'
    pred_proba = np.max(preds)

    print('Explanation for : {} {:0.2f}'.format(pred_str, pred_proba))

    a, b, c = compute_saliency_array(model,
                                     guided_model,
                                     x,
                                     224, 224,
                                     layer_name=layer_to_visualize,
                                     cls=-1)

    return preds, a, b, c


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds, gradCAM1, gradCAM2, gradCAM3 = model_predict(file_path, model)

        # Process your result for human
        #pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #result = str(pred_class[0][0][1])               # Convert to string
        '''
        return {
            'preds': preds,
            'gradCAM1': gradCAM1,
            'gradCAM2': gradCAM2,
            'gradCAM3': gradCAM3
        }
        '''
        return str(preds)
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    IMAGE_SHAPE = (224, 224, 3)
    NUM_CLASSES = 2
    NUM_LAYERS_FROM_VGG = 11
    WEIGHTS_PATH = '/home/hippolyte/Websites/flask-starter/routes/vgg19_depthwise_11.model'

    load_model(IMAGE_SHAPE, NUM_CLASSES, NUM_LAYERS_FROM_VGG, WEIGHTS_PATH)

    #app.run()
    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()