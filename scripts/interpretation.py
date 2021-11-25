from keras import applications
from keras import optimizers
from keras.layers import (
    Dropout, Flatten, Dense, SeparableConv2D, BatchNormalization, MaxPooling2D)
from keras.models import Model

from code.data import DataHelper
from code.gradcam import *
# from code.applications import *
from code.model import *

DATASET_NAME = 'chest_xray'

ROOT_PATH = \
    '/home/hippo/MachineLearning/pneumonia/'
DATASET_PATH = \
    '/home/hippo/MachineLearning/data/' + DATASET_NAME + '/'

MODEL_PATH = ROOT_PATH + 'models/' + DATASET_NAME + '/'
ARRAY_PATH = ROOT_PATH + 'arrays/' + DATASET_NAME + '/'

HEIGHT, WIDTH, CHANNELS = 224, 224, 3

dataclass = DataHelper(DATASET_NAME,
                       DATASET_PATH,
                       ARRAY_PATH,
                       height=HEIGHT,
                       width=WIDTH,
                       channels=CHANNELS,
                       histogram_equalization=False)

print(dataclass.directories)
print(dataclass.labels)

dataclass.get_images_dir('test')
test_x = dataclass.x['test']
test_y = dataclass.y['test']


# Load model and predict
def build_model(image_shape,
                classes,
                num_layers_to_freeze=21,
                base_model_weights_path=None):
    model_name = 'vgg19_{}'.format(num_layers_to_freeze)

    # build the base model
    if base_model_weights_path is None:
        base_model_weights_path = 'imagenet'
    base_model = applications.vgg19.VGG19(input_shape=image_shape,
                                          weights=base_model_weights_path,
                                          include_top=False)
    print('Load VGG19 as base model')

    # load base model weights
    # base_model.load_weights(base_model_weights_path)

    # adding classification block on top of base model
    x = Flatten(input_shape=image_shape, name='flatten')(base_model.output)
    # x = Dense(1024, activation='relu', name='fc1')(x)
    # x = Dropout(0.5, name='dropout1')(x)
    x = Dense(256, activation='relu', name='fc2')(x)
    x = Dropout(0.5, name='dropout2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    # combine both
    model = Model(inputs=base_model.input, outputs=x, name=model_name)

    # load top model weights
    # model.load_weights(top_model_weights_path)

    # freeze layers
    for i, layer in enumerate(model.layers):
        if i <= num_layers_to_freeze:
            layer.trainable = False
        else:
            layer.trainable = True
            print('Layer {} {} is trainable'.format(i, layer.name))

    # compile model
    loss_type = 'binary_' if classes == 2 else 'categorical_'
    loss_type += 'crossentropy'
    model.compile(loss=loss_type,
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    print('Model compiled')
    return model


def build_model_bis(image_shape,
                    classes,
                    num_layers_from_vgg=11):
    model_name = 'vgg19_depthwise_{}'.format(num_layers_from_vgg)

    # build vgg 19 model
    vgg = applications.vgg19.VGG19(input_shape=image_shape,
                                   weights='imagenet',
                                   include_top=False)
    print('Load VGG19 as base model')

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

    print('Load blocks 4 and 5')

    # adding classification block on top
    x = Flatten(input_shape=image_shape, name='flatten')(x)
    x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dropout(0.7, name='dropout1')(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dropout(0.5, name='dropout2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    # combine both
    model = Model(inputs=vgg.input, outputs=x, name=model_name)

    # load top model weights
    # model.load_weights(top_model_weights_path)

    # freeze layers
    for i, layer in enumerate(model.layers):
        if i <= num_layers_from_vgg:
            layer.trainable = False
        else:
            layer.trainable = True
            print('Layer {} {} is trainable'.format(i, layer.name))

    # compile model
    loss_type = 'binary_' if classes == 2 else 'categorical_'
    loss_type += 'crossentropy'
    model.compile(loss=loss_type,
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    print('Model compiled')
    return model


# using models
# save_model_path = MODEL_PATH + 'vgg19_aug_256.model'
# save_history_path = MODEL_PATH + 'vgg19_aug_256.history'
# BASE_MODEL_WEIGHTS_PATH = ROOT_PATH + 'models/vgg19_weights_no_top.h5'

NUM_CLASSES = test_y.shape[1]
ground_truth = np.argmax(test_y, axis=-1)

# depthwise
model_depthwise = build_model_bis((HEIGHT, WIDTH, CHANNELS), classes=NUM_CLASSES)
weights_depthwise = MODEL_PATH + 'vgg19_depthwise_11.model'
history_depthwise = MODEL_PATH + 'vgg19_depthwise_11.history'
model_depthwise.load_weights(weights_depthwise)

preds_depthwise = np.argmax(model_depthwise.predict(test_x), axis=-1)
differences_depthwise = [0 if p == g else 1 for p, g in zip(preds_depthwise, ground_truth)]
print(dict(zip(*np.unique(differences_depthwise, return_counts=True))))
plot_model_performance(preds_depthwise, ground_truth, dataclass.labels.keys())
plot_model_history(history_depthwise)

"""
# vgg19
model_vgg = build_model((HEIGHT, WIDTH, CHANNELS), classes=NUM_CLASSES)
weights_vgg = MODEL_PATH + 'vgg19_blocks_fc1.model'
history_vgg = MODEL_PATH + 'vgg19_blocks_fc1.history'
model_vgg.load_weights(weights_vgg)

preds_vgg = np.argmax(model_vgg.predict(test_x), axis=-1)
differences_vgg = [0 if p == g else 1 for p, g in zip(preds_vgg, ground_truth)]
print(dict(zip(*np.unique(differences_vgg, return_counts=True))))
plot_model_performance(preds_vgg, ground_truth, dataclass.labels.keys())
plot_model_history(history_vgg)
"""


# Grad CAM construction
def get_label_str_from_code(code):
    for label, _code in dataclass.labels.items():
        if code == _code:
            return label


guided_model_depthwise = build_guided_model(model_depthwise)
# guided_model_vgg = build_guided_model(model_vgg)

img_index = 432

true_str = get_label_str_from_code(ground_truth[img_index])
img = test_x[img_index]
img_extended = np.expand_dims(img, axis=0)
layer_to_visualize = 'block5_pool'

"""
### VGG 19
pred_str_vgg = get_label_str_from_code(preds_vgg[img_index])
pred_vgg_proba = round(np.max(model_vgg.predict(img_extended)), 2)

print('VGG 19 GradCAM for layer : {}'.format(layer_to_visualize))
print('Explanation for : {} {:0.2f}'.format(pred_str_vgg, pred_vgg_proba))
print('Ground truth is : {}'.format(true_str))

x, y, z = compute_saliency_array(model_vgg,
                                 guided_model_vgg,
                                 img,
                                 HEIGHT,
                                 WIDTH,
                                 layer_name=layer_to_visualize,
                                 cls=-1,
                                 visualize=True,
                                 save=False)
"""

pred_str_depthwise = get_label_str_from_code(preds_depthwise[img_index])
pred_depthwise_proba = np.max(model_depthwise.predict(img_extended))

print('Custom CNN GradCAM for layer : {}'.format(layer_to_visualize))
print('Explanation for : {} {:0.2f}'.format(pred_str_depthwise, pred_depthwise_proba))
print('Ground truth is : {}'.format(true_str))

x, y, z = compute_saliency_array(model_depthwise,
                                 guided_model_depthwise,
                                 img,
                                 HEIGHT,
                                 WIDTH,
                                 layer_name=layer_to_visualize,
                                 cls=-1,
                                 visualize=True,
                                 save=False)
