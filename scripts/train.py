import pickle

from keras import applications
from keras import optimizers
from keras.models import Model
from keras.layers import (
    Dropout, Flatten, Dense, SeparableConv2D, BatchNormalization, MaxPooling2D)
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

DATA_AUGMENTATION = True
DATASET_NAME = 'chest_xray'

ROOT_PATH = \
    '/data1/home/hippolyte.debernardi/pneumonia/'
DATASET_PATH = \
    '/data1/home/hippolyte.debernardi/data/' + DATASET_NAME + '/'
MODEL_PATH = ROOT_PATH + 'models/' + DATASET_NAME + '/'


def get_generator(directory,
                  image_shape=(224, 224),
                  batch_size=32,
                  should_augment=False,
                  data_gen_args=None):
    # only rescale
    if should_augment is False:
        image_gen = ImageDataGenerator(rescale=(1. / 255))

    # use dictionary to define augmentations
    else:
        if data_gen_args is None:
            data_gen_args = dict(rescale=1. / 255,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True)

        image_gen = ImageDataGenerator(**data_gen_args)

    return image_gen.flow_from_directory(directory,
                                         target_size=image_shape,
                                         batch_size=batch_size)


HEIGHT, WIDTH, CHANNELS = 224, 224, 3
BATCH_SIZE = 32

train_generator = get_generator(DATASET_PATH + 'train',
                                batch_size=BATCH_SIZE,
                                should_augment=DATA_AUGMENTATION)

test_generator = get_generator(DATASET_PATH + 'test',
                               batch_size=BATCH_SIZE)


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
                        name='block4_sepconv1')(
        vgg.layers[num_layers_from_vgg].output)
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


# BASE_MODEL_WEIGHTS_PATH = ROOT_PATH + 'models/vgg19_weights_no_top.h5'
NUM_CLASSES = len(train_generator.class_indices)

# model = build_model(image_shape=(HEIGHT, WIDTH, CHANNELS), classes=NUM_CLASSES)
model = build_model_bis(image_shape=(HEIGHT, WIDTH, CHANNELS),
                        classes=NUM_CLASSES)
model.summary()

# fine-tune the model
EPOCHS = 300
TRAIN_LEN = train_generator.n
TEST_LEN = test_generator.n
CLASS_WEIGHT = {0: 1.0, 1: 0.4}

SAVE_HISTORY_PATH = MODEL_PATH + model.name + '.history'
SAVE_MODEL_PATH = MODEL_PATH + model.name + '.model'


def train_model(save_model_path,
                save_history_path,
                save_history=True):
    early_stopping = EarlyStopping(patience=20,
                                   monitor='val_loss',
                                   restore_best_weights=True)

    checkpoint = ModelCheckpoint(save_model_path,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=False)

    # reduce_lr = ReduceLROnPlateau(monitor='val_ loss', factor=0.2,
    #                              patience=5, min_lr=0.001)

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=TRAIN_LEN // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=test_generator,
        validation_steps=TEST_LEN // BATCH_SIZE,
        verbose=1,
        callbacks=[early_stopping, checkpoint],
        class_weight=CLASS_WEIGHT)

    # save history
    if save_history:
        with open(save_history_path, 'wb') as file:
            pickle.dump(history.history, file)
        print('\n\nSaved history into {}'.format(save_history_path))


train_model(save_model_path=SAVE_MODEL_PATH,
            save_history_path=SAVE_HISTORY_PATH)
