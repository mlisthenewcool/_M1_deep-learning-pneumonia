from keras import applications
from keras.models import Sequential, Model
from keras.layers import (
    InputLayer, Dense, Flatten, Dropout, Input, Conv2D, SeparableConv2D,
    MaxPooling2D, BatchNormalization
)
from keras import optimizers


def build_model(image_shape,
                classes,
                num_layers_to_freeze=21,
                base_model_weights_path=None,
                top_model_weights_path=None):
    model_name = f'vgg19_{num_layers_to_freeze}'

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
            print(f'Layer {i} {layer.name} is trainable')

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
    model_name = f'vgg19_depthwise_{num_layers_from_vgg}'

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
            print(f'Layer {i} {layer.name} is trainable')

    # compile model
    loss_type = 'binary_' if classes == 2 else 'categorical_'
    loss_type += 'crossentropy'
    model.compile(loss=loss_type,
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    print('Model compiled')
    return model


def create_vgg19(num_classes,
                 include_top=False,
                 input_shape=(224, 224, 3),
                 num_frozen_layers=None,
                 frozen_layers=None,
                 optimizers=None):
    """
    TODO
    """
    # create the VGG model
    vgg19_model = applications.vgg19.VGG19(include_top=include_top,
                                           weights='imagenet',
                                           input_shape=input_shape)
    # model name
    frozen_layers_str = '_'.join(frozen_layers) if frozen_layers else ''

    if include_top is True:
        # create a sequential model with corresponding name
        model_name = 'vgg19' + frozen_layers_str
        model = Sequential(name=model_name)

        # add input layer at bottom
        model.add(InputLayer(input_shape=input_shape))

        # transform the vgg model into sequential
        # taking care to not add the last layer containing the dense layer
        for layer in vgg19_model.layers[:-1]:
            model.add(layer)

        # add last layer
        model.add(Dense(num_classes, activation='softmax', name='predictions'))

    else:
        # create a sequential model with corresponding name
        model_name = 'vgg19_custom' + frozen_layers_str
        model = Sequential(name=model_name)

        # add input layer at bottom
        model.add(InputLayer(input_shape=input_shape))

        # add the vgg model to model
        for layer in vgg19_model.layers:
            model.add(layer)

        # add classification block
        model.add(Flatten(name='flatten'))
        model.add(Dense(1024, activation='relu', name='fc1')) #1024
        model.add(Dropout(0.7, name='dropout1'))              #notpresent
        model.add(Dense(512, activation='relu', name='fc2'))  #512
        model.add(Dropout(0.5, name='dropout2'))              #notpresent
        model.add(Dense(num_classes, activation='softmax', name='predictions'))

    ###################
    ### FREEZE PART ###
    ###################

    if frozen_layers is None:
        print('Everything is trainable.')

    else:
        # freeze the first 5 blocks
        if 'blocks' in frozen_layers:
            for i, layer in enumerate(model.layers):
                if layer.name not in ['fc1', 'fc2']:
                    layer.trainable = False

            print('Froze all blocks before classification one.')

        # freeze fc1
        if 'fc1' in frozen_layers:
            model.get_layer('fc1').trainable = False
            print('Froze fc1.')

        # freeze fc2
        if 'fc2' in frozen_layers:
            model.get_layer('fc2').trainable = False
            print('Froze fc2.')

    #####################
    ### COMPILE MODEL ###
    #####################

    loss_type = 'binary' if num_classes == 2 else 'categorical'
    loss_type += '_crossentropy'

    if optimizers is None:
        optimizers = [
            optimizers.Adam(lr=1e-5, decay=1e-5)
        ]

    model.compile(loss=loss_type,
                  metrics=['accuracy'],
                  optimizer=optimizers)

    return model

def create_our_cnn(num_classes,
                   weights_path,
                   input_shape=(224, 224, 3),
                   optimizers=None):
    """
    TODO
    """
    # variables to change for SeparableConv2D
    #  - no activation in xception
    #  - use_bias=False
    img_input = Input(shape=input_shape, name='ImageInput')

    # block 1
    x = Conv2D(64, (3, 3),
               activation='relu',
               padding='same',
               name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3),
               activation='relu',
               padding='same',
               name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), name='block1_pool')(x)

    # block 2
    x = SeparableConv2D(128, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block2_sepconv1')(x)
    x = SeparableConv2D(128, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block2_sepconv2')(x)
    x = MaxPooling2D((2, 2), name='block2_pool')(x)

    # block 3
    x = SeparableConv2D(256, (3, 3),
                        activation='relu',
                        padding='same',
                        #use_bias=False,
                        name='block3_sepconv1')(x)
    x = BatchNormalization(name='block3_conv1_bn')(x)
    x = SeparableConv2D(256, (3, 3),
                        activation='relu',
                        padding='same',
                        #use_bias=False,
                        name='block3_sepconv2')(x)
    x = BatchNormalization(name='block3_conv2_bn')(x)
    x = SeparableConv2D(256, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block3_sepconv3')(x)
    x = MaxPooling2D((2, 2), name='block3_pool')(x)

    # block 4
    x = SeparableConv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block4_sepconv1')(x)
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
    # TODO

    # classification block
    x = Flatten(name='flatten')(x)
    x = Dense(1024, activation='relu', name='fc1')(x) #4096
    x = Dropout(0.7, name='dropout1')(x) # not present
    x = Dense(512, activation='relu', name='fc2')(x) # 4096
    x = Dropout(0.5, name='dropout2')(x) # not present
    x = Dense(num_classes, activation='softmax', name='predictions')(x)

    # create the cnn
    model = Model(img_input, x, name='our_cnn')
    print('Model created.')
    model.summary()

    # load weights
    model.load_weights(weights_path, by_name=True)

    # set the first 2 layers to non-trainable
    for layer in model.layers[1:3]:
        print(layer.name)
        layer.trainable = False

    #####################
    ### COMPILE MODEL ###
    #####################

    loss_type = 'binary' if num_classes == 2 else 'categorical'
    loss_type += '_crossentropy'

    if optimizers is None:
        optimizers = [
            optimizers.SGD(lr=1e-4, momentum=0.9),
            #Adam(lr=1e-5, decay=1e-5)
        ]

    model.compile(loss=loss_type,
                  metrics=['accuracy'],
                  optimizer=optimizers)

    return model
