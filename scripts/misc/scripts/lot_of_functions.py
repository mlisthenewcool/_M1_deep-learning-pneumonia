### SHAP

import shap

# syntaxic sugar
train_x = data['train']
test_x = data['test']
test_y = labels['test']

# random indexes
train_indexes = np.random.choice(train_x.shape[0], 10, replace=False)
test_indexes = np.random.choice(test_x.shape[0], 5, replace=False)

# let shap does its job
e = shap.DeepExplainer(model, train_x[train_indexes])
shap_values = e.shap_values(test_x[test_indexes])

# plot the results !
print('True labels are : {}'.format(test_y[test_indexes]))
shap.image_plot(shap_values, test_x[test_indexes], test_y[test_indexes])

def load_weights_from_pretrained_cnn(model):
    """
    """
    from keras.applications import vgg16
    import h5py

    model_filename = model_path / 'vgg16_weights_notop.h5'
    model_path_local = Path(model_filename)

    # download the model and save weights into a file
    if not model_path_local.is_file():
        vgg16_model = vgg16.VGG16(weights='imagenet', include_top=False)
        vgg16_model.save_weights(model_filename)

    # load the weights from h5py file
    f = h5py.File(model_filename, 'r')

    if verbose == 2:
        list_keys = [key for key in f.keys()]
        for key in list_keys:
            print(f[key])
            for key2 in f[key].keys():
                print('\t {}'.format(f[key][key2]))
                for key3 in f[key][key2].keys():
                    print('\t\t {}'.format(f[key][key2][key3]))

    #f.close()
    #return model

    # block1 conv1
    w = f['block1_conv1']['block1_conv1_2']['bias:0']
    b = f['block1_conv1']['block1_conv1_2']['kernel:0']
    model.layers[1].set_weights = [w, b]

    # block1 conv2
    w = f['block1_conv2']['block1_conv2_2']['bias:0']
    b = f['block1_conv2']['block1_conv2_2']['kernel:0']
    model.layers[2].set_weights = [w, b]

    # block2 conv1
    w = f['block2_conv1']['block2_conv1_2']['bias:0']
    b = f['block2_conv1']['block2_conv1_2']['kernel:0']
    model.layers[4].set_weights = [w, b]

    # block2 conv2
    w = f['block2_conv2']['block2_conv2_2']['bias:0']
    b = f['block2_conv2']['block2_conv2_2']['kernel:0']
    model.layers[5].set_weights = [w, b]

    f.close()
    return model

def create_vgg19custom():
    """
    TODO
    """
    
    #TODO
    # variables to change for SeparableConv2D
    # no activation in xception
    # use_bias=False
    
    img_input = Input(shape=(HEIGHT, WIDTH, CHANNELS))

    # block 1
    # it's exactly the same as in vgg19
    x = Conv2D(64, (3, 3),
               activation='relu',
               padding='same',
               name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3),
               activation='relu',
               padding='same',
               name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # block 2
    x = SeparableConv2D(128, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block2_sepconv1')(x)
    x = SeparableConv2D(128, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block2_sepconv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # block 3
    # conv1
    x = SeparableConv2D(256, (3, 3),
                        activation='relu',
                        padding='same',
                        #use_bias=False,
                        name='block3_sepconv1')(x)
    x = BatchNormalization(name='block3_conv1_bn')(x)
    # conv2
    x = SeparableConv2D(256, (3, 3),
                        activation='relu',
                        padding='same',
                        #use_bias=False,
                        name='block3_sepconv2')(x)
    x = BatchNormalization(name='block3_conv2_bn')(x)
    # conv3
    x = SeparableConv2D(256, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block3_sepconv3')(x)
    x = BatchNormalization(name='block3_conv_3_bn')(x)
    # conv4
    x = SeparableConv2D(256, (3, 3),
                        activation='relu',
                        padding='same',
                        #use_bias=False,
                        name='block3_sepconv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # block 4
    # conv1
    x = SeparableConv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block4_sepconv1')(x)
    x = BatchNormalization(name='block4_conv1_bn')(x)
    # conv2
    x = SeparableConv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block4_sepconv2')(x)
    x = BatchNormalization(name='block4_conv2_bn')(x)
    # conv3
    x = SeparableConv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block4_sepconv3')(x)
    x = BatchNormalization(name='block4_conv3_bn')(x)
    # conv4
    x = SeparableConv2D(512, (3, 3),
                    activation='relu',
                    padding='same',
                    name='block4_sepconv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # block 5
    # conv1
    x = SeparableConv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block5_sepconv1')(x)
    x = BatchNormalization(name='block5_conv1_bn')(x)
    # conv2
    x = SeparableConv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block5_sepconv2')(x)
    x = BatchNormalization(name='block5_conv2_bn')(x)
    # conv3
    x = SeparableConv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block5_sepconv3')(x)
    x = BatchNormalization(name='block5_conv3_bn')(x)
    # conv4
    x = SeparableConv2D(512, (3, 3),
                    activation='relu',
                    padding='same',
                    name='block5_sepconv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # classification block
    x = Flatten(name='flatten')(x)
    x = Dense(1024, activation='relu', name='fc1_custom')(x) #4096
    x = Dropout(0.7, name='dropout1')(x) # not present
    x = Dense(512, activation='relu', name='fc2_custom')(x) # 4096
    x = Dropout(0.5, name='dropout2')(x) # not present
    x = Dense(NUM_LABELS, activation='softmax', name='predictions_custom')(x)

    # create the model
    model = Model(img_input, x, name='vgg19custom')
    
    # load weights
    # !!! WEIGHTS !!!
    weights_path = ROOT_PATH + 'models/vgg19_weights_notop.h5'
    print('Trying to load weights from {} ...'.format(weights_path))
    
    ### TODO cleaner
    from pathlib import Path
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        vgg19_model = vgg19.VGG19(weights='imagenet', include_top=False)
        vgg19_model.save_weights(weights_path)

    # load weights
    model.load_weights(weights_path, by_name=True)
    print('DONE')
    
    return model