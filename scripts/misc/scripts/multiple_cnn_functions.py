def get_cnn_model(model_name: str, **kwargs):
    """
    TODO
    """
    if model_name == 'vgg16':
        return create_vgg16(kwargs)
    
    #elif model_name == 'vgg16custom':
    #    return create_vgg16custom(kwargs)
    
    elif model_name == 'vgg19':
        return create_vgg19(kwargs)
    
    elif model_name == 'vgg19custom':
        return create_vgg19custom(kwargs)
    
    elif model_name == 'xception':
        return create_xception(kwargs)
    
    else:
        raise ValueError(
                'The model {} isn\'t implemented yet'.format(model_name))

################################################################################
def create_vgg16():
    """
    TODO
    """
    base_model = vgg16.VGG16(include_top=False,
                             weights='imagenet',
                             input_shape=(HEIGHT, WIDTH, CHANNELS))

    model = Sequential()
    model.add(base_model)

    # add classification block
    model.add(Flatten(name='flatten'))
    model.add(Dense(1024, activation='relu', name='fc1'))
    model.add(Dropout(0.7, name='dropout1'))
    model.add(Dense(512, activation='relu', name='fc2'))
    model.add(Dropout(0.5, name='dropout2'))
    model.add(Dense(NUM_LABELS, activation='softmax', name='predictions'))

    # see the base model architecture
    #base_model.summary()

    return model

################################################################################
def create_vgg19(_model_name,
                 _frozen_layers=None,
                 _include_top=True):
    """
    TODO
    """
    
    # create the VGG model
    vgg19_model = vgg19.VGG19(include_top=_include_top,
                              weights='imagenet',
                              input_shape=(HEIGHT, WIDTH, CHANNELS))
   
    #vgg19_model.summary()
    
    # create a sequential model
    frozen_layers_str = '_'.join(_frozen_layers) if _frozen_layers else ''
    m_name = 'vgg19_' + frozen_layers_str
    model = Sequential(name=m_name)
    
    # add input layer
    model.add(InputLayer(input_shape=(HEIGHT, WIDTH, CHANNELS)))
    
    # convert vgg19 to sequential and remove the last layer
    for layer in vgg19_model.layers[:-1]:
        model.add(layer)
    
    #model.summary()
    
    # add last layer
    model.add(Dense(NUM_LABELS, activation='softmax', name='predictions'))
    
    #model.summary()
    
    # FREEZE PART
    if _frozen_layers is None:
        print('Everything is trainable')
        return model
    
    # freeze the first 5 blocks
    if 'blocks' in _frozen_layers:
        for i, layer in enumerate(model.layers):
            # freeze everything except fc1 and fc2
            if layer.name not in ['fc1', 'fc2']:
                #print('Not trainable', i, layer.name)
                layer.trainable = False
        
        print('Froze all blocks before classification one')
    
    # freeze fc1
    if 'fc1' in _frozen_layers:
        model.get_layer('fc1').trainable=False
        print('Froze fc1')

    # freeze fc2
    if 'fc2' in _frozen_layers:
        model.get_layer('fc2').trainable=False
        print('Froze fc2')
        
    return model

################################################################################
def create_vgg19custom(_model_name, _frozen_layers=[]):
    """
    TODO
    """
    
    # create the VGG model
    vgg19_model = vgg19.VGG19(include_top=False,
                              weights='imagenet',
                              input_shape=(HEIGHT, WIDTH, CHANNELS))
    
    """
    # freeze the layers according to `frozen_layers`
    for i, layer in enumerate(model.layers):
        if layer.name in _frozen_layers:
            print('Not trainable', i, layer.name)
            layer.trainable = False
        else:
            print('Trainable', i, layer.name)
            layer.trainable = True
    """
    
    model_name = 'vgg19_custom_'
    
    if 'blocks' in _frozen_layers:
        vgg19_model.trainable = False
        model_name += 'blocks'
    

    # create a sequential model and add the vgg19 model at bottom
    model = Sequential(name=model_name)
    model.add(vgg19_model)
    
    # add classification block
    model.add(Flatten(name='flatten'))
    model.add(Dense(1024, activation='relu', name='fc1')) #1024
    model.add(Dropout(0.7, name='dropout1'))              #notpresent
    model.add(Dense(512, activation='relu', name='fc2'))  #512
    model.add(Dropout(0.5, name='dropout2'))              #notpresent
    model.add(Dense(NUM_LABELS, activation='softmax', name='predictions'))
    
    #weights_list = model.get_weights()
    #for i, layer in enumerate(model.get_layer('vgg19').layers):
    #    print(i, layer.name)

    #vgg19_model.summary()
    #model.summary()
    
    return model

def create_xception():
    """
    TODO
    """
    base_model = xception.Xception(include_top=False,
                                   weights='imagenet',
                                   input_shape=(HEIGHT, WIDTH, CHANNELS))

    model = Sequential()
    model.add(base_model)

    # add classification block
    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu', name='fc1')) #1024
    model.add(Dropout(0.7, name='dropout1'))              #notpresent
    model.add(Dense(4096, activation='relu', name='fc2'))  #512
    model.add(Dropout(0.5, name='dropout2'))              #notpresent
    model.add(Dense(NUM_LABELS, activation='softmax', name='predictions'))

    # see the base model architecture
    #base_model.summary()
    #model.summary()

    return model

### XCEPTION
MODEL_NAME = 'xception'

model = create_xception()
model = compile_model(model, MODEL_NAME)

if True:
    # !!! WARNINGS !!!
    # train or not is a question of time
    model, history = train_model(model,
                                 MODEL_NAME,
                                 train_data=data['train'],
                                 train_labels=labels['train'],
                                 val_data=data['test'],
                                 val_labels=labels['test'],
                                 epoch=100,
                                 batch_size=32,
                                 metric='val_acc',
                                 save_best_only=True,
                                 save_weights_only=False,
                                 stop_after=10)


"""
## Try to understand what our CNN learns

https://github.com/slundberg/shap

https://shap.readthedocs.io/en/latest/

https://www.kaggle.com/aakashnain/what-does-a-cnn-see
"""

#! pip install shap

"""
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
"""