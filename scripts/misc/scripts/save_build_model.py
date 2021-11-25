def build_model(image_shape,
                base_model_weights_path=None,
                classifier_model_weights_path=None):
    
    # build the base model
    if base_model_weights_path is None:
        base_model_weights_path = 'imagenet'
    model = applications.VGG19(input_shape=image_shape,
                               weights=base_model_weights_path,
                               include_top=False,
                               classes=2)
    print('Load model VGG19')
    
    # load weights
    #base_model.load_weights(base_model_weights_path)
    
    # transform the base model into a sequential one
    #model = Sequential(name='vgg_fine_tuning')
    #model.add(InputLayer(input_shape=image_shape, name='input_layer'))
    #for layer in base_model.layers:
    #    model.add(layer)

    # add a classifier model on top of base model
    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:], name='flatten'))
    top_model.add(Dense(256, activation='relu', name='fc1'))
    top_model.add(Dropout(0.5, name='dropout1'))
    top_model.add(Dense(2, activation='softmax', name='fc2'))
    print('Classifier model build')
    
    # todo load weights in classifier_model
    #model.load_weights(classifier_model_weights_path)
    
    model = Model(inputs=model.input, output=top_model(model.output))
    
    model.load_weights(MODEL_PATH + 'vgg_fine_tuning.model')
    
    # freeze layers
    for i, layer in enumerate(model.layers):
        if i <= 21:
            layer.trainable = False
        else:
            layer.trainable = True
            print(f'Layer {i} {layer.name} is trainable')

    # compile model
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])
    
    return model
