#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report
from keras.callbacks import EarlyStopping, ModelCheckpoint


def train_model(model,
                train_x,
                train_y,
                val_x,
                val_y,
                save_model_path,
                epochs=50,
                batch_size=32,
                metric='val_loss',
                save_best_only=True,
                save_weights_only=True,
                stop_after=10,
                save_history=True,
                save_history_path=None,
                class_weight_mapping=None):
    """
    TODO
    """
    if save_history and save_history_path is None:
        raise ValueError('You must define a path to save history.')

    # callbacks
    early_stopping = EarlyStopping(patience=stop_after,
                                   monitor=metric,
                                   restore_best_weights=True)

    checkpoint = ModelCheckpoint(save_model_path,
                                 monitor=metric,
                                 verbose=1,
                                 save_best_only=save_best_only,
                                 save_weights_only=save_weights_only)

    # here we are, we'll train the model
    history = model.fit(x=train_x,
                        y=train_y,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=[early_stopping, checkpoint],
                        validation_split=0.0,
                        validation_data=(val_x, val_y),
                        shuffle=True,
                        class_weight=class_weight_mapping)

    # save history
    if save_history:
        with open(save_history_path, 'wb') as file:
            pickle.dump(history.history, file)
        print('\n\nSaved history into {}'.format(save_history_path))

    return model, history


def plot_model_performance(preds, ground_truth, labels):
    """
    """
    print(classification_report(ground_truth,
                                preds,
                                target_names=labels))

    # plot confusion matrix
    cm = confusion_matrix(ground_truth, preds)
    plt.figure()
    plot_confusion_matrix(cm, figsize=(12, 8), cmap=plt.cm.Blues)
    #plt.xticks(range(len(labels)), labels) #, fontsize=16)
    #plt.yticks(range(len(labels)), labels) #, fontsize=16)
    plt.show()


def plot_model_history(history_path):
    """
    """
    print('Loading history from {}'.format(history_path))
    history = pickle.load(open(history_path, 'rb'))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4)) #, constrained_layout=True)
    # accuracy
    axes[0].plot(history['acc'])
    axes[0].plot(history['val_acc'])
    axes[0].set_title('Model Accuracy')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xlabel('Epochs')
    axes[0].legend(['train', 'val'])
    # loss
    axes[1].plot(history['loss'])
    axes[1].plot(history['val_loss'])
    axes[1].set_title('Model Loss')
    axes[1].set_ylabel('Loss')
    axes[1].set_xlabel('Epochs')
    axes[1].legend(['train', 'val'])
    plt.show()
