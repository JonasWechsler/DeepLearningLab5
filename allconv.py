from __future__ import print_function
import tensorflow as tf
from keras.datasets import cifar10
import image
from image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Activation, Conv2D, GlobalAveragePooling2D, merge
from keras.utils import np_utils
from keras.optimizers import SGD
from keras import backend as K
from keras.models import Model
from keras.layers.core import Lambda
from keras.callbacks import ModelCheckpoint
from lsuv_init import LSUVinit
from skimage import data, img_as_float
from skimage import exposure
from PIL import Image
import os
import pandas
import numpy as np

K.set_image_dim_ordering('tf')

def load_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    n_classes = len(set(y_train.flatten()))

    Y_train = np_utils.to_categorical(y_train, n_classes) # Convert to one-hot vector
    Y_test = np_utils.to_categorical(y_test, n_classes)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255 #Normalize
    X_test /= 255

    return (X_train, Y_train, X_test, Y_test)


def preprocess_dataset(X,Y, settings):
    if settings.augment_data:
        datagen = ImageDataGenerator(
                contrast_stretching=True, adaptive_equalization=False, histogram_equalization=False,
                #featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # whitening
                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False) 
    else:
        datagen = ImageDataGenerator()
    
    datagen.fit(X)
    batches = datagen.flow(X, Y, batch_size=settings.batch_size)
    
    return batches, datagen

def make_model(settings, X_train=None):
    model = Sequential()

    if settings.input_dropout:
        model.add(Dropout(0.2, input_shape=(32, 32, 3)))
        model.add(Conv2D(96, (3, 3), padding = 'same'))
    else:
        model.add(Conv2D(96, (3, 3), padding = 'same', input_shape=(32, 32, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(96, (3, 3),padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(96, (3, 3), padding='same', strides = (2,2)))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(192, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (3, 3),padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (3, 3),padding='same', strides = (2,2)))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(192, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (1, 1),padding='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(10, (1, 1), padding='valid'))
    
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sgd = SGD(lr=settings.learning_rates[0], decay=settings.decay, momentum=0.9)

    if settings.weights_path != None and os.path.isfile(weights):
        print("loading weights from checkpoint")
        model.load_weights(weights)

    if settings.orthonormal_init: 
        model = LSUVinit(model, X_train[:settings.batch_size,:,:,:])

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

def run(settings, batches, test_batches, X_train):
    model = make_model(settings, X_train)
    checkpoint = ModelCheckpoint(settings.weights_path, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max')
    history = {'val_loss': [], 'val_acc': [], 'loss': [], 'acc': []}
    
    total_epochs = 0
    iter=0
    for rate, epochs in zip(settings.learning_rates, settings.epoch_lengths):
        K.set_value(model.optimizer.lr, rate)
        history_callback = model.fit_generator(batches,
            #steps_per_epoch=X_train.shape[0]/batch_size,
            epochs=epochs,
            validation_data=test_batches,
            validation_steps=1,
            #validation_steps=X_test.shape[0],
            callbacks=[checkpoint],
            verbose=2)
        next_hist = history_callback.history
        history = {key:history[key] + next_hist[key] for key in history}
        pandas.DataFrame(history).to_csv("history-{}.csv".format(iter))
        iter=iter+1
        total_epochs += epochs
        for key in history:
            assert(len(history[key]) == total_epochs)

    return history

class Settings:
    def __init__(self,
            batch_size=64,
            epoch_lengths=[100],
            learning_rates=[0.01],
            momentum=0.9,
            weights_path=None,
            decay=3e-5,
            input_dropout=False,
            orthonormal_init=True,
            augment_data=True):
        self.batch_size = batch_size
        self.epoch_lengths = epoch_lengths
        self.learning_rates = learning_rates
        self.weights_path=None
        self.decay = decay
        self.input_dropout = input_dropout 
        self.momentum = momentum
        self.orthonormal_init = orthonormal_init
        self.augment_data = augment_data

def param_search(settings_list, batches, test_batches, X_train):
    with open('out.txt', 'a') as f:
        for settings in settings_list:
            history = run(settings, batches, test_batches, X_train)
            pandas.DataFrame(history).to_csv("history-{}-{}.csv".format(settings.decay,settings.learning_rate[0]))
            loss = np.min(history['loss'])
            acc = np.max(history['acc'])
            val_loss = np.min(history['val_loss'])
            val_acc = np.max(history['val_acc'])
            line = ','.join(map(str,[lr, decay, loss, acc, val_loss, val_acc]))
            print(line)
            f.write(line)
            f.write('\n')
            f.flush()

def learner(settings, batches, test_batches, X_train):
    history = run(settings, batches, test_batches, X_train)
    pandas.DataFrame(history).to_csv("history-{}-{}-{}.csv".format(settings.epoch_lengths[0], settings.learning_rates[0], decay))
    model.save('final_model.h5')

def run_our_model():
    settings = Settings(batch_size = 32,
            epoch_lengths = [100],
            learning_rates = [0.015],
            decay = 3e-5,
            input_dropout = False,
            orthonormal_init = True)
    X_train, Y_train, X_test, Y_test = load_data()
    batches, datagen = preprocess_dataset(X_train, Y_train, settings)
    test_batches = (X_test, Y_test)
    learner(settings, batches, test_batches, X_train)

def run_their_model():
    settings = Settings(batch_size = 64,
            epoch_lengths = [200, 50, 50, 50],
            learning_rates = [0.05, 0.005, 0.0005, 0.00005],
            decay = 0.001,
            input_dropout = True,
            orthonormal_init = False)
    X_train, Y_train, X_test, Y_test = load_data()
    batches, datagen = preprocess_dataset(X_train, Y_train, settings)
    test_batches = (X_test, Y_test)
    learner(settings, batches, test_batches, X_train)

if __name__ == "__main__":
    run_our_model()
    #param_search([3e-5, 3e-5, 3e-5, 3e-5, 3e-5, 3e-5], [0.005, 0.01, 0.015, 0.02, 0.025, 0.03], 20)
