from __future__ import print_function
import tensorflow as tf
from keras.datasets import cifar10
#from keras.preprocessing.image import ImageDataGenerator
from image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Activation, Convolution2D, GlobalAveragePooling2D, merge
from keras.utils import np_utils
from keras.optimizers import SGD
from keras import backend as K
from keras.models import Model
from keras.layers.core import Lambda
from keras.callbacks import ModelCheckpoint
from lsuv_init import LSUVinit
import os
import pandas
import numpy as np

def make_model(weights):
    model = Sequential()
    
    model.add(Convolution2D(96, 3, 3, border_mode = 'same', input_shape=(32, 32, 3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(96, 3, 3,border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(96, 3, 3, border_mode='same', subsample = (2,2)))
    model.add(Dropout(0.5))
    
    model.add(Convolution2D(192, 3, 3, border_mode = 'same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(192, 3, 3,border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(192, 3, 3,border_mode='same', subsample = (2,2)))
    model.add(Dropout(0.5))
    
    model.add(Convolution2D(192, 3, 3, border_mode = 'same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(192, 1, 1,border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(10, 1, 1, border_mode='valid'))
    
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9)

    #if os.path.isfile(weights):
    #    print("loading weights from checkpoint")
    #    model.load_weights(weights)
    
    batch_size = 32
    model = LSUVinit(model, X_train[:batch_size,:,:,:])
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

def load_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    n_classes = len(set(y_train.flatten()))

    Y_train = np_utils.to_categorical(y_train, n_classes) # Convert to one-hot vector
    Y_test = np_utils.to_categorical(y_test, n_classes)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 256 #Normalize
    X_test /= 256

    return (X_train, Y_train, X_test, Y_test)

def preprocess_dataset(X,Y,batches):
    datagen = ImageDataGenerator(
            contrast_stretching=True, adaptive_equalization=False, histogram_equalization=False,
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=True,  # whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False) 
    
    datagen.fit(X)
    batches = datagen.flow(X, Y, batch_size=batch_size)
    return batches

def run(model, epoch_lengths, learning_rates):
    checkpoint = ModelCheckpoint(path, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max')
    history = {'val_loss': [], 'val_acc': [], 'loss': [], 'acc': []}
    
    total_epochs = 0
    for rate, epochs in zip(learning_rate, epoch_lengths):
        K.set_value(model.optimizer.lr, rate)
        history_callback = model.fit_generator(batches, samples_per_epoch=X_train.shape[0], nb_epoch=epochs, validation_data=(X_test, Y_test), callbacks=[checkpoint], verbose=2)
        total_epochs += epochs
        next_hist = history_callback.history
        history = {key:history[key] + next_hist[key] for key in history}
        for key in history:
            assert(len(history[key]) == total_epochs)

    return history


K.set_image_dim_ordering('tf')

batch_size = 32
epochs = 350
epochs = 1
rows, cols = 32, 32
channels = 3
learning_rate = [0.25, 0.1, 0.05, 0.01]
epoch_lengths = [200, 50, 50, 50]
learning_rate = [0.01]
epoch_lengths = [100]

path="weights.hdf5"

X_train, Y_train, X_test, Y_test = load_data()
batches = preprocess_dataset(X_train, Y_train, batch_size)
model = make_model(path)

for a, b in zip([X_train, Y_train, X_test, Y_test], ['X','y','X_test','y_test']):
    print('{} shape : {}'.format(b, a.shape))
print(model.summary())

history = run(model, epoch_lengths, learning_rate)
    
print(history)

pandas.DataFrame(history).to_csv("history.csv")
model.save('final_model.h5')
#def resize(infile):
#    from PIL import Image
#    im = Image.open(infile)
#    im = im.resize((224, 224), Image.ANTIALIAS)
#    return np.array(im).astype(np.float32)
#
#im = resize('image.png')
#out = model.predict(im)
#print(np.argmax(out))
