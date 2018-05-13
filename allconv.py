from __future__ import print_function
import tensorflow as tf
from keras.datasets import cifar10
#from keras.preprocessing.image import ImageDataGenerator
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

def make_model(weights):
    model = Sequential()
    
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
    sgd = SGD(lr=0.01, decay=0.001, momentum=0.9)

    #if os.path.isfile(weights):
    #    print("loading weights from checkpoint")
    #    model.load_weights(weights)
    
    #batch_size = 32
    #model = LSUVinit(model, X_train[:batch_size,:,:,:])
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

def load_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    n_classes = len(set(y_train.flatten()))
    #X_train = X_train[1:1000]
    #y_train = y_train[1:1000]
    #X_test = X_test[1:100]
    #y_test = y_test[1:100]

    Y_train = np_utils.to_categorical(y_train, n_classes) # Convert to one-hot vector
    Y_test = np_utils.to_categorical(y_test, n_classes)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255 #Normalize
    X_test /= 255

    return (X_train, Y_train, X_test, Y_test)

def preprocess_data(X, datagen=None):
    if datagen  == None:
        datagen = ImageDataGenerator(zca_whitening=True)
        datagen.fit(X)
    for idx in range(X.shape[0]):
        x = X[idx]
        p2, p98 = np.percentile(x, (2, 98))
        x = exposure.rescale_intensity(x, in_range=(p2, p98))
        x = datagen.standardize(x)
        X[idx] = x
    return datagen
        

def preprocess_dataset(X,Y,batches):
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
    
    datagen.fit(X)
    batches = datagen.flow(X, Y, batch_size=batch_size)
    
    return batches, datagen

def run(model, epoch_lengths, learning_rates, batches, test_batches):
    checkpoint = ModelCheckpoint(path, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max')
    history = {'val_loss': [], 'val_acc': [], 'loss': [], 'acc': []}
    
    total_epochs = 0
    iter=0
    for rate, epochs in zip(learning_rate, epoch_lengths):
        K.set_value(model.optimizer.lr, rate)
        print(X_train.shape[0], batch_size, X_train.shape[0]//batch_size, len(batches), len(test_batches))
        history_callback = model.fit_generator(batches,
            #steps_per_epoch=X_train.shape[0]/batch_size,
            epochs=epochs,
            validation_data=(X_test, Y_test),
            validation_steps=1,
            #validation_steps=X_test.shape[0],
            callbacks=[checkpoint],
            verbose=2)
        total_epochs += epochs
        next_hist = history_callback.history
        history = {key:history[key] + next_hist[key] for key in history}
        pandas.DataFrame(history).to_csv("history-{}.csv".format(iter))
        iter=iter+1
        for key in history:
            assert(len(history[key]) == total_epochs)

    return history


K.set_image_dim_ordering('tf')

batch_size = 64
epochs = 350
epochs = 1
rows, cols = 32, 32
channels = 3
learning_rate = [0.01, 0.001, 0.0001, 0.00001]
epoch_lengths = [200, 50, 50, 50]
#learning_rate = [0.01]
#epoch_lengths = [100]

path="weights.hdf5"

X_train, Y_train, X_test, Y_test = load_data()
#whitener = preprocess_data(X_train)
#preprocess_data(X_test, whitener)

batches, datagen = preprocess_dataset(X_train, Y_train, batch_size)
test_batches = datagen.flow(X_test, Y_test, batch_size = len(X_test[0]))

#for i in range(len(X_test)):
#    X_test[i] = datagen.standardize(X_test[i])

model = make_model(path)

for a, b in zip([X_train, Y_train, X_test, Y_test], ['X','y','X_test','y_test']):
    print('{} shape : {}'.format(b, a.shape))
print(model.summary())

history = run(model, epoch_lengths, learning_rate, batches, test_batches)
    
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
