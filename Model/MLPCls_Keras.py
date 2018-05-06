from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
import sys
cur_path = os.path.dirname(__file__)
rel_path = "..\\"
abs_file_path = os.path.join(cur_path, rel_path)
sys.path.insert(0, abs_file_path)
from myUtility import show_train_history
from myUtility import plot_images_labels_prediction

# data_set = 'iris'
# data_set = 'mnist'
data_set = 'cifar10'

model_path = os.path.join(cur_path, "SaveModel\\MLP\\Keras\\" + data_set + ".h5")

model = Sequential()

try:
    model.load_weights(model_path)
    print("Load old model data")
except:
    print("No old model data, create new one")

if data_set == 'iris':

    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    X_train, X_test, Y_train, Y_test = \
        train_test_split(X, Y, test_size=0.3, random_state=1)
    
    """
    sc = StandardScaler()
    sc.fit(X)
    X_train = sc.transform(X_train)                              
    X_test = sc.transform(X_test)
    """

    X_train_processed = X_train                        
    X_test_processed = X_test

    classes = 3
    input_size = 4
    hidden1_neurons = 200
    hidden2_neurons = 100

    """
    Epoch 200/200
 - 0s - loss: 0.1493 - acc: 0.9681 - val_loss: 0.1628 - val_acc: 0.9091
    """

    # activation_h = 'relu'
    activation_h = 'sigmoid'
    activation_o = 'softmax'
    # optimizer_ = 'sgd'
    optimizer_ = 'adam'
    # optimizer_='adadelta'
    batch_size = 10
    # batch_size = None
    epochs = 200
    valid_data_rate = 0.1

    model.add(Dense(hidden1_neurons, input_dim=input_size, 
                    kernel_initializer='normal', activation=activation_h))
    model.add(Dense(hidden2_neurons, kernel_initializer='normal', 
                    activation=activation_h))
    model.add(Dense(classes, kernel_initializer='normal', 
                    activation=activation_o))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer=optimizer_)

elif data_set == 'mnist':
    from keras.datasets import mnist

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train_processed = X_train/255                             
    X_test_processed = X_test/255

    X_train_img = X_train_processed.reshape(60000, 28, 28)
    X_test_img = X_test_processed.reshape(10000, 28, 28)

    label_dict={0:"0", 1:"1", 2:"2", 3:"3", 4:"4", 
                5:"5", 6:"6", 7:"7", 8:"8", 9:"9"}

    classes = 10
    input_size = 784
    hidden1_neurons = 1000
    hidden2_neurons = 1000

    """
    Epoch 70/70
    - 3s - loss: 0.0624 - acc: 0.9799 - val_loss: 0.0615 - val_acc: 0.9818
    """

    # activation_h = 'relu'
    activation_h = 'sigmoid'
    activation_o = 'softmax'
    # optimizer_ = 'sgd'
    # optimizer_ = 'adam'
    optimizer_='adadelta'
    batch_size = 200
    # batch_size = None
    epochs = 70
    valid_data_rate = 0.1

    model.add(Dense(hidden1_neurons, input_dim=input_size, 
                    kernel_initializer='normal', activation=activation_h))
    model.add(Dropout(0.5))
    model.add(Dense(hidden2_neurons, kernel_initializer='normal', 
                    activation=activation_h))
    model.add(Dropout(0.5))
    model.add(Dense(classes, kernel_initializer='normal', 
                    activation=activation_o))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer=optimizer_)

elif data_set == 'cifar10':
    from keras.datasets import cifar10

    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    X_train = X_train.reshape(50000, 3072)
    X_test = X_test.reshape(10000, 3072)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train_processed = X_train/255                             
    X_test_processed = X_test/255

    X_train_img = X_train_processed.reshape(50000, 32, 32, 3)
    X_test_img = X_test_processed.reshape(10000, 32, 32, 3)

    Y_train = Y_train.squeeze()
    Y_test = Y_test.squeeze()

    label_dict={0:"airplane", 1:"automobile", 2:"bird", 3:"cat", 4:"deer", 
                5:"dog", 6:"frog", 7:"horse", 8:"ship", 9:"truck"}

    classes = 10
    input_size = 3072
    hidden1_neurons = 3000
    hidden2_neurons = 2000

    """
    poch 30/30
    - 7s - loss: 1.4110 - acc: 0.4947 - val_loss: 1.4020 - val_acc: 0.5090
    """

    # activation_h = 'relu'
    activation_h = 'sigmoid'
    activation_o = 'softmax'
    # optimizer_ = 'sgd'
    # optimizer_ = 'adam'
    optimizer_='adadelta'
    batch_size = 200
    # batch_size = None
    epochs = 30
    valid_data_rate = 0.1

    model.add(Dense(hidden1_neurons, input_dim=input_size, 
                    kernel_initializer='normal', activation=activation_h))
    model.add(Dropout(0.5))
    model.add(Dense(hidden2_neurons, kernel_initializer='normal', 
                    activation=activation_h))
    model.add(Dropout(0.5))
    model.add(Dense(classes, kernel_initializer='normal', 
                    activation=activation_o))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer=optimizer_)


Y_train_oneHot = np_utils.to_categorical(Y_train, classes)
Y_test_oneHot = np_utils.to_categorical(Y_test, classes)


print("Model:", model.summary())

train_history = model.fit(X_train_processed, Y_train_oneHot, batch_size=batch_size,
                          epochs=epochs, validation_split=valid_data_rate, verbose=2)

score = model.evaluate(X_train_processed, Y_train_oneHot, verbose=2)
print('Train accuracy:', score[1])

score = model.evaluate(X_test_processed, Y_test_oneHot, verbose=2)
print('Test accuracy:', score[1])

model.save_weights(model_path)

predict_result = model.predict_classes(X_test_processed)
table = pd.crosstab(Y_test, predict_result, rownames=["label"], 
                    colnames=["predict"])
print("cross table:\n", table)

print(train_history.history.keys())

show_train_history(train_history, valid_data_rate=valid_data_rate)

predict_prob = model.predict(X_test_processed)

if data_set != 'iris':
    plot_images_labels_prediction(X_test_img, Y_test, predict_result, predict_prob, label_dict, idx=0)
