from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers.core import Dense, Activation
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


iris = datasets.load_iris()
X = iris.data
Y = iris.target

print("shape of X:", X.shape)
print("shape of Y:", Y.shape)

X_train, X_test, Y_train, Y_test = \
        train_test_split(X, Y, test_size=0.3, random_state=1)
"""
sc = StandardScaler()
sc.fit(X)
X_train_std = sc.transform(X_train)                              
X_test_std = sc.transform(X_test)
"""

classes = 3
Y_train_oneHot = np_utils.to_categorical(Y_train, classes)
Y_test_oneHot = np_utils.to_categorical(Y_test, classes)


input_size = 4
batch_size = 10
hidden1_neurons = 200
hidden2_neurons = 100
epochs = 200

model = Sequential()
model.add(Dense(hidden1_neurons, input_dim=input_size))
model.add(Activation('sigmoid'))
model.add(Dense(hidden2_neurons, input_dim=hidden1_neurons))
model.add(Activation('sigmoid'))
model.add(Dense(classes, input_dim=hidden2_neurons))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
              optimizer='sgd')

print("Model:", model.summary())

train_history = model.fit(X_train, Y_train_oneHot, batch_size=batch_size, 
                          epochs=epochs, verbose=0)

score = model.evaluate(X_test, Y_test_oneHot, verbose=1)

"""
model.fit(X_train_std, Y_train, batch_size=batch_size, epochs=epochs,
verbose=0)
score = model.evaluate(X_test_std, Y_test, verbose=1)
"""

print('Test accuracy:', score[1])

predict_result = model.predict_classes(X_test)
table = pd.crosstab(Y_test, predict_result, rownames=["label"], colnames=["predict"])
print("cross table:\n", table)

print(train_history.history.keys())

show_train_history(train_history)
