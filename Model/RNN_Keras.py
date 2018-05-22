from sklearn.model_selection import train_test_split
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


data_set = 'IMDb'

model_path = os.path.join(cur_path, "SaveModel\\MLP\\Keras\\" + data_set + ".h5")

model = Sequential()

try:
    model.load_weights(model_path)
    print("Load old model data")
except:
    print("No old model data, create new one")


if data_set == 'IMDb':
    
    from keras.preprocessing import sequence
    from keras.preprocessing.text import Tokenizer
    from keras.layers.embeddings import Embedding
    from keras.layers.recurrent import SimpleRNN
    from keras.layers.recurrent import LSTM

    cur_path = os.path.dirname(__file__)
    rel_path = "..\\"
    abs_file_path = os.path.join(cur_path, rel_path)
    sys.path.insert(0, abs_file_path)
    from myDataset import get_IMDb
    from myUtility import text_preprocessor

    imdb_df = get_IMDb()
    imdb_df['review'] = imdb_df['review'].apply(text_preprocessor)
    
    imdb_data = imdb_df.as_matrix()
    
    X = imdb_data[:, 0]
    Y = imdb_data[:, 1]

    X_train, X_test, Y_train, Y_test = \
        train_test_split(X, Y, test_size=0.3, random_state=1)
        
    # print('X_train:', X_train)

    max_words_dict = 3800
    max_words_len = 380
    text_token = Tokenizer(num_words=max_words_dict)
    text_token.fit_on_texts(X_train)

    # print("text_token.document_count:", text_token.document_count)    
    # print("text_token.word_index:", text_token.word_index)
    # print("text_token.word_docs:", text_token.word_docs)

    X_train_seq = text_token.texts_to_sequences(X_train)
    X_test_seq = text_token.texts_to_sequences(X_test)

    # print('X_train[0]:', X_train[0])
    # print('X_train_seq[0]:', X_train_seq[0])

    X_train_processed = sequence.pad_sequences(X_train_seq, maxlen=max_words_len)
    X_test_processed = sequence.pad_sequences(X_test_seq, maxlen=max_words_len)

    classes = 2
    input_size = max_words_dict
    hidden1_neurons = 256
    # hidden2_neurons = 512

    activation_h = 'relu'
    # activation_h = 'sigmoid'
    #activation_o = 'softmax'
    activation_o = 'sigmoid'
    # optimizer_ = 'sgd'
    # optimizer_ = 'adam'
    optimizer_='adadelta'
    batch_size = 100
    # batch_size = None
    epochs = 10
    valid_data_rate = 0.2

    embedding_output = 32
    RNN_layers = 16

    model.add(Embedding(input_dim=input_size, input_length=max_words_len,
                        output_dim=embedding_output))
    model.add(Dropout(0.3))
    # model.add(SimpleRNN(units=RNN_layers))
    model.add(LSTM(units=RNN_layers))
    model.add(Dense(hidden1_neurons, kernel_initializer='normal', 
                    activation=activation_h))
    model.add(Dropout(0.3))
    # model.add(Dense(hidden2_neurons, kernel_initializer='uniform', 
                    # activation=activation_h))
    # model.add(Dropout(0.5))
    model.add(Dense(classes, kernel_initializer='normal', 
                    activation=activation_o))
    model.compile(loss='binary_crossentropy', metrics=['accuracy'],
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

