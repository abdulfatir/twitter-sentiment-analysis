from __future__ import print_function

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input,  Flatten
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, Embedding
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras import optimizers
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix

import os
import csv
import numpy as np
import sys
import pickle
import utils
import random



MAX_SEQUENCE_LENGTH = 40
MAX_NB_WORDS = 20000
seed= 113
vocab=100000
batch_size=1000
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)

def main():
    X_train, Y_train, word_index = get_training_and_validation_sets()
    print("X_train " +str(X_train[0]))
    print("Y_train " + str(Y_train[0]))
    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, embedding_vecor_length, input_length=MAX_SEQUENCE_LENGTH))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    cb = [ModelCheckpoint("weights.h5", save_best_only=True, save_weights_only=False)]
    model.fit(X_train, Y_train, nb_epoch=20, batch_size=128, validation_split=0.1, shuffle=True, callbacks=cb)
    model.save("model.h5")

    #scores = model.evaluate(X_val, Y_val, verbose=0)
    #print("Accuracy: %.2f%%" % (scores[1] * 100))

    x_test = get_test_sets()
    model_test = load_model("model.h5")
    model_test.load_weights("weights.h5")
    print (model.summary())
    predictions = np.array([])
    print("predicting started")
    predict= model_test.predict(x_test,verbose=0, batch_size=128)
    predict= np.round(predict,decimals=0,out=None)
    #for index, txt in enumerate(x_test):
     #   prediction = predict[index] >= 0.5
      #  predictions = np.concatenate((predictions, prediction))
    predictions= [(str(j), int(predict[j]))
                   for j in range(len(x_test))]
    print("j1 :" + str(predictions[0]))
    print("j2 :" + str(predictions[1]))
    print("j3 :" + str(predictions[2]))
    print("j4 :" + str(predictions[3]))
    utils.save_results_to_csv(predictions, 'networks.csv')
    print("Saved to networks.csv")


def get_test_sets():
    X_test=load_test_data_set()
    sequences = tokenizer.texts_to_sequences(X_test)
    Xtest_processed = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    return Xtest_processed


def batch_the_data(X, Y,batch_size=1000,test_file=False):
    num_batches = int(np.ceil(len(X) / float(batch_size)))
    print("num of batches " +str(num_batches))
    X_batch=[]
    Y_batch=[]
    for i in xrange(num_batches):
        if test_file:
            X_batch=X[i * batch_size: (i + 1) * batch_size]
        else:
            X_batch=X[i * batch_size: (i + 1) * batch_size]
            Y_batch=Y[i * batch_size: (i + 1) * batch_size]
            #print("xshape "+str(X_batch))
            #print("Y_batch " +str(Y_batch))
        yield X_batch,Y_batch




def get_training_and_validation_sets():
    X, Y = load_train_data_set()
    X_processed, word_index=tokenize_data(X)
    #X_train, X_val, Y_train, Y_val = split_the_data(X_processed,Y)
    return X_processed, Y, word_index



def split_the_data(X_processed, Y_processed):
    np.random.seed(seed)
    np.random.shuffle(X_processed)
    np.random.seed(seed)
    np.random.shuffle(Y_processed)
    nb_validation_samples = int(0.1 * X_processed.shape[0])
    x_train = X_processed[:-nb_validation_samples]
    y_train = Y_processed[:-nb_validation_samples]
    x_val = X_processed[-nb_validation_samples:]
    y_val = Y_processed[-nb_validation_samples:]

    return x_train, x_val, y_train, y_val


def tokenize_data(X_raw):
    tokenizer.fit_on_texts(X_raw)
    sequences = tokenizer.texts_to_sequences(X_raw)
    word_index = tokenizer.word_index
    X_processed = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    return X_processed, word_index


def load_train_data_set():
    X = []
    Y = []
    train_csv_file="../train-processed.csv"
    with open(train_csv_file, "rb") as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            text = line[2]
            sentiment= line[1]
            X.append(text)
            Y.append(sentiment)
    return X,Y

def load_test_data_set():
    X = []
    test_csv_file="../test-processed.csv"
    with open(test_csv_file, "rb") as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            #print("line " +str(line))
            text = line[1]
            X.append(text)
    return X


if __name__ == "__main__":
    main()
