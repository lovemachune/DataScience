import keras
import pandas as pd
import os
import sys
import numpy as np
import re
import csv
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM
from keras.models import Model, Sequential

def word_split(mystring):
    str_split = []
    for tmp in mystring:
        tmp = tmp.lower()
        tmp = re.sub('[^a-zA-Z0-9\s\?\!]+', '', tmp)
        tmp = tmp.replace('!', ' !')
        tmp = tmp.replace('?', ' ?')
        tmp = tmp.split(' ')
        while True:
            if '' not in tmp:
                break
            tmp.remove('')
        while True:
            if 'the' not in tmp:
                break
            tmp.remove('the')
        while True:
            if 'and' not in tmp:
                break
            tmp.remove('and')
        while True:
            if 'of' not in tmp:
                break
            tmp.remove('of')
        '''
        while True:
            if 'is' not in tmp:
                break
            tmp.remove('is')
        while True:
            if 'are' not in tmp:
                break
            tmp.remove('are')
        '''
        str_split.append(tmp)
    return str_split

def my_model():
    GLOVE_DIR = './'
    MAX_SEQUENCE_LENGTH = 50
    MAX_NB_WORDS = 10000
    EMBEDDING_DIM = 100
    NUM_LSTM_UNITS = 512
    VALIDATION_SPLIT = 0.2

    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print("load glove")
    all_data = pd.read_csv("train.csv")
    data = all_data['Headline']
    label = all_data['Label']
    my_split = word_split(data)
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(my_split)
    mm = keras.models.load_model('my_model.h5')
    test_data = pd.read_csv("test.csv")
    data = test_data['Headline']
    label = test_data['Label']
    test_split = word_split(data)
    sequences = tokenizer.texts_to_sequences(test_split)
    x = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    y_pre = mm.predict(x)
    b = np.arange(1, y_pre.shape[0]+1).reshape(y_pre.shape[0], 1).astype('int32')
    y_pre = np.append(b, y_pre, axis=1).astype(object)
    for i in range(len(y_pre)):
        y_pre[i][0] = int(y_pre[i][0])
    print("predict done")
    with open('309511045.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ID','Label'])
        writer.writerows(y_pre)

if __name__ == '__main__':
    my_model()
