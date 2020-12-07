from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.optimizers import RMSprop

import numpy as np
import pandas as pd
import random
import requests
import sys


# For spam Model
def lstm_gen_spam(label, num, noise):
    data = "spam"
    # Read in the data
    df = pd.read_csv(f"sms/sms_data/{data}_seed_05.csv")
    text_ = " "

    for i in df['text']:
        text_ += i

    text_ = text_.split(" ")

    text = []
    for i in text_:
        text.append(i+" ")


    # Noise
    data_ = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))
    df = pd.DataFrame(data_.data)

    articles = []
    for i in df[0]:
        articles.append(i)

    random.shuffle(articles)
    
    text_ = " "

    for i in articles:
        text_ += i

    text_ = text_.split(" ")

    # Figuring out how much random noise
    # For example, if I want 25% noise
    var = 1-noise # 1 - .25 = .75
    var2 = var/noise # .75 / .25 = 3
    var3 = 1/var2 # So I need a number of token for noise 
    var4 = round(len(text)*var3) # equal to 1/3 of the original dataset

    # So I add that
    for i in range(0, var4):
        text.append(text_[i]+ " ")


    # Encode the data as Chars
    chars = sorted(list(set(text)))
    char_int = dict((c, i) for i, c in enumerate(chars))
    int_char = dict((i, c) for i, c in enumerate(chars))

    # Create the Sequence Data
    maxlen = 40
    steps = 3

    sentences = [] #X
    next_chars = [] #Y

    for i in range(0, len(text)- maxlen, steps):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])

    # Specify X and y
    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_int[char]] = 1
        y[i, char_int[next_chars[i]]] = 1

    # Early Stopping Requirements
    stop = EarlyStopping(monitor='loss', min_delta=0.05, patience=1, mode='auto')

    # Build Model
    model = Sequential()
    model.add(LSTM(300, input_shape=(maxlen, len(chars))))
    model.add(Dense(900, activation='relu'))
    #model.add(Dense(600, activation='relu'))
    model.add(Dense(len(chars), activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adamax')
    
    # Fit Model
    model.fit(x, y,
            batch_size=10,
            epochs=200, 
            callbacks=[stop])

    # Save Model
    model.save(f'sms_{data}_lstm')


    def sample(preds):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / 1 # Could be changed to temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)



    model.save(f"sms_{data}_lstm")
    def gen_data(number):
        data = []
        for i in range(0, number):
            start_index = random.randint(0, len(text) - maxlen - 1)
            generated = ''
            blah = []
            sentence = ''
            #senten = text[:1]
            senten = text[start_index: start_index + maxlen]
            for i in senten:
                sentence += i
            for i in range(1, random.randint(2, 118)):
                x_pred = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(senten):
                    x_pred[0, t, char_int[char]] = 1
                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds)
                next_char = int_char[next_index]
                sentence = sentence[1:] + next_char
                blah.append(next_char)
            data.append(generated+ ''.join(blah))
        return data



    var = pd.DataFrame(gen_data(num), columns=['text'])
    var['label'] = label
    print(var.head())
    print(var.tail())
    noise_ = str(noise)

    var.to_csv(f"sms_{data}_lstm_noise_{noise_[2:]}.csv", index=False)






# For ham Model 
def lstm_gen_ham(label, num, noise):
    # Read in the data
    data = "ham"

    df = pd.read_csv(f"sms/sms_data/{data}_seed_05.csv")
    text_ = " "

    for i in df['text']:
        text_ += i

    text_ = text_.split(" ")

    text = []
    for i in text_:
        text.append(i+" ")

   # Noise
    data_ = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))
    df = pd.DataFrame(data_.data)

    articles = []
    for i in df[0]:
        articles.append(i)

    random.shuffle(articles)
    
    text_ = " "

    for i in articles:
        text_ += i

    text_ = text_.split(" ")

    # Figuring out how much random noise
    # For example, if I want 25% noise
    var = 1-noise # 1 - .25 = .75
    var2 = var/noise # .75 / .25 = 3
    var3 = 1/var2 # So I need a number of token for noise 
    var4 = round(len(text)*var3) # equal to 1/3 of the original dataset

    # So I add that
    for i in range(0, var4):
        text.append(text_[i]+" ")


    # Encode the data as Chars
    chars = sorted(list(set(text)))
    char_int = dict((c, i) for i, c in enumerate(chars))
    int_char = dict((i, c) for i, c in enumerate(chars))

    # Create the Sequence Data
    maxlen = 40
    steps = 3

    sentences = [] #X
    next_chars = [] #Y

    for i in range(0, len(text)- maxlen, steps):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])

    # Specify X and y
    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_int[char]] = 1
        y[i, char_int[next_chars[i]]] = 1

    # Early Stopping Requirements
    stop = EarlyStopping(monitor='loss', min_delta=0.05, patience=1, mode='auto')

    # Build Model
    model = Sequential()
    model.add(LSTM(300, input_shape=(maxlen, len(chars))))
    model.add(Dense(900, activation='relu'))
    #model.add(Dense(600, activation='relu'))
    model.add(Dense(len(chars), activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adamax')
    
    # Fit Model
    model.fit(x, y,
            batch_size=10,
            epochs=200, 
            callbacks=[stop])

    # Save Model
    model.save(f"sms_{data}_lstm")


    def sample(preds):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / 1 # Could be changed to temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)



    model.save(f"sms_{data}_lstm")
    def gen_data(number):
        data_ = []
        for i in range(0, number):
            #start_index = random.randint(0, len(text) - maxlen - 1)
            generated = ''
            blah = []
            sentence = ''
            senten = text[:1]
            #senten = text[start_index: start_index + maxlen]
            for i in senten:
                sentence += i
            for i in range(1, random.randint(9, 146)):
                x_pred = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(senten):
                    x_pred[0, t, char_int[char]] = 1
                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds)
                next_char = int_char[next_index]
                sentence = sentence[1:] + next_char
                blah.append(next_char)
            data_.append(generated+ ''.join(blah))
        return data_



    var = pd.DataFrame(gen_data(num), columns=['text'])
    var['label'] = label
    print(var.head())
    print(var.tail())
    noise_ = str(noise)

    var.to_csv(f"sms_{data}_lstm_noise_{noise_[2:]}.csv", index=False)



def concatonate(noise):
    noise_ = str(noise)
    df1 = pd.read_csv(f"sms_ham_lstm_noise_{noise_[2:]}.csv")
    df2 = pd.read_csv(f"sms_spam_lstm_noise_{noise_[2:]}.csv")

    df = pd.concat([df1, df2])
    df.to_csv(f"sms_lstm_train_noise_{noise_[2:]}.csv", index=False)


def run(noise):
    lstm_gen_spam(1, 1040, noise)
    lstm_gen_ham(0, 2420, noise)
    concatonate(noise)

run(.1)
run(.15)
run(.25)
run(.33)
run(.5)