import pandas as pd
import numpy as np
import urllib.request
import zipfile
import os

# text preprocessing
from nltk.tokenize import word_tokenize
import re
import pickle
import gc
from collections import Counter

# plots and metrics
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

# preparing input to our model
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

# keras layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, \
    GRU, Bidirectional, Flatten, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import nltk

# nltk.download('punkt')

gc.collect()
RANDOM_SEED = 42

# Number of dimensions for word embedding
EMBED_NUM_DIMS = 300 # TODO depending on the word embedding i use (w2v-wiki --> 300, others 100)

# Max input length (max number of words)
MAX_SEQ_LEN = 100

BATCH_SIZE = 32
EPOCHS = 20

# Class names and number
class_names = ['sadness', 'worry', 'surprise', 'love', 'happiness', 'anger']
num_classes = len(class_names)


def dataset():
    data = pd.read_csv(r"C:\Users\georgiapant\PycharmProjects\REBECCA\Datasets\Emotion\emotions_merged.csv",
                       encoding="utf8", low_memory=False)
    data = data[['sentiment', 'content']]
    print(data.sentiment.value_counts())

    d = dict(zip(class_names, range(len(class_names))))
    print(d)
    data['label'] = data['sentiment'].map(d, na_action='ignore').astype('int64')
    data.drop(['sentiment'], inplace=True, axis=1)
    data = data.reset_index(drop=True)

    X = data['content'].values
    y = data['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=RANDOM_SEED,
                                                    stratify=y_test)

    print('Size of training set: %s' % (len(X_train)))
    print('Size of validation set: %s' % (len(X_dev)))
    print('Size of test set: %s' % (len(X_test)))

    print('Train labels distribution: %s' % (Counter(y_train)))
    print('Train labels distribution: %s' % (Counter(y_dev)))
    print('Train labels distribution: %s' % (Counter(y_test)))

    return X_train, X_dev, X_test, y_train, y_dev, y_test


def clean_text(data):  # TODO: Maybe update
    # remove hashtags and @usernames
    # data = re.sub(r"(#[\d\w\.]+)", '', data)
    data = re.sub(r"(@[\d\w\.]+)", '', data)
    data = re.sub(r'&amp;', '&', data)
    data = re.sub(r'\s+', ' ', data).strip()
    data = re.sub(r"https?://[A-Za-z0-9./]+", ' ', data)
    data = re.sub(r"[^a-zA-z.!?'0-9]", ' ', data)
    data = re.sub('\t', ' ', data)
    data = re.sub(r" +", ' ', data)
    data = re.sub(r"won\'t", "will not", data)
    data = re.sub(r"can\'t", "can not", data)
    data = re.sub(r"n\'t", " not", data)
    data = re.sub(r"\'re", " are", data)
    data = re.sub(r"\'s", " is", data)
    data = re.sub(r"\'d", " would", data)
    data = re.sub(r"\'ll", " will", data)
    data = re.sub(r"\'t", " not", data)
    data = re.sub(r"\'ve", " have", data)
    data = re.sub(r"\'m", " am", data)

    # remove stock market tickers like $GE
    data = re.sub(r'\$\w*', '', data)
    # remove old style retweet text "RT"
    data = re.sub(r'^RT[\s]+', '', data)
    # remove hyperlinks
    data = re.sub(r'https?:\/\/.*[\r\n]*', '', data)
    # remove hashtags
    # only removing the hash # sign from the word
    data = re.sub(r'#', '', data)
    data = str(re.sub("\S*\d\S*", "", data).strip())

    # tokenization using nltk
    data = word_tokenize(data)

    return data


def create_embedding_matrix(X_train, X_dev, embed_type='w2v_wiki'):
    # texts = [' '.join(clean_text(text)) for text in data.content]

    texts_train = [' '.join(clean_text(text)) for text in X_train]
    texts_dev = [' '.join(clean_text(text)) for text in X_dev]

    text = np.concatenate((texts_train, texts_dev), axis=0)
    text = np.array(text)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)

    sequence_train = tokenizer.texts_to_sequences(texts_train)
    X_train_pad = pad_sequences(sequence_train, maxlen=MAX_SEQ_LEN)

    sequence_dev = tokenizer.texts_to_sequences(texts_dev)
    X_dev_pad = pad_sequences(sequence_dev, maxlen=MAX_SEQ_LEN)

    index_of_words = tokenizer.word_index
    print('Number of unique words: {}'.format(len(index_of_words)))

    # get embeddings from file
    if embed_type == 'w2v_wiki':
        fpath = 'embeddings/wiki-news-300d-1M.vec'

        if not os.path.isfile(fpath):
            print('Downloading word vectors...')
            urllib.request.urlretrieve(
                'https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip',
                'wiki-news-300d-1M.vec.zip')
            print('Unzipping...')
            with zipfile.ZipFile('wiki-news-300d-1M.vec.zip', 'r') as zip_ref:
                zip_ref.extractall('embeddings')
            print('done.')

            os.remove('wiki-news-300d-1M.vec.zip')

    elif embed_type == 'w2v':
        fpath = r"C:\Users\georgiapant\PycharmProjects\REBECCA\Resources\embedding_word2vec.txt"

    elif embed_type == 'glove':
        fpath = r"C:\Users\georgiapant\PycharmProjects\REBECCA\Resources\glove.6B\glove.6B.100d.txt"
            # pd.read_csv(r"C:\Users\georgiapant\PycharmProjects\REBECCA\Resources\glove.6B\glove.6B.100d.txt")

    # create embedding matrix

    # vacab size is number of unique words + reserved 0 index for padding
    vocab_size = len(index_of_words) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, EMBED_NUM_DIMS))

    with open(fpath, encoding="utf-8") as f:

        for line in f:
            word, *vector = line.split()
            if word in index_of_words:
                idx = index_of_words[word]
                embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:EMBED_NUM_DIMS]


    # # Inspect unseen words
    new_words = 0

    for word in index_of_words:
        entry = embedding_matrix[index_of_words[word]]
        if all(v == 0 for v in entry):
            new_words = new_words + 1

    print('Words found in wiki vocab: ' + str(len(index_of_words) - new_words))
    print('New words found: ' + str(new_words))

    # save tokenizer (to be used for test set)
    with open('tokenizer', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return X_train_pad, X_dev_pad, vocab_size, embedding_matrix


def cnn(vocab_size, embedding_matrix):
    embedding_layer = Embedding(vocab_size, EMBED_NUM_DIMS, input_length=MAX_SEQ_LEN, weights=[embedding_matrix],
                                trainable=False)

    # Convolution
    kernel_size = 3
    filters = 256
    model = Sequential()
    model.add(embedding_layer)
    model.add(Conv1D(filters, kernel_size, activation='relu'))
    model.add(Conv1D(filters=256, kernel_size=4, padding='same', activation='relu'))
    model.add(Conv1D(filters=256, kernel_size=5, padding='same', activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def lstm(vocab_size, embedding_matrix):
    gru_output_size = 128
    bidirectional = True

    embedding_layer = Embedding(vocab_size, EMBED_NUM_DIMS, input_length=MAX_SEQ_LEN, weights=[embedding_matrix],
                                trainable=False)

    # Embedding Layer, LSTM or biLSTM, Dense, softmax
    model = Sequential()
    model.add(embedding_layer)

    if bidirectional:
        # model.add(Bidirectional(GRU(units=gru_output_size, dropout=0.2, recurrent_dropout=0.2)))
        model.add(Bidirectional(LSTM(64, return_sequences=False)))
    else:
        # model.add(GRU(units=gru_output_size, dropout=0.2, recurrent_dropout=0.2))
        model.add((LSTM(64, return_sequences=False)))

    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model

def training(X_train, y_train, X_dev, y_dev):
    X_train_pad, X_dev_pad, vocab_size, embedding_matrix = create_embedding_matrix(X_train, X_dev,
                                                                                   embed_type='w2v_wiki')

    y_train_categorical = to_categorical(y_train)
    y_dev_categorical = to_categorical(y_dev)

    # model = cnn(vocab_size, embedding_matrix)
    model = lstm(vocab_size, embedding_matrix)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    # save the best model
    mc = ModelCheckpoint('model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    history = model.fit(np.array(X_train_pad), np.array(y_train_categorical), epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_data=(np.array(X_dev_pad), np.array(y_dev_categorical)), callbacks=[es, mc])

    # Accuracy plot
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # Loss plot
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def evaluation(y_test, predictions):
    acc = accuracy_score(y_test, predictions)
    fscore_micro = f1_score(y_test, predictions, average='micro')
    fscore_macro = f1_score(y_test, predictions, average='macro')
    # fscore = f1_score(y_test, predictions, average=None)
    prec = precision_score(y_test, predictions, average='macro')
    rec = recall_score(y_test, predictions, average='macro')
    # auc_sc = roc_auc_score(y_test, probas, average='macro')
    print('\nBest accuracy: %f F1-score micro: %f F1-score macro: %f Precision: %f Recall: %f\n' % (
        acc, fscore_micro, fscore_macro, prec, rec))
    # print('F1-score[0]: %f' % fscore[0])
    # print('\nF1-score[1]: %f\n' % fscore[1])
    print("\nClassification report:\n", classification_report(y_test, predictions))


def prediction(X_test, y_test):
    # load tokenizer
    with open('tokenizer', 'rb') as handle:
        tok = pickle.load(handle)

    # load pretrained model
    model = load_model('model.h5', compile=False)

    # preprocessing
    X_test = [' '.join(clean_text(text)) for text in X_test]
    # X_test = clean_text(X_test)

    # tokenization
    sequences = tok.texts_to_sequences(X_test)
    X_test_pad = pad_sequences(sequences, maxlen=EMBED_NUM_DIMS)

    # prediction
    probabilities = model.predict(np.array(X_test_pad))
    # if the probability is over 0.5, it is considered as class 1, else as class 0

    predictions = np.argmax(probabilities, axis=1)
    evaluation(y_test, predictions)



def main():
    X_train, X_dev, X_test, y_train, y_dev, y_test = dataset()
    training(X_train, y_train, X_dev, y_dev)
    prediction(X_test, y_test)


if __name__ == '__main__':
    main()
