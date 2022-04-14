import pandas as pd
import numpy as np

# text preprocessing
import nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

import re

# plots and metrics
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report
from collections import Counter

# feature extraction / vectorization
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# save and load a file
import pickle

# gc.collect()
RANDOM_SEED = 42
BATCH_SIZE = 32
# EPOCHS = 20

# Class names and number
class_names = ['sadness', 'worry', 'surprise', 'love', 'happiness', 'anger']
num_classes = len(class_names)


def datasets():
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y)

    print('Size of training set: %s' % (len(X_train)))
    print('Size of test set: %s' % (len(X_test)))

    print('Train labels distribution: %s' % (Counter(y_train)))
    print('Train labels distribution: %s' % (Counter(y_test)))

    return data, X_train, X_test, y_train, y_test


def preprocess_and_tokenize(data):
    # remove html markup
    data = re.sub("(<.*?>)", "", data)

    # remove urls
    data = re.sub(r'http\S+', '', data)

    # remove hashtags and @names
    data = re.sub(r"(#[\d\w\.]+)", '', data)
    data = re.sub(r"(@[\d\w\.]+)", '', data)

    # remove punctuation and non-ascii digits
    data = re.sub("(\\W|\\d)", " ", data)

    # remove whitespace
    data = data.strip()

    # tokenization with nltk
    data = word_tokenize(data)

    # stemming with nltk
    porter = PorterStemmer()
    stem_data = [porter.stem(word) for word in data]

    return stem_data


def tf_idf(data, X_train, X_test):
    # TFIDF, unigrams and bigrams
    vect = TfidfVectorizer(tokenizer=preprocess_and_tokenize, sublinear_tf=True, norm='l2', ngram_range=(1, 2))

    # fit on our complete corpus
    vect.fit_transform(data.content)

    # transform testing and training datasets to vectors
    X_train_vect = vect.transform(X_train)
    X_test_vect = vect.transform(X_test)

    return X_train_vect, X_test_vect, vect


# Naive Bayes
def nb(X_train_vect, y_train):

    nb = MultinomialNB()
    nb.fit(X_train_vect, y_train)

    return nb


# Random Forest
def rf(X_train_vect, y_train):

    rf = RandomForestClassifier(n_estimators=50)
    rf.fit(X_train_vect, y_train)

    return rf


# Logistic Regression
def lr(X_train_vect, y_train):

    log = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=400)
    log.fit(X_train_vect, y_train)

    return log


# Linear Support Vector
def lsvc(X_train_vect, y_train):

    lsvc = LinearSVC(tol=1e-05)
    lsvc.fit(X_train_vect, y_train)

    return lsvc

def evaluation(y_test, predictions):
    acc = accuracy_score(y_test, predictions)
    fscore_micro = f1_score(y_test, predictions, average='micro')
    fscore_macro = f1_score(y_test, predictions, average='macro')
    rec = recall_score(y_test, predictions, average='macro')
    prec = precision_score(y_test, predictions, average='macro')
    print('\nBest accuracy: %f \nF1-score micro: %f \nF1-score macro: %f \nRecall: %f \nPrecision: %f\n' % (
        acc, fscore_micro, fscore_macro, rec, prec))
    print("\nClassification report:\n", classification_report(y_test, predictions))


def main():
    data, X_train, X_test, y_train, y_test = datasets()
    X_train_vect, X_test_vect, vect = tf_idf(data, X_train, X_test)

    # model = nb(X_train_vect, y_train)
    # model = lr(X_train_vect, y_train)
    model = rf(X_train_vect, y_train)
    # model = lsvc(X_train_vect, y_train)

    predictions = model.predict(X_test_vect)
    evaluation(y_test, predictions)

    # Save model
    save_model = Pipeline([('tfidf', vect), ('clf', model)])
    filename = r"C:\Users\georgiapant\PycharmProjects\REBECCA\Models\tfidf_lsvc.sav"
    pickle.dump(save_model, open(filename, 'wb'))

if __name__ == '__main__':
    main()