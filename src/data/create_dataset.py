import pandas as pd
from collections import Counter
import json
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import compute_class_weight
from src.config import data_path, project_root_path
from src.features.translate import oversample_with_back_translation


def calculating_class_weights(y_true, class_names):
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in range(len(class_names)):
        weights[i] = compute_class_weight('balanced', classes=[0, 1], y=y_true[class_names[i]])
    return weights


def ekman_dataset_categorical(RANDOM_SEED):
    # Load data and set labels
    data = pd.read_csv(data_path + "/ekman_emotions.csv", encoding="utf8", low_memory=False)
    # data = pd.read_csv(data_path + "/go_emo_ekman_emotions_w_neutral.csv", encoding="utf8", low_memory=False)
    data = data.drop(['sentiment'], axis=1)#[0:10000]

    X = data['content'].values
    y = data.drop(columns=['content'])

    stats = pd.DataFrame()
    stats['Complete'] = y.sum(axis=0, skipna=True)

    # load train, test and validation data]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED,
                                                        stratify=y['disgust'])
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=RANDOM_SEED,
                                                    stratify=y_test['disgust'])

    # remove no emotion from validation and test sets

    # X_val = pd.DataFrame(X_val, columns=["content"])
    # y_val.reset_index(drop=True, inplace=True)
    # val = pd.concat([X_val,y_val], axis=1)
    #
    # val.drop(val[val['no emotion'] == 1].index, inplace=True)
    # val = val.drop(columns=['no emotion'])
    # X_val = val['content']
    # y_val = val.drop(columns=['content'])
    #
    # X_test = pd.DataFrame(X_test, columns=["content"])
    # y_test.reset_index(drop=True, inplace=True)
    # test = pd.concat([X_test,y_test], axis=1)
    # test.drop(test[test['no emotion'] == 1].index, inplace=True)
    # test = test.drop(columns=['no emotion'])
    # X_test = test['content']
    # y_test = test.drop(columns=['content'])

    #######

    stats['Train'] = y_train.sum(axis=0)
    stats['Val'] = y_val.sum(axis=0)
    stats['Test'] = y_test.sum(axis=0)
    print(f'The statistics of the dataset are:\n {stats}')

    return X_train, y_train,  X_val, y_val, X_test, y_test


def ekman_dataset(RANDOM_SEED, oversampling=False):
    # Load data and set labels
    data = pd.read_csv(data_path + "/ekman_emotions.csv", encoding="utf8", low_memory=False)

    data = data[['sentiment', 'content']]
    print('Dataset shape %s' % Counter(data['sentiment']), flush=True)

    # Create statistics of dataset
    stats = pd.DataFrame()
    stats['count'] = data.groupby('sentiment').count()['content']
    stats['percent'] = 100 * stats['count'] / len(data)
    stats['sentiment'] = stats.index
    stats = stats[['count', 'percent', 'sentiment']]
    # stats.plot.pie(y='percent')
    stats = stats.reset_index(drop=True)
    print(stats)

    # Transform text labels to numbers
    # d = dict(zip(emotions, range(0, num_labels)))

    with open(r"C:\Users\georgiapant\PycharmProjects\GitHub\rebecca\data\interim\emotion_mappings.json") as file:
        file_data = file.read()
        mapping = json.loads(file_data)

    # mapping = json.loads(r"../../data/interim/emotion_mappings.json")
    data['label'] = data['sentiment'].map(mapping, na_action='ignore').astype('int64')
    data.drop(['sentiment'], inplace=True, axis=1)
    print(mapping)

    # weights = calculating_class_weights(data.drop(columns=['content']), np.unique(data['label']))

    if oversampling:

        train, test = train_test_split(data, test_size=0.2, random_state=RANDOM_SEED, stratify=data['label'].values)
        val, test = train_test_split(test, test_size=0.5, random_state=RANDOM_SEED, stratify=test['label'].values)

        oversampled_train = oversample_with_back_translation(train)
        X_train = oversampled_train['content'].values
        y_train = oversampled_train['label'].values.tolist()
        X_val = val['content'].values
        y_val = val['label'].values.tolist()
        X_test = test['content'].values
        y_test = test['label'].values.tolist()
    else:

        X = data['content'].values
        y = data['label'].values

        # load train, test and validation data]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=RANDOM_SEED,
                                                        stratify=y_test)

    weights = compute_class_weight('balanced', np.unique(y_train), y_train)
    # print(f'The weights of the classes are:\n {weights}')

    return X_train, y_train, X_val, y_val, X_test, y_test, weights


def isear_dataset(RANDOM_SEED, emotions):
    # Load data and set labels

    data = pd.read_csv(
        data_path + "\ISEAR_DATA.csv",
        encoding="utf8", low_memory=False)

    data = data[['SIT', 'Field1']]
    data.rename(columns={'SIT': 'Text', 'Field1': 'emotion'}, inplace=True)

    X = data['Text'].values
    y = data.drop(columns=['Text'])
    # y['emotion'] = (y.iloc[:,1:] ==1).idxmax(1)
    y = y['emotion']

    stats = pd.DataFrame()
    stats['count'] = y.value_counts()
    stats['percent'] = 100 * stats['count'] / len(y)
    stats['emotion'] = stats.index
    stats = stats[['count', 'percent', 'emotion']]
    # stats.plot.pie(y='percent')
    stats = stats.reset_index(drop=True)
    print(stats)
    # Transform text labels to numbers
    d = dict(zip(emotions, range(0, len(emotions))))
    print(d)
    y['label'] = y.map(d, na_action='ignore').astype('int64')
    y = y['label']

    # load train, test and validation data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=RANDOM_SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=RANDOM_SEED,
                                                    stratify=y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


def merged_dataset(RANDOM_SEED, emotions):
    # Load data and set labels
    data = pd.read_csv(
        data_path+ "\emotions_merged.csv",
        encoding="utf8", low_memory=False)

    data = data[['sentiment', 'content']]

    print('Original dataset shape %s' % Counter(data['sentiment']), flush=True)

    # Create statistics of dataset
    stats = pd.DataFrame()
    stats['count'] = data.groupby('sentiment').count()['content']
    stats['percent'] = 100 * stats['count'] / len(data)
    stats['sentiment'] = stats.index
    stats = stats[['count', 'percent', 'sentiment']]
    # stats.plot.pie(y='percent')
    stats = stats.reset_index(drop=True)
    print(stats)

    # Transform text labels to numbers
    d = dict(zip(emotions, range(0, len(emotions))))
    data['label'] = data['sentiment'].map(d, na_action='ignore').astype('int64')
    data.drop(['sentiment'], inplace=True, axis=1)

    X = data['content'].values
    y = data['label'].values

    # load train, test and validation data]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=RANDOM_SEED,
                                                    stratify=y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


def goemotions_with_weights_dataset(emotions):
    # Load data and set labels
    stats = pd.DataFrame()

    train = pd.read_csv(
        data_path+ "\GoEmotions\train.tsv",
        encoding="utf8", low_memory=False, sep='\t', names=['Text', 'sentiment_id', 'id'])

    validation = pd.read_csv(
        data_path+ "\GoEmotions\dev.tsv",
        encoding="utf8", low_memory=False, sep='\t', names=['Text', 'sentiment_id', 'id'])

    test = pd.read_csv(
        data_path+"\GoEmotions\test.tsv",
        encoding="utf8", low_memory=False, sep='\t', names=['Text', 'sentiment_id', 'id'])

    # Loading emotion labels for GoEmotions taxonomy
    with open(data_path+"\GoEmotions\emotions.txt", "r") as file:
        GE_taxonomy = file.read().split("\n")

    size_train = train.shape[0]
    size_val = validation.shape[0]
    size_test = test.shape[0]

    df_all = pd.concat([train, validation, test], axis=0).reset_index(drop=True).drop(['id'], axis=1)
    df_all['sentiment_id'] = df_all['sentiment_id'].apply(lambda x: x.split(','))

    def idx2class(idx_list):
        arr = []
        for i in idx_list:
            arr.append(GE_taxonomy[int(i)])
        return arr

    df_all['sentiment'] = df_all['sentiment_id'].apply(idx2class)

    # OneHot encoding for multi-label classification
    for emo in GE_taxonomy:
        df_all[emo] = np.zeros((len(df_all), 1))
        df_all[emo] = df_all['sentiment'].apply(lambda x: 1 if emo in x else 0)

    df_all = df_all.drop(['sentiment_id', 'sentiment'], axis=1)
    print(df_all)
    weights = calculating_class_weights(df_all.drop(columns=['Text']), emotions)
    print(f'The weights of the classes are:\n {weights}')

    X_train = df_all.iloc[:size_train, :]['Text']
    X_val = df_all.iloc[size_train:size_train + size_val, :]['Text']
    X_test = df_all.iloc[size_train + size_val:size_train + size_val + size_test, :]['Text']

    y_train = df_all.iloc[:size_train, :].drop(columns=['Text'])
    y_val = df_all.iloc[size_train:size_train + size_val, :].drop(columns=['Text'])
    y_test = df_all.iloc[size_train + size_val:size_train + size_val + size_test, :].drop(columns=['Text'])

    stats['Train'] = y_train.sum(axis=0)
    stats['Val'] = y_val.sum(axis=0)
    stats['Test'] = y_test.sum(axis=0)
    print(f'The statistics of the dataset are:\n {stats}')

    return X_train, y_train, X_val, y_val, X_test, y_test, weights


def ec_with_weights_dataset(emotions):
    # Load data and set labels
    # dataset from the SemEval-2018 Task1
    data_ec_val = pd.read_csv(
        data_path+"\2018-E-c\2018-E-c-En-dev.txt",
        sep='	', encoding="utf-8", header=0)
    data_ec_test = pd.read_csv(
        data_path+"\2018-E-c\2018-E-c-En-test-gold.txt",
        sep='	', encoding="utf-8", header=0)
    data_ec_train = pd.read_csv(
        data_path+"\2018-E-c\2018-E-c-En-train.txt",
        sep='	', encoding="utf-8", header=0)

    # data = pd.concat([data_ec_train, data_ec_val], ignore_index=True)
    #
    # data.rename(columns={"Tweet": "Text"}, inplace=True)

    data_ec_val.rename(columns={"Tweet": "Text"}, inplace=True)
    data_ec_test.rename(columns={"Tweet": "Text"}, inplace=True)
    data_ec_train.rename(columns={"Tweet": "Text"}, inplace=True)

    X_train = data_ec_train['Text'].values
    y_train = data_ec_train.drop(columns=['Text', 'ID'])

    X_val = data_ec_val['Text'].values
    y_val = data_ec_val.drop(columns=['Text', 'ID'])

    X_test = data_ec_test['Text'].values
    y_test = data_ec_test.drop(columns=['Text', 'ID'])

    stats = pd.DataFrame()

    weights = calculating_class_weights(y_train, emotions)
    print(f'The weights of the classes are:\n {weights}')

    stats['Train'] = y_train.sum(axis=0)
    stats['Val'] = y_val.sum(axis=0)
    stats['Test'] = y_test.sum(axis=0)
    print(f'The statistics of the dataset are:\n {stats}')

    return X_train, y_train, X_val, y_val, X_test, y_test, weights

def main():

    X_train, y_train, X_val, y_val, X_test, y_test = ekman_dataset(42, oversampling=True)

if __name__ == '__main__':
    main()