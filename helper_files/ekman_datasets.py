import pandas as pd
import numpy as np
from collections import Counter

"""
Merge the three different emotions datasets to one. Rename some of the classes as well as drop others to have only EKMAN emotions
Final classes distribution: 
   count    percent  sentiment
0   4139   8.457467      anger
1  13383  27.346288  happiness
2   5508  11.254827       love
3  11657  23.819449    sadness
4   2976   6.081040   surprise
5  11276  23.040929      worry

"""

# Second dataset from kaggle
data_kaggle_train = pd.read_csv(
        r"C:\Users\georgiapant\PycharmProjects\REBECCA\Datasets\Emotion\kaggle_emotion_nlp\train.txt",
        encoding="utf8", low_memory=False, delimiter=";", header=None)
data_kaggle_val = pd.read_csv(
        r"C:\Users\georgiapant\PycharmProjects\REBECCA\Datasets\Emotion\kaggle_emotion_nlp\test.txt",
        encoding="utf8", low_memory=False, delimiter=";", header=None)
data_kaggle_test = pd.read_csv(
        r"C:\Users\georgiapant\PycharmProjects\REBECCA\Datasets\Emotion\kaggle_emotion_nlp\val.txt",
        encoding="utf8", low_memory=False, delimiter=";", header=None)
data_kaggle = pd.concat([data_kaggle_train,data_kaggle_test, data_kaggle_val], ignore_index=True)
mapping = {"love":"joy", "joy":"joy", "fear":"fear","anger":"anger","sadness":"sadness","surprise":"surprise"}
data_kaggle[1] = data_kaggle[1].map(mapping, na_action='ignore')#.astype('int64')
data_kaggle.columns=['content', 'sentiment']
# data_kaggle.to_csv(r"C:\Users\georgiapant\PycharmProjects\REBECCA\Datasets\Emotion\data_kaggle.csv", index=False)

print(Counter(data_kaggle['sentiment']))

# dataset from the SemEval-2018 Task1
data_ec_val = pd.read_csv(
        r"C:\Users\georgiapant\PycharmProjects\REBECCA\Datasets\Emotion\2018-E-c\2018-E-c-En-dev.txt",
        sep='	', encoding="utf8", header=0)
# data_ec_test = pd.read_csv(
#         r"C:\Users\georgiapant\PycharmProjects\REBECCA\Datasets\Emotion\2018-E-c\2018-E-c-En-test.txt",
#         sep='	', encoding="utf8", header=0)
data_ec_train = pd.read_csv(
        r"C:\Users\georgiapant\PycharmProjects\REBECCA\Datasets\Emotion\2018-E-c\2018-E-c-En-train.txt",
        sep='	', encoding="utf8", header=0)

data_ec = pd.concat([data_ec_train,data_ec_val], ignore_index=True)
# data_ec.to_csv(r"C:\Users\georgiapant\PycharmProjects\REBECCA\Datasets\Emotion\EC.csv", index=False)

data_ec.rename(columns={"pessimism":"fear", "anticipation":"joy", "optimism":"joy", "trust":"joy", "Tweet":"content"}, inplace=True)
# print(data_ec)
# data_ec_labels = data_ec[["content", "joy", "sadness", "anger", "fear", "disgust", "surprise"]]
# def sjoin(x): return x[x.max()].astype(int)
data_ec_labels = data_ec[["joy", "sadness", "anger", "fear", "disgust", "surprise"]]
# data_ec_labels = data_ec_labels.groupby(level=0, axis=1).apply(lambda x: x.apply(sjoin, axis=1))
# data_ec = data_ec.groupby(data_ec.columns, axis=1).sum()
data_ec_labels = data_ec_labels.groupby(data_ec_labels.columns, axis=1).max()
data_ec = pd.concat([data_ec_labels, data_ec['content']], axis=1)

# data_ec = data_ec[["content", "happiness", "sadness", "anger", "worry", "love", "surprise"]]
data_ec["sum"] = data_ec["joy"]+data_ec["sadness"]+data_ec["anger"]+data_ec["fear"]+data_ec["disgust"]+data_ec["surprise"]
# print(data_ec_labels)
data_ec = data_ec[data_ec['sum'] == 1]
data_ec.drop(['sum'], inplace=True, axis=1)
data_ec['sentiment'] = (data_ec.iloc[:, 1:] == 1).idxmax(1)
# print(data_ec)
print(Counter(data_ec['sentiment']))


# GoEmotions
goemotions = pd.read_csv(
        r"C:\Users\georgiapant\PycharmProjects\REBECCA\Datasets\Emotion\GoEmotions.csv",
        encoding="utf8", low_memory=False)
#
goemotions.drop(goemotions[goemotions['neutral'] == 1].index, inplace=True)
goemotions = goemotions.drop(columns=['neutral'])
#
# data.drop(data[data['relief'] == 1].index, inplace=True)
# data = data.drop(columns=['relief'])
#
# data.drop(data[data['nervousness'] == 1].index, inplace=True)
# data = data.drop(columns=['nervousness'])
#
# data.drop(data[data['pride'] == 1].index, inplace=True)
# data = data.drop(columns=['pride'])
#
# data.drop(data[data['grief'] == 1].index, inplace=True)
# data = data.drop(columns=['grief'])
goemotions.rename(columns={"Text":"content"}, inplace=True)
goemotions["sum"] = goemotions.sum(axis=1)
goemotions = goemotions[goemotions['sum'] == 1]
goemotions.drop(['sum'], inplace=True, axis=1)
goemotions['sentiment'] = (goemotions.iloc[:, 1:] == 1).idxmax(1)
goemotions = goemotions[['content', 'sentiment']]

mapping = {"annoyance":"anger", "disapproval":"anger", "anger":"anger", "disgust":"disgust", "joy":"joy","amusement":"joy","approval":"joy",
           "excitement":"joy", "gratitude":"joy",  "love":"joy", "optimism":"joy", "relief":"joy", "pride":"joy", "admiration":"joy", "desire":"joy", "caring":"joy",
           "sadness":"sadness", "disappointment":"sadness", "embarrassment":"sadness", "grief":"sadness",  "remorse":"sadness",
           "surprise":"surprise", "realization":"surprise", "confusion":"surprise", "curiosity":"surprise", "fear":"fear", "nervousness":"fear"}

goemotions['sentiment'] = goemotions['sentiment'].map(mapping, na_action='ignore')#.astype('int64')

dataset = pd.concat([data_ec, goemotions, data_kaggle], ignore_index=True)
print(Counter(dataset['sentiment']))
dataset.to_csv(r"C:\Users\georgiapant\PycharmProjects\REBECCA\Datasets\Emotion\ekman_emotions.csv", index=False)

stats = pd.DataFrame()
stats['count'] = dataset.groupby('sentiment').count()['content']
stats['percent'] = 100 * stats['count'] / len(dataset)
stats['sentiment'] = stats.index
stats = stats[['count', 'percent', 'sentiment']]
# stats.plot.pie(y='percent')
stats = stats.reset_index(drop=True)
print(stats)

# data = data[['HandLabel', 'text']]
# # data['labels'] = data['HandLabel']
#
# sentiments = ['Positive', 'Negative', 'Neutral']
# d = dict(zip(sentiments, range(0, 3)))
# data['labels'] = data['HandLabel'].map(d, na_action='ignore').astype('int64')