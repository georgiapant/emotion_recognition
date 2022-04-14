import pandas as pd
from collections import Counter

"""
Merge the three different emotions datasets to one. Rename some of the classes as well as drop others 
Final classes distribution: 
   count    percent  sentiment
0   4139   8.457467      anger
1  13383  27.346288  happiness
2   5508  11.254827       love
3  11657  23.819449    sadness
4   2976   6.081040   surprise
5  11276  23.040929      worry


Datasets:
- twitter_text_emotion (maybe bad quality)
- kaggle_emotion_nlp
- SemEval - 2018-Ec
- 

"""

#First dataset - twitter text emotion
dataset_emo = pd.read_csv(
        r"C:\Users\georgiapant\PycharmProjects\REBECCA\Datasets\Emotion\twitter_text_emotion.csv",
        encoding="utf8")
dataset_emo= dataset_emo[["content","sentiment"]]
dataset_emo = dataset_emo.loc[dataset_emo['sentiment'].isin(['love','worry','happiness','anger','sadness','surprise'])]
# print(Counter(dataset_emo['sentiment']))

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
data_kaggle.columns=['content', 'sentiment']
data_kaggle.to_csv(r"C:\Users\georgiapant\PycharmProjects\REBECCA\Datasets\Emotion\data_kaggle.csv", index=False)
merge = {'joy': 'happiness', 'sadness': 'sadness', 'anger': 'anger', 'fear': 'worry', 'love': 'love', 'surprise':'surprise'}
data_kaggle['sentiment'] = data_kaggle['sentiment'].map(merge, na_action='ignore')
# print(Counter(data_kaggle['sentiment']))

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
data_ec.to_csv(r"C:\Users\georgiapant\PycharmProjects\REBECCA\Datasets\Emotion\EC.csv", index=False)

data_ec.rename(columns={"fear":"worry", "joy":"happiness", "Tweet":"content"}, inplace=True)
data_ec = data_ec[["content", "happiness", "sadness", "anger", "worry", "love", "surprise"]]
data_ec["sum"] = data_ec["happiness"]+data_ec["sadness"]+data_ec["anger"]+data_ec["worry"]+data_ec["love"]+data_ec["surprise"]
data_ec = data_ec[data_ec['sum'] == 1]
data_ec.drop(['sum'], inplace=True, axis=1)
data_ec['sentiment'] = (data_ec.iloc[:, 1:] == 1).idxmax(1)
data_ec = data_ec[["content","sentiment"]]
# print(Counter(data_ec['sentiment']))


dataset = pd.concat([data_ec, dataset_emo, data_kaggle], ignore_index=True)
# print(Counter(dataset['sentiment']))
# dataset.to_csv(r"C:\Users\georgiapant\PycharmProjects\REBECCA\Datasets\Emotion\emotions_merged.csv", index=False)

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