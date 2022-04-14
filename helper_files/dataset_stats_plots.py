import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

dataset_emo = pd.read_csv(
        r"C:\Users\georgiapant\PycharmProjects\REBECCA\Datasets\Emotion\twitter_text_emotion.csv",
        encoding="utf8")
dataset_emo= dataset_emo[["content","sentiment"]]
print(Counter(dataset_emo['sentiment']))



stats = pd.DataFrame()
stats['count'] = dataset_emo.groupby('sentiment').count()['content']
stats['percent'] = 100 * stats['count'] / len(dataset_emo)
stats['sentiment'] = stats.index
stats = stats[['count', 'percent', 'sentiment']]

stats.set_index('sentiment')[['percent']].T.plot(kind='bar', stacked=True)
stats['total'] = sum(stats['count'])

labels = [f'{l}, {s:0.4f}% ({j}/{k})' for l, s, j, k in zip(stats['sentiment'], stats['percent'], stats['count'], stats['total'])]
plt.legend(bbox_to_anchor=(0.85, 1), loc='upper left', labels=labels)

# plt.pie(stats['count'], labels=stats['sentiment'],startangle=90)
plt.show()
stats = stats.reset_index(drop=True)
print(stats)