# Create dataset from ISEAR data

import pandas as pd
from collections import Counter

dataset = pd.read_csv(
        r"C:\Users\georgiapant\PycharmProjects\REBECCA\Datasets\Emotion\ISEAR_DATA.csv",
        encoding="utf8")
dataset= dataset[["SIT","Field1"]]
dataset.rename(columns={"SIT":"content", "Field1":"sentiment"}, inplace=True)
print(Counter(dataset['sentiment']))