import tweepy
import json
import pandas as pd
import math
from io import StringIO

def lookup_tweets(tweet_IDs, api):
    full_tweets = []
    tweet_count = len(tweet_IDs)
    try:
        for i in range(math.floor(tweet_count / 100) + 1):
            # Catch the last group if it is less than 100 tweets
            end_loc = min((i + 1) * 100, tweet_count)
            full_tweets.extend(
                api.statuses_lookup(id_=tweet_IDs[i * 100:end_loc])
            )
            print(i)
        return full_tweets
    except tweepy.TweepError:
        print('Something went wrong, quitting...')

consumer_key = 'j6jmABpLvO11iXyD3gxxG7Ovl'
consumer_secret = 'bxSN6WvU9OIhdsc5hcfUN5w3TgOYjOWf2odsNSeLK5ioyrDYAt'
access_token = '1895048976-3SLFBk6T0OuHa4aSVOZ5QW0s8G5KkThFvb9MhtO'
access_token_secret = 'lOLu2fDgMIrg66DilZMTxrZ4hZlANBEdHFkyViaenIvcY'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

df = pd.read_csv(r"C:\Users\georgiapant\PycharmProjects\REBECCA\Datasets\Sentiment\data_twitter_sentiment\Spanish_Twitter_sentiment.csv", encoding = 'utf-8')

good_tweet_ids = [i for i in df.TweetID] #tweet ids to look up
results = lookup_tweets(good_tweet_ids, api) #apply function

#Wrangle the data into one dataframe
temp = json.dumps([status._json for status in results]) #create JSON
new_df = pd.read_json(StringIO(temp), orient='records')
new_df.to_csv(path_or_buf=r'C:\Users\georgiapant\PycharmProjects\REBECCA\Datasets\Sentiment\data_twitter_sentiment\Spanish_Twitter_sentiment_raw.csv', index=True, header=True)
full = pd.merge(df, new_df, left_on='TweetID', right_on='id', how='left').drop('id', axis=1)
full.to_csv(path_or_buf=r'C:\Users\georgiapant\PycharmProjects\REBECCA\Datasets\Sentiment\data_twitter_sentiment\Spanish_Twitter_sentiment_all.csv', index=True, header=True)
