import sys
import time

sys.path.append("C:/Users/georgiapant/PycharmProjects/GitHub/rebecca")
from src.data.get_data_from_API import Facebook, Web, Youtube
from datetime import datetime
from dateutil.relativedelta import relativedelta
from models.predict.multiclass_emotion_predictor_NRC_VAD import MulticlassEmotionIdentification
import pandas as pd
from pandas.core.common import SettingWithCopyWarning
import json
from pymongo import MongoClient
import schedule
import warnings
from src.features.translate import translate
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


client = MongoClient('localhost', 27017)
db = client.emotion_index_database
emotions_collection = db.emotions  # collection including all text with the predicted emotion
activities_all_collection = db.emotions
queries_comments_reactions_collection = db.emotions
fb_websites_visited_collection = db.emotions
urls_collection = db.emotions

web = Web()
fb = Facebook()
yt = Youtube()
classification = MulticlassEmotionIdentification()

# Dummy parameters and input #################

# user_id = "d2n611d6-l5v4-9371-b808-4d1632f29422" # TODO: when I have access to the user_id list update this
# user_id = "f4b135b3-c5f6-4261-b808-4d1632f29422"
# dummy_queries = [('6284f95d64ded14744af29f4΄,'1653022155','how to improve the mood of a friend'),
#            ('6284f95d64ded14744af29f5','1652870955','I am sad on funerals'),
#            ('6284f95d64ded14744af29f6',yesterday,'This is awful')]
# dummy_queries = pd.DataFrame(queries, columns=['session_id', 'timestamp', 'query'])

a_year_ago = str(int((datetime.today() - relativedelta(years=1)).timestamp()))
since = a_year_ago
users = ["f4b135b3-c5f6-4261-b808-4d1632f29422", "d2n611d6-l5v4-9371-b808-4d1632f29422"]

##############################################

# yesterday = str(int((datetime.today() - timedelta(days=1)).timestamp()))
# since = yesterday
today = str(int(datetime.now().timestamp()))
until = today
page = "0"
nperpage = "100"


def job():
    # Initialise dataframes, to be used as schema for the db collections
    queries_comments_reactions = pd.DataFrame(
        columns=['_id', 'user_id', 'session_id', 'timestamp', 'activity_type', 'on', 'content'])
    websites_visits = pd.DataFrame(
        columns=['_id', 'session_id', 'timestamp', 'type', 'description', 'name', 'user_id', 'activity_type'])
    web_urls = pd.DataFrame(
        columns=['_id', 'tab', 'session_id', 'timestamp', 'title', 'url', 'user_id', 'activity_type'])
    activities_all = pd.DataFrame(
        columns=['session_id', 'timestamp', 'user_id', 'activity_type'])

    for user_id in users:
        web_queries, _ = web.get_queries(user_id, since, until, page, nperpage)
        web_queries['user_id'] = user_id
        web_queries['activity_type'] = 'web_query'
        web_queries['on'] = ''
        web_queries.rename(columns={'query': 'content'}, inplace=True)

        fb_queries, _ = fb.get_queries(user_id, since, until, page, nperpage)
        fb_queries['user_id'] = user_id
        fb_queries['activity_type'] = 'fb_query'
        fb_queries['on'] = ''
        fb_queries.rename(columns={'query': 'content'}, inplace=True)

        yt_queries, _ = yt.get_queries(user_id, since, until, page, nperpage)
        yt_queries['user_id'] = user_id
        yt_queries['activity_type'] = 'yt_query'
        yt_queries['on'] = ''
        yt_queries.rename(columns={'query': 'content'}, inplace=True)

        fb_comments, _ = fb.get_comments(user_id, since, until, page, nperpage)
        fb_comments['user_id'] = user_id
        fb_comments['activity_type'] = 'fb_comment'
        fb_comments.rename(columns={'comment': 'content', 'post': 'on'}, inplace=True)

        yt_comments, _ = yt.get_comments(user_id, since, until, page, nperpage)
        yt_comments['user_id'] = user_id
        yt_comments['activity_type'] = 'yt_comment'
        yt_comments.rename(columns={'comment': 'content', 'video': 'on'}, inplace=True)

        fb_reactions, _ = fb.get_reactions(user_id, since, until, page, nperpage)
        fb_reactions['user_id'] = user_id
        fb_reactions['activity_type'] = 'fb_reaction'
        fb_reactions.rename(columns={'reaction': 'content', 'post': 'on'}, inplace=True)

        yt_reactions, _ = yt.get_reactions(user_id, since, until, page, nperpage)
        yt_reactions['user_id'] = user_id
        yt_reactions['activity_type'] = 'yt_reaction'
        yt_reactions.rename(columns={'reaction': 'content', 'title': 'on'}, inplace=True)

        queries_comments_reactions = pd.concat([queries_comments_reactions, web_queries, fb_reactions, fb_comments,
                                                fb_queries, yt_reactions, yt_comments, yt_queries], axis=0)

        websites_visits_user, count = fb.get_visits(user_id, since, until, page, nperpage)
        websites_visits_user['user_id'] = user_id
        websites_visits_user['activity_type'] = 'website_visit'
        # websites_visits.rename(columns={'name': 'content'}, inplace=True)
        websites_visits = pd.concat([websites_visits, websites_visits_user], axis=0)

        web_urls_user, _ = web.get_urls(user_id, since, until, page, nperpage)
        web_urls_user['user_id'] = user_id
        web_urls_user['activity_type'] = 'web_urls'
        web_urls = pd.concat([web_urls, web_urls_user], axis=0)

    # Create all activities timeline
    queries_comments_reactions_limited_info = queries_comments_reactions[
        ['session_id', 'timestamp', 'user_id', 'activity_type']]
    websites_visits_limited_info = websites_visits[['session_id', 'timestamp', 'user_id', 'activity_type']]
    web_urls_limited_info = web_urls[['session_id', 'timestamp', 'user_id', 'activity_type']]
    activities_all = pd.concat(
        [activities_all, queries_comments_reactions_limited_info, websites_visits_limited_info, web_urls_limited_info],
        axis=0)

    # Add to database
    activities_all.reset_index(inplace=True)
    activities_all.drop(columns=['index'], inplace=True)
    activities_all_for_collection = json.loads(activities_all.T.to_json()).values()
    db.activities_all_collection.insert_many(activities_all_for_collection)

    queries_comments_reactions.reset_index(inplace=True)
    queries_comments_reactions.drop(columns=['index'], inplace=True)
    queries_comments_reactions_for_collection = json.loads(queries_comments_reactions.T.to_json()).values()
    db.queries_comments_reactions_collection.insert_many(queries_comments_reactions_for_collection)

    web_urls.reset_index(inplace=True)
    web_urls.drop(columns=['index'], inplace=True)
    web_urls_for_collection = json.loads(web_urls.T.to_json()).values()
    db.urls_collection.insert_many(web_urls_for_collection)

    websites_visits.reset_index(inplace=True)
    websites_visits.drop(columns=['index'], inplace=True)
    websites_visits_for_collection = json.loads(websites_visits.T.to_json()).values()
    db.fb_websites_visited_collection.insert_many(websites_visits_for_collection)

    # Predict emotion on text - evaluate what emotion the queries express ##############################################

    activities = queries_comments_reactions # create new dataframe for emotions
    merged_text = activities['content'] + ' on ' + activities['on']
    translated_text = translate(merged_text)
    emotion = classification.inference(translated_text)  # get emotion
    activities['emotion'] = emotion

    # # add to db
    activities.reset_index(inplace=True)
    activities.drop(columns=['index'], inplace=True)
    activities_for_collection = json.loads(activities.T.to_json()).values()
    db.emotions_collection.insert_many(activities_for_collection)

    # Predict depression on text - evaluate what emotion the queries express ########################################### TODO

    # Predict anxiety on text - evaluate what emotion the queries express ############################################## TODO

    return


# For scheduling to execute every day at 00:01
# # schedule.every(20).seconds.do(job) #for testing
schedule.every().day.at("00:01").do(job)
while True:
    schedule.run_pending()
    time.sleep(1)