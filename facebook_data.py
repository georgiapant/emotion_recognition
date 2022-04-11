import os
import json
import codecs
from datetime import datetime
import matplotlib.pyplot as plt

import pandas as pd
import sys
import unicodedata
import six
import re
import emoji
import contractions
from googletrans import Translator

# import matplotlib.pyplot as plt
from collections import Counter

def get_reactions(usr_id):
    type_of_data = "comments_and_reactions"
    filename = "posts_and_comments"

    path = r"D:\CERTH\REBECCA\WP3\Data\FACEBOOK\ADC_PRT_" + usr_id + "_Facebook"
    file = os.path.join(path, type_of_data, filename + ".json")

    with codecs.open(file, 'r', 'utf-8') as infile:
        data = json.load(infile)

    reactions = pd.DataFrame(columns=['user_id', 'timestamp', 'date_time', 'content', 'type'])

    for i in range(len(data['reactions_v2'])):
        temp = {'user_id': usr_id,
                'timestamp': data['reactions_v2'][i]['timestamp'],
                'date_time': datetime.fromtimestamp(data['reactions_v2'][i]['timestamp']),
                'content': data['reactions_v2'][i]['data'][0]['reaction']['reaction'],
                'type': 'reactions'}
        reactions = reactions.append(temp, ignore_index=True)
    return reactions


def get_comments(usr_id):
    type_of_data = "comments_and_reactions"
    filename = "comments"

    path = r"D:\CERTH\REBECCA\WP3\Data\FACEBOOK\ADC_PRT_" + usr_id + "_Facebook"
    file = os.path.join(path, type_of_data, filename + ".json")

    with codecs.open(file, 'r', 'utf-8') as infile:
        data = json.load(infile)

    comments = pd.DataFrame(columns=['user_id', 'timestamp', 'date_time', 'content', 'type'])

    for i in range(len(data['comments_v2'])):
        temp = {'user_id': usr_id,
                'timestamp': data['comments_v2'][i]['timestamp'],
                'date_time': datetime.fromtimestamp(data['comments_v2'][i]['timestamp']),
                'content': data['comments_v2'][i]['data'][0]['comment']['comment'],
                'type': 'comments'}
        comments = comments.append(temp, ignore_index=True)
    return comments


def get_posts(usr_id):
    """
    There are several data that are not utilised yet. Such as attachment data
    """
    type_of_data = "posts"
    filename = "your_posts_1"

    path = r"D:\CERTH\REBECCA\WP3\Data\FACEBOOK\ADC_PRT_" + usr_id + "_Facebook"
    file = os.path.join(path, type_of_data, filename + ".json")

    with codecs.open(file, 'r', 'utf-8') as infile:
        data = json.load(infile)

    posts = pd.DataFrame(columns=['user_id', 'timestamp', 'date_time', 'content', 'type'])
    # print(len(data))

    for i in range(len(data)):
        # df2 = pd.DataFrame.from_dict(data[i], orient='index')
        # print(df2)
        try:
            temp = {'user_id': usr_id,
                    'timestamp': data[i]['timestamp'],
                    'date_time': datetime.fromtimestamp(data[i]['timestamp']),
                    'content': data[i]['data'][0]['post'],
                    'type': 'posts'}
            posts = posts.append(temp, ignore_index=True)
        except:
            continue

    return posts


def get_posts_group(usr_id):
    """
    There are several data that are not utilised yet. Such as attachment data
    """
    type_of_data = "groups"
    filename = "your_posts_in_groups"

    path = r"D:\CERTH\REBECCA\WP3\Data\FACEBOOK\ADC_PRT_" + usr_id + "_Facebook"
    file = os.path.join(path, type_of_data, filename + ".json")

    with codecs.open(file, 'r', 'utf-8') as infile:
        data = json.load(infile)

    posts_group = pd.DataFrame(columns=['user_id', 'timestamp', 'date_time', 'content', 'type'])
    # print(len(data['group_posts_v2']))

    for i in range(len(data['group_posts_v2'])):
        try:

            temp = {'user_id': usr_id,
                    'timestamp': data['group_posts_v2'][i]['timestamp'],
                    'date_time': datetime.fromtimestamp(data['group_posts_v2'][i]['timestamp']),
                    'content': data['group_posts_v2'][i]['data'][0]['post'],
                    'type': 'posts_groups'}
            posts_group = posts_group.append(temp, ignore_index=True)
        except:
            continue

    return posts_group


def get_comments_group(usr_id):
    """
    There are several data that are not utilised yet. Such as attachment data
    """
    type_of_data = "groups"
    filename = "your_comments_in_groups"

    path = r"D:\CERTH\REBECCA\WP3\Data\FACEBOOK\ADC_PRT_" + usr_id + "_Facebook"
    file = os.path.join(path, type_of_data, filename + ".json")

    with codecs.open(file, 'r', 'utf-8') as infile:
        data = json.load(infile)

    comments_group = pd.DataFrame(columns=['user_id', 'timestamp', 'date_time', 'content', 'type'])

    for i in range(len(data['group_comments_v2'])):
        try:
            temp = {'user_id': usr_id,
                    'timestamp': data['group_comments_v2'][i]['timestamp'],
                    'date_time': datetime.fromtimestamp(data['group_comments_v2'][i]['timestamp']),
                    'content': data['group_comments_v2'][i]['data'][0]['comment']['comment'],
                    'type': 'comments_groups'}
            comments_group = comments_group.append(temp, ignore_index=True)
        except:
            # print("one did not have data")
            # print(data['group_comments_v2'][i])
            continue

    return comments_group


def get_searches(usr_id):
    type_of_data = "search"
    filename = "your_search_history"

    path = r"D:\CERTH\REBECCA\WP3\Data\FACEBOOK\ADC_PRT_" + usr_id + "_Facebook"
    file = os.path.join(path, type_of_data, filename + ".json")

    with codecs.open(file, 'r', 'utf-8') as infile:
        data = json.load(infile)

    searches = pd.DataFrame(columns=['user_id', 'timestamp', 'date_time', 'content', 'type'])

    for i in range(len(data['searches_v2'])):
        try:
            temp = {'user_id': usr_id,
                    'timestamp': data['searches_v2'][i]['timestamp'],
                    'date_time': datetime.fromtimestamp(data['searches_v2'][i]['timestamp']),
                    'content': data['searches_v2'][i]['data'][0]['text'],
                    'type': 'searches'}
            searches = searches.append(temp, ignore_index=True)
        except:
            # print("one did not have data")
            # print(data['searches_v2'][i])
            continue

    return searches


def convert_to_unicode(text):
    """
    Converts `text` to Unicode (if it's not already), assuming UTF-8 input.
    """
    # six_ensure_text is copied from https://github.com/benjaminp/six
    def six_ensure_text(s, encoding='utf-8', errors='strict'):
        if isinstance(s, six.binary_type):
            return s.decode(encoding, errors)
        elif isinstance(s, six.text_type):
            return s
        else:
            raise TypeError("not expecting type '%s'" % type(s))

    return six_ensure_text(text, encoding="utf-8", errors="ignore")


def run_strip_accents(text):
    """
    Strips accents from a piece of text.
    """
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)


def text_preprocess(text):
    text = re.sub("[\xc2-\xf4][\x80-\xbf]+", lambda m: m.group(0).encode('latin1').decode('utf8'), text)
    text = convert_to_unicode(text.rstrip().lower())
    text = run_strip_accents(text)

    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
    text = re.sub(r"[^a-zA-z.!?'0-9]", ' ', text)
    text = re.sub('\t', ' ', text)
    text = re.sub(r" +", ' ', text)

    text = text.lower()
    # Demojize
    text = emoji.demojize(text)

    # Expand contraction
    text = contractions.fix(text)

    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)

    # remove hashtags
    # only removing the hash # sign from the word
    text = re.sub(r'#', '', text)
    text = str(re.sub("\S*\d\S*", "", text).strip())

    return text

def main():
    usr = "02"
    reactions = get_reactions(usr)
    comments = get_comments(usr)
    posts = get_posts(usr)
    posts_groups = get_posts_group(usr)
    comments_groups = get_comments_group(usr)
    searches = get_searches(usr)

    activities_all = pd.concat([comments, comments_groups, posts,posts_groups,searches, reactions])
    activities_all['date'] = activities_all['date_time'].dt.date
    path = r"D:\CERTH\REBECCA\WP3\Data\processed_data\FACEBOOK"
    activities_all.to_csv(os.path.join(path, usr + '_activities_all.csv'))

    # Plot all activities per time
    # s = activities['date'].value_counts().sort_index()
    # plt.plot(s.index, s.values)
    # plt.show()

    # Plot separate lines for each activity in time
    # date_type = activities_all.groupby(['date', 'type']).size().unstack(fill_value=0)
    # print(date_type)
    # new.plot()
    # plt.show()

    # print(Counter(reactions['content']))

    posts_comments = pd.concat([comments, comments_groups, posts,posts_groups])
    posts_comments['processed'] = posts_comments.apply(lambda row: text_preprocess(row['content']), axis=1)

    translator = Translator()
    posts_comments['translated'] = posts_comments.apply(lambda row: translator.translate(row['processed']).text, axis=1)
    print(posts_comments)

    path = r"D:\CERTH\REBECCA\WP3\Data\processed_data\FACEBOOK\posts_comments"
    posts_comments.to_csv(os.path.join(path, usr+'_posts_comments_translated.csv'))
    # translator = Translator()
    # searches['preprocessed'] = searches.apply(lambda row: text_preprocess(row['content']), axis=1)
    # searches['translated'] = searches.apply(lambda row: translator.translate(row['preprocessed']).text, axis=1)
    # print(searches)

if __name__ == '__main__':
    main()
