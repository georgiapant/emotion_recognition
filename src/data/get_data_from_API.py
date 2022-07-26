from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import requests
import json
from src.config import api_url

# api_url = "http://160.40.51.26:3000"

class Facebook:

    def __init__(self):
        self.api_url = api_url

    def get_data(self, user_id): # TODO: fix when the rest of the response is avl
        topic = "/get_fb_data?user_id=" + user_id
        response = requests.get(self.api_url + topic)
        data = json.loads(response.content)

        if data:
            conversation = pd.DataFrame(data["conversations"], columns=['conversations'])
            conversation_count = len(conversation)
            pages = pd.DataFrame(data["pages"], columns=['pages'])
            pages_count = len(pages)
            shortcuts = pd.DataFrame(data["shortcuts"], columns=['shortcuts'])
            shortcuts_count = len(shortcuts)

        else:
            conversation = pd.DataFrame(columns=['conversations'])
            conversation_count = 0
            pages = pd.DataFrame(columns=['pages'])
            pages_count = 0
            shortcuts = pd.DataFrame(columns=['shortcuts'])
            shortcuts_count = 0

        return conversation, conversation_count, pages, pages_count, shortcuts, shortcuts_count

    def get_visits(self, user_id, since, until, page, nperpage): # todo: when all info are avl
        topic = "/get_fb_visits?user_id=" + user_id + "&query=" + "&since=" + since + "&until=" + until + "&page=" \
                + page + "&nperpage=" + nperpage
        response = requests.get(self.api_url + topic)
        data = json.loads(response.content)

        if data['count']:
            websites_visited_count = data['count']
            websites_visited = pd.DataFrame(data['visits'])
            # print(websites_visited)

            if websites_visited_count > int(nperpage):
                # check if there are more than one pages
                for i in range(websites_visited_count - 1):
                    page = int(page) + 1
                    topic = "/get_query?user_id=" + user_id + "&query=" + "&since=" + since + "&until=" + until + \
                            "&page=" + str(page) + "&nperpage=" + nperpage
                    response = requests.get(self.api_url + topic)
                    websites_visited = pd.concat([pd.DataFrame(json.loads(response.content)['visits'])
                                                     , websites_visited])

        else:
            websites_visited = pd.DataFrame(columns=['_id', 'session_id', 'timestamp', 'type', 'description', 'name'])
            websites_visited_count = 0

        return websites_visited, websites_visited_count

    def get_queries(self, user_id, since, until, page, nperpage): # todo: when all info are avl
        topic = "/get_fb_query?user_id=" + user_id + "&query=" + "&since=" + since + "&until=" + until + "&page=" \
                + page + "&nperpage=" + nperpage
        response = requests.get(self.api_url + topic)
        data = json.loads(response.content)

        if data['count']:
            fb_queries = pd.DataFrame(data['queries'])
            fb_queries_count = data['count']

            if fb_queries_count > int(nperpage):
                # check if there are more than one pages
                for i in range(fb_queries_count - 1):
                    page = int(page) + 1
                    topic = "/get_query?user_id=" + user_id + "&query=" + "&since=" + since + "&until=" + until + \
                            "&page=" + str(page) + "&nperpage=" + nperpage
                    response = requests.get(self.api_url + topic)
                    fb_queries = pd.concat([pd.DataFrame(json.loads(response.content)['queries'])
                                                     , fb_queries_count])
        else:
            fb_queries = pd.DataFrame(columns=['_id','session_id','timestamp','query'])
            fb_queries_count = 0

        return fb_queries, fb_queries_count

    def get_comments(self, user_id, since, until, page, nperpage): # todo: when all info are avl
        topic = "/get_fb_comment?user_id=" + user_id + "&query=" + "&since=" + since + "&until=" + until \
                + "&page=" + page + "&nperpage=" + nperpage
        response = requests.get(self.api_url + topic)
        data = json.loads(response.content)
        # print(data)

        if data['count']:
            fb_comments = pd.DataFrame(data['comments'])
            fb_comments_count = data['count']

            if fb_comments_count > int(nperpage):
                # check if there are more than one pages
                for i in range(fb_comments_count - 1):
                    page = int(page) + 1
                    topic = "/get_query?user_id=" + user_id + "&query=" + "&since=" + since + "&until=" + until + \
                            "&page=" + str(page) + "&nperpage=" + nperpage
                    response = requests.get(self.api_url + topic)
                    fb_comments = pd.concat([pd.DataFrame(json.loads(response.content)['queries'])
                                                     , fb_comments_count])

        else:
            fb_comments = pd.DataFrame(columns=['_id','session_id','timestamp','post','comment'])
            fb_comments_count = 0

        return fb_comments, fb_comments_count

    def get_reactions(self, user_id, since, until, page, nperpage): # todo: when all info are avl
        topic = "/get_fb_reaction?user_id=" + user_id + "&query=" + "&since=" + since + "&until=" + until + "&page=" + page + "&nperpage=" + nperpage
        response = requests.get(self.api_url + topic)
        data = json.loads(response.content)

        if data['count']:
            fb_reactions = pd.DataFrame(data['reactions'])
            fb_reactions_count = data['count']

            if fb_reactions_count > int(nperpage):
                # check if there are more than one pages
                for i in range(fb_reactions_count - 1):
                    page = int(page) + 1
                    topic = "/get_query?user_id=" + user_id + "&query=" + "&since=" + since + "&until=" + until + \
                            "&page=" + str(page) + "&nperpage=" + nperpage
                    response = requests.get(self.api_url + topic)
                    fb_reactions = pd.concat([pd.DataFrame(json.loads(response.content)['queries'])
                                                     , fb_reactions_count])

        else:
            fb_reactions = pd.DataFrame(columns=['_id','session_id','timestamp','post','reaction'])
            fb_reactions_count = 0

        return fb_reactions, fb_reactions_count


class Youtube:

    def __init__(self):
        self.api_url = api_url

    def get_subscriptions(self, user_id): # TODO: fix when the rest of the response is avl
        topic = "/get_yt_subscriptions?user_id=" + user_id
        response = requests.get(self.api_url + topic)
        data = json.loads(response.content)

        if data['data']:
            subscriptions = pd.DataFrame(data['data'])
            subscriptions_count = len(subscriptions['subscriptions'])

        else:
            subscriptions = pd.DataFrame()
            subscriptions_count = 0

        return subscriptions, subscriptions_count

    def get_queries(self, user_id, since, until, page, nperpage): # todo: when all info are avl
        topic = "/get_yt_query?user_id=" + user_id + "&query=" + "&since=" + since + "&until=" + until \
                + "&page=" + page + "&nperpage=" + nperpage
        response = requests.get(self.api_url + topic)
        data = json.loads(response.content)

        if data['count']:
            yt_queries = pd.DataFrame(data['queries'])
            yt_queries_count = data['count']

            if yt_queries_count > int(nperpage):
                # check if there are more than one pages
                for i in range(yt_queries_count - 1):
                    page = int(page) + 1
                    topic = "/get_query?user_id=" + user_id + "&query=" + "&since=" + since + "&until=" + until + \
                            "&page=" + str(page) + "&nperpage=" + nperpage
                    response = requests.get(self.api_url + topic)
                    yt_queries = pd.concat([pd.DataFrame(json.loads(response.content)['queries'])
                                                     , yt_queries_count])
        else:
            yt_queries = pd.DataFrame(columns=['_id','session_id','timestamp', 'query'])
            yt_queries_count = 0

        return yt_queries, yt_queries_count

    def get_comments(self, user_id, since, until, page, nperpage): # todo: when all info are avl
        topic = "/get_yt_comment?user_id=" + user_id + "&query=" + "&since=" + since + "&until=" + until \
                + "&page=" + page + "&nperpage=" + nperpage
        response = requests.get(self.api_url + topic)
        data = json.loads(response.content)
        # print(data)

        if data['count']:
            yt_comments = pd.DataFrame(data['comments'])
            yt_comments_count = data['count']

            if yt_comments_count > int(nperpage):
                # check if there are more than one pages
                for i in range(yt_comments_count - 1):
                    page = int(page) + 1
                    topic = "/get_query?user_id=" + user_id + "&query=" + "&since=" + since + "&until=" + until + \
                            "&page=" + str(page) + "&nperpage=" + nperpage
                    response = requests.get(self.api_url + topic)
                    yt_comments = pd.concat([pd.DataFrame(json.loads(response.content)['queries'])
                                                     , yt_comments_count])
        else:
            yt_comments = pd.DataFrame(columns=['_id','session_id','timestamp', 'video','comment'])
            yt_comments_count = 0

        return yt_comments, yt_comments_count

    def get_reactions(self, user_id, since, until, page, nperpage):
        topic = "/get_yt_reaction?user_id=" + user_id + "&query=" + "&since=" + since + "&until=" + until + "&page=" + page + "&nperpage=" + nperpage
        response = requests.get(self.api_url + topic)
        data = json.loads(response.content)

        if data['count']:
            yt_reactions = pd.DataFrame(data['reactions']) # todo: when all info are avl
            yt_reactions_count = data['count']

            if yt_reactions_count > int(nperpage):
                # check if there are more than one pages
                for i in range(yt_reactions_count - 1):
                    page = int(page) + 1
                    topic = "/get_query?user_id=" + user_id + "&query=" + "&since=" + since + "&until=" + until + \
                            "&page=" + str(page) + "&nperpage=" + nperpage
                    response = requests.get(self.api_url + topic)
                    yt_reactions = pd.concat([pd.DataFrame(json.loads(response.content)['queries'])
                                                     , yt_reactions_count])
        else:
            yt_reactions = pd.DataFrame(columns=['_id','session_id','timestamp','title','reaction'])
            yt_reactions_count = 0

        return yt_reactions, yt_reactions_count


class Web:

    def __init__(self):
        self.api_url = api_url

    def get_queries(self, user_id, since, until, page, nperpage):
        """
        @param user_id:
        @param since:
        @param until:
        @param page:
        @param nperpage:
        @return:  queries, count
            queries: a dataframe with the information regarding the queries made by the user
            count: the amount of urls for the specified timeframe
        """
        topic = "/get_query?user_id=" + user_id + "&query=" + "&since=" + since + "&until=" + until + "&page=" \
                + page + "&nperpage=" + nperpage
        response = requests.get(self.api_url + topic)
        data = json.loads(response.content)

        if data['count']:
            queries = pd.DataFrame(data['queries'])
            queries_count = data['count']

            if queries_count > int(nperpage):
                # check if there are more than one pages
                for i in range(queries_count - 1):
                    page = int(page) + 1
                    topic = "/get_query?user_id=" + user_id + "&query=" + "&since=" + since + "&until=" + until + \
                            "&page=" + str(page) + "&nperpage=" + nperpage
                    response = requests.get(self.api_url + topic)
                    queries = pd.concat([pd.DataFrame(json.loads(response.content)['queries']), queries])
        else:
            queries = pd.DataFrame(columns=['_id', 'session_id', 'timestamp', 'query'])
            queries_count = 0

        return queries, queries_count

    def get_urls(self, user_id, since, until, page, nperpage):
        """
        @param user_id:
        @param since:
        @param until:
        @param page:
        @param nperpage:
        @return:  urls, count
            urls: a dataframe with the information regarding the URLs accessed by the user
            count: the amount of urls for the specified timeframe
        """

        topic = "/get_url?user_id=" + user_id + "&query=" + "&since=" + since + "&until=" + until + "&page=" \
                + page + "&nperpage=" + nperpage
        response = requests.get(self.api_url + topic)
        data = json.loads(response.content)

        if data['count']:
            urls = pd.DataFrame(data['urls'])
            urls_count = data['count']

            if urls_count > int(nperpage):
                # check if there are more than one pages
                for i in range(urls_count - 1):
                    page = int(page) + 1
                    topic = "/get_query?user_id=" + user_id + "&query=" + "&since=" + since + "&until=" + until + \
                            "&page=" + str(page) + "&nperpage=" + nperpage
                    response = requests.get(self.api_url + topic)
                    urls = pd.concat([pd.DataFrame(json.loads(response.content)['queries']), urls])

        else:
            urls = pd.DataFrame(columns=['_id','tab','session_id','timestamp','title','url'])
            urls_count = 0

        return urls, urls_count


def main():
    web = Web()
    fb = Facebook()
    yt = Youtube()

    # user_id = "d2n611d6-l5v4-9371-b808-4d1632f29422" # TODO: when I have access to the user_id list update this

    user_id = "f4b135b3-c5f6-4261-b808-4d1632f29422"

    # yesterday = str(int((datetime.today() - timedelta(days=1)).timestamp()))
    # since = yesterday

    a_year_ago = str(int((datetime.today() - relativedelta(years=1)).timestamp()))
    since = a_year_ago
    today = str(int(datetime.now().timestamp()))
    until = today
    page = "0"
    nperpage = "100"

    # urls, count_urls = web.get_urls(user_id, since, until, page, nperpage)
    # print("web urls count {}".format(count_urls))
    # print(urls)
    #
    # queries, count_queries = web.get_queries(user_id, since, until, page, nperpage)
    # print("web queries count {}".format(count_queries))
    # print(queries)

    # fb_conversation, fb_conversation_count, fb_pages, fb_pages_count, fb_shortcuts, fb_shortcuts_count = fb.get_data(user_id)
    # print("facebook data conv {}, pages {} and shortcuts {}".format(fb_conversation_count, fb_pages_count, fb_shortcuts_count))
    # print(fb_conversation)
    # print(fb_pages)
    # print(fb_shortcuts)

    websites_visited, websites_visited_count = fb.get_visits(user_id, since, until, page, nperpage)
    print("web sites visited from fb count {}".format(websites_visited_count))
    print(websites_visited)
    #
    # fb_comments, count_fb_comments = fb.get_comments(user_id, since, until, page, nperpage)
    # print("fb comments count {}".format(count_fb_comments))
    # print(fb_comments)
    #
    # fb_reactions, count_fb_reactions = fb.get_reactions(user_id, since, until, page, nperpage)
    # print("fb reactions count {}".format(count_fb_reactions))
    # print(fb_reactions)
    #
    # fb_queries, count_fb_queries = fb.get_queries(user_id, since, until, page, nperpage)
    # print("fb queries count {}".format(count_fb_queries))
    # print(fb_queries)

    # yt_queries, count_yt_queries = yt.get_queries(user_id, since, until, page, nperpage)
    # print("yt queries count {}".format(count_yt_queries))
    # print(yt_queries)
    #
    # yt_comments, count_yt_comments = yt.get_comments(user_id, since, until, page, nperpage)
    # print("yt comments count {}".format(count_yt_comments))
    # print(yt_comments)
    #
    # yt_reactions, count_yt_reactions = yt.get_reactions(user_id, since, until, page, nperpage)
    # print("yt reactions count {}".format(count_yt_reactions))
    # print(yt_reactions)


if __name__ == '__main__':
    main()
