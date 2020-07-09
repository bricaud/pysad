import pandas as pd


class NodeInfo:
    def update(self, new_info):
        # dummy no-op
        return

    def get_nodes(self):
        return pd.DataFrame()


class TwitterNodeInfo(NodeInfo):
    def __init__(self, user_hashtags={}, user_tweets={}, tweets_meta=pd.DataFrame()):
        self.user_hashtags = user_hashtags
        self.user_tweets = user_tweets
        self.tweets_meta = tweets_meta

    def update(self, new_info):
        self.user_hashtags.update(new_info.user_hashtags)
        self.user_tweets.update(new_info.user_tweets)

    def get_nodes(self):
        return self.tweets_meta
