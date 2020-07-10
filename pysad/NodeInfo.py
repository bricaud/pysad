import pandas as pd


class NodeInfo:  # abstract interface
    def update(self, new_info):
        raise NotImplementedError

    def get_nodes(self):
        raise NotImplementedError


class SynthNodeInfo(NodeInfo):
    def __init__(self, nodes=pd.DataFrame()):
        self.nodes = nodes

    def update(self, new_info):
        return

    def get_nodes(self):
        return self.nodes


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
