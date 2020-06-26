# Import the Twython class
from twython import Twython
import json
import pandas as pd
from datetime import datetime, timedelta, time
from twython import TwythonError, TwythonRateLimitError, TwythonAuthError  # to check the returned API errors

from tqdm import tqdm


class twitter_network:

    def __init__(self, credential_file):

        # Load credentials from json file
        # "twitter_credentials.json"
        with open(credential_file, "r") as file:
            creds = json.load(file)

        # Instantiate an object
        self.twitter_handle = Twython(creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'])
        self.rules = {'min_mentions': 0, 'max_day_old': None, 'max_tweets_per_user': 200, 'nb_popular_tweets': 10}

    def get_neighbors(self, user):
        if not isinstance(user, str):
            return pd.DataFrame(), pd.DataFrame()
        tweets_dic, tweets_meta = self.get_user_tweets(user)
        edges_df, node_info = self.edges_nodes_from_user(tweets_meta, tweets_dic)
        return node_info, edges_df

    def filter(self, node_info, edges_df):
        # filter edges according to node properties
        # filter according to edges properties
        edges_df = self.filter_edges(edges_df)
        return node_info, edges_df

    def filter_edges(self, edges_df):
        # filter edges according to their properties
        if edges_df.empty:
            return edges_df
        return edges_df[edges_df['weight'] >= self.rules['min_mentions']]

    def neighbors_list(self, edges_df):
        # print(edges_df)
        # print(edges_df['mention'].unique())
        users_connected = edges_df.index.droplevel(0).tolist()
        return users_connected

    def neighbors_with_weights(self, edges_df):
        user_list = self.neighbors_list(edges_df)
        return dict.fromkeys(user_list, 1)

    ###############################################################
    # Functions for extracting tweet info from the twitter API
    ###############################################################

    def fill_retweet_info(self, tweet_dic, raw_retweet):
        # handle the particular structure of a retweet to get the full text retweeted
        tweet_dic['retweeted_from'] = raw_retweet['user']['screen_name']
        if raw_retweet['truncated']:
            full_text = raw_retweet['extended_tweet']['full_text']
        else:
            full_text = raw_retweet['full_text']
        return tweet_dic, full_text

    def get_full_url(self, url_dic):
        if 'unwound' in url_dic:
            return url_dic['unwound']['url']
        return url_dic['expanded_url']

    def extract_tweet_infos(self, raw_tweet):
        # make a dic from the json raw tweet with the needed information

        tweet_dic = {}
        time_struct = datetime.strptime(raw_tweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
        ts = time_struct.strftime('%Y-%m-%d %H:%M:%S')

        tweet_dic['user'] = raw_tweet['user']['screen_name']
        tweet_dic['name'] = raw_tweet['user']['name']
        tweet_dic['user_details'] = raw_tweet['user']['description']
        tweet_dic['date'] = ts
        tweet_dic['favorite_count'] = raw_tweet['favorite_count']
        tweet_dic['retweet_count'] = raw_tweet['retweet_count']
        tweet_dic['user_mentions'] = [user['screen_name'] for user in raw_tweet['entities']['user_mentions']]
        tweet_dic['urls'] = [self.get_full_url(url) for url in raw_tweet['entities']['urls']]
        tweet_dic['hashtags'] = [htg['text'] for htg in raw_tweet['entities']['hashtags']]
        # if raw_tweet['entities']['hashtags']:
        #    print([htg['text'] for htg in raw_tweet['entities']['hashtags']])
        # print(raw_tweet)
        if 'place' in raw_tweet and raw_tweet['place'] is not None:
            tweet_dic['place'] = raw_tweet['place']['name']
        else:
            tweet_dic['place'] = None

        # Handle text and retweet data
        if raw_tweet['truncated']:
            if 'extended_tweet' not in raw_tweet:
                raise ValueError(
                    'Missing extended tweet information. Make sure you set options to get extended tweet from the API.')
            full_text = raw_tweet['extended_tweet']['full_text']
        elif 'full_text' in raw_tweet:
            full_text = raw_tweet['full_text']
        else:
            full_text = raw_tweet['text']
        if 'retweeted_status' in raw_tweet:
            tweet_dic, full_text = self.fill_retweet_info(tweet_dic, raw_tweet['retweeted_status'])
        else:
            tweet_dic['retweeted_from'] = None
        tweet_dic['text'] = full_text
        return tweet_dic

    def filter_old_tweets(self, tweets):
        max_day_old = self.rules['max_day_old']
        if not max_day_old:
            return tweets

        days_limit = datetime.now() - timedelta(days=max_day_old)
        tweets_filt = filter(lambda t: datetime.strptime(t['created_at'], '%a %b %d %H:%M:%S +0000 %Y') >= days_limit,
                             tweets)
        return list(tweets_filt)

    def get_user_tweets(self, username):
        # Collect tweets from a username

        count = self.rules['max_tweets_per_user']

        # Test if ok
        try:
            user_tweets_raw = self.twitter_handle.get_user_timeline(screen_name=username,
                                                                    count=count, include_rts=True,
                                                                    tweet_mode='extended', exclude_replies=False)
            # remove old tweets
            user_tweets_filt = self.filter_old_tweets(user_tweets_raw)
            # make a dictionary
            user_tweets = {x['id']: x for x in user_tweets_filt}
            tweets_metadata = \
                map(lambda x: (x[0], {'user': x[1]['user']['screen_name'],
                                      'mentions': list(map(lambda y: y['screen_name'], x[1]['entities']['user_mentions'])),
                                      'hashtags': list(map(lambda y: y['text'], x[1]['entities']['hashtags'])),
                                      'retweet_count': x[1]['retweet_count'],
                                      'favorite_count': x[1]['favorite_count'], 'created_at': x[1]['created_at']}),
                    user_tweets.items())
            return user_tweets, dict(tweets_metadata)
        except TwythonAuthError as e_auth:
            print('Cannot access to twitter API, authentification error. {}'.format(e_auth.error_code))
            if e_auth.error_code == 401:
                print('Unauthorized access to user {}. Skipping.'.format(username))
                return {}, {}
            raise
        except TwythonRateLimitError as e_lim:
            print('API rate limit reached')
            print(e_lim)
            wait_time = int(e_lim.retry_after) - time.time()
            print('Retry after {} seconds.'.format(wait_time))
            print('Entering sleep mode at:', time.ctime())
            print('Waking up at:', time.ctime(e_lim.retry_after + 1))
            time.sleep(wait_time + 1)
        except TwythonError as e:
            print('Twitter API returned error {} for user {}.'.format(e.error_code, username))
            return {}, {}

    def edges_nodes_from_user(self, tweets_meta, tweets_dic):
        # Make an edge and node property dataframes
        edges_df = self.get_edges(tweets_meta)
        user_info = self.get_nodes_properties(tweets_meta, tweets_dic)
        return edges_df, user_info

    def get_edges(self, tweets_meta):
        # Create the user -> mention table with their properties fom the list of tweets of a user
        meta_df = pd.DataFrame.from_dict(tweets_meta, orient='index').explode('mentions').dropna()
        # Some bots to be removed from the collection
        userstoremove = ['threader_app', 'threadreaderapp']

        filtered_meta_df = meta_df[~meta_df['mentions'].isin(userstoremove) &
                                   ~meta_df['mentions'].isin(meta_df['user'])]

        # group by mentions and keep list of tweets for each mention
        tmp = filtered_meta_df.groupby(['user', 'mentions']).apply(lambda x: (x.index.tolist(), len(x.index)))
        edge_df = pd.DataFrame(tmp.tolist(), index=tmp.index) \
            .rename(columns={0: 'tweet_id', 1: 'weight'}) \
            .sort_values('weight', ascending=False)

        return edge_df

    def get_nodes_properties(self, tweet_meta, tweets_dic):
        nb_popular_tweets = self.rules['nb_popular_tweets']
        # global properties
        meta_df = pd.DataFrame.from_dict(tweet_meta, orient='index') \
            .sort_values('retweet_count', ascending=False)
        # hashtags statistics
        ht_df = meta_df.explode('hashtags').dropna()
        htgb = ht_df.groupby(['hashtags']).size()
        user_hashtags = pd.DataFrame(htgb).rename(columns={0: 'count'})\
            .sort_values('count', ascending=False)\
            .to_dict()
        user_hashtags['user'] = meta_df['user'].iloc[0]
        tweets_meta_kept = meta_df.head(nb_popular_tweets)
        tweets_kept = {k: tweets_dic[k] for k in tweets_meta_kept.index.to_list()}
        # Get most popular tweets of user
        return {'user_tweets': tweets_kept, 'tweets_meta': tweets_meta_kept, 'user_hashtags': user_hashtags}


#####################################################
## Utils functions for the graph
#####################################################


def reshape_node_data(node_df):
    node_df = node_df[['user', 'name', 'user_details', 'all_hashtags', 'spikyball_hop']]
    node_df = node_df.drop_duplicates(subset='user')
    node_df.set_index('user', inplace=True)
    return node_df


def reshape_edge_data(edge_df, min_weight):
    edge_grouped = edge_df.groupby(['user', 'mention'])
    edge_list = []
    # grouping together the user->mention and summing their weights
    for name, group in edge_grouped:
        tweets = group.to_json()
        # tweets = group
        edge_dic = {'user': name[0], 'mention': name[1], 'weight': group['weight'].sum(),
                    'tweets': tweets}
        if edge_dic['weight'] < min_weight:
            continue
        edge_list.append(edge_dic)
    edge_df = pd.DataFrame(edge_list)
    edge_df.set_index(['user', 'mention'], inplace=True)
    return edge_df


#############################################################
# Functions for managing twitter accounts to follow
#############################################################

class initial_accounts:
    """ Handle the initial twitter accounts (load ans save them)
    """

    def __init__(self, accounts_file=None):
        if accounts_file is None:
            accounts_file = 'initial_accounts.txt'  # Default account file
        self.accounts_file = accounts_file
        self.accounts_dic = {}
        self.load()

    def accounts(self, label=None):
        # if not self.accounts_dic:
        # self.load()
        if label is None:
            return self.accounts_dic
        self.check_label(label)
        return self.accounts_dic[label]

    def list(self):
        return list(self.accounts_dic.keys())

    def add(self, label, list_of_accounts):
        self.accounts_dic[label] = list_of_accounts

    def remove(self, label):
        self.check_label(label)
        del self.accounts_dic[label]

    def save(self):
        with open(self.accounts_file, 'w') as outfile:
            json.dump(self.accounts_dic, outfile)
        print('Wrote', self.accounts_file)

    def load(self):
        with open(self.accounts_file) as json_file:
            self.accounts_dic = json.load(json_file)

    def check_label(self, label):
        if label not in self.accounts_dic:
            print('ERROR. Key "{}" is not in the list of accounts.'.format(label))
            print('Possible choices are: {}'.format([key for key in self.accounts_dic.keys()]))
            raise KeyError('ERROR. Key "{}" is not in the list of accounts.'.format(label))
