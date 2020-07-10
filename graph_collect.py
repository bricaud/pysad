import argparse
import json
from pysad.twitter import TwitterNetwork, reshape_node_data
from pysad import collect
from pysad import graph
from pysad.NodeInfo import TwitterNodeInfo
import networkx as nx
import pandas as pd
import logging
from pymongo import MongoClient
from urllib.parse import quote_plus
from pymongo.errors import ConnectionFailure, BulkWriteError
from datetime import datetime
import sys
import os

# logging.basicConfig(level=logging.ERROR)
# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.ERROR)


def read_config_file(filename):
    with open(filename, 'r') as f:
        cfg = json.load(f)
    return cfg


def write_json_output(dict, filename):
    with open(filename, 'w') as f:
        json.dump(dict, f)


def write_mongodb_output(cfg, tweets):
    mongo_url = 'mongodb://{}:{}@{}/default_db?authSource={}'.format(quote_plus(cfg['user']),
                                                                     quote_plus(cfg['password']),
                                                                     cfg['host'], cfg['database'])
    try:
        client = MongoClient(mongo_url, cfg['port'])
        coll = client[cfg['database']][cfg['collection']]
        # add _id field
        tweets_mongo = [{**v, '_id': k} for k, v in tweets.items()]
        coll.insert_many(tweets_mongo, ordered=False)
    except BulkWriteError:
        logger.info('Duplicates were found.')
    except ConnectionFailure:
        logger.error('Server not available {}@{}:{}'.format(cfg['user'], cfg['host'], cfg['port']))


def get_date_range(tweets):
    kmax = max(tweets.keys(),
               key=(lambda k: datetime.strptime(tweets[k]['created_at'], '%a %b %d %H:%M:%S +0000 %Y')))
    kmin = min(tweets.keys(),
               key=(lambda k: datetime.strptime(tweets[k]['created_at'], '%a %b %d %H:%M:%S +0000 %Y')))
    return tweets[kmin]['created_at'], tweets[kmax]['created_at']


def create_graph(graph_handle, nodes_df, edges_df, hashtags, cfg):
    g = graph.graph_from_edgeslist(edges_df, min_weight=cfg['min_weight'])
    g = graph.add_edges_attributes(g, edges_df)
    g = graph.add_node_attributes(g, reshape_node_data(nodes_df), hashtags)
    g = graph.reduce_graph(g, cfg['min_degree'])
    g = graph.handle_spikyball_neighbors(g, graph_handle)  # ,remove=False)
    return g


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', nargs=1, help='configuration (json) file path')
    parser.add_argument('--twitter-credentials', nargs='?',
                        help='twitter credential file (if missing, set environment variables)', default='')
    parser.add_argument('--graph-output', nargs=1, help='destination (gexf) file path')
    parser.add_argument('--tweets-output', nargs='?', help='destination (json) file path', default='')
    parser.add_argument('--mongodb-config',
                        nargs='?', help='configuration (json) file path for (optional) MongoDB output',
                        default='')
    parser.add_argument('-v', '--verbosity', help='increase output verbosity',
                        action='count')
    parser.add_argument('accounts', help='initial accounts (csv) file path')
    # read parameters
    args = parser.parse_args()
    cfg = read_config_file(args.config[0])

    handler = logging.StreamHandler(sys.stdout)
    if args.verbosity == 1:
        logger.setLevel(logging.INFO)
    elif args.verbosity >= 2:
        logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    if args.twitter_credentials:
        twitter_creds = read_config_file(args.twitter_credentials)
    else:
        twitter_creds = {
            'CONSUMER_KEY': os.getenv('TWITTER_CONSUMER_KEY', ''),
            'CONSUMER_SECRET': os.getenv('TWITTER_CONSUMER_SECRET', '')
        }
    graph_handle = TwitterNetwork(twitter_creds)
    # flatten dictionary
    initial_accounts = pd.read_csv(args.accounts).iloc[:, 0].values.tolist()
    graph_handle.rules = cfg['rules']
    user_list, nodes_df, edges_df, tweets_info = collect.spiky_ball(initial_accounts,
                                                                    graph_handle,
                                                                    exploration_depth=cfg['collection_settings'][
                                                                        'exploration_depth'],
                                                                    mode=cfg['collection_settings']['mode'],
                                                                    random_subset_size=cfg['collection_settings'][
                                                                        'random_subset_size'],
                                                                    spread_type=cfg['collection_settings'][
                                                                        'spread_type'],
                                                                    node_acc=TwitterNodeInfo()
                                                                    )
    logger.info('Total number of users mentioned: {}'.format(len(user_list)))
    start_date, end_date = get_date_range(tweets_info.user_tweets)
    logger.info('Range of tweets date from {} to {}'.format(start_date, end_date))

    # create graph from edge list
    g = create_graph(graph_handle, nodes_df, edges_df, tweets_info.user_hashtags, cfg['graph'])
    g.graph['end_date'] = end_date
    g.graph['start_date'] = start_date

    # perform community detection
    g, clusters = graph.detect_communities(g)
    g.nb_communities = len(clusters)
    g = graph.remove_small_communities(g, clusters, min_size=cfg['graph']['min_community_size'])

    # save graph
    nx.write_gexf(g, args.graph_output[0])

    # tweets output
    if args.tweets_output:
        write_json_output(tweets_info.user_tweets, args.tweets_output)
    if args.mongodb_config:
        cfg_mongo = read_config_file(args.mongodb_config)
        write_mongodb_output(cfg_mongo, tweets_info.user_tweets)


if __name__ == '__main__':
    main()
