import argparse
import json
import twitter
import collect
import graph
import logging
import itertools
from datetime import datetime
import sys

# logging.basicConfig(level=logging.ERROR)
# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.ERROR)


def read_config_file(filename):
    with open(filename, 'r') as f:
        cfg = json.load(f)
    return cfg


def get_date_range(tweets):
    kmax = max(tweets.keys(),
               key=(lambda k: datetime.strptime(tweets[k]['created_at'], '%a %b %d %H:%M:%S +0000 %Y')))
    kmin = min(tweets.keys(),
               key=(lambda k: datetime.strptime(tweets[k]['created_at'], '%a %b %d %H:%M:%S +0000 %Y')))
    return tweets[kmin]['created_at'], tweets[kmax]['created_at']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='configuration (json) file path')
    parser.add_argument('-v', '--verbosity', help='increase output verbosity',
                        action='count')
    # read parameters
    args = parser.parse_args()
    cfg = read_config_file(args.config)

    handler = logging.StreamHandler(sys.stdout)
    if args.verbosity == 1:
        logger.setLevel(logging.INFO)
    elif args.verbosity >= 2:
        logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    graph_handle = twitter.twitter_network(cfg['credentials_file'])
    # flatten dictionary
    initial_accounts = list(itertools.chain.from_iterable(read_config_file(cfg['accounts_file']).values()))
    graph_handle.rules = cfg['rules']
    user_list, nodes_df, edges_df, hashtags, tweets = collect.spiky_ball(initial_accounts,
                                                                         graph_handle,
                                                                         exploration_depth=cfg['collection_settings']['exploration_depth'],
                                                                         mode=cfg['collection_settings']['mode'],
                                                                         random_subset_size=cfg['collection_settings']['random_subset_size'],
                                                                         spread_type=cfg['collection_settings']['spread_type'],
                                                                         )
    logger.info('Total number of users mentioned: {}'.format(len(user_list)))
    start_date, end_date = get_date_range(tweets)
    logger.info('Range of tweets date from {} to {}'.format(start_date, end_date))
    #nodes_df = twitter.reshape_node_data(nodes_df)
    g = graph.graph_from_edgeslist(edges_df)


if __name__ == '__main__':
    main()
