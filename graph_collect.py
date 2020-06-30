import argparse
import json
import twitter
import collect


def read_config_file(filename):
    with open(filename, 'r') as f:
        cfg = json.load(f)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='configuration (json) file path')
    parser.add_argument("--verbose", help="increase output verbosity",
                        action="store_true")
    # read parameters
    args = parser.parse_args()
    cfg = read_config_file(args.config)

    graph_handle = twitter.twitter_network(cfg['credentials_file'])
    initial_accounts = read_config_file(cfg['accounts_file'])
    graph_handle.rules = cfg['rules']
    user_list, nodes_df, edges_df, hashtags, tweets = collect.spiky_ball(initial_accounts.values(),
                                                                         graph_handle,
                                                                         exploration_depth=cfg['collection_settings']['exploration_depth'],
                                                                         mode=cfg['collection_settings']['mode'],
                                                                         random_subset_size=cfg['collection_settings']['random_subset_size'],
                                                                         spread_type=cfg['collection_settings']['spread_type'],
                                                                         logger_level='verbose' if parser.verbose else 'error')


if __name__ == '__main__':
    main()
