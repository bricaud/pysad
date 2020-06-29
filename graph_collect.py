import argparse
import json
import twitter


def read_config_file(filename):
    with open(filename, 'r') as f:
        cfg = json.load(f)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='configuration (json) file path')

    # read parameters
    args = parser.parse_args()
    cfg = read_config_file(args.config)

    graph_handle = twitter.twitter_network(cfg['credentials_file'])
    initial_accounts = read_config_file(cfg['accounts_file'])
    graph_handle.rules = cfg['rules']



if __name__ == '__main__':
    main()
