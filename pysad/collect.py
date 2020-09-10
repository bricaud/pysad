import pandas as pd
from tqdm import tqdm
import numpy as np
import operator
import os
import logging
from .NodeInfo import NodeInfo


def combine_dicts(a, b, op=operator.add):
    return {**a, **b, **{k: op(a[k], b[k]) for k in a.keys() & b.keys()}}


# return dict(a.items() + b.items() +
#    [(k, op(a[k], b[k])) for k in set(b) & set(a)])

def process_hop(graph_handle, node_list, nodes_info_acc):
    """ collect the tweets and tweet info of the users in the list username_list
    """
    new_node_dic = {}
    total_edges_df = pd.DataFrame()
    total_nodes_df = pd.DataFrame()

    # Display progress bar if needed
    disable_tqdm = logging.root.level >= logging.INFO
    logging.info('processing next hop with {} nodes'.format(len(node_list)))
    for node in tqdm(node_list, disable=disable_tqdm):
        # Collect neighbors for the next hop
        node_info, edges_df = graph_handle.get_neighbors(node)
        node_info, edges_df = graph_handle.filter(node_info, edges_df)

        total_nodes_df = total_nodes_df.append(node_info.get_nodes())
        nodes_info_acc.update(node_info)  # add new info

        total_edges_df = total_edges_df.append(edges_df)
        neighbors_dic = graph_handle.neighbors_with_weights(edges_df)
        new_node_dic = combine_dicts(new_node_dic, neighbors_dic)

    return new_node_dic, total_edges_df, total_nodes_df, nodes_info_acc


def random_subset(node_dic, mode, random_subset_size=None):
    nb_nodes = len(node_dic)
    node_list = list(node_dic.keys())
    node_weights = list(node_dic.values())
    # Renormalize node weights
    node_weights = np.array(node_weights) / np.sum(np.array(node_weights))
    # print(node_weights,np.sum(node_weights))
    if mode == 'constant':
        if isinstance(random_subset_size, int) and (nb_nodes > random_subset_size):
            # Only explore a random subset of users
            logging.debug('---')
            logging.debug(
                'Too many users mentioned ({}). Keeping a random subset of {}.'.format(nb_nodes, random_subset_size))
            r_node_list = np.random.choice(node_list, random_subset_size, p=node_weights, replace=False)
        else:
            r_node_list = node_list
    elif mode == 'percent':
        if random_subset_size <= 1 and random_subset_size > 0:
            nb_samples = round(nb_nodes * random_subset_size)
            if nb_samples < 2:  # in case the number of nodes is too small
                nb_samples = nb_nodes
            # print(nb_samples)
            r_node_list = np.random.choice(node_list, nb_samples, p=node_weights, replace=False)
        else:
            raise Exception('the value must be between 0 and 1.')
    else:
        raise Exception('Unknown mode. Choose "constant" or "percent".')
    r_node_dic = {node: node_dic[node] for node in r_node_list}
    return r_node_dic


def spiky_ball(username_list, graph_handle, exploration_depth=4,
               mode='percent', random_subset_size=None,
               node_acc=NodeInfo()):
    """ Collect the tweets of the users and their mentions
        make an edge list user -> mention
        and save each user edge list to a file
    """

    if graph_handle.rules:
        logging.debug('---')
        logging.debug('Parameters:')
        for key, value in graph_handle.rules.items():
            logging.debug(key, value)
        logging.debug('---')

    total_username_list = []
    # total_username_list += username_list
    new_username_list = username_list.copy()
    total_node_dic = {}
    new_node_dic = {node: 1 for node in username_list}
    total_edges_df = pd.DataFrame()
    total_nodes_df = pd.DataFrame()

    for depth in range(exploration_depth):
        logging.debug('')
        logging.debug('******* Processing users at {}-hop distance *******'.format(depth))


        if not new_node_dic:
            break
        new_node_dic = random_subset(new_node_dic, mode=mode, random_subset_size=random_subset_size)
        # Avoid visiting twice the same edges
        # and remove the nodes already collected
        new_node_list = list(set(new_node_dic.keys()).difference(set(total_node_dic.keys())))
        total_node_dic = combine_dicts(total_node_dic, new_node_dic)


        new_node_dic, edges_df, nodes_df, node_acc = process_hop(graph_handle, new_node_list, node_acc)
        if nodes_df.empty:
            continue
        nodes_df['spikyball_hop'] = depth  # Mark the depth of the spiky ball on the nodes
        total_edges_df = total_edges_df.append(edges_df)
        total_nodes_df = total_nodes_df.append(nodes_df)

    total_edges_df = total_edges_df.sort_values('weight', ascending=False)
    total_node_list = list(total_node_dic.keys())  # set of unique nodes
    return total_node_list, total_nodes_df, total_edges_df, node_acc


def save_data(nodes_df, edges_df, data_path):
    # Save to json file
    edgefilename = os.path.join(data_path, 'edges_data.json')
    nodefilename = os.path.join(data_path, 'nodes_data.json')
    logging.debug('Writing', edgefilename)
    edges_df.reset_index().to_json(edgefilename)
    logging.debug('Writing', nodefilename)
    nodes_df.reset_index().to_json(nodefilename)
    return None


def load_data(data_path):
    nodesfilename = os.path.join(data_path, 'nodes_data.json')
    edgesfilename = os.path.join(data_path, 'edges_data.json')
    logging.debug('Loading', nodesfilename)
    nodes_df = pd.read_json(nodesfilename)
    logging.debug('Loading', edgesfilename)
    edges_df = pd.read_json(edgesfilename)
    return nodes_df, edges_df
