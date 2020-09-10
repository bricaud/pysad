import pandas as pd
from tqdm import tqdm
import numpy as np
import operator
import os
import logging
from .NodeInfo import NodeInfo

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def split_edges(edges_df, node_list):
    # split edges between the ones connecting already collected nodes and the ones connecting new nodes
    edges_df_in = edges_df[edges_df['target'].isin(node_list)]
    edges_df_out = edges_df[~(edges_df['target'].isin(node_list))]
    return edges_df_in, edges_df_out 

def remove_edges_with_target_nodes(edges_df, node_list):
    new_edges_df = edges_df[edges_df['target'].isin(node_list)]
    return new_edges_df

def get_node_info(graph_handle, node_list, nodes_info_acc):
    """ collect the node info and neighbors for the nodes in node_list
    """
    total_edges_df = pd.DataFrame()
    total_nodes_df = pd.DataFrame()

    # Display progress bar if needed
    disable_tqdm = logging.root.level > logging.INFO
    logging.info('processing next hop with {} nodes'.format(len(node_list)))
    for node in tqdm(node_list, disable=disable_tqdm):
        # Collect neighbors for the next hop
        node_info, edges_df = graph_handle.get_neighbors(node)
        node_info, edges_df = graph_handle.filter(node_info, edges_df)
        
        total_nodes_df = total_nodes_df.append(node_info.get_nodes())
        nodes_info_acc.update(node_info)  # add new info
        total_edges_df = total_edges_df.append(edges_df)

    return total_edges_df, total_nodes_df, nodes_info_acc

def degree_weight(node_type, edges_df):
    edges_df.reset_index(inplace=True)
    degree_df = edges_df[[node_type,'weight']].groupby([node_type]).sum()
    degree_df.columns = ['degree_'+node_type]
    # needs reset_index and set_index to keep the initial index.
    edges_df = edges_df.merge(degree_df, on=node_type)
    edges_df.set_index('index',inplace=True)
    edges_df.sort_index(inplace=True)
    #edges_df2['weight_over_degree'] = edges_df2['weight']/edges_df2['degree']
    degree_vec = np.array(edges_df['degree_'+node_type].tolist())
    return degree_vec, edges_df

def probability_function(edges_df, balltype, coeff):
     # Taking the weights into account for the random selection
    if balltype =='spikyball':       
        source_coeff, edge_coeff, target_coeff = 0, 1, 0
    elif balltype =='hubball':
        source_coeff, edge_coeff, target_coeff = coeff, 1, 0
    elif balltype == 'coreball':
        source_coeff, edge_coeff, target_coeff = 0, 1, coeff
    elif balltype == 'fireball':
        source_coeff, edge_coeff, target_coeff = -1, 1, 0
    elif balltype == 'firecoreball':
        source_coeff, edge_coeff, target_coeff = -1, 1, coeff        
    else:
        raise ValueError('Unknown ball type.')

    weight_vec = np.array(edges_df['weight'].tolist())
    target_degree_vec, edges_df = degree_weight('target', edges_df)
    source_degree_vec, edges_df = degree_weight('source', edges_df)

    source_func = source_degree_vec.astype(float) ** source_coeff 
    weight_func = weight_vec.astype(float) ** edge_coeff
    target_func = target_degree_vec.astype(float) ** target_coeff
    proba_unormalized = source_func * weight_func * target_func
    proba_f = proba_unormalized / np.sum(proba_unormalized) # Normalize weights
    
    return edges_df.index.tolist(), proba_f, edges_df

def random_subset(edges_df, balltype, mode, coeff, mode_value=None):

    # TODO handle balltype
    nb_edges = len(edges_df)
    if nb_edges == 0:
        return [], pd.DataFrame()
    edges_df.reset_index(drop=True,inplace=True) # needs unique index values for random choice
    edges_indices, proba_f, edges_df = probability_function(edges_df, balltype, coeff)

    if mode == 'constant':
        random_subset_size = mode_value
        if isinstance(random_subset_size, int) and (nb_edges > random_subset_size):
            # Only explore a random subset of users
            logging.debug('---')
            logging.debug(
                'Too many edges ({}). Keeping a random subset of {}.'.format(nb_edges, random_subset_size))
        else:
            random_subset_size = nb_edges
    elif mode == 'percent':
        if mode_value <= 100 and mode_value > 0:
            ratio = 0.01*mode_value
            random_subset_size = round(nb_edges * ratio)
            if random_subset_size < 2:  # in case the number of edges is too small
                random_subset_size = min(nb_edges,10)
                print('Fallback used!')
        else:
            raise Exception('the value must be between 0 and 100.')
    else:
        raise Exception('Unknown mode. Choose "constant" or "percent".')
    r_edges_idx = np.random.choice(edges_indices, random_subset_size, p=proba_f, replace=False)
    r_edges_df = edges_df.loc[r_edges_idx,:]
    #print(proba_f)
    #print(edges_indices)
    #print(r_edges_df)
    nodes_list = r_edges_df['target'].unique().tolist()
    return nodes_list, r_edges_df


def spiky_ball(initial_node_list, graph_handle, exploration_depth=4,
               mode='percent', random_subset_size=None, balltype='spikyball', coeff=0,
               node_acc=NodeInfo(), number_of_nodes=False):
    """ Sample the graph by exploring from an initial node list
    """

    if graph_handle.rules:
        logging.debug('---')
        logging.debug('Parameters:')
        for key, value in graph_handle.rules.items():
            logging.debug(key, value)
        logging.debug('---')


    # Initialization
    new_node_list = initial_node_list.copy()
    total_node_list = [] #new_node_list

    total_edges_df = pd.DataFrame()
    total_nodes_df = pd.DataFrame()
    new_edges = pd.DataFrame()

    # Loop over layers
    for depth in range(exploration_depth):
        logging.debug('')
        logging.debug('******* Processing users at {}-hop distance *******'.format(depth))

        # Option to choose the number of nodes in the final graph
        if number_of_nodes:
            if len(total_node_list + new_node_list) > number_of_nodes:
                # Truncate the list of new nodes
                max_nodes = number_of_nodes - len(total_node_list)
                if max_nodes <=0:
                    break
                logging.info('-- max nb of nodes reached in iteration {} --'.format(depth))
                #print('nodes info',len(total_node_list),len(new_node_list),max_nodes)
                new_node_list = new_node_list[:max_nodes]
                new_edges = remove_edges_with_target_nodes(new_edges, new_node_list)
                #print('new node list',len(new_node_list))


        edges_df, nodes_df, node_acc = get_node_info(graph_handle, new_node_list, node_acc)
        if nodes_df.empty:
            break
        nodes_df['spikyball_hop'] = depth  # Mark the depth of the spiky ball on the nodes    
        
        total_node_list = total_node_list + new_node_list
        #print(edges_df)
        edges_df_in,edges_df_out = split_edges(edges_df, total_node_list)

        # Equivalent of add to graph
        total_edges_df = total_edges_df.append(edges_df_in)
        total_nodes_df = total_nodes_df.append(nodes_df)
        # add the edges linking the new nodes
        total_edges_df = total_edges_df.append(new_edges)
        
        new_node_list, new_edges = random_subset(edges_df_out, balltype, mode=mode, mode_value=random_subset_size, coeff=coeff)
        print('new edges:{} subset:{} in_edges:{}'.format(len(edges_df_out), len(new_edges), len(edges_df_in)))

    print('Nb of layers reached:', depth)
    total_edges_df = total_edges_df.sort_values('weight', ascending=False)
    #total_node_list = list(total_node_dic.keys())  # set of unique nodes
    return total_node_list, total_nodes_df, total_edges_df, node_acc


def save_data(nodes_df, edges_df, data_path):
    # Save to json file
    edgefilename = os.path.join(data_path, 'edges_data.json')
    nodefilename = os.path.join(data_path, 'nodes_data.json')
    logging.debug('Writing', edgefilename)
    edges_df.to_json(edgefilename)
    logging.debug('Writing', nodefilename)
    nodes_df.to_json(nodefilename)
    return None


def load_data(data_path):
    nodesfilename = os.path.join(data_path, 'nodes_data.json')
    edgesfilename = os.path.join(data_path, 'edges_data.json')
    logging.debug('Loading', nodesfilename)
    nodes_df = pd.read_json(nodesfilename)
    logging.debug('Loading', edgesfilename)
    edges_df = pd.read_json(edgesfilename)
    return nodes_df, edges_df
