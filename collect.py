
import pandas as pd
import random
from tqdm import tqdm
import numpy as np
import operator

import logging

#logging.basicConfig(level=logging.ERROR)
#Creating an object 
logger=logging.getLogger() 
  
#Setting the threshold of logger to DEBUG 
logger.setLevel(logging.ERROR) 
#logger.setLevel(logging.DEBUG) 


def combine_dicts(a, b, op=operator.add):
	return {**a, **b, **{k: op(a[k], b[k]) for k in a.keys() & b.keys()}}
    #return dict(a.items() + b.items() +
    #    [(k, op(a[k], b[k])) for k in set(b) & set(a)])

def process_hop(graph_handle, node_list):
	""" collect the tweets and tweet info of the users in the list username_list
	"""
	new_node_dic = {}
	#empty_tweets_users = []
	total_edges_df = pd.DataFrame()
	total_nodes_df = pd.DataFrame()

	# Display progress bar if needed
	disable_tqdm = False
	if logger.root.level > logging.INFO:
		disable_tqdm = True

	for node in tqdm(node_list, disable=disable_tqdm):
		# Collect neighbors for the next hop		
		node_df, edges_df = graph_handle.get_neighbors(node)
		node_df, edges_df = graph_handle.filter(node_df,edges_df)
		if not node_df.empty: # list of node properties and filter edges			
			total_nodes_df = total_nodes_df.append(node_df)	
			if not edges_df.empty: # list of edges and their properties
				total_edges_df = total_edges_df.append(edges_df)
				neighbors_dic = graph_handle.neighbors_with_weights(edges_df)
				new_node_dic = combine_dicts(new_node_dic,neighbors_dic)

	total_edges_df.reset_index(drop=True, inplace=True)
	total_nodes_df.reset_index(drop=True, inplace=True)

	return new_node_dic, total_edges_df, total_nodes_df


def random_subset(node_dic, mode, random_subset_size=None):
	nb_nodes = len(node_dic)
	node_list = list(node_dic.keys())
	node_weights = list(node_dic.values())
	node_weights = np.array(node_weights) / np.sum(np.array(node_weights))
	#print(node_weights,np.sum(node_weights))
	if mode == 'constant':
		if isinstance(random_subset_size,int) and (nb_nodes>random_subset_size):
			# Only explore a random subset of users
			logger.debug('---')
			logger.debug('Too many users mentioned ({}). Keeping a random subset of {}.'.format(nb_nodes,random_subset_size))
			r_node_list = np.random.choice(node_list, random_subset_size, p=node_weights, replace=False)
		else:
			r_node_list = node_list		
	elif mode == 'percent':
		if random_subset_size <= 1 and random_subset_size>0:
			nb_samples = round(nb_nodes * random_subset_size)
			if nb_samples < 2: # in case the number of nodes is too small
				nb_samples = nb_nodes
			#print(nb_samples)
			r_node_list = np.random.choice(node_list, nb_samples, p=node_weights, replace=False)
		else:
			raise Exception('the value must be between 0 and 1.')
	else:
		raise Exception('Unknown mode. Choose "constant" or "percent".')
	r_node_dic = {node:node_dic[node] for node in r_node_list}
	return r_node_dic

def spiky_ball(username_list, graph_handle, exploration_depth=4, 
				mode='percent',random_subset_size=None,spread_type='sharp'):
	""" Collect the tweets of the users and their mentions
		make an edge list user -> mention
		and save each user edge list to a file
	"""
	if graph_handle.rules:
		print('---')
		print('Parameters:')
		for key,value in graph_handle.rules.items():
			print(key,value)
		print('---')
	total_username_list = []
	#total_username_list += username_list
	new_username_list = username_list.copy()
	total_node_dic = {}
	new_node_dic = {node:1 for node in username_list}
	total_edges_df = pd.DataFrame()
	total_nodes_df = pd.DataFrame()
	for depth in range(exploration_depth):
		logger.debug('')
		logger.debug('******* Processing users at {}-hop distance *******'.format(depth))
		
		if spread_type == 'sharp':
			if not new_node_dic:
				break
			new_node_dic = random_subset(new_node_dic, mode=mode, random_subset_size=random_subset_size)
			# Avoid visiting twice the same edges
			# and remove the nodes already collected
			new_node_list = list(set(new_node_dic.keys()).difference(set(total_node_dic.keys()))) 
			total_node_dic = combine_dicts(total_node_dic,new_node_dic)

		elif spread_type == 'broad':
			total_node_dic = combine_dicts(total_node_dic,new_node_dic)
			new_node_dic = random_subset(total_node_dic, mode=mode, random_subset_size=random_subset_size)
			new_node_list = list(new_node_dic.keys())						
		else:
			raise Exception('Unknown spread type, use spread_type="sharp" or "broad".')

		new_node_dic, edges_df, nodes_df = process_hop(graph_handle, new_node_list)
		
		nodes_df['spikyball_hop'] = depth # Mark the depth of the spiky ball on the nodes
		total_edges_df = total_edges_df.append(edges_df)
		total_nodes_df = total_nodes_df.append(nodes_df)
	

	total_edges_df.reset_index(drop=True, inplace=True)
	total_nodes_df.reset_index(drop=True, inplace=True)
	total_node_list = list(total_node_dic.keys()) # set of unique nodes
	return total_node_list, total_nodes_df, total_edges_df


def save_data(nodes_df,edges_df,data_path):
	# Save to json file
	edgefilename = data_path + 'edges_data' + '.json'
	nodefilename = data_path + 'nodes_data' + '.json'
	print('Writing',edgefilename)
	edges_df.to_json(edgefilename)
	print('Writing',nodefilename)
	nodes_df.to_json(nodefilename)
	return None

def load_data(data_path):
	nodesfilename = data_path + 'nodes_data.json'
	edgesfilename =  data_path + 'edges_data.json'
	print('Loading',nodesfilename)
	nodes_df = pd.read_json(nodesfilename)
	print('Loading',edgesfilename)
	edges_df = pd.read_json(edgesfilename)
	return nodes_df,edges_df
