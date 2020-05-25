
import pandas as pd
import networkx as nx
import community
import numpy as np
import json
import glob
import pysad.collect

from tqdm import tqdm
#############################################################
# Functions for the graph of users
#############################################################

# def load_collected_data(data_path, graph_object='edge'):
#     data_df = pd.DataFrame()
#     if graph_object == 'node':
#         filestring = '_userinfo'
#     elif graph_object == 'edge':
#         filestring = '_mentions'
#     else:
#         print('Type unknown. graph_type only accept "node" or "edge".')
#         raise
			
#     for filename in tqdm(glob.glob(data_path + '*' + filestring + '*' + '.json')):
#         new_data_df = pd.read_json(filename)
#         #print('{} with {} tweets.'.format(filename,len(new_data_df)))
#         data_df = data_df.append(new_data_df)
#     data_df.reset_index(drop=True, inplace=True)
#     return data_df



def converttojson(edge_df):
	""" Check if column type is list or dict and convert it to json
		list or dict can not be saved using gexf or graphml format.
	"""
	edge_df_str = edge_df.copy()
	for idx,col in enumerate(edge_df.columns):
		first_row_element = edge_df.iloc[0,idx]
		if isinstance(first_row_element,list) or isinstance(first_row_element,dict):
			edge_df_str[col] = edge_df[col].apply(json.dumps)
			print('Field "{}" of class {} converted to json string'.format(col,type(first_row_element)))
		#else:
		#	print(col,type(edge_df[col][0]))
	return edge_df_str

#def aggregate_edges(edge_df):
	#TODO



def graph_from_edgeslist(edge_df, min_weight):
	print('Creating the graph from the edge list')
	# The indices in the dataframe are source and target for the edges
	edge_df = edge_df.rename_axis(['source','target']).reset_index()
	G = nx.from_pandas_edgelist(edge_df,source='source',target='target', create_using=nx.DiGraph)
	print('Nb of nodes:',G.number_of_nodes())
	return G

# def shape_attributes(G,data_dic):
# 	prop_list = []
# 	for propname, propdic in data_dic.items():
# 		prop = {}
# 		for key, value in propdic.items():
# 			if isinstance(value,list):
# 				prop[key] = json.dumps(value)
# 			else:
# 				prop[key] = value
# 		#nodevaluedic = {k: json.dumps(v) for k, v in nodevaluedic.items()}
# 		prop.append()
# 		nx.set_node_attributes(G,nodeprop,name=propname)
# 	return prop_list

def attributes_tojson(data_dic):
	for propname, propdic in data_dic.items():
		for key, value in propdic.items():
			if isinstance(value,list):
				data_dic[propname][key] = json.dumps(value)
			else:
				data_dic[propname][key] = value
	return data_dic

def add_node_attributes(G,node_df):
	node_dic = node_df.to_dict()
	node_dic = attributes_tojson(node_dic)
	for propname,propdic in node_dic.items():
		nx.set_node_attributes(G,propdic,name=propname)
	return G

def add_edges_attributes(G,edges_df):
	edge_dic = edges_df.to_dict()
	#edge_dic = attributes_tojson(edge_dic)
	for propname,propdic in edge_dic.items():
		nx.set_edge_attributes(G,propdic,name=propname)
	return G


def reduce_graph(G,degree_min):
	# Drop node with small degree
	remove = [node for node,degree in dict(G.degree()).items() if degree < degree_min]
	G.remove_nodes_from(remove)
	print('Nb of nodes after removing nodes with degree strictly smaller than {}: {}'.format(degree_min,G.number_of_nodes()))
	isolates = list(nx.isolates(G))
	G.remove_nodes_from(isolates)
	print('removed {} isolated nodes.'.format(len(isolates)))
	if G.is_directed():
		print('Warning: the graph is directed.')
	return G

def handle_spikyball_neighbors(G,graph_handle,remove=True):
	# Complete the info of the nodes not collected
	sp_neighbors = [node for node,data in G.nodes(data=True) if 'spikyball_hop' not in data]
	print('Number of neighbors of the spiky ball:',len(sp_neighbors))

	# 2 options: 1) remove the neighbors or 2) rerun the collection to collect the missing node info
	if remove == True:
		# Option 1:
		print('Removing spiky ball neighbors...')
		G.remove_nodes_from(sp_neighbors)
		print('Number of nodes after removal:',G.number_of_nodes())
	else:
		# Option 2: collect the missing node data
		print('Collecting info for neighbors...')
		new_nodes_founds, edges_df, nodes_df = pysad.collect.process_hop(graph_handle, sp_neighbors)
		G = add_node_attributes(G,nodes_df)
		sp_nodes_dic = {node:-1 for node in sp_neighbors}
		nx.set_node_attributes(G,sp_nodes_dic,name='spikyball_hop')
		print('Node info added to the graph.')
	# Check integrity
	i=0
	for node,data in G.nodes(data=True):
		if 'spikyball_hop' not in data:
			print('Missing information for node',node)
	return G

def detect_communities(G):
	#first compute the best partition
	if isinstance(G,nx.DiGraph):
		Gu = G.to_undirected()
	else:
		Gu = G
	partition = community.best_partition(Gu, weight='weight')
	nx.set_node_attributes(G,partition,name='community')
	print('Communities saved on the graph as node attributes.')
	nb_partitions = max(partition.values())+1
	print('Nb of partitions:',nb_partitions)
	# Create a dictionary of subgraphs, one per community
	community_dic = {}
	for idx in range(nb_partitions):
		subgraph = G.subgraph([key for (key,value) in partition.items() if value==idx])
		community_dic[idx] = subgraph
	#clusters_modularity = community.modularity(partition, Gu)
	return G, community_dic #,clusters_modularity

def remove_small_communities(G,community_dic,min_size):
	community_tmp = {k:v.copy() for k,v in community_dic.items()}
	nb_removed = 0
	for key in community_tmp:
		graph = community_tmp[key]
		if graph.number_of_nodes() <= min_size:
			G.remove_nodes_from(graph.nodes())
			nb_removed +=1
	print('removed {} community(ies) smaller than {} nodes.'.format(nb_removed, min_size))
	return G

#############################################################
## Functions for cluster analysis
#############################################################

def cluster_connectivity(G, weight='weight'):
	""" Compute the ratio nb of edges inside the community over nb edges pointing outside,
		for each community
	"""
	# 1) indexing the edges by community
	sum_edges_dic = { com : {} for com in range(G.nb_communities)}
	for node1,node2 in G.edges():
		comm1 = G.nodes[node1]['community']
		comm2 = G.nodes[node2]['community']
		if comm2 not in sum_edges_dic[comm1]:
			sum_edges_dic[comm1][comm2] = 0
			sum_edges_dic[comm2][comm1] = 0
		else:
			if weight == None:
				sum_edges_dic[comm1][comm2] += 1
				sum_edges_dic[comm2][comm1] += 1
			else:	
				sum_edges_dic[comm1][comm2] += G.edges[node1,node2][weight]
				sum_edges_dic[comm2][comm1] += G.edges[node1,node2][weight]
	c_connectivity = {}
	# 2) computing the connectivity
	for com in sum_edges_dic:
		in_out_edges = sum(sum_edges_dic[com].values())
		c_connectivity[com] = round(- np.log2(sum_edges_dic[com][com] / in_out_edges),3)   
	return c_connectivity



#############################################################
## Functions for Community data (comparing clusters)
#############################################################

def community_data(G):
	# get the hashtags for each community and inter-communities
	# return them in dics of dics 
	# import ast
	tags_dic = {}
	dates_dic = {}
	url_dic = {}
	text_dic = {}
	for node1,node2,data in G.edges(data=True):
		if node1 == node2:
			print('Self edge',node1)
		n1_com = G.nodes[node1]['community']
		n2_com = G.nodes[node2]['community']
		# Convert string to list
		#x = ast.literal_eval(data['hashtags'])
		#d = ast.literal_eval(data['date'])
		#u = ast.literal_eval(data['urls'])
		#keywords = [n.strip() for n in x]
		#date_list = [n.strip() for n in d]
		#urls = [n.strip() for n in u]
		keywords = json.loads(data['hashtags'])
		date_list = json.loads(data['date'])
		urls = json.loads(data['urls'])
		texts = json.loads(data['text'])

		# fill the dics of dics
		if n1_com not in tags_dic:
			tags_dic[n1_com] = {}
			dates_dic[n1_com] = {}
			url_dic[n1_com] = {}
			text_dic[n1_com] = {}
		if n2_com not in tags_dic[n1_com]:
			tags_dic[n1_com][n2_com] = keywords
			dates_dic[n1_com][n2_com] = date_list
			url_dic[n1_com][n2_com] = urls
			text_dic[n1_com][n2_com] = texts
		else:
			tags_dic[n1_com][n2_com] += keywords 
			dates_dic[n1_com][n2_com] += date_list
			url_dic[n1_com][n2_com] += urls
			text_dic[n1_com][n2_com] += texts
	return tags_dic, dates_dic, url_dic,text_dic

def compute_meantime(date_list):
	# return mean time and standard deviation of a list of dates in days
	# import numpy as np
	d_list = [ datetime.strptime(dt,'%Y-%m-%d %H:%M:%S') for dt in date_list]
	second_list = [x.timestamp() for x in d_list]
	meand = np.mean(second_list)
	stdd = np.std(second_list)
	return datetime.fromtimestamp(meand), timedelta(seconds=stdd)

def communities_date_hashtags(dates_dic, tags_dic):
	# Create a table with time and popular hashtags for each community
	# from collections import Counter
	comm_list = []
	nb_partitions = len(tags_dic.keys())
	for key in range(nb_partitions):
		most_common = Counter(tags_dic[key][key]).most_common(5)
		meandate,stddate = compute_meantime(dates_dic[key][key])
		#print('Community',key)
		#print(most_common)
		#print('Average date: {} and std deviation: {} days'.format(meandate.date(),stddate.days))
		comm_dic = {'Community':key, 'Average date':meandate.date(), 'Deviation (days)':stddate.days}
		for htag_nb in range(5): # filling the table with the hashtags
			if htag_nb < len(most_common):
				comm_dic['hashtag'+str(htag_nb)] = most_common[htag_nb][0]
			else:
				comm_dic['hashtag'+str(htag_nb)] = ''
		comm_list.append(comm_dic)
	community_table = pd.DataFrame(comm_list)
	return community_table

### Handling urls

def get_urls(url_df):
	# Dataframe with the urls of each cluster
	urltocomm = []
	for index_c, row in url_df.iterrows():
		for url in row['urls']:
			urltocomm.append([url,index_c,1])
	url_table = pd.DataFrame(urltocomm, columns=['url','Community','Occurence'])
	url_table = url_table.groupby(['url','Community']).agg(Occurence=('Occurence',sum))
	url_table = url_table.reset_index()
	return url_table

def communities_urls(url_dic):
	# Dataframe with the urls of each cluster
	urltocomm = []
	for key in url_dic:
		for url in url_dic[key][key]:
			urltocomm.append([url,key,1])
	url_table = pd.DataFrame(urltocomm, columns=['url','Community','Occurence'])
	url_table = url_table.groupby(['url','Community']).agg(Occurence=('Occurence',sum))
	url_table = url_table.reset_index()
	return url_table


def save_graph(graph,graphfilename):
	nx.write_gexf(graph,graphfilename)
	print('Graph saved to',graphfilename)
