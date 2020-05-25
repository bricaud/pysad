
import networkx as nx
import pandas as pd

class graph:


	def __init__(self,graph):
		# Instantiate an object
		self.G = graph
		self.rules = {}
		self.rules['min_degree'] = 1


	def get_neighbors(self,node_id):
		# collect info on the node and its (out going) edges
		# return 2 dataframes, one with edges info and the other with the node info
		G = self.G
		if node_id not in G:
			return pd.DataFrame(),pd.DataFrame()
		# node data
		node_df = pd.DataFrame([{'source':node_id, **G.nodes[node_id]}])
		# Edges and edge data		
		if nx.is_directed(G):
			#inedges = G.in_edges(node_id, data=True)
			edges = G.out_edges(node_id, data=True)
		else:
			edges = G.edges(node_id, data=True)
		edgeprop_dic_list = []
		#nodeprop_dic_list = []
		for source,target,data in edges:
			edge_dic = {'source': source, 'target': target, **data}
			edgeprop_dic_list.append(edge_dic)
		edges_df = pd.DataFrame(edgeprop_dic_list)
		return node_df, edges_df

	def filter(self,node_df,edges_df):
		if len(edges_df) < self.rules['min_degree']:
			# discard the node
			node_df = pd.DataFrame()
			edges_df = pd.DataFrame()
		# filter the edges
		edges_df = self.filter_edges(edges_df)
		return node_df,edges_df

	def filter_edges(self,edges_df):
		#edges_g = self.group_edges(edges_df)	
		#users_to_remove = edges_g['mention'][edges_g['weight'] < self.rules['min_mentions']]
		# Get names of indexes for which column Age has value 30
		#indexNames = edges_df[ edges_df['user'].isin(users_to_remove) ].index
		# Delete these row indexes from dataFrame
		#edges_df.drop(indexNames , inplace=True)
		return edges_df

	def neighbors_list(self,edges_df):
		neighbors = edges_df['target'].unique().tolist()
		return neighbors


	def neighbors_with_weights(self, edges_df):
		node_list = self.neighbors_list(edges_df)
		node_dic = {}
		for node in node_list:
			node_dic[node] = len(edges_df) # degree of the node
		return node_dic


def reshape_node_data(nodes_df):
	nodes_df.set_index('source', inplace=True)
	return nodes_df

def reshape_edge_data(edge_df):
	edge_df.set_index(['source','target'], inplace=True)
	return edge_df