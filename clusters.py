

import pandas as pd
import numpy as np
import networkx as nx
import preprocessor as tweetpre
from . import utils as pu

from nltk.corpus import stopwords
from string import punctuation
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

from collections import Counter

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
		if in_out_edges == 0 or sum_edges_dic[com][com] == 0:
			if sum_edges_dic[com][com] == 0:
				c_connectivity = 0
			else:
				c_connectivity = 1000
		else:
			c_connectivity[com] = round(- np.log2(sum_edges_dic[com][com] / in_out_edges),3)   
	return c_connectivity

def cluster_attributes(cluster_graph):
	""" Compute and store structure properties in the cluster (to nodes and edges)
	"""
	cg = cluster_graph
	nx.set_node_attributes(cg,dict(nx.degree(cg)),'degree')
	if cg.is_directed():
		nx.set_node_attributes(cg,dict(cg.in_degree()),'in degree')
		nx.set_node_attributes(cg,dict(cg.out_degree()),'out degree')
	nx.set_node_attributes(cg,dict(nx.degree(cg,weight='weight')),'degree_w')
	nx.set_node_attributes(cg,nx.betweenness_centrality(cg),'bcentrality')
	nx.set_node_attributes(cg,nx.pagerank(cg),'pagerank')
	if cg.is_directed():
		nx.set_node_attributes(cg,nx.pagerank(cg.reverse()),'pagerank inv')
	nx.set_edge_attributes(cg,nx.edge_betweenness_centrality(cg),'bcentrality')
	return cg

def compute_cluster_indicators(subgraph):
	gc = subgraph
	gc_size = gc.number_of_nodes()
	ck = nx.algorithms.core.core_number(gc)
	max_k = max(ck.values())
	kcurve = [len([key for (key,value) in ck.items() if value==idx]) for idx in range(max_k+1)]
	max_k_core_size = kcurve[-1]
	density = gc.number_of_edges()/(gc_size*(gc_size-1))
	mean_edge_weight = gc.size(weight='weight')/gc.number_of_edges()
	sink_list = [node for node, out_degree in gc.out_degree() if out_degree == 0]
	source_list = [node for node, in_degree in gc.in_degree() if in_degree == 0]
	info_dic = {'nb_nodes': gc_size, 'k_max':max_k, 'max_kcore':max_k_core_size,
				'norm_kcore':max_k_core_size/gc_size,'density': density,
				'activity_per_edge': mean_edge_weight, 'nb_influencers': len(sink_list),
				'nb anthousiasts': len(source_list),
				'hierarchy':max_k*1/(max_k_core_size/gc_size), 'hierarchy2': max_k**2/gc_size}
	return info_dic

def indicator_table(clusters_dic):
	comm_list = []
	for c in clusters_dic:
		gc = clusters_dic[c]
		comm_dic = {'Community': c}
		info_dic = compute_cluster_indicators(gc)
		comm_dic = {**comm_dic,**info_dic}
		comm_list.append(comm_dic)
	community_table = pd.DataFrame(comm_list)
	return community_table

def cluster_textandinfo(subgraph):
	user_text = {}
	hashtags = []
	date_list = []
	urls = []
	for node1,node2,data in subgraph.edges(data=True):
		if node1 == node2:
			print('Self edge',node1)
		hashtags += json.loads(data['hashtags'])
		date_list += json.loads(data['date'])
		urls += json.loads(data['urls'])
		texts = json.loads(data['text'])
		# associate text to writer
		if node1 not in user_text:
			user_text[node1] = texts
		else:
			user_text[node1] += texts
	return user_text, hashtags, date_list, urls


def community_tags_dic(tags_list,nb_tags=None):
	# Create a dict with popular hashtags for each community
	# from collections import Counter
	htag_dic = {}
	most_common = Counter(tags_list).most_common(nb_tags)
	if nb_tags is None: # take all the hashtags
		nb_tags = len(most_common)
	for htag_idx in range(nb_tags): # filling the table with the hashtags
		if htag_idx < len(most_common): 
			htag_dic['hashtag'+str(htag_idx)] = most_common[htag_idx][0]
		else: # less hashtags than required
			htag_dic['hashtag'+str(htag_idx)] = ''
	return htag_dic

def hashtag_count_table(tags_list):
	# Create a table with hashtags and their count
	# from collections import Counter
	htag_list = []
	most_common = Counter(tags_list).most_common()
	for htag_idx in range(len(most_common)): # creting a list of dic with the hashtags
		htag_dic = {'hashtag': most_common[htag_idx][0], 'count': most_common[htag_idx][1]}
		htag_list.append(htag_dic)
	htag_table = pd.DataFrame(htag_list)
	return htag_table

def community_date_stats(dates_list):
	# Create a dict with mean time and deviation
	meandate,stddate = compute_meantime(dates_list)
	date_dic = {'Average date':meandate.date(), 'Deviation (days)':stddate.days}
	return date_dic


def dates_tags_table(clusters_dic):
	comm_list = []
	for c in clusters_dic:
		gc = clusters_dic[c]
		comm_dic = {'Community': c}
		user_text, hashtags, date_list, urls = cluster_textandinfo(gc)
		hash_dic = community_tags_dic(hashtags,nb_tags=5)
		date_dic = community_date_stats(date_list)
		comm_dic = {**comm_dic, **date_dic, **hash_dic, 'urls': urls, 'text': user_text}
		comm_list.append(comm_dic)
	community_table = pd.DataFrame(comm_list)
	return community_table		



def count_order_items(item_list,item_name):
	dic_list = []
	most_commons = Counter(item_list).most_common()
	for item_idx in range(len(most_commons)): # creating a list of dic with the hashtags
		item_dic = {item_name: most_commons[item_idx][0], 'count': most_commons[item_idx][1]}
		dic_list.append(item_dic)
	item_table = pd.DataFrame(dic_list)
	return item_table

def tokenize(text):
	#from nltk import word_tokenize
	#from nltk.corpus import stopwords
	stop_words = stopwords.words('french') + list(punctuation)
	words = word_tokenize(text)
	words = [w.lower() for w in words]
	return [w for w in words if w not in stop_words and not w.isdigit()]

def most_common_words(text_table):
	""" Requires nltk
	"""
	fulltext = ''
	for text in text_table:
		fulltext += ' ' + text
	
	tktext = tokenize(fulltext)
	word_table = count_order_items(tktext,'word')
	# Calculate frequency distribution
	#fdist = nltk.FreqDist(tktext)
	#return fdist.most_common()
	return word_table



def extract_info_from_cluster_table(cluster_edge_table):
	text_list = []
	htag_list = []
	url_list = []
	for index,row in cluster_edge_table.iterrows():
		username = row['user']
		#if not 'tweets' in row:
		#	continue # pass to the next user
		tweet_df = pd.read_json(row['tweets'])
		
		for idx,tweet in tweet_df.iterrows():
			htags = tweet['hashtags']
			urls = tweet['urls']
			text = tweet['text']
			retweet_count = tweet['retweet_count']
			favorite_count = tweet['favorite_count']

			parsed_tweet = tweetpre.parse(text)
			# extract emojis
			emojis = []
			if parsed_tweet.emojis is not None:
				emojis = [emo.match for emo in parsed_tweet.emojis]
			tweetpre.set_options(tweetpre.OPT.MENTION, tweetpre.OPT.URL)
			filtered_text = tweetpre.clean(text)
			tweetpre.set_options()
			#emojis = parsed_tweet.emojis.match ???
			url_c = [pu.convert_bitly_url(url_string) for url_string in urls]
			text_list.append({'text': text, 'user': username, 'url': url_c , 'emojis':emojis , 
				'retweet_count':retweet_count, 'favorite_count': favorite_count, 'filtered text': filtered_text, 
				'bcentrality': row['bcentrality']})
			htag_list += htags
			url_list += urls
	if not text_list:
		empty_df = pd.DataFrame()
		return {'text': empty_df, 'hashtags': empty_df, 'words': empty_df, 'urls': empty_df}
	text_df = pd.DataFrame(text_list)
	mostcommon_words_df = most_common_words(text_df['filtered text'])
	hashtags_df = count_order_items(htag_list,'hashtag')
	url_df = count_order_items(url_list,'url')
	url_df = pu.convert_bitly_table(url_df)
	filtered_url_df = pu.drop_twitter_urls(url_df)
	return {'text': text_df, 'hashtags': hashtags_df, 'words': mostcommon_words_df, 'urls': filtered_url_df}


def cluster_tables(cluster_graph):
	edge_data_list = []
	cluster_users_df = pd.DataFrame.from_dict(dict(cluster_graph.nodes(data=True)),orient='index').sort_values('pagerank',ascending=False)
	cluster_users_df = cluster_users_df.drop('community',axis=1)
	cluster_users_df = cluster_users_df.reset_index().rename(columns={'index':'username'})
	for node1,node2,data in cluster_graph.edges(data=True):
		# Add writer
		data['user'] = node1
		edge_data_list.append(data)
	cluster_edge_info = pd.DataFrame(edge_data_list)
	cluster_edge_info = cluster_edge_info.sort_values('bcentrality',ascending=False)
	table_dic = extract_info_from_cluster_table(cluster_edge_info)
	return {'users': cluster_users_df, **table_dic}

def clutersprop2graph(G, cluster_info_dic, clusters):
	G.graph['clusters'] = {}
	for c_id in cluster_info_dic:
		if not cluster_info_dic[c_id]:
			continue
		cluster_info = {}
		info_table = cluster_info_dic[c_id]['info_table']
		#info_table['keywords'] = keyword_dic[c_id]
		cluster_info['hashtags'] = info_table['hashtags']['hashtag'].to_list()
		cluster_info['keywords'] = info_table['keywords']['keyword'].to_list()
		#cluster_info['keywords'] = keywords_dic[c_id]['keyword'].to_list()
		cluster_info['urls'] = info_table['urls']['url'].to_list()
		cluster_indicators = compute_cluster_indicators(clusters[c_id])
		cluster_info['indicators'] = cluster_indicators

		G.graph['clusters'][c_id] = cluster_info		
	return G


#############################################################
## Text processing from clusters info
#############################################################

def get_corpus(clusters_dic):
	""" Return a corpus with one document per cluter.
		Each document contains all the cluster tweets concatenated.
	"""
	corpus = []
	for c_id in clusters_dic:
		document = ''
		if clusters_dic[c_id]:
			table_dic = clusters_dic[c_id]['info_table']
			if 'filtered text' not in table_dic['text']:
				document = ''
			else:
				tweet_texts = table_dic['text']['filtered text']
				for text in tweet_texts: # concatenate tweets
					document += text + ' '
		corpus.append(document)
	return corpus

def tfidf(corpus,max_keywords=20):
	# from sklearn.feature_extraction.text import TfidfVectorizer
	corpus_len = len(corpus)
	vectorizer = TfidfVectorizer(max_df=corpus_len//2)
	X = vectorizer.fit_transform(corpus)
	keywords_dic = {}
	for idx in range(corpus_len):
		# place tf-idf values in a pandas data frame
		df = pd.DataFrame(X[idx].T.todense(), index=vectorizer.get_feature_names(), columns=["tfidf"])
		df = df.sort_values(by=["tfidf"],ascending=False).head(max_keywords)
		keywords_dic[idx] = df.reset_index().rename(columns={'index':'keyword'})
	return keywords_dic




#################
## Saving
################

def save_excel(table_dic,filename, table_format='full'):
	#import pandas.io.formats.excel
	#pandas.io.formats.excel.header_style = None

	with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:

		if table_format is 'full':
			for tablename in table_dic:
				col_size_dic = {'B:E':25, 'A:A':100}
				write_excel_sheet(table_dic[tablename],tablename,writer,col_size_dic)
		else:
			#table_dic = excel_reshape(table_dic)
			tablename = 'cluster'
			col_size_dic = {'A:B':10, 'C:C': 15,'D:E':20, 'F:F':20, 'G:G': 20, 'H:H':25 ,'J:J': 25}
			write_excel_sheet(table_dic[tablename],tablename,writer,col_size_dic)
			tablename = 'tweets'
			col_size_dic = {'A:A':100, 'B:B':20, 'C:D':50, 'E:E':25}
			if 'filtered text' in table_dic[tablename].columns:
				table_dic[tablename].drop(labels='filtered text',axis=1, inplace=True)
			write_excel_sheet(table_dic[tablename],tablename,writer,col_size_dic)
			tablename = 'indicators'
			col_size_dic = {'A:B':10, 'C:C': 15,'D:I':20}
			write_excel_sheet(table_dic[tablename],tablename,writer,col_size_dic)
			tablename = 'users'
			col_size_dic = {'A:B':10, 'C:C': 15,'D:I':20}
			write_excel_sheet(table_dic[tablename],tablename,writer,col_size_dic)
		#for tablename in table_dic:
		#	write_excel_sheet(table_dic[tablename],tablename,writer,col_size_dic)
			#table_dic[tablename].to_excel(writer, sheet_name=tablename, index=False)
			#worksheet = writer.sheets[tablename]
			#worksheet.set_column('B:E',column_width,format1)
			#worksheet.set_column('A:A',100,format1)
	print('Data saved to',filename)		
	 

def write_excel_sheet(df,sheet_name,writer,column_size_dic):
	df.to_excel(writer, sheet_name=sheet_name, index=False)
	workbook  = writer.book
	# Add a header format.
	format1 = workbook.add_format({
			#'bold': True,
			'text_wrap': True,
			#'valign': 'top',
			#'fg_color': '#D7E4BC',
			'border': 1}) 
	# column sizes
	worksheet = writer.sheets[sheet_name]
	for column in column_size_dic:
		column_width = column_size_dic[column]
		worksheet.set_column(column,column_width,format1)

def excel_reshape(table_dic):
	# Sort table_dic
	hashtags_keywords = pd.concat([table_dic['cluster'],
		table_dic['hashtags'],table_dic['keywords']],axis=1)
	tweets = table_dic['text']
	table_dic_s = {'cluster': table_dic['cluster'],'hashtags': hashtags_keywords,
					'text': tweets}
	return table_dic_s

