
import os
import requests


####################################################################
# Other utils
####################################################################

def initialize_folder(path_folder_list, erase=True):
	# create or clean the path
	# import os
	folder_concat = ''
	for folder in path_folder_list[:-1]:
		folder_concat += folder + '/'
		if not os.path.isdir(folder_concat):
			os.mkdir(folder_concat)
			print('Path created:',folder_concat)
	# Special treatment for the last folder
	folder_concat += path_folder_list[-1] + '/'
	if not os.path.isdir(folder_concat):
		os.mkdir(folder_concat)
		print('Path created:',folder_concat)
	elif erase==True:
		for f in os.listdir(folder_concat):
			os.remove(os.path.join(folder_concat, f))
		print('Cleaned path',folder_concat)
	return folder_concat


def convert_bitly_url(url_string):
	# return the true URL from the bit.ly one
	#import requests

	session = requests.Session()  # so connections are recycled

	if 'bit.ly' in url_string:
		try:
			resp = session.head(url_string, allow_redirects=True)
			return resp.url
		except requests.exceptions.RequestException as e: 
			print(' exception raised for url', url_string)
			print(e)
			return url_string
	return url_string


def convert_bitly_table(url_table):
	# create a dataframe of correspondances for bit.ly urls 
	import requests

	session = requests.Session()  # so connections are recycled

	for index, row in url_table.iterrows():
		url = row['url']
		if 'bit.ly' in url:
			try:
				resp = session.head(url, allow_redirects=True)
				url_table.loc[index,'url_c'] = resp.url
			except requests.exceptions.RequestException as e:  # This is the correct syntax
				print(' exception raised for url',url)
				print(e)
				url_table.loc[index,'url_c'] = url
		else:
			url_table.loc[index,'url_c'] = url
	return url_table

def drop_twitter_urls(url_table):
	# Drop the references to twitter web site
	if url_table.empty:
		return url_table
	twitterrowindices = url_table[url_table['url_c'].str.contains('twitter.com')].index
	return url_table.drop(twitterrowindices)

