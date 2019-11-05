# -*- coding: utf-8 -*-
from __future__ import division
from pymongo import MongoClient
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import operator
import geocoder as gc
import requests

# Establish connection with database
client = MongoClient()
db = client.test
col = db.brexitSearch

#######################################################
# Retrieve data from the mongodb database, choosing
# the fields you'll need afterwards
#######################################################
my_tweets = db.brexitSearch.find({},{'lang':1, '_id':1, 'text':1, 'entities.hashtags':1,
'in_reply_to_status_id':1, 'is_quote_status':1, 'retweeted_status':1, 'user.screen_name':1
, 'user.location':1, 'geo':1, 'osm_location':1})
numTweets = db.brexitSearch.count()

####################################################
# Plot of Languages (autodetected by Twitter)
####################################################
langsList = []
for t in my_tweets:
	langsList.append(t['lang'])

D = Counter(langsList) 
D_sorted = sorted(dict(D).iteritems(), key=operator.itemgetter(1), reverse=True)
# ----------- Bar Plot ------------------------
plt.bar(range(len(D_sorted)), [val[1] for val in D_sorted], align='center')
plt.xticks(range(len(D_sorted)), [val[0] for val in D_sorted])
plt.title('Languages spoken in the tweets captured')
plt.show()

##############################################################
# Plot how many of them are retweets, replies,
# quotations or original tweets
##############################################################
my_tweets.rewind() #Reset cursor
retweets = 0
replies = 0
quotations = 0
originals = 0
for t in my_tweets:
	if t.get('retweeted_status') is not None:
		retweets=retweets+1
	elif t['is_quote_status'] is not False:
		quotations = quotations+1
	elif t.get('in_reply_to_status_id') is not None:
		replies = replies+1
	else:
		originals = originals+1

# ----------- Pie Chart ------------------------
labels = 'Original Content', 'Retweets', 'Quotations', 'Replies'
sizes = [originals, retweets, quotations, replies]
frequencies = [x/numTweets for x in sizes]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0.1, 0, 0, 0)  # explode 1st slice
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
		autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.title('Percentage of Tweets depending on how the content is generated')
plt.show()


##################################################################
# Plot secondary hashtags
##################################################################
my_tweets.rewind()
hashList = []
for t in my_tweets:
    for e in t['entities']['hashtags']:
        h = e['text']
        if h.lower() != 'brexit':
			hashList.append(h.lower())
D = Counter(hashList)
subset = dict(D.most_common(15))
sorted_subset = sorted(subset.iteritems(), key=operator.itemgetter(1))

# ----------- Horizontal Bar Plot ------------------------
pos = range(len(sorted_subset))
plt.barh(pos, [val[1] for val in sorted_subset], align = 'center', color = 'yellowgreen')
plt.yticks(pos, [val[0] for val in sorted_subset])
plt.title('Top 15 of hashtags captured')
plt.tight_layout()
plt.show()

##################################################################
# Plotting most appeared countries
##################################################################

##################################################################
# START - Getting information from geolocation on user's profile
##################################################################
my_tweets.rewind()
locations = []
tweets_without_location = []
for t in my_tweets:
	if 'osm_location' not in t.keys():
		tweets_without_location.append(t)
		if t['user']['location'] != None:
			locations.append(t['user']['location'])

geocoded = dict()
loc_count = len(locations)
with requests.Session() as session:
	i = 1
	for location in locations:
		print('Checking location %s. (%s/%s)' %  (location, i, loc_count))
		if location.lower() not in geocoded.keys():
			g = gc.osm(location)
			if (g != None and g.ok):
				geocoded[location.lower()] = g

		i = i + 1

for t in tweets_without_location:
	try:
		if t['user']['location'] == None or t['user']['location'].lower() not in geocoded.keys() or geocoded[t['user']['location'].lower()].current_result.country_code == None:
			db.brexitSearch.update({"_id" : t['_id'] }, {"$set" : {"osm_location": None}})
		else:
			db.brexitSearch.update({"_id" : t['_id'] }, {"$set" : {"osm_location": geocoded[t['user']['location'].lower()].current_result.country_code.upper()}})
	except: pass
##################################################################
# END - Getting information from geolocation on user's profile
##################################################################


my_tweets.rewind()
countries = []
for t in my_tweets:
	country = t['osm_location']
	if country != None:
		countries.append(country.upper())

D = Counter(countries)
subset = dict(D.most_common(15))
sorted_subset = sorted(subset.iteritems(), key=operator.itemgetter(1))

# ----------- Horizontal Bar Plot ------------------------
pos = range(len(sorted_subset))
plt.barh(pos, [val[1] for val in sorted_subset], align = 'center', color = 'yellowgreen')
plt.yticks(pos, [val[0] for val in sorted_subset])
plt.title('Top 15 of countries captured')
plt.tight_layout()
plt.show()
