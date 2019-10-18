from __future__ import print_function
import tweepy
import json
from pymongo import MongoClient

MONGO_HOST= 'mongodb://localhost/test'  # assuming you have mongoDB installed locally
                                        # and a database called 'test'
                                        # if 'test' doesn't exist, it will be created

# Change your hashtags here
WORDS = ['#brexit'] # This is an OR relation

# Insert your keys here
CONSUMER_KEY = "0DWf1zhR0otv2apbXEUOGklbA"
CONSUMER_SECRET = "6j96Dn4X1w7ujezPtGpkZlTr7jFjNZXTf10gxw08cOSmXywwB6"
ACCESS_TOKEN = "42145302-CQOGwdsFEyh7puMSSOmp46XDHvolmn4sd4LN8Gvx2"
ACCESS_TOKEN_SECRET = "ek7v59Se8EshBWCLxwbYkkhi8OLqp7YFfdT9avdSpbiln"


class StreamListener(tweepy.StreamListener):    
    #This is a class provided by tweepy to access the Twitter Streaming API. 

    def on_connect(self):
        # Called initially to connect to the Streaming API
        print("You are now connected to the streaming API.")
 
    def on_error(self, status_code):
        # On error - if an error occurs, display the error / status code
        print('An Error has occured: ' + repr(status_code))
        return False
 
    def on_data(self, data):
        #This is the meat of the script...it connects to your mongoDB and stores the tweet
        try:
            client = MongoClient(MONGO_HOST)
            
            # Use test database. If it doesn't exist, it will be created.
            db = client.test
    
            # Decode the JSON from Twitter
            datajson = json.loads(data)
            
            #grab the 'created_at' data from the Tweet to use for display
            created_at = datajson['created_at']
            username = datajson['user']['screen_name']

            #print out a message to the screen that we have collected a tweet
            print("Tweet collected at " + str(created_at) + " from user @" + username)
            
            #insert the data into the mongoDB into a collection called twitter_search
            #if twitter_search doesn't exist, it will be created.
            db.brexitSearch.insert(datajson)
        except Exception as e:
           print(e)

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
#Set up the listener. The 'wait_on_rate_limit=True' is needed to help with Twitter API rate limiting.
listener = StreamListener(api=tweepy.API(wait_on_rate_limit=True)) 
streamer = tweepy.Stream(auth=auth, listener=listener)
print("Tracking: " + str(WORDS))
streamer.filter(track=WORDS)


