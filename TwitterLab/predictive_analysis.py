# %% Imports
import statistics

import pandas as pd
from pymongo import MongoClient
from sklearn import metrics, neighbors, svm, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection._search import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import paramselector as psel
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# %% Establish connection with database
client = MongoClient()
db = client.test
col = db.brexitSearch

# %%
#######################################################
# Retrieve data from the mongodb database, choosing
# the fields you'll need afterwards
#######################################################
brexitTweets = db.brexitSearch.find({"$and": [{"lang": "en"}, {"osm_location": {"$ne": None}}]},
                                    {'lang': 1, '_id': 1, 'text': 1, 'entities.hashtags': 1,
                                     'in_reply_to_status_id': 1, 'is_quote_status': 1, 'retweeted_status': 1, 'user.screen_name': 1,
                                     'user.location': 1, 'geo': 1, 'osm_location': 1, 'user.statuses_count': 1, 'user.verified': 1,
                                     'user.followers_count': 1, 'user.created_at': 1, 'user.verified': 1,
                                     'retweeted_status.id_str': 1})
tweets = list(brexitTweets)

# %%
#######################################################
# Transfering Tweets to a Pandas dataframe, for easier
# manipulation.
#######################################################

df = pd.DataFrame()

items = [tweet['entities'].get('hashtags') for tweet in tweets]
hashtags = list()
for item in items:
    hashtags.append(' '.join([h['text'].lower()
                              for h in item if h['text'].lower() != 'brexit']))

df['Hashtags'] = hashtags
df['UserStatusCount'] = [tweet['user']['statuses_count'] for tweet in tweets]
df['UserVerified'] = [tweet['user']['verified'] for tweet in tweets]
df['TweetText'] = [tweet['text'] for tweet in tweets]
df['UserFollowersCount'] = [tweet['user']['followers_count']
                            for tweet in tweets]
df['UserLocation'] = [tweet['osm_location'] for tweet in tweets]
#df['UserCreatedAt'] = [tweet['user']['created_at'] for tweet in tweets]

df['UserVerified'] = pd.Categorical(df['UserVerified'])
df = pd.concat(
    [df, pd.get_dummies(df['UserVerified'], prefix='UserVerified_')], axis=1)


df['UserLocation'] = pd.Categorical(df['UserLocation'])
df = pd.concat(
    [df, pd.get_dummies(df['UserLocation'], prefix='UserLoc_')], axis=1)

# Keeping only twitters that have more than one hashtag (!= brexit)
df = df[df['Hashtags'] != '']


# %%
compound_score = list()
sentiments = list()
analyser = SentimentIntensityAnalyzer()
for idx, row in df.iterrows():
    score = analyser.polarity_scores(row['TweetText'])['compound']
    if(score > 0):
        sentiments.append('Positive')
    elif(score < 0):
        sentiments.append('Negative')
    else:
        sentiments.append('Neutral')

    compound_score.append(score)

#df['CompoundScore'] = compound_score
df['TextSentiment'] = sentiments

df = df.drop('TweetText', axis=1)
df = df.drop('UserVerified', axis=1)
df = df.drop('UserVerified__False', axis=1)
df = df.drop('UserLocation', axis=1)
df = df.drop('UserStatusCount', axis=1)

# %%
# df['TextSentiment'].value_counts().plot(kind='bar')

#Define the count vectorizer using default parameters
cv1 = CountVectorizer()
#Apply the count vectorizer for the bag of words to the dataframe feature
cv1_text = cv1.fit_transform(df['Hashtags'])
#This is the feature space for TweetText
print('Shape:', cv1_text.shape)
#This is the type of matrix that is returned for TweetText
print('Type:', type(cv1_text))

cv_df = pd.DataFrame(cv1_text.toarray(), columns = cv1.get_feature_names())
cv_df.reset_index(drop=True,inplace=True)
# %%
skf = StratifiedKFold(n_splits=10)
y=df.TextSentiment
df = df.drop('TextSentiment', axis=1).drop('Hashtags', axis=1)
df.reset_index(drop=True, inplace=True)
X = pd.concat([df, cv_df], axis=1)
#%% Run model

cv_scores = []
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model = RandomForestClassifier(criterion='entropy', max_depth=20, 
                                   max_features='sqrt', n_estimators= 600)

    # model = svm.SVC(C = 1, gamma = 0.01, kernel='rbf')
    print(model)
    model.fit(X_train, y_train)

    # make predictions
    clf1_expected = y_test
    clf1_predicted = model.predict(X_test)
    score = model.score(X_test, y_test)
    # summarize the fit of the model
    print("accuracy: " + str(metrics.accuracy_score(clf1_expected, clf1_predicted)))
    print(metrics.classification_report(clf1_expected, clf1_predicted))
    cv_scores.append(score)

print(f'Avg Accuracy: {sum(cv_scores) / len(cv_scores)}')

#%%
# rndFC = RandomForestClassifier()

# param_grid = { 
#     'n_estimators': [100,200, 300, 400],
#     'max_features': ['auto', 'sqrt', 'log2'],
#     'max_depth' : [1, 3, 7, 13, 20],
#     'criterion' :['gini', 'entropy']
# }

# clf, bestparams  = psel.param_selection(rndFC,param_grid, X, y, 2, print_averages=False)

# %%
