#Saman
#Nathan
#Daniel
#Pip
#Georgios
#Abraham
#Dume

#Install Tweepy
!pip install tweepy
#Installation of graph interface
!pip install matplotlib


import re
import tweepy

#importing the graph interface
import matplotlib.pyplot as plt

from tweepy import OAuthHandler
from datetime import datetime
from pytz import timezone



#Key inputs from Twitter
consumer_api_key = 'WnOse4KW0dwPNcrYtymJ11jMQ'
consumer_api_secret = 's9DhKqHASJISJETy462fupugBRMRDoTzRxvUfLJCU4N5fe3Jfl' 
access_token = '1315961217740091392-ZXiF60BdxeQ8irqRAItWyZ567b0HDQ'
access_token_secret ='wL2NRBt7sl3ldb8eVHjjxGEZgY2jgqslZOPROcQiG3U0L'

#Create authorizer file
authorizer = OAuthHandler(consumer_api_key, consumer_api_secret)
authorizer.set_access_token(access_token, access_token_secret)

#Get recent tweet and find tweets with search terms
api = tweepy.API(authorizer ,timeout=15)
all_tweets = []
SearchValue = ""

tweet_incoming = api.search(q="@Covid_16_Bot", count=1)

final_query = tweet_incoming[0].text.replace("@Covid_16_Bot ","").strip()

if "OR" in final_query or "or" in final_query:
  SearchValue = final_query.split(" OR ")
else:
  SearchValue = [final_query]

for ix in SearchValue:
  for tweet_object in tweepy.Cursor(api.search,q=ix+" -filter:retweets",lang='en',result_type='recent').items(200):
    all_tweets.append(tweet_object.text)

print(all_tweets)


#imports and getting dataset from a sentimental analysis done by Kaggle on tweets relating to airlines in US
import numpy as np 
import pandas as pd 
import re  
import nltk # an amazing library to play with natural language
nltk.download('stopwords')  
from nltk.corpus import stopwords 

tweets = pd.read_csv("https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv")
print(tweets)

X = tweets.iloc[:, 10].values  
y = tweets.iloc[:, 1].values

print(X)

processed_tweets = []

#Removes formatting
for tweet in range(0, len(X)):  
    # Remove all the special characters
    processed_tweet = re.sub(r'\W', ' ', str(X[tweet]))
 
    # remove all single characters
    processed_tweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_tweet)
 
    # Remove single characters from the start
    processed_tweet = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_tweet) 
 
    # Substituting multiple spaces with single space
    processed_tweet= re.sub(r'\s+', ' ', processed_tweet, flags=re.I)
 
    # Removing prefixed 'b'
    processed_tweet = re.sub(r'^b\s+', '', processed_tweet)
 
    # Converting to Lowercase
    processed_tweet = processed_tweet.lower()
 
    processed_tweets.append(processed_tweet)

#Sentimental analysis
from sklearn.feature_extraction.text import TfidfVectorizer  
tfidfconverter = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))  
X = tfidfconverter.fit_transform(processed_tweets).toarray()

from sklearn.ensemble import RandomForestClassifier
text_classifier = RandomForestClassifier(n_estimators=100, random_state=0)  
text_classifier.fit(X, y)

PositiveCount = 0
NeutralCount = 0
NegativeCount = 0

#Remove formating from tweets
for tweet in all_tweets:
    # Remove all the special characters
    processed_tweet = re.sub(r'\W', ' ', tweet)
 
    # remove all single characters
    processed_tweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_tweet)
 
    # Remove single characters from the start
    processed_tweet = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_tweet) 
 
    # Substituting multiple spaces with single space
    processed_tweet= re.sub(r'\s+', ' ', processed_tweet, flags=re.I)
 
    # Removing prefixed 'b'
    processed_tweet = re.sub(r'^b\s+', '', processed_tweet)
 
    # Converting to Lowercase
    processed_tweet = processed_tweet.lower()
 
    sentiment = text_classifier.predict(tfidfconverter.transform([ processed_tweet]).toarray())
    print(processed_tweet ,":", sentiment)

#Count of negative/positive/neutral
    if "neutral" in sentiment:
      NeutralCount += 1
    elif "negative" in sentiment:
      NegativeCount += 1
    else:
      PositiveCount += 1

#Percentage
PosPer = PositiveCount/len(all_tweets)*100
NegPer = NegativeCount/len(all_tweets)*100
NeuPer = NeutralCount/len(all_tweets)*100

#Prints out the sentiment analysis of the tweets
print("Negative Count:", NegativeCount, "- {:.2f}%".format(NegPer))
print("Neutral Count:", NeutralCount, "- {:.2f}%".format(NeuPer))
print("Positive Count: ", PositiveCount, "- {:.2f}%".format(PosPer))

#Creating pie chart using mathplotlib
labels = "Negative", "Positive", "Neutral"
sections = [NegPer, PosPer, NeuPer]
colors = ["r", "g", "y"]

plt.pie(sections, labels=labels, colors=colors,
        startangle=90,
        explode = (0, 0.1, 0),
        autopct = "%1.2f%%")

#creates the pi chart
plt.title("Type of Tweets")
plt.savefig("graph.png")  #Saves so it can be tweeted later
plt.show()

#Finds the exact date and time for the tweet
fmt = "%Y-%m-%d %H:%M:%S %Z%z"
now_utc = datetime.now(timezone('UTC'))

now_uk = now_utc.astimezone(timezone('Europe/London'))
print(now_uk.strftime(fmt))

#creates the text to be shown in our tweet
ourtweet =  'Search: "' + final_query + '"  @ ' + now_uk.strftime(fmt) + '\n' + \
            'Positive Count: {} - {:.2f}%\n'.format(PositiveCount, PosPer) + \
            'Negative Count: {} - {:.2f}%\n'.format(NegativeCount, NegPer) + \
            'Neutral Count: {} - {:.2f}%\n\n#UoE2020LovecaceChallengeWeek'.format(NeutralCount, NeuPer)


#Replies to incoming tweet
for s in tweet_incoming:
  sn = s.user.screen_name
  m = "@{}\n{}".format(sn, ourtweet)
  api.update_with_media("graph.png", m, s.id)
