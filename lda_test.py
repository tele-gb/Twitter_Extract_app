import nltk
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords
import numpy as np
import sklearn
import matplotlib.pyplot as plt
#%matplotlib inline
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy
import re
import string
import sqlite3
from sqlite3 import Error
pd.options.display.max_colwidth
#nlp packages
#remove stop words
import re
import nltk
from nltk.corpus import stopwords
# For sending GET requests from the API
import requests
# For saving access tokens and for file management when creating and adding to the dataset
import os
# For dealing with json responses we receive from the API
import json
# For displaying the data after
import pandas as pd
# For saving the response data in CSV format
import csv
# For parsing the dates received from twitter in readable formats
import datetime
import dateutil.parser
import unicodedata
#To add wait time between requests
import time
from google.cloud import bigquery

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF

client = bigquery.Client()
sql = """
        SELECT
        *
        FROM `twitter-bank-sentiment.twitter_bank_sent.tweets_main` 


"""

df = client.query(sql).to_dataframe()

print(df.head())

#PreProcessing on tweets
#remove stop words
import re
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

#Remove any rows with a "nan" in them
df = df.dropna(axis=0, how = 'any')

#Make it so that any non readable text gets converted into nothing
def removetext(text):
    return ''.join([i if ord(i) < 128 else '' for i in text])

#Here I am doing the actual removing
df['text'] = df['text'].apply(removetext)

#remove @monzo etc
def clean_tweet_handles(raw_tweet):
    handles_only = re.sub(r"()@\w+",r"\1",raw_tweet)
    #print(handles_only)
    words = handles_only.lower().split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops] 
    return( " ".join( meaningful_words )) 

#remove URLS
def remove_urls(raw_tweet):
    tweet = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', raw_tweet,flags=re.MULTILINE)
#     re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE
    #print(handles_only)
    words = tweet.lower().split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops] 
    return( " ".join( meaningful_words )) 

def tweet_to_words(raw_tweet):
    letters_only = re.sub("[^a-zA-Z]", " ",raw_tweet) 
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops] 
    return( " ".join( meaningful_words )) 

def stemming(raw_tweet):
    letters_only = re.sub("[^a-zA-Z]", " ",raw_tweet) 
    words = letters_only.lower().split()   
    stops = set(stopwords.words("english"))      
    ps = PorterStemmer()
    stemmed_words = [ps.stem(w) for w in words if not w in stops]
    return( " ".join( stemmed_words)) 
    
def lemmatizer(raw_tweet):
    letters_only = re.sub("[^a-zA-Z]", " ",raw_tweet) 
    words = letters_only.lower().split()   
    stops = set(stopwords.words("english"))      
#     ps = PorterStemmer()
#     stemmed_words = [ps.stem(w) for w in words if not w in stops]   
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in words if not w in stops]
    return( " ".join( lemma_words)) 

#removing links

def clean_tweet_length(raw_tweet):
    letters_only = re.sub("[^a-zA-Z]", " ",raw_tweet) 
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops] 
    return(len(meaningful_words)) 

df_clean=df
df_clean['no_handles']=df_clean['text'].apply(lambda x: clean_tweet_handles(x))
df_clean['remove_urls']=df_clean['no_handles'].apply(lambda x: remove_urls(x))
df_clean['clean_tweet']=df_clean['remove_urls'].apply(lambda x: tweet_to_words(x))
df_clean['stemmed_tweet']=df_clean['clean_tweet'].apply(lambda x: stemming(x))
df_clean['lemmatizer_tweet']=df_clean['clean_tweet'].apply(lambda x: lemmatizer(x))
df_clean['Tweet_length']=df_clean['text'].apply(lambda x: clean_tweet_length(x))

print(df_clean.head(10))

from joblib import dump, load
#local loads
# clf = load('C:/Dev_And_Apps/Twitter_Extract_app/Trained_Models/LDA_Model.joblib') 
# savevector = load("C:/Dev_And_Apps/Twitter_Extract_app/Trained_Models/vectorizer.joblib")

#vm loads
clf = load('/home/kryz_wosik/Twitter_extract_app/Trained_Models/LDA_Model.joblib') 
# load(/kryz_wosik/Twitter_extract_app/Trained_Models/LDA_Model.joblib)
savevector = load("/home/kryz_wosik/Twitter_extract_app/Trained_Models/vectorizer.joblib")

#recall model
# the dataset to predict on (first two samples were also in the training set so one can compare)
data_samples = df_clean['lemmatizer_tweet'].tolist()
#print(data_samples)

# Vectorize the training set using the model features as vocabulary

# transform method returns a matrix with one line per document, columns being topics weight
predict = clf.transform(savevector.transform(data_samples))
#print(predict)

topic_pred = []
for n in range(predict.shape[0]):
    topic_most_pr = predict[n].argmax()
#     print("doc: {} topic: {}\n".format(n,topic_most_pr))
    topic_pred.append(topic_most_pr)
#     print(topic_pred)

# predictdf = pd.DataFrame()
# predictdf['text'] = data_samples
# predictdf['topic'] = topic_pred
# print(predictdf.head())

df_clean['topic_predict'] = topic_pred
print(df_clean.head())

def append_data_from_para(bq_client, dataset, table_name, file_path, file_name):
    """
    Ingest data to BQ table from CSV file
    """

    dataset_ref = bq_client.dataset(dataset)
    table_ref = dataset_ref.table(table_name)
    
    # try:
    job_config = bigquery.LoadJobConfig(source_format=bigquery.SourceFormat.PARQUET)


    full_file_path = os.path.join(file_path, file_name)
    with open(full_file_path, "rb") as source_file:
        job = bq_client.load_table_from_file(source_file, table_ref, job_config=job_config)

    job.result()  # Waits for table load to complete.

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

#-----sentiment model
print(test_clean_tweet[0:2])
#load sentiment models
clfsent = load('/home/kryz_wosik/Twitter_extract_app/Trained_Models/sent_Model.joblib') 
sentvector = load("/home/kryz_wosik/Twitter_extract_app/Trained_Models/sentvectorizer.joblib")

#process data
test_clean_tweet=[]
for tweet in df_clean['lemmatizer_tweet']:
    test_clean_tweet.append(tweet)
    
#transform the features
new_features=sentvector.transform(test_clean_tweet)
predict = clfsent.predict(new_features)

df_clean['sent_predict']=predict
PRINT(df_clean.head(2))

# df_clean.to_parquet('bq_load.gzip',compression="gzip")

# append_data_from_para(client,"twitter_bank_sent","tweets_topic_test","./","bq_load.gzip")