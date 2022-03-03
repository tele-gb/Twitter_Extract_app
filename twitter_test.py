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

pd.options.display.max_colwidth

# setup database 
#removing because will convert to BQ
# import sqlite3
# from sqlite3 import Error


# db_file = "C:\SQLLite\Twitter_database.db"
# conn = sqlite3.connect(db_file)  

#max tweet if currently in db
# cur = conn.cursor()
# cur.execute("SELECT MAX(TWEET_ID) FROM Tweets")
# maxval, = cur.fetchone()
# max_val=str(maxval)


#Add bearer token
os.environ['TOKEN'] = 'AAAAAAAAAAAAAAAAAAAAADl%2FZAEAAAAA%2BLQvqIB%2F9y9COL8H%2FPAwq%2BEh1dI%3DkR5VfO2ng7B3VTDgxZEYC3TcxV5LTGR8EnIbGE342hnoUDCzyn'

#API Code to extract the JSON

#API Code to extract the JSON

def auth():
    return os.getenv('TOKEN')

def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers

def create_url(keyword,max_results):
# def create_url(keyword,end_date, max_results = 100):   
    search_url = "https://api.twitter.com/2/tweets/search/recent" #Change to the endpoint you want to collect data from

    #change params based on the endpoint you are using
    query_params = {'query': keyword,
                    ##'start_time': start_date,
                    ##'end_time': end_date,
                    'max_results': max_results,
                    #'sort_order':'relevancy',
                    'since_id': max_val,
                    'expansions': 'author_id,in_reply_to_user_id,geo.place_id',
                    'tweet.fields': 'id,text,author_id,in_reply_to_user_id,geo,conversation_id,created_at,lang,public_metrics,referenced_tweets,reply_settings,source',
                    'user.fields': 'id,name,username,created_at,description,public_metrics,verified',
                    'place.fields': 'full_name,id,country,country_code,geo,name,place_type',
                    'next_token': {}}
    return (search_url, query_params)

def connect_to_endpoint(url, headers, params, next_token = None):
    params['next_token'] = next_token   #params object received from create_url function
    response = requests.request("GET", url, headers = headers, params = params)
    print("Endpoint Response Code: " + str(response.status_code))
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()

#Inputs for the request
bearer_token = auth()
headers = create_headers(bearer_token)
keyword = "(@HSBC_UK OR @HalifaxBank OR @santanderukhelp OR @BarclaysUK OR @CooperativeBank OR @RevolutApp OR @LloydsBank OR @NatWest_Help OR @Monzo OR @StarlingBank OR @RBS_Help OR @TSB) lang:en -is:retweet -is.reply "
max_results = 100
# start_time = "2021-03-01T00:00:00.000Z"

url = create_url(keyword, max_results)
#print(url)
json_response = connect_to_endpoint(url[0], headers, url[1])


author_ls = []
created_at_ls=[]
created_at_raw_ls=[]
geo_ls=[]
tweet_id_ls=[]
lang_ls=[]
like_count_ls=[]
quote_count_ls=[]        
reply_count_ls=[]       
retweet_count_ls=[]
source_ls=[]
text_ls =[]

def create_tweet_df(json_response):
    
    #A counter variable
    counter = 0

    #Loop through each tweet
    for tweet in json_response['data']:
        
        # We will create a variable for each since some of the keys might not exist for some tweets
        # So we will account for that

        # 1. Author ID
        author_id = tweet['author_id']
        author_ls.append(int(author_id))

        # 2. Time created
        created_at = dateutil.parser.parse(tweet['created_at'])
        created_at_ls.append(created_at)
        created_at_raw=tweet['created_at']
        created_at_raw_ls.append(created_at_raw)

        # 3. Geolocation
        if ('geo' in tweet):   
            geo = tweet['geo']['place_id']
        else:
            geo = " "
        geo_ls.append(geo)

        # 4. Tweet ID
        tweet_id = tweet['id']
        tweet_id_ls.append(int(tweet_id))

        # 5. Language
        lang = tweet['lang']
        lang_ls.append(lang)                                            

        # 6. Tweet metrics
        retweet_count = tweet['public_metrics']['retweet_count']                   
        reply_count = tweet['public_metrics']['reply_count']
        like_count = tweet['public_metrics']['like_count']
        quote_count = tweet['public_metrics']['quote_count']
        like_count_ls.append(int(like_count))
        quote_count_ls.append(int(quote_count))       
        reply_count_ls.append(int(reply_count))
        retweet_count_ls.append(int(retweet_count))

        # 7. source
        source = tweet['source']
        source_ls.append(source)

        # 8. Tweet text
        text = tweet['text']
        text_ls.append(text)
        
        # Assemble all data in a list
        #res = [author_id, created_at,created_at_raw, geo, tweet_id, lang, like_count, quote_count, reply_count, retweet_count, source, text]
        #print(res)

def upload():
    
    json_response = connect_to_endpoint(url[0], headers, url[1])
    create_tweet_df(json_response)
    print(len(author_ls))
    
    final_df=pd.DataFrame(list(zip(author_ls,created_at_ls,created_at_raw_ls,geo_ls,tweet_id_ls,lang_ls,like_count_ls,quote_count_ls,
                              reply_count_ls,retweet_count_ls,source_ls,text_ls)),
               columns=['author_id', 'created_at','created_at_raw','geo', 'tweet_id', 'lang', 'like_count', 'quote_count',
                        'reply_count', 'retweet_count', 'source', 'text'])
    final_df['normalised_date'] = final_df['created_at'].dt.normalize()
    print(final_df['created_at'])
    
    author_ls.clear()
    created_at_ls.clear()
    created_at_raw_ls.clear()
    geo_ls.clear()
    tweet_id_ls.clear()
    lang_ls.clear()
    like_count_ls.clear()
    quote_count_ls.clear()
    reply_count_ls.clear()
    retweet_count_ls.clear()
    source_ls.clear()
    text_ls.clear()
    print(len(author_ls))
    
    #final_df.to_sql('Tweets', conn, if_exists='append',index=False)
    
    #clears the dataframe
    #final_df = final_df[0:0] 
    
    print(final_df['created_at'])
    
    
upload()