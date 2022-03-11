from google.cloud import bigquery



def query_stackoverflow():
    client = bigquery.Client()
    query_job = client.query(
        """
        SELECT
        *
        FROM `twitter-bank-sentiment.twitter_bank_sent.tweets`
        WHERE upper(text) like '%@HSBC%'
        LIMIT 10"""
    )

    results = query_job.result()  # Waits for job to complete.

    for row in results:
        print(row)

query_stackoverflow()

def max_tweet():
    client = bigquery.Client()
    query_job = client.query(
        """
        SELECT
        max(tweet_id)
        FROM `twitter-bank-sentiment.twitter_bank_sent.tweets`
        WHERE upper(text) like '%@HSBC%'
        LIMIT 10"""
    )

    results = query_job.result()  # Waits for job to complete.

    for row in results:
        print(row)

max_tweet()
