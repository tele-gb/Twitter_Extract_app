from google.cloud import bigquery



def BQ_READ():
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

BQ_READ()

query_job = client.query(
        """
        SELECT
        *
        FROM `twitter-bank-sentiment.twitter_bank_sent.tweets`
        WHERE upper(text) like '%@HSBC%'
        LIMIT 10"""
    )

bqclient = bigquery.Client()

dataframe = (
    bqclient.query(query_job)
    .result()
    .to_dataframe(
        # Optionally, explicitly request to use the BigQuery Storage API. As of
        # google-cloud-bigquery version 1.26.0 and above, the BigQuery Storage
        # API is used by default.
        create_bqstorage_client=True,
    )
)
print(dataframe.head())




