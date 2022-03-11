from google.cloud import bigquery   
import os
#from google.api_core.exeptions import BadRequest
import pandas as pd
#import pandas_gbq

# def BQ_READ():
#     client = bigquery.Client()
#     query_job = client.query(
#         """
#         SELECT
#         *
#         FROM `twitter-bank-sentiment.twitter_bank_sent.tweets`
#         WHERE upper(text) like '%@HSBC%'
#         LIMIT 10"""
#     )

#     results = query_job.result()  # Waits for job to complete.

#     for row in results:
#         print(row)

# BQ_READ()

client = bigquery.Client()
query_job = """
    SELECT
    *
    FROM `twitter-bank-sentiment.twitter_bank_sent.tweets`
    WHERE upper(text) like '%@HSBC%'
    LIMIT 10"""
    

dataframe = (
    client.query(query_job)
    .result()
    .to_dataframe(
        # Optionally, explicitly request to use the BigQuery Storage API. As of
        # google-cloud-bigquery version 1.26.0 and above, the BigQuery Storage
        # API is used by default.
        create_bqstorage_client=True,
    )
)
print(dataframe.head())

dataframe.to_csv('bq_test.csv',sep=",",index=False,encoding='utf-8',quotechar="'")




def append_data_from_CSV(bq_client, dataset, table_name, file_path, file_name):
    """
    Ingest data to BQ table from CSV file
    """

    dataset_ref = bq_client.dataset(dataset)
    table_ref = dataset_ref.table(table_name)
    
    # try:
    job_config = bigquery.LoadJobConfig()
    job_config.source_format = bigquery.SourceFormat.CSV
    job_config.skip_leading_rows = 1  # header skipped by default for append
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND  # append data to table
    job_config.allow_quoted_newlines = True  

    # job_config.ignoreUnknownValues = True
    # job_config.autodetect = True

    full_file_path = os.path.join(file_path, file_name)
    with open(full_file_path, "rb") as source_file:
        job = bq_client.load_table_from_file(source_file, table_ref, job_config=job_config)

    job.result()  # Waits for table load to complete.
    #log.info("Loaded {} rows into {}:{}.".format(job.output_rows, dataset, table_name))
    
    # except Exception as Ex:
    #     pass

append_data_from_CSV(client,"twitter_bank_sent","tweets2","./","bq_test.csv")





