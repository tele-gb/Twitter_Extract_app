# setup database
import sqlite3
import pandas as pd
from sqlite3 import Error
db_file = "C:\SQLLite\Twitter_database.db"
conn = sqlite3.connect(db_file)  

df = pd.read_sql_query("select count(*) from Tweets;", conn)

print(df)