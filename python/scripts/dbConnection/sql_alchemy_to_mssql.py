
from sqlalchemy import create_engine, inspect
import pandas as pd
import urllib

server_name = "server"
db_name="db_name"
params = urllib.parse.quote_plus("Driver={SQL Server};"
                                 f"Server={server_name};"
                                 "DATABASE={db_name};"
                                 "Trusted_Connection=yes;"
                                 "PWD=password")
#Parsing sql server params to sqlalchemy
engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")
inspector = inspect(engine)
schemas = inspector.get_schema_names()

try:
    #After exploring the schemas
    df = pd.concat(pd.read_sql_table("SQl_table_name", con=engine, schema=schemas[0],chunksize = 100000))
    #Concat the result of using chunksize. Otherwise, it returns a dataframe directly
except Exception as e:
    print("hubo un error",e)
    engine.rollback()

query_columns_information = '''
select column_name,
data_type
from information_schema.columns
where table_name = 'SQL_table_name';
'''