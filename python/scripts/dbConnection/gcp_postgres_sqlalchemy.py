import sqlalchemy
from sqlalchemy import event
import pandas as pd


ssl_mode ='verify-ca'
root = "gcp_credentials/server-ca.pem"
cert = "gcp_credentials/client-cert.pem"
key  = "gcp_credentials/client-key.pem"

ssl_args ={"sslmode":ssl_mode,
"sslrootcert": root,
"sslcert": cert,
"sslkey":key}

#db params
user_name ="postgres"
passwrd = "***"
host_address = "35.289.147.301"
port_chosen = "5432"
database_name = "DB_ERP"


pool = sqlalchemy.create_engine(
    # Equivalent URL:
    # mysql+pymysql://<db_user>:<db_pass>@<db_host>:<db_port>/<db_name>
    sqlalchemy.engine.url.URL(
        drivername="postgresql+psycopg2", #Important
        username=user_name,  # e.g. "my-database-user"
        password=passwrd,  # e.g. "my-database-password"
        host=host_address,  # e.g. "127.0.0.1"
        port=port_chosen,  # e.g. 3306
        database=database_name,  # e.g. "my-database-name"
    ),connect_args=ssl_args
)
#Fast execution of query
@event.listens_for(pool, 'before_cursor_execute')
def receive_before_cursor_execute(conn, cursor, statement, params, context, executemany):
    if executemany:
        cursor.fast_executemany = True
        cursor.commit()

#Example of to_sql command
etapasaplicacion= pd.DataFrame(columns=["column1","column2"])
etapasaplicacion.to_sql(name="etapasaplicacion",con=pool,if_exists='replace',index=False,method='multi')