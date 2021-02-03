import pandas as pd
import psycopg2

#Import credentials from another script taylored for each user
user_name ="postgres"
passwrd = "***"
host_address = "35.289.147.301"
port_chosen = "5432"
database_name = "DB_ERP"
ssl_mode ='verify-ca' #Important configuration
root = "gcp_credentials/server-ca.pem"
cert = "gcp_credentials/client-cert.pem"
key  = "gcp_credentials/client-key.pem"
try:
    connection = psycopg2.connect(user=user_name,
                                    password=passwrd,
                                    host=host_address,
                                    port=port_chosen,
                                    database=database_name,
                                    sslmode=ssl_mode,
                                    sslrootcert= root,
                                    sslcert = cert,
                                    sslkey = key)

    cursor = connection.cursor()

except psycopg2.OperationalError as e:
    print(e)
    print("network error")
else:
    query_sintax = "select * from example"
    try:
        query_result = pd.read_sql_query(query_sintax,connection)
    except psycopg2.OperationalError as e1:
        print(e1)
        print("Failed connection")
    except Exception as e:

        print("Another error occurred", e)
        connection.rollback()