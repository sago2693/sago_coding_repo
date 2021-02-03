import pyodbc
import pandas as pd

server_name = "DESKTOP-LIQPS7G\SQLEXPRESS" #Writen on sql server start
data_base = "DB_ERP"

conn = pyodbc.connect('Driver={SQL Server};'
                      'Server='+ server_name + ';'
                      'Database=' + data_base +';'
                      'Trusted_Connection=yes;')

cursor = conn.cursor()
cursor.execute('SELECT * FROM DB_ERP.dbo.example')

for row in cursor:
    print(row)