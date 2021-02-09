#Source: https://hakibenita.com/fast-load-data-python-postgresql
from typing import Iterator, Optional, Dict, Any
import io
import pandas as pd 
import time
from functools import wraps
from memory_profiler import memory_usage

#Optional decorator to measure time and memory usage
def profile(fn):
    @wraps(fn)
    def inner(*args, **kwargs):
        fn_kwargs_str = ', '.join(f'{k}={v}' for k, v in kwargs.items())
        print(f'\n{fn.__name__}({fn_kwargs_str})')

        # Measure time
        t = time.perf_counter()
        retval = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t
        print(f'Time   {elapsed:0.4}')

        # Measure memory
        mem, retval = memory_usage((fn, args, kwargs), retval=True, timeout=200, interval=1e-7)

        print(f'Memory {max(mem) - min(mem)}')
        return retval

    return inner


class StringIteratorIO(io.TextIOBase):
    def __init__(self, iter: Iterator[str]):
        self._iter = iter
        self._buff = ''

    def readable(self) -> bool:
        return True

    def _read1(self, n: Optional[int] = None) -> str:
        while not self._buff:
            try:
                self._buff = next(self._iter)
            except StopIteration:
                break
        ret = self._buff[:n]
        self._buff = self._buff[len(ret):]
        return ret

    def read(self, n: Optional[int] = None) -> str:
        line = []
        if n is None or n < 0:
            while True:
                m = self._read1()
                if not m:
                    break
                line.append(m)
        else:
            while n > 0:
                m = self._read1(n)
                if not m:
                    break
                n -= len(m)
                line.append(m)
        return ''.join(line)

import io

def object_columns_to_datetime (df): #In case date columns are turned to object when reading the sql table
    df_resultado = df.copy()
    object_cols = [col for col, col_type in df.dtypes.iteritems() if col_type == 'object']
    df_resultado.loc[:, object_cols] = df_resultado[object_cols].combine_first(df_resultado[object_cols].apply(pd.to_datetime, errors="ignore"))
    return df_resultado

def pandas_to_postgres (df):
    cadena = ''' '''
    for index,row in df.dtypes.items():
        new_row=""
        if row=="object": #Varchar length for empty columns = 1 (minimum)
            if df[index].isnull().all():
                new_row=f"VARCHAR(1)"
            else:
                max_length = max(int(df[index].map(len,na_action="ignore").max()),1) #Ignores na when mapping to prevent errors
                #Max between max length and 1 in case the column has only empty strings
                new_row=f"VARCHAR({max_length})"
        elif row=="float64":
            new_row="FLOAT8" #inexact but more efficient
        elif row=="int64":
            if (df.max()[index] < 32767) and (df.min()[index]>-32768): #limit values for smallint variables
                new_row="SMALLINT"
            else:
                new_row="INTEGER"
        elif row=="bool":
            new_row="BOOLEAN"
        elif row=="datetime64[ns]":
            new_row="TIMESTAMP"
        else:
            new_row="parametrizar" #In case another pandas data type is missing
            print(index, row)
        
        cadena+=index.lower() + " " + new_row + "," #lower column names to migrate to postgresql
    
    return cadena[1:-1] #Deletes leading space and last comma



def create_staging_table(cursor,table_name,df ) -> None: #the function does not return a value
    cursor.execute(f"""
        DROP TABLE IF EXISTS {table_name};
        CREATE TABLE {table_name} (
            {pandas_to_postgres(df)}
        );
    """)


def clean_csv_value(value: Optional[Any]) -> str:
    if value is None or pd.isnull(value): #pd is null catches NaT values which are problematic
        return r'\N'
    if "\r" in str(value): #This scape sequence is problematic in progress
        return str(value).replace("\r", ' ').replace('\n', '\\n')
    return str(value).replace('\n', '\\n')

#@profile
def copy_string_iterator(connection,table_name, df: Iterator[Dict[str, Any]], size: int = 8192) -> None:
    with connection.cursor() as cursor:
        create_staging_table(cursor=connection,table_name=table_name,df=df ) #Deletes table if it exists and creates it again
        df_string_iterator = StringIteratorIO((
            '|'.join(map(clean_csv_value,row)) + '\n'
            for row in df.to_records(index=False)
        )) #I could probably map the function to all the records instead of looping 
        cursor.copy_from(df_string_iterator, table_name, sep='|', size=size) #feeds copy_from function with iterable and uploads it