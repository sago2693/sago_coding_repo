from concurrent.futures import ThreadPoolExecutor
import asyncio
import nest_asyncio
import pandas as pd
from datetime import datetime

def time_function(original_function):

    def wrap_function(*args, **kwargs):
        start = datetime.now()
        print("Hora de inicio: " + start.strftime("%H:%M:%S"))

        result = original_function(*args, **kwargs)

        end = datetime.now()
        print("Hora de finalización: " + end.strftime("%H:%M:%S"))
        print((end-start).seconds/60)
        return result
        
    return wrap_function


async def get_data_asynchronous(function, offers,browser,workers_number):

    with ThreadPoolExecutor(max_workers=workers_number) as executor:

        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(
                executor,
                function,
                *(browser, offer) 
            )
            for offer in offers
        ]
        for response in await asyncio.gather(*tasks):
            global df_result
            if response.empty == False:
                df_result = df_result.append(response,ignore_index=True)

@time_function
def run_in_parallel(function,offers,browser,workers_number,df_columns):
    nest_asyncio.apply()
    global df_result
    df_result = pd.DataFrame(columns=df_columns)
    loop = asyncio.get_event_loop()
    future = asyncio.ensure_future(get_data_asynchronous(function=function,offers=offers,browser=browser,workers_number=workers_number))
    loop.run_until_complete(future)
    return df_result
