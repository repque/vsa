
def make_data_folder( folder ):
    ''' Creates folder if it doesn't exist already '''
    from pathlib import Path
    Path(folder).mkdir(parents=True, exist_ok=True)

import contextlib

@contextlib.contextmanager
def atomic_overwrite(filename):
    import shutil
    temp = filename + '~'
    with open(temp, "w") as f:
        yield f
    shutil.move(temp, filename) # this will only happen if no exception was raised

def is_bday(x):
    from pandas.tseries.offsets import BDay, Day 
    return x == x + Day(1) - BDay(1)

def last_bday(end_date=None):
    from pandas.tseries.offsets import BDay
    from datetime import datetime

    end_date = end_date or datetime.now().date()
    return end_date if is_bday(end_date) else (end_date - BDay(1)).date() # find the last business date

def from_file( ticker, file_name, end=None ):
    ''' loads dataframe from csv file, pads missing most-recent data if any '''
    import pandas as pd
    from pandas.tseries.offsets import BDay
    from datetime import datetime
    
    df = pd.read_csv( file_name )
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)  
    
    if end is not None:
        start = pd.to_datetime( df.tail(1).index.values[0] ).date()
        end_date = datetime.strptime( end, '%Y-%m-%d' ).date()
        last = last_bday( end_date )
        if start < last:
            new_start = (start + BDay(1)).date()
            new_data = get_ticker_data( ticker, new_start.strftime('%Y-%m-%d'), last.strftime('%Y-%m-%d') )
            combined = pd.concat( [df, new_data], axis=0 )           
            with atomic_overwrite(file_name) as f:
                combined.to_csv( f )         
            
            df = from_file( ticker, file_name )
    return df

def get_ticker_data( ticker, start, end ):
    import pandas_datareader as web    
    print('Downloading', ticker, start, end)
    df = web.DataReader( ticker, 'yahoo', start=start, end=end )    
    return df

def get_current_price( ticker ):
    end = last_bday()
    data = get_historical_data( [ticker], start=end.strftime('%Y-%m-%d') )[ticker]
    return data['Close']

def get_historical_data( symbols, start='2000-01-01', end=None, folder='D:/dev/db/daily_data' ):
    from datetime import datetime
    import pandas as pd
    import os.path
    
    make_data_folder( folder )
    
    # end date is today
    end = end or datetime.now().date().strftime('%Y-%m-%d')
    
    data = {} # returns dict of individual dataframes, one per ticker symbol    
    for ticker in symbols:
        file_name = os.path.join(folder, ticker + ".csv")
        if os.path.exists(file_name):
            df = from_file( ticker, file_name, end )
        else:
            df = get_ticker_data( ticker, start, end )
            df.to_csv( file_name )
        data[ticker] = df
    return data

def compress_data( df, res='W' ):
    ''' Can compress daily into weekly (W) or monthly (M) bars '''
    logic = {'Open'  : 'first',
             'High'  : 'max',
             'Low'   : 'min',
             'Close' : 'last',                              
             'Volume': 'sum'}

    return df.resample(res).apply(logic)    