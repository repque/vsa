from typing import Dict, Optional
from pathlib import Path
from datetime import date
import pandas as pd
import logging

from exceptions import DataFetchError

logger = logging.getLogger(__name__)

def make_data_folder(folder: str) -> None:
    """Creates folder if it doesn't exist already.

    Args:
        folder: Path to folder to create
    """
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

def from_file( ticker, file_name, end=None, active=True ):
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
            if not new_data is None:
                combined = pd.concat( [df, new_data], axis=0 )           
                with atomic_overwrite(file_name) as f:
                    combined.to_csv( f )
            else:
                if active:
                    return None
            
                df = from_file( ticker, file_name )
    return df

def get_ticker_data(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Fetch ticker data from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol
        start: Start date string
        end: End date string

    Returns:
        DataFrame with OHLCV data or None if fetch fails
    """
    import pandas_datareader as web
    try:
        logger.info(f'Downloading {ticker}: {start} - {end}')
        df = web.DataReader(ticker, 'yahoo', start=start, end=end)
        logger.info(f'{ticker} download complete')
        return df
    except Exception as e:
        logger.warning(f'Failed to download {ticker}: {str(e)}')
        return None

def get_last_row( ticker ):
    ''' get last row values for the ticker '''
    end = last_bday()
    data = get_historical_data( [ticker], start=end.strftime('%Y-%m-%d') )[ticker]
    return data.tail(1)

def get_current_price( ticker ):
    row = get_last_row( ticker )
    end = last_bday()
    data = get_historical_data( [ticker], start=end.strftime('%Y-%m-%d') )[ticker]
    return row['Close'].values[0]

def get_historical_data(symbols: list, start: str = '2000-01-01', end: Optional[str] = None, folder: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """Fetch historical OHLCV data for symbols with local caching.

    Args:
        symbols: List of ticker symbols
        start: Start date in YYYY-MM-DD format
        end: End date in YYYY-MM-DD format (defaults to today)
        folder: Cache directory path (uses config default if None)

    Returns:
        Dictionary mapping ticker symbols to DataFrames
    """
    from datetime import datetime
    import os.path
    from config import DATA_CACHE_DIR

    folder = folder or DATA_CACHE_DIR
    make_data_folder(folder)
    
    # end date is today
    end = end or datetime.now().date().strftime('%Y-%m-%d')
    
    data = {} # returns dict of individual dataframes, one per ticker symbol    
    for ticker in symbols:
        file_name = os.path.join(folder, ticker + ".csv")
        try:
            df = None
            if os.path.exists(file_name):
                df = from_file( ticker, file_name, end )
            else:
                df = get_ticker_data( ticker, start, end )
                if not df is None:
                    df.to_csv( file_name )
            if not df is None:
                data[ticker] = df
        except:
            continue
    return data

def compress_data(df: pd.DataFrame, res: str = 'W') -> pd.DataFrame:
    """Compress daily OHLCV data into weekly or monthly bars.

    Args:
        df: DataFrame with daily OHLCV data
        res: Resampling resolution ('W' for weekly, 'M' for monthly)

    Returns:
        DataFrame with compressed OHLCV data
    """
    logic = {'Open'  : 'first',
             'High'  : 'max',
             'Low'   : 'min',
             'Close' : 'last',
             'Volume': 'sum'}

    return df.resample(res).apply(logic)    

def clean_file( filename ):
    seen = set()
    lines = []
    for line in reversed(list(open(filename))):
        dt = line.rstrip().split(',')[0]
        if not dt in seen:
            seen.add(dt)
            lines.append( line )

    with atomic_overwrite(filename) as output: 
        for line in reversed(lines):
            output.write(line)