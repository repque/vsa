from typing import List
import pandas as pd

def get_ratios(df: pd.DataFrame, periods: int = 10) -> pd.DataFrame:
    """Calculates the following ratios:

    Args:
        df: DataFrame with OHLCV data
        periods: Number of periods for rolling calculations

    Returns:
        DataFrame with VSA ratio indicators

    Ratios calculated:
        range_ratio  - how big the bar is compared to the average
        close_ratio  - where was the close in relation to high-low range
        volume_ratio - volume compared to the average
    """
    
    bar_range  = df['High'] - df['Low']
    # bar range compared to the average
    range_ratio  = bar_range / bar_range.rolling(window=periods, min_periods=periods).mean()
    wide_bar = range_ratio > 2 if range_ratio is not None else None
    # close price in relation to high-low range
    close_ratio  = (df['Close'] - df['Low'])/(df['High'] - df['Low'])
    up_close = close_ratio > 0.75
    down_close = close_ratio < 0.25
    up_bar = df['High'].diff().gt(0) & df['Low'].diff().gt(0)
    down_bar = df['High'].diff().lt(0) & df['Low'].diff().lt(0) 
    inside_bar = df['High'].diff().le(0) & df['Low'].diff().ge(0)  
    outside_bar = df['High'].diff().gt(0) & df['Low'].diff().lt(0)
    # volume compared to the average
    volume_ratio = df['Volume'] / df['Volume'].rolling(window=periods, min_periods=periods).mean()
    high_volume = volume_ratio > 2 if volume_ratio is not None else None
    low_volume = volume_ratio < 0.5
    
    data = [range_ratio, 
            wide_bar.astype(float),
            up_close.astype(float),
            down_close.astype(float),
            up_bar.astype(float),
            down_bar.astype(float),
            inside_bar.astype(float),
            outside_bar.astype(float),
            high_volume.astype(float),            
            low_volume.astype(float)]
    
    df_out = pd.concat( data, axis=1, keys=['range_ratio', 'wb','uc','dc','ub','db','ib','ob','hv','lv'] )
    df_out = df_out.dropna( how='any' )
    del df_out['range_ratio']
    return df_out

def get_y(df: pd.DataFrame, offset: int = 4, target: int = 3) -> pd.Series:
    """Calculates binary label for 4 consecutive weeks of price increases.

    Args:
        df: DataFrame with OHLCV data
        offset: Offset for returns calculation
        target: Target threshold

    Returns:
        Series with binary labels (1 = bullish, 0 = bearish)
    """

    returns = df['Close'].diff(offset)
    y = (df['Close'].diff(1).gt(0) & 
         df['Close'].diff(2).gt(0) & 
         df['Close'].diff(3).gt(0) & 
         df['Close'].diff(4).gt(0) ) 
    return y.astype(int)

def lagged(df: pd.DataFrame, periods: int = 4, skip: List[str] = None) -> pd.DataFrame:
    """Add lagged features to DataFrame.

    Args:
        df: DataFrame with features
        periods: Number of lagged periods to create
        skip: List of columns to skip from lagging

    Returns:
        DataFrame with lagged features added
    """
    if skip is None:
        skip = []
    for c in df.columns:
        if c not in skip:
            for p in range(1, periods):
                df[str(c) + '_t_' + str(p)] = df[c].shift(p)
    return df

def make_features(df: pd.DataFrame, with_label: bool = False) -> pd.DataFrame:
    """Create VSA features from OHLCV data.

    Args:
        df: DataFrame with OHLCV data
        with_label: Whether to include target labels

    Returns:
        DataFrame with VSA features and optional labels
    """
    ratios = get_ratios(df)
    if with_label:
        ratios['y'] = get_y(df)

    # add lagging ratios
    return lagged(ratios, skip=['y'], periods=5).dropna(how='any')    
