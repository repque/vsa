from typing import List, Tuple, Generator, Optional, Any
import pandas as pd
from sklearn.ensemble import VotingClassifier

from data import get_historical_data, compress_data
from features import make_features
from config import MODEL_PATH

stocks = ['AA', 'AAL', 'AAPL', 'ABBV', 'ABT', 'ADBE', 'ADI', 'ADM', 'AEO', 'AEP', 'AES', 'AFL', 'AGNC', 'AIG', 'AKER', 'ALKS', 'ALLY', 'AMAT', 'AMD', 'AMZN', 'ANF', 'APA', 'APPS', 'ARMK', 'ATI', 'ATVI', 
          'AXP', 'BA', 'BAC', 'BAX', 'BBBY', 'BBY', 'BEN', 'BERY', 'BK', 'BKE', 'BMRN', 'BMY', 'BSX', 'BX', 'C', 'CAG', 'CALM', 'CAT', 'CCL', 'CDE', 'CFG', 'CL', 'CLF', 'CMCSA', 'CNC', 'CNK', 'CNP', 
          'COG', 'COP', 'CORT', 'Cost', 'CPB', 'CPE', 'CRM', 'CSCO', 'CSX', 'CTL', 'CTSH', 'CVS', 'CVX', 'CWH', 'D', 'DAL', 'DD', 'DDD', 'DE', 'DELL', 'DFS', 'DHI', 'DIS', 'DISCA', 'DISH', 'DLPH', 
          'DOCU', 'DOW', 'DRI', 'DUK', 'DVAX', 'DVN', 'EBAY', 'ED', 'EIX', 'EL', 'EMR', 'EOG', 'EPD', 'EQT', 'ETFC', 'ETSY', 'EVH', 'EXC', 'EXPE', 'EXPI', 'F', 'FANG', 'FB', 'FCX', 'FDX', 'FE', 
          'FEYE', 'FIS', 'FISV', 'FIT', 'FITB', 'FL', 'FLR', 'FNB', 'FTI', 'GDDY', 'GE', 'GEL', 'GILD', 'GLUU', 'GLW', 'GM', 'GME', 'GOLD', 'GPS', 'GRPN', 'GRUB', 'GT', 'HAL', 'HBAN', 'HBI', 
          'HD', 'HFC', 'HLT', 'HOME', 'HON', 'HP', 'HPE', 'HPQ', 'HRB', 'HSIC', 'HST', 'HUN', 'IBM', 'ICE', 'INCY', 'INTC', 'IP', 'IPG', 'JBLU', 'JCI', 'JNJ', 'JNPR', 'JPM', 'JWN', 'KEY', 
          'KEYS', 'KGC', 'KHC', 'KIM', 'KIRK', 'KKR', 'KLAC', 'KMI', 'KO', 'KR', 'KSS', 'LB', 'LEN', 'LKQ', 'LLY', 'LOW', 'LRCX', 'LUV', 'M', 'MA', 'Mac', 'MAR', 'MCD', 'MCHP', 'MDT', 'MGM', 'MIK', 'MMC', 'MMM',
          'MO', 'MOBL', 'MOS', 'MPC', 'MRK', 'MRO', 'MRVL', 'MS', 'MSFT', 'MTCH', 'MU', 'MUR', 'MXIM', 'MYL', 'NBL', 'NCLH', 'NEM', 'NFLX', 'NI', 'NKE', 'NLSN', 'NOV', 'NOW', 'NRZ', 'NTAP', 'NUAN', 
          'NVAX', 'NVDA', 'NWL', 'OKE', 'OLN', 'ON', 'ORCL', 'OSTK', 'OXY', 'PBCT', 'PBF', 'PCG', 'PE', 'PEB', 'PEG', 'PEP', 'PFE', 'PG', 'PGR', 'PGRE', 'PHM', 'PK', 'PLAY', 'PM', 'PPL', 'PSTG', 'PSX', 
          'PYPL', 'QCOM', 'QSR', 'RAD', 'RCII', 'RCL', 'RF', 'RLGY', 'ROKU', 'ROST', 'RRC', 'RUN', 'SAVE', 'SBUX', 'SCHW', 'SE', 'SFM', 'SHAK', 'SHO', 'SIRI', 'SLB', 'SLM', 'SLV', 'SNPS', 'SO', 
          'SPG', 'SPR', 'SQ', 'SRNE', 'SSI', 'STAG', 'STOR', 'STX', 'SYF', 'SYY', 'T', 'TEVA', 'TGT', 'TJX', 'TLRY', 'TMUS', 'TOL', 'TRGP', 'TSLA', 'TTMI', 'TWTR', 'TXN', 'UA', 'UAL', 'UNH', 'UNP', 'UPS', 
          'USB', 'USFD', 'V', 'VER', 'VLO', 'VNO', 'VZ', 'W', 'WBA', 'WDC', 'WEN', 'WFC', 'WMB', 'WMT', 'WNC', 'WPX', 'WU', 'WY', 'WYNN', 'X', 'XOM', 'XRAY', 'XRX', 'ZION', 'ZNGA']

def train(stocks: List[str]) -> VotingClassifier:
    """Train ensemble ML model on stock data.

    Args:
        stocks: List of stock ticker symbols

    Returns:
        Trained VotingClassifier ensemble model
    """
    data = get_historical_data(stocks)

    weekly = {}
    for ticker in data:
        compressed = compress_data( data[ticker] )
        weekly[ticker]  = compressed

    from sklearn.model_selection import TimeSeriesSplit
    import numpy as np

    tssp = TimeSeriesSplit(n_splits = 3)

    tmp = make_features( weekly['AMD'], with_label=False )

    y = np.zeros(1).astype(int)
    x_data = np.zeros( (1, len( tmp.columns ) ) )

    for ticker in stocks:
        features = make_features( weekly[ticker], with_label = True )
        y = np.concatenate( [ y, features['y'].values ] )
        x_data = np.concatenate( [ x_data, features.drop('y', axis=1) ] )

    
    for train_index, test_index in tssp.split(x_data):
        X_train, X_test = x_data[train_index, :], x_data[test_index,:]
        y_train, y_test = y[train_index], y[test_index]



    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import VotingClassifier
    from sklearn.metrics import classification_report

    clf1 = LogisticRegression(random_state=1)
    clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
    clf3 = BernoulliNB()
    clf4 = KNeighborsClassifier()

    eclf = VotingClassifier(estimators=[('lr', clf1), 
                                        ('rf', clf2), 
                                        ('nb', clf3), 
                                        ('knn', clf4)],
                            voting='soft')

    clf1 = clf1.fit(X_train, y_train)
    clf2 = clf2.fit(X_train, y_train)
    clf3 = clf3.fit(X_train, y_train)
    clf4 = clf4.fit(X_train, y_train)
    eclf = eclf.fit(X_train, y_train)

    epred = eclf.predict(X_test)
    print( classification_report( y_test, epred ) )
    return eclf

def save(clf: Any, file_name: str) -> None:
    """Save trained model to disk.

    Args:
        clf: Trained classifier model
        file_name: Path to save model file
    """
    import joblib
    joblib.dump(clf, file_name)

def load(file_name: str) -> Any:
    """Load trained model from disk.

    Args:
        file_name: Path to model file

    Returns:
        Loaded classifier model
    """
    import joblib
    return joblib.load(file_name)

def plot(df: pd.DataFrame, title: str = '') -> None:
    """Plot candlestick chart with volume and predictions.

    Args:
        df: DataFrame with OHLCV data
        title: Chart title
    """
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    pred = model.predict( make_features( df ).values )

    candles = go.Candlestick(x=df.index,
                        open=df.Open,
                        high=df.High,
                        low=df.Low,
                        close=df.Close,
                        name='Price')

    volume = go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        marker_color='lightslategray'
        )

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_width=[1, 3] )

    fig.append_trace(candles, 1, 1)  
    fig.append_trace(volume, 2, 1)    

    fig.append_trace(go.Scatter(x=df.iloc[np.where(pred)].index, y=[min(df.Low)-2]*len(df),
                            mode='markers',
                            textposition="bottom center",
                            name='Predictions',    
                            marker_size=7,
                            marker=dict(color='blue', symbol='triangle-up')), 1, 1 )
    

    fig['layout']['yaxis1'].update(range=[min(df.Low)-2,max(df.High)+2])
    
    fig.update_layout(
        title=title,
    )

    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.show()

def make_predictions(stocks: List[str], file_name: Optional[str] = None) -> Generator[Tuple[float, str], None, None]:
    """Generate predictions for stocks.

    Args:
        stocks: List of stock ticker symbols
        file_name: Path to model file (uses config default if None)

    Yields:
        Tuple of (probability, ticker) for positive predictions
    """
    file_name = file_name or MODEL_PATH
    data = get_historical_data(stocks)
    model = load(file_name)

    for ticker in data:
        compressed = compress_data( data[ticker] )
        current = compressed.tail(14)
        features = make_features( current, with_label = False )
        if len(features) == 0:
            print( 'Skipping {}'.format( ticker ) )
            continue
        result = model.predict(features)[-1]
        if result:
            proba = round(model.predict_proba(features)[-1][-1], 2)
            yield proba, ticker
