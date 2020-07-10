from data import get_historical_data, compress_data
from features import make_features

stocks=['AMD', 'HAL', 'APA', 'CL', 'AAPL', 'KLAC', 'GS', 'BAC', 'PG', 'NVDA', 'AMZN', 'LUV', 'LMT', 'GD', 'T', 'BIIB', 'AMGN', 'ALXN', 'ODFL', 'HD', 'KO', 'JNJ', 'MMM', 'KGC', 'NEM', 'CI', 'UNH', 'CVS']

def train( stocks ):
    data = get_historical_data( stocks )

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

def save( clf, file_name ):
    import joblib
    joblib.dump( clf, file_name )

def load( file_name ):
    import joblib
    return joblib.load( file_name )

def plot( df, title='' ):
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

def make_predictions( stocks, file_name='C:/Users/gera/vsa.sav' ):
    data = get_historical_data( stocks )
    model = load( file_name )

    for ticker in data:
        compressed = compress_data( data[ticker] )
        current = compressed.tail(14)
        features = make_features( current, with_label = False )
        result = model.predict(features)[-1]
        if result:
            yield ticker