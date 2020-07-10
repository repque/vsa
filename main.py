import mail
from model import stocks, make_predictions

all_stocks = set( stocks + ['TSLA','NFLX','HEAR'] )

pred = list( make_predictions( all_stocks ) )
mail.send( body=pred )
