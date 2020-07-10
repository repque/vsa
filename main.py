import mail
from model import stocks, make_predictions

all_stocks = set( stocks + ['TSLA','NFLX','HEAR'] )

pred = sorted( list( make_predictions( all_stocks ) ), reversed=True )
mail.send( body=pred )
