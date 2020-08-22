import mail
from model import stocks, make_predictions


pred = sorted( list( make_predictions( stocks ) ), reverse=True )
mail.send( body=pred )
