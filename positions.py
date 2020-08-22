
from collections import namedtuple
from datetime import datetime, timedelta
from tinydb import TinyDB, Query
import pandas as pd

from data import get_current_price

Pnl = namedtuple('Pnl', ['mtm', 'realized', 'total'])

class Position( object ):
    ''' Position per instrument. 
        Used to keep track of open positions, holding time, reset, pnl '''

    db = TinyDB('db.json')
    
    @classmethod
    def all_open( cls ):
        records = cls.db.all()
        positions = { Position.load(r['ticker']) for r in records if r['ticker'] not in ['TEST'] }
        return [ p for p in positions if p.is_open ]

    @classmethod
    def as_df( cls ):
        pnl_data = [ (p.ticker, *p.pnl()) for p in cls.all_open() ]
        return pd.DataFrame(pnl_data, columns =['ticker', 'mtm', 'realized', 'total'])               

    @classmethod
    def expiring( cls, days=1 ):
        positions = cls.all_open()
        if positions:            
            return [ (p, p.expiration)  for p in positions if (datetime.strptime( p.expiration, '%Y-%m-%d' ).date() - datetime.today().date()).days <= days ]
    
    @classmethod
    def load( cls, symbol ):
        query = Query()
        results = cls.db.search(query.ticker == symbol)
        return cls( symbol, results )

    @classmethod
    def record( cls, symbol, qty, price, date = None, is_entry = True ):
        date = date or datetime.today().date().strftime('%Y-%m-%d')
        expiration = (datetime.strptime( date, '%Y-%m-%d' ).date() + timedelta(days=31)).strftime('%Y-%m-%d') if is_entry else None
        cls.db.insert({'ticker': symbol, 'qty': qty, 'price': price, 'date':date, 'is_entry': is_entry, 'expiration': expiration})
        return cls.load( symbol )

    @classmethod    
    def reset_expiration( cls, symbol, date = None ):
        p = cls.load( symbol )
        if not p.is_open:
            raise ValueError('Can only reset expiration for an open position!')
            
        date = date or datetime.today().date().strftime('%Y-%m-%d')
        cls.db.update({'expiration': date}, doc_ids=[p._last_id()])
        return Position.load( symbol )
    
    def __init__( self, symbol, rows ):                
        self.ticker = symbol
        self.rows = rows
        
        self.is_open = rows[-1]['is_entry'] if rows else False
        if rows:
            self.holding_days = ( datetime.today().date() - datetime.strptime( rows[-1]['date'], '%Y-%m-%d' ).date() ).days if self.is_open else ( datetime.strptime( rows[-1]['date'], '%Y-%m-%d' ).date() - datetime.strptime( rows[-2]['date'], '%Y-%m-%d' ).date() ).days
            self.expiration = rows[-1]['expiration']

    def __repr__(self):
        return self.ticker
    
    def __hash__(self):
        return hash(self.ticker)
    
    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and
            self.ticker == other.ticker )
    
    def _last_id( self ):
        return self.rows[-1].doc_id

    def pnl( self, price=None ):
        mtm_pl = realized_pl = total_pl = 0.0
        if self.rows:
            prev = None
            for row in self.rows:
                if row['is_entry']:
                    prev = row
                else:
                    if row['qty'] != prev['qty']:
                        raise Exception('Quantities do not match: \n{}\n{}'.format( prev, row ))
                    realized_pl = (row['price'] - prev['price']) * row['qty']
            if price is None:
                price = get_current_price( self.ticker )
            mtm_pl = (price - prev['price']) * prev['qty']
            total_pl = realized_pl + mtm_pl
        return Pnl( round(mtm_pl, 2), round(realized_pl, 2), round(total_pl, 2) )