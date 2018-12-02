import pandas as pd
from cassandra.cluster import Cluster

from order import Order
from datetime import datetime as dt

class DBWorker(object):
    def __init__(self):
        self.cluster = Cluster(['192.168.84.132'])
        # self.cluster = Cluster(['localhost'])
        self.session = self.cluster.connect('market')
        self.cluster.register_user_type('market', 'market_order', Order)

    def insert_trade(self, trade, timestamp):

        query = '''INSERT INTO market.trades (id, ticker, price, size, bid, ask, time) 
                  VALUES (%(id)s,%(ticker)s, %(price)s, %(size)s, %(bid)s, %(ask)s, %(time)s);'''
        params = {
            'id': trade.id,
            'ticker': trade.ticker,
            'price' :trade.price,
            'size': trade.size,
            'bid': trade.bid,
            'ask': trade.ask,
            'time': timestamp
        }

        self.session.execute(query, params)
        return

    def insert_order_book_snapshot(self, ticker, bids, asks, time):
        query = '''INSERT INTO market.order_book (ticker, bids, asks,  time) 
                  VALUES (%(ticker)s, %(bids)s, %(asks)s, %(time)s);'''

        # bids = map(lambda x: x.__dict__, bids)
        # print bids[0]
        params = {
            'ticker': ticker,
            'bids': bids,
            'asks': asks,
            'time': time
        }

        self.session.execute(query, params)
        return

    def get_trades(self, ticker, limit):

        query = '''SELECT * FROM trades    
                   WHERE ticker = %(ticker)s order by time desc limit %(limit)s;'''

        params = {
            'ticker': ticker,
            'limit': limit
        }

        rows = self.session.execute(query, params)

        df = pd.DataFrame(columns = ['Close'])

        for row in rows:
            df.loc[row.time] = row.price

        df = df.sort_index()

        return df

    def get_trades_for_period(self, ticker, start_date, end_date = None):

        if end_date == None:
            end_date = dt.today()

        query = '''SELECT * FROM trades    
                   WHERE ticker = %(ticker)s AND time>= %(start_date)s and time <= %(end_date)s;'''

        params = {
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date
        }

        rows = self.session.execute(query, params)

        df = pd.DataFrame(columns = ['Close', 'Volume'])

        for row in rows:
            df.loc[row.time] = [row.price, row.size]
            # df.loc[row.time]['Volume'] = row.size

        df = df.sort_index()
        return df

    def get_latest_order_book(self, ticker):
        query = '''SELECT bids, asks FROM order_book    
                   WHERE ticker = %(ticker)s ORDER BY time DESC LIMIT 1;'''

        params = {
            'ticker': ticker,
        }

        rows = self.session.execute(query, params)


        [(bids, asks)] = [(row.bids, row.asks) for row in rows]


        return bids, asks




