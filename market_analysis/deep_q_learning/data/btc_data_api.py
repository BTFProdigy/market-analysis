import json
import time
import uuid
from datetime import datetime as dt
from threading import Timer

import requests

from data_api import DataApi
from db_worker import DBWorker
from order import Order
from trade import Trade


class BtcDataApi(DataApi):

    def __init__(self, db_worker):
        self.timeout = 30
        self.db_worker = db_worker

    def start_collecting(self):
        initial_delay = 0
        t = Timer(initial_delay, self.collect_trade_data_and_order_book)
        t.start()

    def get_trade(self, ticker="BTC-EUR"):
        url = "https://api.gdax.com/products/{}/ticker".format(ticker)

        response = requests.get(url)
        json_response = json.loads(response.content)

        ts = time.time()
        timestamp = dt.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

        return  Trade(str(json_response['trade_id']), ticker, float(json_response['price']), float(json_response['size']),
                      timestamp, float(json_response['bid']), float(json_response['ask']))

    def collect_trade_data_and_order_book(self, ticker="BTC-EUR"):

        trade = self.get_trade()
        bids, asks = self.get_order_book()


        ts = time.time()
        timestamp = dt.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

        self.db_worker.insert_trade(trade, timestamp)
        self.db_worker.insert_order_book_snapshot(ticker, bids, asks, timestamp)

        Timer(self.timeout, self.collect_trade_data_and_order_book).start()

        return

    def get_order_book(self, ticker="BTC-EUR", level=2):
        url = "https://api.gdax.com/products/{}/book".format(ticker)

        params = {
            "level": level,
        }

        response = requests.get(url, params)
        json_response = json.loads(response.content)

        print json_response

        bids_received = json_response['bids']
        asks_received = json_response['asks']

        bids = []
        asks = []

        # bids = map(lambda x: Order(ticker, int(x[0]), int(x[1]), "Bid"), bids)
        size = 10
        for i in range(size):
            bid = bids_received[i]
            amount = bid[2]

            for _ in range(amount):
                bids.append(Order(str(uuid.uuid4()), ticker, float(bid[0]), float(bid[1]), "Bid"))

            ask = asks_received[i]
            amount = ask[2]

            for _ in range(amount):
                asks.append(Order(str(uuid.uuid4()), ticker, float(ask[0]), float(ask[1]), "Ask"))

        return bids, asks


# get_order_book()
db_worker = DBWorker()
data_getter = BtcDataApi(db_worker)
data_getter.start_collecting()


