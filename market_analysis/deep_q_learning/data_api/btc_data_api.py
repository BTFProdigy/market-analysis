import json
import os.path
import time
import uuid
from datetime import datetime as dt
from threading import Timer

import configparser
import requests

from data_api import DataApi
from db_worker import DBWorker
from market_analysis.deep_q_learning import config_getter
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

    def get_trades(self, ticker="BTC-EUR", start_date = None, end_date = None):
        url = "https://api.gdax.com/products/{}/ticker".format(ticker)

        response = requests.get(url)
        json_response = json.loads(response.content)

        ts = time.time()
        timestamp = dt.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

        return  Trade(str(json_response['trade_id']), ticker, float(json_response['price']), float(json_response['size']),
                      timestamp, float(json_response['bid']), float(json_response['ask']))

    def collect_trade_data_and_order_book(self, ticker="BTC-EUR"):

        trade = self.get_trades()
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

        bids_received = json_response['bids']
        asks_received = json_response['asks']

        bids = []
        asks = []

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


def get_config(file):
    config = configparser.ConfigParser()
    config.read(os.path.dirname(os.path.dirname(__file__)) + '/' + file)


config = config_getter.get_config('config')
db_address = config.getint('DB','db_address')
db_name = config.getint('DB','db_name')
db_worker = DBWorker(db_address, db_name)
# get_order_book()

data_getter = BtcDataApi(db_worker)
data_getter.start_collecting()


