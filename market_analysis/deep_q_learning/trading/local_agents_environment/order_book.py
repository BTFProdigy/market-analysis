import time
from datetime import datetime as dt
from market_analysis.deep_q_learning.data_api.trade import Trade

class OrderBook:

    def __init__(self):
        self.trade_requests = []

        self.trades= []

        self.bids = []
        self.asks = []

    def generate_bids_and_asks(self, bids, asks):
        self.bids = sorted(bids, key=lambda x:x.price, reverse=True)
        self.asks = sorted(asks, key=lambda x:x.price, reverse=False)
        return

    def receive_order(self, order):
        self.create_a_trade_request(order)
        if order.type == "Market":
            self.create_market_trade(order)

    def create_a_trade_request(self, order):
        self.trade_requests.append(order)

    def remove_order(self, order):
        if order.side == "Ask":
            self.asks.remove(order)
        elif order.side == "Bid":
            self.bids.remove(order)

    def change_order_stocks(self, index, stocks, side):
        if side == "Ask":
            self.asks[index].stocks = stocks
        elif side == "Bid":
            self.bids[index].stocks = stocks

    def create_market_trade(self, order):
        try:
            while(order.stocks > 0):
                if order.side == "Ask":
                    best_order = self.bids[0]
                    # best_order = self.find_order_with_maximal_bid_price(order)
                elif order.side == "Bid":
                    # best_order = self.find_order_with_minimal_ask_price(order)
                    best_order = self.asks[0]
                stocks_found = best_order.stocks
                stocks_request= order.stocks

                stocks_left = stocks_request - stocks_found
                stocks_passed = 0

                if stocks_left > 0:
                    self.remove_order(best_order)
                    stocks_passed = stocks_found

                if stocks_left == 0:
                    self.remove_order(best_order)
                    stocks_passed = stocks_found

                if stocks_left<0:
                    index = self.get_order_index(best_order.order_id, best_order.side)
                    self.change_order_stocks(index, abs(stocks_left), best_order.side)
                    stocks_passed = stocks_request

                ts = time.time()
                timestamp = dt.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

                self.trades.append(Trade(order.order_id, order.ticker, best_order.price, stocks_passed , timestamp))
                order.stocks-=stocks_passed
        except Exception, e:
            print 'No asks or bids in order book'
        return

    def get_order_index(self, id, side):
        if side == "Ask":
            ids = map(lambda x: x.order_id, self.asks)
        elif side == "Bid":
            ids = map(lambda x: x.order_id, self.bids)

        index = ids.index(id)

        return index

    def get_mid_spread(self):
        return (self.bids[0].price+self.asks[0].price)/2