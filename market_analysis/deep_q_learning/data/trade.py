

class Trade:

    def __init__(self, id, ticker, price, size, time, bid=None, ask=None, side=None, taker =None, maker=None):
        self.id = id
        self.size = size
        self.price = price
        self.side = side

        self.ticker = ticker
        self.bid = bid
        self.ask = ask
        self.time = time

        self.taker = taker
        self.maker = maker