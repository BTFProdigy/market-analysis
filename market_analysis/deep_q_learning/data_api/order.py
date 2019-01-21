import uuid

from cassandra.cqlengine.usertype import UserType


class Order(UserType):

    def __init__(self, order_id, ticker, price, stocks, side, type="Market", maker=None):
        self.order_id = order_id
        self.ticker = ticker
        self.price = price
        self.stocks = stocks

        self.side = side
        self.type = type
        self.maker = maker


