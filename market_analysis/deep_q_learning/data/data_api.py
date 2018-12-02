class DataApi:

    def start_collecting(self):
        raise Exception("Implemented in subclass")

    def get_trade(self, ticker):
        raise Exception("Implemented in subclass")