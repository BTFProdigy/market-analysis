class DataApi:

    def start_collecting(self):
        raise Exception("Implemented in subclass")

    def get_trades(self, ticker, start_date, end_date):
        raise Exception("Implemented in subclass")