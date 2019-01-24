class FakeRealTimeTradingDataGetter:
    def __init__(self, db_worker, data_preprocessor, start_timestamp, end_timestamp = None):

        self.db_worker = db_worker
        self.data_preprocessor = data_preprocessor
        self.start_time = start_timestamp
        self.limit = 1
        # self.data = self.get_fake_real_data(ticker, start_timestamp)

    def get_data(self, ticker):
        instance= self.db_worker.get_trades_for_fake_trading(ticker, self.start_time, self.limit)
        self.limit+=1

        return instance

    def is_new_data_present(self, ticker):
        return True


    def get_new_data(self, ticker):

        return self.get_data(ticker)