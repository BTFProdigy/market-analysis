class RealTimeTradingDataGetter:
    def __init__(self, db_worker, data_preprocessor):
        self.db_worker = db_worker
        self.data_preprocessor = data_preprocessor

    def get_data(self, ticker):
        data = self.db_worker.get_trades(ticker, 50)
        # preprocessed = self.data_preprocessor.preprocess_data(data,
        #                                                       self.agent_state.num_of_stocks,
        #                                                       self.agent_state.budget,
        #                                                       False)
        # return self.data_preprocessor.transform_price(data.ix[-1].values[0])
        return data.iloc[-1]
