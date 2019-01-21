
import quandl


# def get_data(start, ticker):
#     data = quandl.get(ticker, start_date=start)
#     # data.columns=['Date', 'Close', 'Open', 'High', 'Low', 'Volume']
#     data.columns= ['Close']
#     # data = data.set_index('Date')
#     data = data.sort_index()
#     print data
#     return data
from market_analysis.deep_q_learning.data_api.data_api import DataApi


class QuandlDataApi(DataApi):

    def get_trades(self, ticker, start_date, end_date = None):
        quandl.ApiConfig.api_key = 'ddxfB5FM3NhZ3FN4PxTm'
        data = quandl.get_table('WIKI/PRICES', ticker = [ticker],
                                qopts = { 'columns': ['Date', 'adj_close', 'adj_open', 'adj_high', 'adj_low', 'volume'] },
                                date = { 'gte': start_date},
                                paginate=True)

        data.columns=['Date', 'Close', 'Open', 'High', 'Low', 'Volume']
        data = data.set_index('Date')
        data = data.sort_index()

        return data