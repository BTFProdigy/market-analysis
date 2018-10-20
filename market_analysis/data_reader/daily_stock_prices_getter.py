# import fix_yahoo_finance as yf
# data = yf.download('AAPL','2016-01-01','2018-01-01')
# data.Close.plot()
# plt.show()


# start = datetime.datetime(2010, 1, 1)
#
# end = datetime.datetime(2013, 1, 27)
#
# f = web.DataReader('F', 'google', start, end)
#
# f.Close.plot()
# plt.show()



import quandl


# def get_data(start, ticker):
#     data = quandl.get(ticker, start_date=start)
#     # data.columns=['Date', 'Close', 'Open', 'High', 'Low', 'Volume']
#     data.columns= ['Close']
#     # data = data.set_index('Date')
#     data = data.sort_index()
#     print data
#     return data



def get_data(ticker, start_date):
    quandl.ApiConfig.api_key = 'ddxfB5FM3NhZ3FN4PxTm'
    data = quandl.get_table('WIKI/PRICES', ticker = [ticker],
                            qopts = { 'columns': ['Date', 'adj_close', 'adj_open', 'adj_high', 'adj_low', 'volume'] },
                            date = { 'gte': start_date},
                            paginate=True)

    data.columns=['Date', 'Close', 'Open', 'High', 'Low', 'Volume']
    data = data.set_index('Date')
    data = data.sort_index()

    return data