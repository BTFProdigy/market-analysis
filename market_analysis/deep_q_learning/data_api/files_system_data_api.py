import traceback

import pandas as pd

from market_analysis.deep_q_learning.data_api.data_api import DataApi


class FileSystemDataApi(DataApi):

    def __init__(self, path):
        self.path = path

    def get_trades(self, stock, start_date, end_date = None):
        try:
            filename = self.path + stock.lower() + ".us.txt"
            df = pd.read_csv(filename, index_col="Date", parse_dates=True)

            if end_date != None:
                return df[start_date : end_date]
            else:
                return df[start_date : ]

        except Exception, e:
            traceback.print_exc()
            raise e



    def read_market_data(self, path, market, start_date, end_date):

        data = self.get_trades(path, market, start_date, end_date)
        return data