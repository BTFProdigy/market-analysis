import traceback

import pandas as pd

from market_data_reader import DataReader


class DataReaderImpl(DataReader):

    def read_data(self, path, stock, start_date, end_date = None):
        # dates = pd.date_range(start_date, end_date)
        try:
            filename = path + stock.lower() + ".us.txt"
            df = pd.read_csv(filename, index_col="Date", parse_dates=True)

            if end_date != None:
                return df[start_date : end_date]
            else:
                return df[start_date : ]

        except Exception, e:
            traceback.print_exc()
            raise e



    def read_market_data(self, path, market, start_date, end_date):
        data = self.read_data(path, market, start_date, end_date)
        return data
        # for comparson