from sklearn.preprocessing import MinMaxScaler, StandardScaler

from market_analysis.features import DateFrameBuilder
from market_analysis.preprocessing import DataTransforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.externals import joblib
class DataPreprocessor:

    _instance = None

    @staticmethod
    def get_instance():
        if DataPreprocessor._instance is None:
            DataPreprocessor._instance = DataPreprocessor()
        return DataPreprocessor._instance

    def __init__(self):
        self.minmaxscaler = MinMaxScaler()
        self.stocks_scaler = MinMaxScaler()
        self.budget_scaler = MinMaxScaler()

        self.data_transforms = DataTransforms()
        self.scaling = True

    def preprocess_data(self, data, stocks, budget, training = True):
        if data.size != 0:
            dataframe = self.build_dataframe(data)

            dataframe.dropna(inplace=True)
            # dataframe = data_transforms.remove_outliers(dataframe, 2.5)
            # dataframe = data_transforms.normalize_data(dataframe)

            dataframe = self.data_transforms.smooth_data(dataframe, 10)
            dataframe = self.data_transforms.fill_missing_data(dataframe)

            dataframe.plot()
            plt.show()

            if self.scaling:
                budget_min_max = np.array([0, budget*2])
                stocks_min_max = np.array([0, stocks*2])

                if training:
                    self.minmaxscaler.fit_transform(dataframe)
                    self.stocks_scaler.fit(stocks_min_max.reshape(-1,1))
                    self.budget_scaler.fit(budget_min_max.reshape(-1,1))
                else:
                    self.minmaxscaler.transform(dataframe)
            # return dataframe, self.transform_stocks(stocks), self.transform_budget(budget)

            # return  dataframe.diff().fillna(0)
            return dataframe

        else: raise ValueError("There is no data")

    def transform_stocks(self, stocks):
        if self.scaling:
            return self.stocks_scaler.transform(stocks)[0][0]
        else:
            return stocks

    def transform_price_batch(self, data):
        if self.scaling:
            return pd.DataFrame(self.minmaxscaler.transform(data), index = data.index, columns = data.columns)
            # return data.diff().fillna(0)
        else:
            return data
    def transform_price(self, price):
        if self.scaling:
            return self.minmaxscaler.transform(price)[0][0]
        else:
            return price

    def transform_budget(self, budget):
        if self.scaling:
            return self.budget_scaler.transform(budget)[0][0]
        else:
            return budget

    def inverse_transform_stocks(self, stocks):
        if self.scaling:
            return self.stocks_scaler.inverse_transform(stocks)[0][0]
        else:
            return stocks


    def inverse_transform_price(self, data):
        if self.scaling:
            return self.minmaxscaler.inverse_transform(data)[0][0]
        else:
            return data

    def inverse_transform_budget(self, budget):
        if self.scaling:
            return self.budget_scaler.inverse_transform(budget)[0][0]
        else:
            return budget

    def build_dataframe(self, data):
        dataframe = (
            DateFrameBuilder(data)
                # .add_returns()
                # .add_bolinger_bands_diff(7)
                # .add_sharp_ratio(30)
                # .add_cummulative_daily_returns()
                # .add_daily_returns()
                .build()
        )
        return dataframe

    def save_scalars(self, folder):
        if folder[-1] != '/':
            folder = folder + '/'
        joblib.dump(self.minmaxscaler, folder + "price")
        joblib.dump(self.stocks_scaler, folder + "stocks")
        joblib.dump(self.budget_scaler, folder + "budget")

    def load_scalers(self, folder):
        if folder[-1] != '/':
            folder = folder + '/'
        self.minmaxscaler = joblib.load(folder + "price")
        self.stocks_scaler = joblib.load(folder + "stocks")
        self.budget_scaler =joblib.load(folder + "budget")



