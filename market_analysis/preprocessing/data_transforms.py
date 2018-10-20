import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DataTransforms:

    def normalize_data(self, data, min_max_scaler = None):

        min_max_scaler = MinMaxScaler() if not min_max_scaler else min_max_scaler
        normalized = pd.DataFrame(min_max_scaler.fit_transform(data), index = data.index, columns = data.columns)
        return normalized

    def denormalize_data(self, data, min_max_scaler):
        inverse = min_max_scaler.inverse_transform(data)
        inverse = pd.DataFrame(inverse, index = data.index, columns = data.columns)
        return inverse

    def smooth_data(self, data, factor):
        rolling = data.rolling(window=factor)
        rolling_mean = rolling.mean()
        return  rolling_mean

    def fill_missing_data(self, df):
        df.fillna(method="ffill", inplace=True)
        df.fillna(method= "bfill", inplace=True)

        return df

    def remove_outliers(self, data, num_std=2):
        d = data.copy()
        for column in data.columns:
            self.remove_outliers_for_column(d[column], num_std)
        return d

    def remove_outliers_for_column(self, column_data, num_std=2):
        a = abs(column_data-column_data.mean()) > num_std * column_data.std()
        column_data[abs(column_data-column_data.mean()) > num_std * column_data.std()] = None
        self.fill_missing_data(column_data)

