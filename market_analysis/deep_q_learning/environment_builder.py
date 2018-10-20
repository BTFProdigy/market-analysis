from sklearn.preprocessing import StandardScaler, MinMaxScaler

from market_analysis.deep_q_learning.environment import Environment
from market_analysis.main import DateFrameBuilder

from market_analysis.preprocessing.data_transforms import DataTransforms

class EnvironmentBuilder:

    def __init__(self, reward):
        self.reward = reward

    def build_environment(self, num_of_stocks, budget, data):

        if data.size != 0:
            original_close= data.Close
            dataframe = self.build_dataframe(data)
            dataframe.dropna(inplace=True)
            data_transforms = DataTransforms()
            # dataframe = data_transforms.remove_outliers(dataframe, 2.5)
            # dataframe = data_transforms.normalize_data(dataframe)
            dataframe = data_transforms.smooth_data(dataframe, 5)
            dataframe = data_transforms.fill_missing_data(dataframe)
            minmaxscaler = MinMaxScaler(copy=False)
            minmaxscaler.fit_transform(dataframe[['Close']])
            standscaler = StandardScaler(copy=False)
            standscaler.fit_transform(dataframe[['Bollinger Band Diff', 'Daily Returns']])
            # standscaler.fit_transform(dataframe)

            environment = Environment(original_close, dataframe, self.reward, num_of_stocks, budget)
            return environment

        else: raise ValueError("There is no data")


    def build_dataframe(self, data):
        dataframe = (
            DateFrameBuilder(data)
                .add_daily_returns()
                .add_bolinger_bands_diff(7)
                .build()
        )
        return dataframe