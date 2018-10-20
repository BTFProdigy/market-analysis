import pandas as pd

from technical_features_calculator import TechnicalFeaturesCalculator


class DateFrameBuilder:

    def __init__(self, data):
        self.tfc = TechnicalFeaturesCalculator()

        self.data = data
        self.close_prices = self.data['Close']

        self.dataframe = pd.DataFrame()
        self.dataframe = pd.concat([self.close_prices], axis =1)

    #     self.dispatcher = {"bolinger_bands_diff": tfc.get_bolinger_bands_diff,
    #                        "bolinger_bands": tfc.get_bolinger_bands,
    #
    #                        "rolling_volatility": tfc.get_rolling_volatility,
    #                        "roc": tfc.get_price_rate_of_change,
    #
    #                        "momentum": tfc.get_momentum,
    #                        "rsi_index": tfc.get_rsi_index,
    #                        "trend": tfc.get_price_volume_trend,
    #
    #                        "volume": tfc.get_volume,
    #                        "price_volume_trend": tfc.get_price_volume_trend,
    #
    #                        "high_low_ratio": tfc.get_high_low_ratio,
    #                        "acceleration_bands": tfc.get_acceleration_bands}
    #
    # def build_dataframe(self, prices, features, **kwargs):
    #     for feature in features:
    #         self.get_feature(feature, prices,)
    #
    #
    # def get_feature(self, feature):
    #     self.dispatcher[feature]

    # def concat_dataframes(self, dataframe1):
    #     self.dataframe = pd.concat([self.dataframe, dataframe1])

    def add_bolinger_bands(self, window):
        bolinger_bands = self.tfc.get_bolinger_bands(self.close_prices, window)
        self.dataframe["Bollinger Lower Band"] = bolinger_bands[0]
        self.dataframe["Bollinger Higher Band"] = bolinger_bands[1]
        return self

    def add_bolinger_bands_diff(self, window, num_std = 2):
        self.dataframe["Bollinger Band Diff"] = self.tfc.get_bolinger_bands_diff(self.close_prices, window, num_std)
        return self

    def add_rolling_volatility(self, window):
        self.dataframe["Volatility"] = self.tfc.get_rolling_volatility(self.close_prices, window)
        return self

    def add_roc(self, window=1):
        self.dataframe["Price Rate of Change"] = self.tfc.get_price_rate_of_change(self.close_prices, window)
        return self

    def add_momentum(self, window=1):
        self.dataframe["Momentum"] =self.tfc.get_momentum(self.close_prices, window)
        return self

    def add_rsi_index(self, period = 14):
        self.dataframe["RSI Index"] = self.tfc.get_rsi_index(self.close_prices, period)
        return self

    def add_trend(self, window):
        self.dataframe["Trend"] = self.tfc.get_trend(self.close_prices, window)
        return self

    def add_distance_from_sma(self, window):
        self.dataframe["Distance from SMA"] = self.tfc.get_distance_from_sma(self.close_prices, window)
        return self

    def add_volume(self):
        self.dataframe["Volume"] = self.tfc.get_volume(self.data.Volume)
        return self

    def add_price_volume_trend(self):
        self.dataframe["Price Volume Trend"] = self.tfc.get_price_volume_trend(self.close_prices, self.data.Volume)
        return self

    def add_high_low_ratio(self):
        self.dataframe["High/Low ratio"] = self.tfc.get_high_low_ratio(self.data.High, self.data.Low)
        return self

    def add_acceleration_bands(self):
        acceleration_bands = self.tfc.get_acceleration_bands(self.data.High, self.data.Low)

        self.dataframe["Acceleration Lower Band"] = acceleration_bands[0]
        self.dataframe["Acceleration Higher Band"] = acceleration_bands[1]

        return self

    def add_daily_returns(self):
        self.dataframe["Daily Returns"] = self.tfc.get_daily_returns(self.data.Close)
        return self

    def build(self):
        return self.dataframe



