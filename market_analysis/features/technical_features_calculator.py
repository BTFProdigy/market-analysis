import math

from returns import ReturnsStatistics


class TechnicalFeaturesCalculator:

    def get_sharpe_ratio(self, prices):
        daily_returns_statistics = ReturnsStatistics()
        daily_returns = daily_returns_statistics.compute_returns(prices)

        return math.sqrt(252) * daily_returns.mean() / daily_returns.std()

    def get_sharpe_ratio_series(self, prices, resample_factor, resample_unit):
        daily_returns_statistics = ReturnsStatistics()
        daily_returns = daily_returns_statistics.compute_returns(prices)

        window = 20
        mean = daily_returns.rolling(window=window).mean()
        std = daily_returns.rolling(window=window).std()

        if resample_unit == "d":
            factor = 252
            factor /= resample_factor

        if resample_unit == "s":
            factor = 252*24*60*60
            factor /= resample_factor

        # return math.sqrt(factor) * mean / std
        return mean / std

    def get_bolinger_bands_diff(self, prices, window, num_std = 2):
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()

        return (prices - sma) / num_std*std

    def get_bolinger_bands(self, prices, window=20):
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()

        return (sma - 2*std, sma + 2*std)

    def get_cummulative_returns(self, prices):
        daily_returns_statistics = ReturnsStatistics()
        return daily_returns_statistics.get_cummulative_return(prices)

    def get_general_volatility(self, prices):
        return prices.std()

    def get_rolling_volatility(self, prices, window):
        daily_returns = self.get_returns(prices)
        std = daily_returns.rolling(window=window).std()
        return std

    def get_price_rate_of_change(self, prices, window=1):
        return prices / prices.shift(window) - 1

    def get_momentum(self, prices, window=1):
        return prices - prices.shift(window)

    def get_rsi_index(self, prices, period=14):
        diff = prices.diff()

        up, down = diff[1:].copy(), diff[1:].copy()

        up[up<0] = 0
        up_c = up.rolling(period).mean()

        down[down>0] = 0
        down_c = down.rolling(period).mean().abs()

        rs = up_c / down_c
        # return 100 * up_c / (up_c + down_c)
        return 100 - 100/(1+rs)


    def get_simple_moving_averge(self, prices, window):
        return prices.rolling(window = window).mean()


    def get_trend(self, prices, window):
            sma = prices.rolling(window = window).mean()
            return sma /sma.shift(5)

    def get_distance_from_sma(self, prices, window):
        return prices / prices.rolling(window=window).mean() - 1

############## Volume

    def get_volume(self, volume):
        return volume.fillna(0)

    def get_price_volume_trend(self, prices, volumes):

        diff = prices.diff().fillna(0)

        previous_closes = prices.shift(1)
        diff_close_previous_close_ratio = diff / previous_closes
        diff_close_previous_close_ratio*=volumes

        return diff_close_previous_close_ratio / previous_closes

############## High and Low

    def get_high_low_ratio(self, high, low):
        return high/low

    def get_acceleration_bands(self, high, low):
        high_band = high* (1+4* (high-low)/(high+low))
        high_band_average = high_band.rolling(window = 20).mean()

        low_band = low*(1-4*(high - low)/(high+low))
        low_band_average = low_band.rolling(window = 20).mean()

        return (low_band_average, high_band_average)

    def get_returns(self, prices):
        daily_returns_statistics = ReturnsStatistics()
        return daily_returns_statistics.compute_returns(prices) * 100

    def get_weekly_returns(self, prices):
        returns_statistics = ReturnsStatistics()
        return returns_statistics.get_weekly_returns(prices).values

    def get_daily_returns(self, prices):
        returns_statistics = ReturnsStatistics()
        returns = returns_statistics.get_daily_returns(prices)

        return returns