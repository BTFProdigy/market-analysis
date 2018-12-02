import math

from returns import ReturnsStatistics


class TechnicalFeaturesCalculator:

    # one number
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

    # odstupanja su u nekom delu veca, u nekom manja
    # higher and lower volatility
    # kada presece bb, tj.kada je ovo 1, to je signal
    def get_bolinger_bands_diff(self, prices, window, num_std = 2):
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()

        return (prices - sma) / num_std*std

    # for visualization only
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

    # over some number of days how much has the price changed

    # momentum and roc show trend (and risk?)
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


    # strong momentum and crossing the line - sign for technicians
    def get_simple_moving_averge(self, prices, window):
        return prices.rolling(window = window).mean()

# it is estimated that stocks only trend about 30% of the time.
# The rest of the time they move sideways in trading ranges. This is what a trading range looks like:
# you only want trade stocks that are trending

    def get_trend(self, prices, window):
            sma = prices.rolling(window = window).mean()
            return sma /sma.shift(5)

        # kao i momentum, varira o -0.5 do 0.5
        # sma is like true value ofthe company

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

    # high i low se koriste vise za daily trading
    def get_high_low_ratio(self, high, low):
        return high/low

    # ako je rizicno, da li vise tradevati ili manje (daily ili redje)
    # zelimo da prodamo rizicnu

    def get_acceleration_bands(self, high, low):
        high_band = high* (1+4* (high-low)/(high+low))
        high_band_average = high_band.rolling(window = 20).mean()

        low_band = low*(1-4*(high - low)/(high+low))
        low_band_average = low_band.rolling(window = 20).mean()

        return (low_band_average, high_band_average)

    def get_returns(self, prices):
        # daily_returns =  prices / prices.shift(1) - 1
        #
        # return daily_returns.ix[1:]

        daily_returns_statistics = ReturnsStatistics()
        return daily_returns_statistics.compute_returns(prices) * 100

    def get_weekly_returns(self, prices):
        returns_statistics = ReturnsStatistics()
        return returns_statistics.get_weekly_returns(prices).values

    def get_daily_returns(self, prices):
        returns_statistics = ReturnsStatistics()
        returns = returns_statistics.get_daily_returns(prices)

        # returns = returns.resample(str(resample_factor)+str(resample_unit)).bfill()
        return returns