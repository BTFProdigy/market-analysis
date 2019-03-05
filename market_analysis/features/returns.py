import matplotlib.pyplot as plt


class ReturnsStatistics:
    def compute_returns(self, prices):
        daily_returns =  prices / prices.shift(1) - 1
        daily_returns.ix[0] = 0
        return daily_returns.ix[1:]

    def get_monthly_returns(self, prices):
        data = prices.copy()
        monthly = data.resample("BM").apply(lambda x: x[-1])
        monthly.pct_change()
        return  monthly

    def get_weekly_returns(self, prices):
        data = prices.copy()
        daily_returns = data.pct_change()
        weekly = daily_returns.resample("W").sum()
        return  weekly

    def get_daily_returns(self, prices):
        data = prices.copy()
        daily_returns = data.pct_change()
        daily = daily_returns.resample("D").sum()
        return  daily

    def get_cummulative_return(self, prices):

        cumm = prices / prices.iloc[0]
        cumm.ix[0]=0
        return cumm

    def get_daily_returns_related_to_market(self, prices, market_prices):
        daily_returns = self.compute_returns(prices)

        market_daily_returns = self.compute_returns(market_prices)

        daily_returns.plot(label = "Stock")
        market_daily_returns.plot(label = "Market")

        plt.legend()
        plt.show()

        return

