import matplotlib.pyplot as plt


class ReturnsStatistics:
    def compute_returns(self, prices):
        daily_returns =  prices / prices.shift(1) - 1
        daily_returns.ix[0] = 0
        return daily_returns.ix[1:]
    # moze da se uporedi daily return sa indeksom ili drugim stock

    def get_monthly_returns(self, prices):
        data = prices.copy()
        monthly = data.resample("BM").apply(lambda x: x[-1])
        monthly.pct_change()
        return  monthly

    def get_weekly_returns(self, prices):
        data = prices.copy()
        daily_returns = data.pct_change()
        weekly = daily_returns.resample("W").sum()
        # weekly = data.resample("W").apply(lambda x: x[-1])
        # weekly.pct_change()
        return  weekly

    def get_daily_returns(self, prices):
        data = prices.copy()
        daily_returns = data.pct_change()
        daily = daily_returns.resample("D").sum()
        # weekly = data.resample("W").apply(lambda x: x[-1])
        # weekly.pct_change()
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

    # bolinger bands, momentum, pe ratio
    # predict change in price, market relative change in price
    # sharpe ratio
    # standard deviation of daily return is the sharp ratio
    #
    # idemo nazad po test podacima, trazimo korelaciju sa vremenskim serijama, uzmemo rezidual
    #
    # idemo nazad na sezone, nadjemo korelacije
    # ukljuciti i market
