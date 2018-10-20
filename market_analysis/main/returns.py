import matplotlib.pyplot as plt


class ReturnsStatistics:
    def compute_daily_returns(self, prices):
        daily_returns =  prices / prices.shift(1) - 1
        daily_returns.ix[0] = 0
        return daily_returns.ix[1:]
    # moze da se uporedi daily return sa indeksom ili drugim stock


    def get_monthly_returns(self, prices):
        data = prices.copy()
        monthly = data.resample("BM").apply(lambda x: x[-1])
        monthly.pct_change()
        return  monthly

    # def get_cummulative_day_return(self, prices):
    #     return

    def get_daily_returns_related_to_market(self, prices, market_prices):
        daily_returns = self.compute_daily_returns(prices)

        market_daily_returns = self.compute_daily_returns(market_prices)

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
