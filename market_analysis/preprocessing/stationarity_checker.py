import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller

class StationarityChecker:
    def is_stationary(self, timeseries):
        dftest = adfuller(timeseries, regression = "ct", autolag='AIC')

        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        test_statistic = dfoutput['Test Statistic']
        # print "PP " + str(dfoutput['p-value'])
        critical_value = dftest[4]['5%']
        # print "Test statistic: " +timeseries.name + " "+ str(test_statistic)
        # print "Critical value " + str(critical_value)
        return test_statistic < critical_value

    def are_all_series_stationary(self, data_frame):
        data_frame.dropna(inplace=True)
        all_series_stationary = True

        for column in data_frame.columns:
            is_stationary = self.is_stationary(data_frame[column])
            all_series_stationary &= is_stationary
        return all_series_stationary

    def test_stationarity_visually_for_all_parameters(self, data):
        for column in data.columns:
            self.test_stationarity_visually(data[column])
        return

    def test_stationarity_visually(self, timeseries):

        print 'Results of Dickey-Fuller Test:' + timeseries.name
        dftest = adfuller(timeseries, regression = "ct", autolag='AIC')

        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        print dfoutput

        rolmean = timeseries.rolling(window=30, center=False).mean()

        rolstd = timeseries.rolling(window=30).std()

        timeseries.plot( color='blue', label='Original')
        rolmean.plot(color='red', label='Rolling Mean')
        rolstd.plot(color='black', label = 'Rolling Std')
        plt.legend(loc='best')

        plt.title('Rolling Mean & Standard Deviation ' + timeseries.name)
        plt.show()
