import datetime
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose

from market_analysis.preprocessing import StationarityChecker


class SARIMAX:

    def __init__(self):
        self.stationarity_checker = StationarityChecker()

    def import_data(self, resampled):
        self.resampled = resampled

    def create_model(self, endog):
        self.data = endog
        self.data.dropna(inplace=True)
        # self.plot_acf(self.data[self.data.columns[0]].diff().dropna())
        # self.plot_pacf(self.data[self.data.columns[0]].diff().dropna())
        self.data.index = pd.DatetimeIndex(self.data.index)
        d = 0
        self.examine_process_order(self.data, 0, 0)
        # if not (self.stationarity_checker.is_stationary(endog[endog.columns[0]])):
        #     d = 1
        # model = sm.tsa.statespace.SARIMAX(self.data.values,
        #                                   trend='n',
        #                                   order=(0, 0, 1),
        #                                   seasonal_order=(0, 0, 0, 0), enforce_stationarity=True)
        #
        #
        # self.results = model.fit()


    def examine_process_order(self, endog, seasonal_d, d):
        min_aic = 99999999
        pq = self.get_p_q_combinations()

        for combination in pq:
            print combination
            try:
                model = sm.tsa.statespace.SARIMAX(endog.values,

                                                  order=(combination[0], d, combination[1]),
                                                  seasonal_order=(0, seasonal_d, 0, 0))

                results = model.fit(disp=False)
                if results.aic < min_aic:
                    print min_aic
                    min_aic = results.aic
                    min_param = combination
            except Exception, e:
                continue

        model = sm.tsa.statespace.SARIMAX(endog.values,

                                          trend='n',
                                          order=(min_param[0], d, min_param[1]),
                                          seasonal_order=(0, 0, 0, 0))

        self.results = model.fit()
        # print(self.results.summary())
        # print min_param
        # print(self.results.mle_retvals)

        # pred = self.results.get_prediction(start=pd.to_datetime('1998-01-01'), dynamic=False)
        # self.results.predict()


        # self.results.plot_diagnostics(figsize=(15, 12))
        plt.show()

    def predict(self, steps):

        # pred = self.results.get_prediction(start=self.data.shape[0], end = self.data.shape[0]+steps)
        pred = self.results.forecast(steps = steps)
        # pred = self.results.get_forecast(steps = steps, exog = exogenous_variables_future_values)
        forecasted_formatted = [float(format(value, 'f')) for value in pred]
        forecasted_frame = pd.DataFrame(forecasted_formatted, columns = self.data.columns)

        forecasted_frame.index = self.get_n_future_timestamps(self.data.index[-1], steps)
        return forecasted_frame
        return pred

    def get_future_timestamp(self, last_timestamp, index):
        future_timestamp  =last_timestamp + datetime.timedelta(days = index)
        return future_timestamp

    def get_n_future_timestamps(self, last_timestamp, n):
        week_day = last_timestamp.weekday()

        # indices = np.array(range(n))
        days = np.array(range(week_day+1, week_day + n+1))
        days /= 5
        days*=2

        indices = np.array(range(1, n+1))
        indices += days

        future_timestamps = []

        for i in indices:
            future_timestamps.append(self.get_future_timestamp(last_timestamp, i))

        return future_timestamps

    def get_p_q_combinations(self):

        p = q = range(0, 3)
        combinations= list(itertools.product(p, q))

        return combinations

    # def get_seasonal_p_q_combinations(self):
    #     p = q = range(0, 20)
    #     pq= list(itertools.product(p, q))
    #     return pq

    # while not self.stationarity_checker.are_all_series_stationary(data):
    #     data = data.diff()
    #     self.configuration.transformations.append("differencing")

    def get_seasonality(self, offset_in_minutes):
        return self.get_daily_seasonality(offset_in_minutes)

    def get_daily_seasonality(self, offset_in_minutes):
        return 24*60/offset_in_minutes

    def get_d(self):
        d = self.stationarity_checker.get_differencing_order(self.data)
        return d

    def plot_acf(self, series):
        # plot.autocorrelation_plot(self.data)

        plot_acf(series, lags=10)
        plt.title(series.name)
        plt.show()

    def plot_pacf(self, series):

        plot_pacf(series, lags = 10)
        plt.show()

    def check_components(self):

        result = seasonal_decompose(self.resampled.values, model='additive', freq = 24*60/15*1)

        result.plot()

        plt.show()

    def get_model_residuals(self):
        return self.results.resid
