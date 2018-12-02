import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.api import VAR
from market_analysis.predictions.algorithm.prediction_algorithm import PredictionAlgorithm

class VARPredictor (PredictionAlgorithm):

    def create_model(self, data):
        self.data = data.copy()
        print self.data.shape
        self.data.dropna(inplace=True)

        if self.data.__len__() == 0:
            raise ValueError("No enough data to create a model.")

        model = VAR(self.data)

        # for column in data.columns:
        #     self.plot_acf(data[column])
        #     self.plot_pacf(data[column])

        self.results = model.fit(ic = "aic", maxlags=10)

        # self.results.plot_acorr(nlags = self.results.k_ar)
        plt.show()

        # print self.results.resid_acorr()
        # print self.results.forecast_interval(self.results.y, steps = 1)
        # self.evaluate_model_visually()
        # mape = self.get_mape()
        # print mape
        return

    def plot_acf(self, series):

        plot_acf(series, lags = 10)
        plt.title("acf: " + series.name)
        plt.show()

    def plot_pacf(self, series):

        plot_pacf(series, lags = 10)
        plt.title("pacf: " + series.name)
        plt.show()

    def evaluate_model_visually(self):
        rmses = []

        for column in self.data.columns:
            rmse = self.evaluate_parameter_and_calculate_error(column)
            rmses.append(rmse)

        self.average_error = np.mean(rmses)
        print "Average rmse: " + str(np.mean(rmses))

    def get_lag_order(self):
        return self.results.k_ar

    def get_fitted_values(self):
        # return pd.DataFrame(self.results.fittedvalues, copy=True)
        return self.results.fittedvalues

    # def get_average_rmse_error(self):
    #     residuals = self.results.resid.values
    #     return np.sqrt(np.sum((residuals)**2)/len(self.data))
        # return abs(self.results.resid).values.mean()

    def predict(self, num_of_values):
        # self.results.plot_forecast(num_of_values)
        # plt.show()
        lag_order = self.results.k_ar
        print "Lag order: {}".format(lag_order)

        if lag_order > self.data.values.__len__():
            lag_order = self.data.values.__len__()

        forecasted = self.results.forecast(self.data.values[-lag_order:], num_of_values)

        forecasted_formatted = [[float(format(value, 'f')) for value in one_moment] for one_moment in forecasted]
        forecasted_frame = pd.DataFrame(forecasted_formatted, columns = self.data.columns)

        forecasted_frame.index = self.get_n_future_timestamps(self.data.index[-1], num_of_values)
        return forecasted_frame

    # def predict_interval(self, num_of_values):
    #     lag_order = self.results.k_ar
    #     forecast_interval = self.results.forecast_interval(self.data.values[-lag_order:], num_of_values)
    #
    #     by_intervals = [[instance for instances in bound] for bound in forecast_interval[0:3:2]]
    #     lower = forecast_interval[0]
    #     higher = forecast_interval[2]
    #
    #     forecasted_lower = [[float(format(value, 'f')) for value in one_moment] for one_moment in lower]
    #     forecasted_higher = [[float(format(value, 'f')) for value in one_moment] for one_moment in higher]
    #
    #
    #
    #     low_high_columns = []
    #     for cols in self.data.columns():
    #         low_high_columns.append(cols+ "-High")
    #         low_high_columns.append(cols + "-Low")
    #
    #     forecasted_frame = pd.DataFrame(columns = low_high_columns)
    #
    #
    #     forecasted_frame.index = self.get_n_future_timestamps(self.data.index[-1], num_of_values)


    def get_mse(self, number_of_steps_ahad):
        return self.results.mse(number_of_steps_ahad)

    def get_mape(self):
        residuals = self.results.resid.values
        data = self.data[self.get_lag_order():].as_matrix()

        res = residuals[:, 0]
        data = data[:, 0]

        fitted = self.results.fittedvalues.as_matrix()[:, 0]
        return 2 * np.sum(abs(res)/(data + fitted)) / len(data)

    def get_model_results(self):
        return self.results

    def get_future_timestamp(self, last_timestamp, index):
        future_timestamp  =last_timestamp + datetime.timedelta(days = index)
        return future_timestamp

    def get_n_future_timestamps(self, last_timestamp, n):
        week_day = last_timestamp.weekday()
        days = np.array(range(week_day+1, week_day + n+1))
        days /= 5
        days*=2

        indices = np.array(range(1, n+1))
        indices += days

        future_timestamps = []

        for i in indices:
            future_timestamps.append(self.get_future_timestamp(last_timestamp, i))

        return future_timestamps

