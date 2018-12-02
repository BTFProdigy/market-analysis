import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

from market_analysis.preprocessing import DataTransforms


class ModelEvaluator:

    def __init__(self):
        self.data_transforms = DataTransforms()

    def evaluate_model(self, predictions, data):
        rmses = []

        for column in data.columns:
            rmse = self.evaluate_parameter_and_calculate_error(data[column], predictions[column], column)
            rmses.append(rmse)

        self.average_error = np.mean(rmses)
        print "Average rmse: " + str(np.mean(rmses))

    def evaluate_parameter_and_calculate_error(self, data, predictions, column):
        # predictions = pd.DataFrame(model.fittedvalues[column], copy=True)

        # if (self.configuration.contains_transformation("differencing")):
        #     predictions_arrima_cumsum = predictions.cumsum()
        #
        #     last_row= data[column].ix[-1]
        #
        #     predictions_original = pd.DataFrame(last_row, index = predictions_arrima_cumsum.index, columns = [column])
        #     predictions = predictions_original.add(predictions_arrima_cumsum, fill_value = 0)

        # if self.configuration.contains_transformation("scaling"):
        #     predictions = self.data_transforms.denormalize_data(predictions)

        self.plot_predicted_vs_real_for_column(column, predictions, data)

        root_mean_squared_dev = np.sqrt(np.sum((predictions - data.dropna())**2)/len(data))

        return root_mean_squared_dev


    def plot_predicted_vs_real_for_column(self, column, predicted, original):
        ax = original.plot()

        predicted.plot(ax = ax)
        ax.legend(["Real", "Predicted"])
        # ax.set_xlim(predicted.index[0], predicted.index[-1])

        plt.title(column)
        plt.show()

    # def plot_rmse_during_training(self, rmse):
    #     return

    def get_rmse(self, predicted, real):
        return np.sqrt(np.sum((predicted-real)**2)/len(real))

    def plot_residuals(self, residuals):
        residuals.plot()
        plt.title("Residuals")
        plt.show()
        return

    def inspect_residuals(self, resid):
        for column in resid.columns:
            self.plot_autocorrelation(resid[column])
            rolmean = resid[column].rolling(window=5, center=False).mean()

            rolstd = resid[column].rolling(window=5).std()

            resid[column].plot( color='blue', label='Original')
            rolmean.plot(color='red', label='Rolling Mean')
            rolstd.plot(color='black', label = 'Rolling Std')
            plt.legend(loc='best')

            plt.title('Rolling Mean & Standard Deviation, ' + column)
            plt.show()

    def ljung_box(self, resid):
        for column in resid.columns:
            print "Checking Ljung-Box"
            a = acorr_ljungbox(resid[column])
            print a

    def plot_scatter_real_predicted(self, real, predicted):
        real = real[predicted.index[0]:]
        predicted = predicted[:real.index[-1]]


        diff = list(set(real.index)-set(predicted.index))
        for d in diff:
            real.drop(index = d, inplace=True)

        diff1 = list(set(predicted.index)-set(real.index))
        for d in diff1:
            print d
            predicted.drop(index = d, inplace=True)

        for column in real.columns:
            plt.scatter(real[column], predicted[column], alpha=0.5)
            plt.title('Scatter plot ' + column)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()

    # def get_mape(self, predicted, real, lag_order):
    #
    #     real = real[lag_order:].as_matrix()
    #
    #
    #     return 2 * np.sum(abs(real-predicted)/(real + predicted)) / len(real)

    def get_mape(self, predicted, real):
        # residuals = self.results.resid.values
        # real = real[lag_order:].as_matrix()
        diff = list(set(real.index)-set(predicted.index))
        for d in diff:
            real.drop(index = d, inplace=True)

        diff1 = list(set(predicted.index)-set(real.index))
        for d in diff1:
            print d
            predicted.drop(index = d, inplace=True)

        return 2 * np.sum(abs((real-predicted)/(real + predicted))) / len(real)
        # return np.sum(abs((real-predicted)/real)) / len(real)

    def plot_cross_correlation(self, data, predictions, num_of_steps):
        diff = list(set(data.index)-set(predictions.index))
        for dr in diff:
            data.drop(index = dr, inplace=True)
        data.columns=["Shifted real"]

        d = data.shift(num_of_steps)
        pr = pd.concat([predictions, d], axis = 1)
        print pr.corr()


