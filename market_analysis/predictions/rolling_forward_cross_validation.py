import pandas as pd

from market_analysis.predictions.algorithm.arima import Arima
from market_analysis.predictions import ModelConfiguration, ModelEvaluator
from market_analysis.postprocessing import PostprocessingSteps
from market_analysis.preprocessing import PreprocessingSteps


class RollingForwardCrossValidation:
    def __init__(self):
        self.predictor = Arima()
        self.predicted = pd.DataFrame()
        self.model_evaluator = ModelEvaluator()

    def validation(self, test_set_size, data):

        self.rmses = []
        self.mapes = []
        prediction_counter = 0
        training_data_len = int(0.7 * data.__len__())
        # training_data_len = 80
        max_data = data.__len__()
        while (training_data_len < max_data):
            print training_data_len, max_data
            configuration = ModelConfiguration()
            num_of_predictions_ahead = configuration.transformation_parameters["predictions_ahead"]
            training_data = data[:training_data_len]
            try:
                if prediction_counter % num_of_predictions_ahead == 0:
                    self.train(configuration, training_data, num_of_predictions_ahead)
                    print configuration.transformations
                prediction_counter+=1
                training_data_len += 1

                print training_data_len
            except ValueError, e:
                print e.message
        # self.predicted.columns = data.columns

        self.model_evaluator.evaluate_model(self.predicted, data)

        self.model_evaluator.plot_scatter_real_predicted(data, self.predicted)
        print "mape"
        print self.model_evaluator.get_mape(self.predicted, data)
        print "rmse"
        print self.model_evaluator.get_rmse(self.predicted, data)
        print self.model_evaluator.plot_cross_correlation(data, self.predicted, num_of_predictions_ahead)

    # def get_average_rmse(self):
    # #     take the average rmse of all the models
    #     return sum(self.rmses)/len(self.rmses)
    #
    # def get_average_mape(self):
    #     #     take the average rmse of all the models
    #     return sum(self.mapes)/len(self.mapes)

    def train(self, configuration, training_data, num_of_predictions_ahead):

        preprocessing_steps = PreprocessingSteps(configuration)
        postprocessing_steps = PostprocessingSteps(configuration)

        preprocessed = preprocessing_steps.preprocess_data(training_data)

        self.predictor.create_model(preprocessed)
        predictions = self.predictor.predict(num_of_predictions_ahead)
        postprocessed = postprocessing_steps.postprocess(predictions, training_data)

        # rmse = self.model_evaluator.get_rmse(postprocessed.values, training_data.values)
        # self.rmses.append(rmse)
        self.predicted = pd.concat([self.predicted, postprocessed])

        # mape = self.model_evaluator.get_mape(postprocessed, training_data)
        # self.mapes.append(mape)
