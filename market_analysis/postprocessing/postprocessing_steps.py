import pandas as pd

from market_analysis.preprocessing import DataTransforms


class PostprocessingSteps:
    def __init__(self, configuration):
        self.configuration = configuration
        self.data_transforms = DataTransforms()

    def get_undifferenced_predictions(self, predictions, data, number):
        pred = pd.DataFrame(predictions, copy=True)
        cumsum = pred.cumsum()

        # for column in data.columns:
        #     cumsum[column] = cumsum[column].apply(lambda x: x + data[column].ix[-1])
        #     if number == 2:
        #         cumsum[column] = cumsum[column].apply(lambda x: x + data[column].diff().ix[-1])


        cumsum= cumsum.apply(lambda x: x + data.ix[-1])
        return cumsum

    def postprocess(self, forecasted, original_data):
        if self.configuration.contains_transformation("differencing"):
            number = self.configuration.transformations.count("differencing")
            forecasted = self.get_undifferenced_predictions(forecasted, original_data, number)


            # return cumsum
        if self.configuration.contains_transformation("scaling"):
            forecasted = self.data_transforms.denormalize_data(forecasted, self.configuration.get_transformation_parameter('scaling_model'))

        return forecasted