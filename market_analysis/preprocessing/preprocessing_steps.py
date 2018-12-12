
from sklearn.preprocessing import MinMaxScaler

from data_transforms import DataTransforms
from stationarity_checker import StationarityChecker


class PreprocessingSteps:
    def __init__(self, configuration):
        self.configuration = configuration
        self.stationarity_checker = StationarityChecker()
        self.data_transforms = DataTransforms()

    def preprocess_data(self, data):

        self.data_transforms.fill_missing_data(data)

        if self.configuration.contains_transformation("scaling"):
            min_max_scaler = MinMaxScaler()

            data = self.data_transforms.normalize_data(data, min_max_scaler)
            self.configuration.transformation_parameters["scaling_model"] = min_max_scaler
            self.scaled =data

        if self.configuration.contains_transformation("smoothing") and \
                self.configuration.contains_transformation_parameter("smoothing_factor"):
            data = self.data_transforms.smooth_data(data, self.configuration.transformation_parameters["smoothing_factor"])
        # data = self.data_transforms.remove_outliers(data)
        # return self.convert_to_stationary(data)
        return data

    def convert_to_stationary(self, data):
        data.dropna(inplace=True)
        num = 0
        for column in data.columns:
            timeseries = data[column]
            self.stationarity_checker.test_stationarity_visually(timeseries)
        # while not self.stationarity_checker.are_all_series_stationary(data):
        #     if (num<1):
        #         data = data.diff()
        #         self.configuration.transformations.append("differencing")
        #     else: break
        #     num+=1
        if not not self.stationarity_checker.are_all_series_stationary(data):
            data = data.diff()
            self.configuration.transformations.append("differencing")


        data.dropna(inplace=True)
        return data

    def convert_series_to_stationary(self, data):
        data.dropna(inplace=True)
        num = 0
        self.stationarity_checker.test_stationarity_visually(data)
        if not self.stationarity_checker.is_stationary(data):
            data = data.diff()
            self.configuration.transformations.append("differencing")

        data.dropna(inplace=True)
        return data