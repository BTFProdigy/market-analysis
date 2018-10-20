
class ModelConfiguration:
    def __init__(self):
        self.transformations = []
        self.transformation_parameters= {"predictions_ahead":5, "smoothing_factor":10}
        # self. transformation_parameters= {}

    def contains_transformation(self, transformation):
        return self.transformations.__contains__(transformation)

    def contains_transformation_parameter(self, parameter):
        return self.transformation_parameters.has_key(parameter)

    def get_transformation_parameter(self, parameter):
        return self.transformation_parameters[parameter]