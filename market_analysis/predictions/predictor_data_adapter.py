from itertools import groupby
import numpy as np
import pandas as pd

class PredictorDataAdapter:
    def adapt_data(self, data):

        for x in data:
            x[x['name']] = x['value']
            del x['name']
            del x['value']

        data.sort(key = lambda x : x['timestamp'])

        parameters_aggregated = []
        for timestamp,values in groupby(data, key = lambda x : x['timestamp']):
            grouped_parameters = list(values)

            if len(grouped_parameters) > 1:
                for index in range(1, len(grouped_parameters)):
                    grouped_parameters[0].update(grouped_parameters[index])
            parameters_aggregated.append(grouped_parameters[0])

        data_frame = pd.DataFrame(parameters_aggregated, index = [x['timestamp'] for x in parameters_aggregated], dtype = np.float64)

        if data_frame.columns.contains('timestamp'):
            del data_frame['timestamp']

        sorted_data_frame = data_frame.sort_index()
        sorted_data_frame = sorted_data_frame.ix['2018-12-19 23:18:50':'2018-12-20 04:36:35',]

        return sorted_data_frame