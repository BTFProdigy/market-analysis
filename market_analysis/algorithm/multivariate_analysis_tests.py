from johansen.johansen import Johansen

class MultivariateAnalysisTests:
    def cointegration_exists(self, data):
        data.dropna(inplace=True)
        parameters = data.columns

        johansen=Johansen(data, model = 2)
        johansen_result = johansen.johansen()
        print "Johansen result "
        print johansen_result

    def granger_causality_test(self, data, results):
        parameters = data.columns.get_values()
        parameters = list(data.columns.get_values())
        print "Granger"

        for parameter in parameters:
            chosen_parameter = set()
            chosen_parameter.add(parameter)

            other_parameters = list(set(parameters) - chosen_parameter)

            for parameter2 in parameters:
                # grangercausalitytests(data[[parameter, parameter2],[parameter2, parameter]], results.k_ar)
                print results.test_causality(parameter, parameter2)
            results.test_causality(parameter, other_parameters)
        return