import cPickle

from market_analysis.deep_q_learning import paths
# from market_analysis.deep_q_learning.neural_net.neural_net_keras import NeuralNetwork
from market_analysis.deep_q_learning.neural_net.neural_net import NeuralNet


class ModelPersister:
    @staticmethod
    def restore_model(model):
        model_path = paths.get_models_path()+model
        model_parameters = ModelPersister.load_model_parameters(model_path + "parameters")
        nn = NeuralNet(model_parameters.input_size, model_parameters.output_size,
                           [model_parameters.num_hidden_nodes1, model_parameters.num_hidden_nodes2, model_parameters.num_hidden_nodes2],
                           [model_parameters.activation_function1, model_parameters.activation_function2, model_parameters.activation_function2])

        # nn = NeuralNetwork(*model_parameters.__dict__.values)
        nn = nn.restore_model(model_path)
        return nn

    @staticmethod
    def load_model_parameters(file_name):
        with open(file_name, 'rb') as file:
            return cPickle.load(file)

