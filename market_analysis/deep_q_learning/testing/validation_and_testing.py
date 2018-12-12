import cPickle
from collections import namedtuple

import numpy as np

from market_analysis.deep_q_learning import paths
from market_analysis.deep_q_learning.action import Action
from market_analysis.deep_q_learning.deep_q import DeepQ
from market_analysis.deep_q_learning.deep_q_statistics import DeepQStatistics
from market_analysis.deep_q_learning.environment.data_preprocessor import DataPreprocessor
from market_analysis.deep_q_learning.evaluation import Evaluation
from market_analysis.deep_q_learning.exploration.greedy_strategy import GreedyStrategy
from market_analysis.deep_q_learning.neural_net import ActivationFunction
from market_analysis.deep_q_learning.neural_net.neural_network import NeuralNetwork
from market_analysis.deep_q_learning.replay_memory import ReplayMemory

Conf = namedtuple('Conf', ['hidden_nodes1', 'hidden_nodes2', 'act_f1', 'act_f2'])

class TestAndValidation:
    def __init__(self, env_builder, num_of_stocks, budget, tester):
        self.env_builder = env_builder
        self.init_num_of_of_stocks = num_of_stocks
        self.init_budget = budget
        self.tester = tester

    def split_train_test(self, data):
        size = data.shape[0]
        train_size = int(0.7*size)

        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]

        return train_data, test_data

    def split_train_validation_test(self, data):
        size = data.shape[0]
        train_size = int(0.5*size)
        validation_size = int(0.3*size)

        train_data = data.iloc[:train_size]
        validation_data = data.iloc[train_size:train_size+validation_size]
        test_data = data.iloc[train_size+validation_size:]

        return train_data, validation_data, test_data

    def train_validate_test(self, data):
        train_data, val_data, test_data = self.split_train_validation_test(data)

        confs_parameters= [(20, 20, ActivationFunction.Relu, ActivationFunction.Relu),
                          (20, 20, ActivationFunction.Tanh, ActivationFunction.Tanh),
                          (10, 10, ActivationFunction.Relu, ActivationFunction.Relu),
                          (10, 10, ActivationFunction.Tanh, ActivationFunction.Tanh)]

        confs = [Conf(*conf_parameters) for conf_parameters in confs_parameters]

        models = self.train(confs, train_data)

        best_model = self.validate(models, val_data)

        self.test(best_model, test_data)
        return best_model

    def validate(self, models, data):
        validation_rewards = []

        for model in models:
            env = self.env_builder.build_train_environment(self.init_num_of_of_stocks, self.init_budget, data, DataPreprocessor.get_instance())

            reward, budget, stocks = self.tester.test(model, env, data)
            validation_rewards.append(reward)

        best_model_index = np.argmax(np.array(validation_rewards))
        best_model = models[best_model_index]
        return best_model


    def train_test(self, data):
        train_data, test_data = self.split_train_test(data)
        confs = [Conf(20, 20, ActivationFunction.Relu, ActivationFunction.Relu)]

        net = self.train(confs, train_data)[0]
        self.test(net, test_data)

        return net

    def train(self, confs, data):
        models = []

        for conf in confs:
            train_env = self.env_builder.build_train_environment(self.init_num_of_of_stocks, self.init_budget, data, DataPreprocessor.get_instance())
            model = self.get_model(conf, train_env, data)

            models.append(model)

        return models

    def test(self, net, data):
        test_env = self.env_builder.build_train_environment(self.init_num_of_of_stocks, self.init_budget, data, DataPreprocessor.get_instance())

        return self.tester.test(net, test_env, data)

    def create_net(self, num_of_features, num_of_actions, hidden_nodes1, hidden_nodes2, act_f1, act_f2):
        nn = NeuralNetwork(num_of_features, num_of_actions, hidden_nodes1, hidden_nodes2, act_f1, act_f2)
        return nn

    def get_model(self, conf, env, data):
        num_of_actions = Action.num_of_actions

        mem = ReplayMemory(2000)
        evaluation = Evaluation()
        statistics = DeepQStatistics(env.get_num_of_states_per_training_episode())

        nn = self.create_net(env.num_of_features, num_of_actions, *conf)

        num_of_iterations = 3
        epsilon_strategy = GreedyStrategy(num_of_actions, num_of_iterations, env.get_num_of_states_per_training_episode())

        deep_q = DeepQ(nn, env, mem, statistics, num_of_actions, env.num_of_features, epsilon_strategy, num_of_iterations)
        deep_q.iterate_over_states()

        evaluation.plot_actions_during_time(data['Close'], statistics.actions_for_last_iteration)

        return deep_q.neural_network

    def test_existing_model(self, model, data):
        model = self.restore_model(model)
        return self.test(model, data)

    def validate_existing_models(self,models_names, data):
        models = []
        for model_name in models_names:
            model = self.restore_model(model_name)
            models.append(model)
        return self.validate(models, data)


    def load_model_parameters(self, file_name):
        with open(file_name, 'rb') as file:
            return cPickle.load(file)

    def restore_model(self, model):
        model_path = paths.get_models_path()+model
        model_parameters = self.load_model_parameters(model_path + "parameters")
        nn = NeuralNetwork(model_parameters.input_size, model_parameters.output_size,
                           model_parameters.num_hidden_nodes1, model_parameters.num_hidden_nodes2,
                           model_parameters.activation_function1, model_parameters.activation_function2)

        # nn = NeuralNetwork(*model_parameters.__dict__.values)
        nn = nn.restore_model(model_path + "model")
        return nn


