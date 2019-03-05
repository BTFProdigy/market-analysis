from collections import namedtuple

import numpy as np

from market_analysis.deep_q_learning.evaluation.deep_q_statistics import DeepQStatistics
from market_analysis.deep_q_learning.evaluation.evaluation import Evaluation
from market_analysis.deep_q_learning.exploration.linear_greedy_strategy import LinearGreedyStrategy
from market_analysis.deep_q_learning.neural_net.model_persister import ModelPersister
from market_analysis.deep_q_learning.neural_net.neural_net import NeuralNet
from market_analysis.deep_q_learning.preprocessing.data_preprocessor import DataPreprocessor
from market_analysis.deep_q_learning.reinforcement.action import Action
from market_analysis.deep_q_learning.reinforcement.deep_q import DeepQ
from market_analysis.deep_q_learning.reinforcement.replay_memory import ReplayMemory

Conf = namedtuple('Conf', ['hidden_nodes', 'act_f'])

class TestAndValidation:
    def __init__(self, env_builder, num_of_stocks, budget, dataset_splitter, evaluator):
        self.env_builder = env_builder
        self.init_num_of_of_stocks = num_of_stocks
        self.init_budget = budget
        self.dataset_splitter = dataset_splitter
        self.evaluator = evaluator

    def train_validate_test(self, data, confs):
        train_data, val_data, test_data = self.dataset_splitter.split_train_validation_test(data, 0.5, 0.3)

        models = self.train(confs, train_data)
        best_model = self.validate(models, val_data)

        self.test(best_model, test_data)
        return best_model

    def validate(self, models, data):
        validation_rewards = []

        for model in models:
            env = self.env_builder.build_batch_environment(self.init_num_of_of_stocks, self.init_budget, data, DataPreprocessor.get_instance())

            reward, budget, stocks = self.evaluator.test(model, env, data)
            validation_rewards.append(reward)

        best_model_index = np.argmax(np.array(validation_rewards))
        best_model = models[best_model_index]
        return best_model

    def train_test(self, data, confs):
        train_data, test_data = self.dataset_splitter.split_train_test(data, 0.7)

        net = self.train(confs, train_data)[0]
        self.test(net, test_data)

        return net

    def train(self, confs, data):
        models = []

        for conf in confs:
            train_env = self.env_builder.build_batch_environment(self.init_num_of_of_stocks, self.init_budget, data, DataPreprocessor.get_instance())
            model = self.get_model(conf, train_env, data)

            models.append(model)

        return models

    def test(self, net, data):
        test_env = self.env_builder.build_batch_environment(self.init_num_of_of_stocks, self.init_budget, data, DataPreprocessor.get_instance())

        return self.evaluator.test(net, test_env, data)

    def create_net(self, num_of_features, num_of_actions, hidden_nodes, act_f):
        nn = NeuralNet(num_of_features, num_of_actions, hidden_nodes, act_f)
        return nn

    def get_model(self, conf, env, data):
        num_of_actions = Action.num_of_actions

        mem = ReplayMemory(2000)
        evaluation = Evaluation()
        statistics = DeepQStatistics(env.get_num_of_states_per_training_episode())

        nn = self.create_net(env.num_of_features, num_of_actions, *conf)

        num_of_iterations = 3
        epsilon_strategy = LinearGreedyStrategy(num_of_actions, num_of_iterations, env.get_num_of_states_per_training_episode())

        deep_q = DeepQ(nn, env, mem, statistics, num_of_actions, env.num_of_features, epsilon_strategy, num_of_iterations)
        deep_q.train()

        evaluation.plot_actions_during_time(data['Price'], statistics.actions_for_last_iteration)

        return deep_q.neural_network

    def test_existing_model(self, model, data):
        model = ModelPersister.restore_model(model)
        return self.test(model, data)

    def validate_existing_models(self,models_names, data):
        models = []
        for model_name in models_names:
            model = ModelPersister.restore_model(model_name)
            models.append(model)
        return self.validate(models, data)