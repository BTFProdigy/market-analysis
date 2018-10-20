import tensorflow as tf
import numpy as np
from market_analysis.deep_q_learning.action import Action
from market_analysis.deep_q_learning.deep_q import DeepQ
from market_analysis.deep_q_learning.deep_q_statistics import DeepQStatistics
from market_analysis.deep_q_learning.evaluation import Evaluation
from market_analysis.deep_q_learning.neural_network import NeuralNetwork
from market_analysis.deep_q_learning.replay_memory import ReplayMemory

class TestAndValidation:
    def __init__(self, env_builder, num_of_stocks, budget, tester):
        self.env_builder = env_builder
        self.init_num_of_of_stocks = num_of_stocks
        self.init_budget = budget
        self.tester = tester

    def split_train_test(self, data):
        size = data.shape[0]
        train_size = int(0.7*size)

        self.train_data = data.iloc[:train_size]
        self.test_data = data.iloc[train_size:]

    def split_train_validation_test(self, data):
        size = data.shape[0]
        train_size = int(0.5*size)
        validation_size = int(0.3*size)

        self.train_data = data.iloc[:train_size]
        self.validation_data = data.iloc[train_size:train_size+validation_size]
        self.test_data = data.iloc[train_size+validation_size:]



    def train_validate_test(self, data):
        self.split_train_validation_test(data)
        confs = [{"hidden_nodes": 10, "learning_rate":0.6}, {"hidden_nodes": 20, "learning_rate":0.6}]

        models = self.train(confs)

        best_model = self.validate(models)

        self.test(best_model)
        return best_model

    def validate(self, models):
        validation_rewards = []

        for model in models:
            env = self.env_builder.build_environment(self.init_num_of_of_stocks, self.init_budget, self.validation_data)

            reward, budget, stocks = self.tester.test(model, env)
            validation_rewards.append(reward)

        best_model_index = np.argmax(np.array(validation_rewards))
        best_model = models[best_model_index]
        return best_model

    def train_test(self, data):
        self.split_train_test(data)
        confs = [{"hidden_nodes": 10, "learning_rate":0.6}]

        net = self.train(confs)[0]
        return self.test(net)

    def train(self, confs):
        models = []

        for conf in confs:
            train_env = self.env_builder.build_environment(self.init_num_of_of_stocks, self.init_budget, self.train_data)
            model = self.get_model(conf, train_env)

            models.append(model)

        return models

    def test(self, net):
        test_env = self.env_builder.build_environment(self.init_num_of_of_stocks, self.init_budget, self.test_data)

        self.tester.test(net, test_env)

    def create_net(self, num_of_features, num_of_actions, hidden_nodes, learning_rate):
        nn = NeuralNetwork(num_of_features, num_of_actions, hidden_nodes, hidden_nodes, learning_rate)
        return nn

    def get_model(self, conf, env):
        num_of_actions = Action.num_of_actions


        mem = ReplayMemory(3000)
        evaluation = Evaluation()
        statistics = DeepQStatistics()
        nn = self.create_net(env.num_of_features, num_of_actions, conf['hidden_nodes'], conf['learning_rate'])

        deep_q = DeepQ(nn, env, mem, statistics, num_of_actions, env.num_of_features)
        deep_q.iterate_over_states()

        evaluation.plot_actions_during_time(env.original_close, statistics.actions)

        return deep_q.neural_network