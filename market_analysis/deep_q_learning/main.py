import tensorflow as tf

from market_analysis.data_reader import DataReaderImpl
from market_analysis.deep_q_learning.action import Action
from market_analysis.deep_q_learning.deep_q import DeepQ
from market_analysis.deep_q_learning.deep_q_statistics import DeepQStatistics
from market_analysis.deep_q_learning.environment_builder import EnvironmentBuilder
from market_analysis.deep_q_learning.evaluation import Evaluation
from market_analysis.deep_q_learning.neural_network import NeuralNetwork
from market_analysis.deep_q_learning.replay_memory import ReplayMemory
from market_analysis.deep_q_learning.reward import Reward
from market_analysis.deep_q_learning.tester import Tester
from market_analysis.deep_q_learning.validation_and_testing import TestAndValidation
import market_analysis.data_reader.daily_stock_prices_getter as data_getter

# def init_session(sess):
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     return sess

def import_data():
    data_reader = DataReaderImpl()
    # data = data_reader.read_data("/home/nissatech/Documents/Market Analysis Data/prices/Data/Stocks/",
    #                          "aapl",
    #                          '2014-06-24')
    data = data_getter.get_data("AAPL", "2014-06-24")
    return data

def create_env():
    reward = Reward()
    env_builder = EnvironmentBuilder(reward)
    initial_budget = 500
    initial_stocks = 10

    data = import_data()
    env = env_builder.build_environment(initial_stocks, initial_budget, data)

    return env

def create_net(num_of_features, num_of_actions):
    hidden_nodes = 20
    learning_rate = 0.5
    nn = NeuralNetwork(num_of_features, num_of_actions, hidden_nodes, hidden_nodes, learning_rate)
    return nn

def run():
    num_of_actions = Action.num_of_actions

    env = create_env()

    mem = ReplayMemory(3000)
    evaluation = Evaluation()
    statistics = DeepQStatistics()
    nn =create_net(env.num_of_features, num_of_actions)

    deep_q = DeepQ(nn, env, mem, statistics, num_of_actions, env.num_of_features)
    deep_q.iterate_over_states()

    evaluation.plot_actions_during_time(env.original_close, statistics.actions)
    evaluation.evaluate(statistics)

def test():
    reward = Reward()
    env_builder = EnvironmentBuilder(reward)
    data = import_data()
    init_num_of_stocks, init_budget = 10, 1000

    tester = Tester()
    test = TestAndValidation(env_builder, init_num_of_stocks, init_budget, tester)
    # test.train_test(data)
    test.train_validate_test(data)
run()
# test()
# if __name__ == "__main__":
#     # test()
#     run()