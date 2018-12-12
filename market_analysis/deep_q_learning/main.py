import cPickle
from threading import Timer

import numpy as np

from market_analysis.data_reader import DataReaderImpl
from market_analysis.deep_q_learning.action import Action
from market_analysis.deep_q_learning.deep_q import DeepQ
from market_analysis.deep_q_learning.deep_q_statistics import DeepQStatistics
from market_analysis.deep_q_learning.environment.agent_state import AgentState
from market_analysis.deep_q_learning.environment.data_preprocessor import DataPreprocessor
from market_analysis.deep_q_learning.environment.environment_builder import EnvironmentBuilder
from market_analysis.deep_q_learning.environment.real_time_trading_data_getter import RealTimeTradingDataGetter
from market_analysis.deep_q_learning.evaluation import Evaluation
from market_analysis.deep_q_learning.exploration.greedy_strategy import GreedyStrategy
from market_analysis.deep_q_learning.data.db_worker import DBWorker
from market_analysis.deep_q_learning.local_agents_environment.null_action_performer import NullActionPerformer
from market_analysis.deep_q_learning.neural_net import ActivationFunction
from market_analysis.deep_q_learning.neural_net.neural_net1 import NeuralNet
from market_analysis.deep_q_learning.neural_net.neural_network import NeuralNetwork
from market_analysis.deep_q_learning.replay_memory import ReplayMemory
from market_analysis.deep_q_learning.reward import Reward
from market_analysis.deep_q_learning.testing.tester import Tester
from market_analysis.deep_q_learning.testing.validation_and_testing import TestAndValidation
from datetime import datetime as dt

from market_analysis.deep_q_learning.trader import Trader

import paths
def import_data():
    # data_reader = DataReaderImpl()
    # data = data_reader.read_data("/home/nissatech/Documents/Market Analysis Data/prices/Data/Stocks/",
    #                          "aapl",
    #                          '2014-06-24')
    # data = data_getter.get_data("AAPL", "2014-06-24")
    db_worker = DBWorker()
    start_date = dt.strptime("2018-11-24 15:00:00", '%Y-%m-%d %H:%M:%S')

    end_date = dt.strptime("2018-11-24 19:00:00", '%Y-%m-%d %H:%M:%S')
    data = db_worker.get_trades_for_period('BTC-EUR', start_date, end_date)
    # data = data.resample('30s').mean()
    # data = db_worker.get_trades('BTC-EUR', 50)
    return data

def create_env(data, preprocessor):
    reward = Reward(preprocessor)
    env_builder = EnvironmentBuilder(reward)
    initial_budget = 40000
    initial_stocks = 30

    env = env_builder.build_train_environment(initial_stocks, initial_budget, data, preprocessor)

    return env

def create_net(num_of_features, num_of_actions):
    hidden_nodes1 = 32
    hidden_nodes2 = 32
    nn = NeuralNet(num_of_features, num_of_actions, hidden_nodes1, hidden_nodes2, ActivationFunction.Relu, ActivationFunction.Relu)
    return nn

def train():
    num_of_actions = Action.num_of_actions

    data = import_data()
    env = create_env(data, DataPreprocessor.get_instance())

    mem = ReplayMemory(2000)
    evaluation = Evaluation()
    statistics = DeepQStatistics(env.get_num_of_states_per_training_episode())
    nn =create_net(env.num_of_features, num_of_actions)
    target_net = create_net(env.num_of_features, num_of_actions)

    num_of_iterations = 200
    epsilon_strategy = GreedyStrategy(num_of_actions, num_of_iterations, env.get_num_of_states_per_training_episode())
    deep_q = DeepQ(nn, env, mem, statistics, num_of_actions, env.num_of_features, epsilon_strategy, num_of_iterations, target_net)
    deep_q.iterate_over_states()

    evaluation.plot_actions_during_time(data['Close'], statistics.actions_for_last_iteration)
    evaluation.evaluate(statistics)

    agent_state = env.agent_state
    print '''Budget: {},
                Num of stocks: {},
                Reward: {}, 
                Num of stocks bought: {}, 
                Num of stocks sold: {}'''.format(agent_state.budget,
                                    agent_state.num_of_stocks,
                                    statistics.rewards_history[-1],
                                    agent_state.num_of_stocks_bought,
                                    agent_state.num_of_stocks_sold)
    nn.save_model(paths.get_models_path()+"model_novo4/")

def test():
    reward = Reward(DataPreprocessor.get_instance())
    env_builder = EnvironmentBuilder(reward)
    data = import_data()
    init_num_of_stocks, init_budget = 20, 3000

    tester = Tester()
    test = TestAndValidation(env_builder, init_num_of_stocks, init_budget, tester)
    # model = test.train_test(data)

    test.test_existing_model("model_novo4", data)
    # model = test.train_validate_test(data)
    # model.save_model(paths.get_models_path()+"model_novo5")
    #
    # DataPreprocessor.get_instance().save_scalars(paths.get_scalars_path())

def trade():
    trader = Trader("test")
    action_performer = NullActionPerformer()
    num_of_stocks= 10
    budget = 20000

    agent_state = AgentState(num_of_stocks, budget)
    trader.trade(agent_state, "BTC-EUR", action_performer, "")

train()
# test()

# trade()

# if __name__ == "__main__":
#     # test()
#     run()