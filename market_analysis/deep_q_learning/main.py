import os.path

from market_analysis.deep_q_learning import config_getter
from market_analysis.deep_q_learning.evaluation.evaluation import Evaluation
from market_analysis.deep_q_learning.exploration.linear_greedy_strategy import LinearGreedyStrategy
from market_analysis.deep_q_learning.neural_net.neural_net import NeuralNet
from market_analysis.deep_q_learning.neural_net.neural_net_keras import NeuralNetwork
from market_analysis.deep_q_learning.reinforcement.action import Action
from market_analysis.deep_q_learning.reinforcement.deep_q import DeepQ
from market_analysis.deep_q_learning.evaluation.deep_q_statistics import DeepQStatistics
from market_analysis.deep_q_learning.environment.agent_state import AgentState
from market_analysis.deep_q_learning.preprocessing.data_preprocessor import DataPreprocessor
from market_analysis.deep_q_learning.environment.environment_builder import EnvironmentBuilder
from market_analysis.deep_q_learning.data_api.db_worker import DBWorker
from market_analysis.deep_q_learning.reinforcement.fixed_taget import FixedTarget
from market_analysis.deep_q_learning.trading.actions.null_action_performer import NullActionPerformer
from market_analysis.deep_q_learning.neural_net import ActivationFunction
from market_analysis.deep_q_learning.reinforcement.replay_memory import ReplayMemory
from market_analysis.deep_q_learning.reinforcement.reward import Reward
from market_analysis.deep_q_learning.testing.tester import Tester
from market_analysis.deep_q_learning.testing.validation_and_testing import TestAndValidation
from datetime import datetime as dt

from market_analysis.deep_q_learning.trading.trader import Trader

import paths
import configparser
model_name = "model_randommmm/"

def import_data(db_worker):
    # data_reader = DataReaderImpl()
    # data = data_reader.read_data("/home/nissatech/Documents/Market Analysis Data/prices/Data/Stocks/",
    #                          "aapl",
    #                          '2014-06-24')
    # data = data_getter.get_data("AAPL", "2014-06-24")
    start_date = dt.strptime("2018-11-24 15:00:00", '%Y-%m-%d %H:%M:%S')

    end_date = dt.strptime("2018-11-24 21:00:00", '%Y-%m-%d %H:%M:%S')
    data = db_worker.get_trades_for_period('BTC-EUR', start_date, end_date)
    # data = data.resample('30s').mean()
    # data = db_worker.get_trades('BTC-EUR', 50)
    return data

def create_env(data, preprocessor):
    reward = Reward(preprocessor)
    env_builder = EnvironmentBuilder(reward)
    initial_budget = 50000
    initial_stocks = 30

    env = env_builder.build_train_environment(initial_stocks, initial_budget, data, preprocessor)

    return env

def create_net(num_of_features, num_of_actions):
    hidden_nodes = [12, 12, 12]
    acts = [ActivationFunction.Relu, ActivationFunction.Relu, ActivationFunction.Relu]
    hidden_nodes1 = 12
    hidden_nodes2 = 12
    nn = NeuralNetwork(num_of_features, num_of_actions, hidden_nodes, acts)

    # nn = NeuralNet(num_of_features, num_of_actions, hidden_nodes1, hidden_nodes2, ActivationFunction.Relu, ActivationFunction.Relu)
    return nn

def train(db_worker):
    num_of_actions = Action.num_of_actions

    data = import_data(db_worker)
    data_preprocessor = DataPreprocessor.get_instance()
    env = create_env(data, data_preprocessor)
    # data_preprocessor.save_scalars(paths.get_scalars_path())
    mem = ReplayMemory(1000)
    evaluation = Evaluation()
    statistics = DeepQStatistics(env.get_num_of_states_per_training_episode())
    nn = create_net(env.num_of_features, num_of_actions)


    num_of_iterations = 3
    epsilon_strategy = LinearGreedyStrategy(num_of_actions, num_of_iterations, env.get_num_of_states_per_training_episode())
    deep_q = DeepQ(nn, env, mem, statistics, num_of_actions, env.num_of_features, epsilon_strategy, num_of_iterations)

    target_net = create_net(env.num_of_features, num_of_actions)
    # deep_q = FixedTarget(deep_q, target_net)

    deep_q.train()

    evaluation.plot_actions_during_time(data['Price'], statistics.actions_for_last_iteration)
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

    nn.save_model(paths.get_models_path()+model_name)


def import_test_data(db_worker):
    start_date = dt.strptime("2018-11-24 20:00:00", '%Y-%m-%d %H:%M:%S')

    end_date = dt.strptime("2018-11-24 23:00:00", '%Y-%m-%d %H:%M:%S')
    data = db_worker.get_trades_for_period('BTC-EUR', start_date, end_date)
    # data = data.resample('30s').mean()
    # data = db_worker.get_trades('BTC-EUR', 50)
    return data

def test(db_worker):
    reward = Reward(DataPreprocessor.get_instance())
    env_builder = EnvironmentBuilder(reward)
    data = import_test_data(db_worker)
    init_num_of_stocks, init_budget = 20, 40000

    tester = Tester()
    test = TestAndValidation(env_builder, init_num_of_stocks, init_budget, tester)
    # model = test.train_test(data)

    # test.test_existing_model("model_novo14/", data)

    test.test_existing_model(model_name, data)
    # model = test.train_validate_test(data)
    # model.save_model(paths.get_models_path()+"model_novo5")
    #
    # DataPreprocessor.get_instance().save_scalars(paths.get_scalars_path())

def trade(db_worker):
    trader = Trader(db_worker, "test", False)
    action_performer = NullActionPerformer()
    num_of_stocks= 10
    budget = 20000

    agent_state = AgentState(num_of_stocks, budget)
    # trader.trade(agent_state, "BTC-EUR", action_performer, "model_novo14/")
    trader.trade(agent_state, "BTC-EUR", action_performer, model_name)

config = config_getter.get_config('config')
db_address = config.get('DB','db_address')
db_name = config.get('DB','db_name')
db_worker = DBWorker(db_address, db_name)


# train(db_worker)
test(db_worker)

# trade(db_worker)


# if __name__ == "__main__":
#     # test()
#     run()