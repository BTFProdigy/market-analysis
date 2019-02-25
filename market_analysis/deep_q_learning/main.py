import random
from datetime import datetime as dt

import math

import paths
from market_analysis.deep_q_learning import config_getter
from market_analysis.deep_q_learning.data_api.db_worker import DBWorker
from market_analysis.deep_q_learning.environment.agent_state import AgentState
from market_analysis.deep_q_learning.environment.environment_builder import EnvironmentBuilder
from market_analysis.deep_q_learning.environment.data_getter.fake_real_time_trading_data_getter import \
    FakeRealTimeTradingDataGetter
from market_analysis.deep_q_learning.evaluation.deep_q_statistics import DeepQStatistics
from market_analysis.deep_q_learning.evaluation.evaluation import Evaluation
from market_analysis.deep_q_learning.exploration.exp_greedy_strategy import ExpGreedyStrategy
from market_analysis.deep_q_learning.exploration.linear_greedy_strategy import LinearGreedyStrategy
from market_analysis.deep_q_learning.neural_net import ActivationFunction
from market_analysis.deep_q_learning.neural_net.neural_net import NeuralNet
from market_analysis.deep_q_learning.neural_net.neural_net_keras import NeuralNetwork
from market_analysis.deep_q_learning.preprocessing.data_preprocessor import DataPreprocessor
from market_analysis.deep_q_learning.reinforcement.action import Action
from market_analysis.deep_q_learning.reinforcement.deep_q import DeepQ
from market_analysis.deep_q_learning.reinforcement.replay_memory import ReplayMemory
from market_analysis.deep_q_learning.reinforcement.reward import Reward
from market_analysis.deep_q_learning.testing.dataset_splitter import DataSetSplitter
from market_analysis.deep_q_learning.testing.evaluator import Evaluator
from market_analysis.deep_q_learning.testing.validation_and_testing import TestAndValidation, Conf
from market_analysis.deep_q_learning.trading.actions.null_action_performer import NullActionPerformer
from market_analysis.deep_q_learning.trading.local_agents_environment.local_agents import LocalAgents
from market_analysis.deep_q_learning.trading.trader import Trader
import matplotlib.pyplot as plt
def import_data(db_worker):
    # data_reader = DataReaderImpl()
    # data = data_reader.read_data("/home/nissatech/Documents/Market Analysis Data/prices/Data/Stocks/","aapl",'2014-06-24')
    # data = data_getter.get_data("AAPL", "2014-06-24")
    start_date = dt.strptime("2018-11-25 11:00:00", '%Y-%m-%d %H:%M:%S')
    end_date = dt.strptime("2018-11-25 23:00:00", '%Y-%m-%d %H:%M:%S')
    data = db_worker.get_trades_for_period('BTC-EUR', start_date, end_date)

    # plt.plot(data['Price'])
    # data = data.resample('30s').mean()
    # data = db_worker.get_trades('BTC-EUR', 50)
    return data

def create_env(data, preprocessor):
    reward = Reward()
    env_builder = EnvironmentBuilder(reward.get_reward)
    initial_budget = 50000
    initial_stocks = 30

    env = env_builder.build_batch_environment(initial_stocks, initial_budget, data, preprocessor)

    return env

def create_net(num_of_features, num_of_actions):
    hidden_nodes = [14, 10, 10]
    acts = [ActivationFunction.Relu, ActivationFunction.Relu, ActivationFunction.Relu]
    nn = NeuralNet(num_of_features, num_of_actions, hidden_nodes, acts)
    # nn = NeuralNet(num_of_features, num_of_actions, hidden_nodes1, hidden_nodes2, ActivationFunction.Relu, ActivationFunction.Relu)
    return nn

def train(db_worker):
    num_of_actions = Action.num_of_actions

    data = import_data(db_worker)

    data['Price'].plot()
    data_preprocessor = DataPreprocessor.get_instance()
    env = create_env(data, data_preprocessor)
    # data_preprocessor.save_scalars(paths.get_scalars_path())
    mem = ReplayMemory(1000)
    evaluation = Evaluation()
    statistics = DeepQStatistics(env.get_num_of_states_per_training_episode())
    nn = create_net(env.num_of_features, num_of_actions)


    num_of_iterations = 1

    epsilon_strategy = LinearGreedyStrategy(num_of_actions, num_of_iterations, env.get_num_of_states_per_training_episode())


    deep_q = DeepQ(nn, env, mem, statistics, num_of_actions, env.num_of_features, epsilon_strategy, num_of_iterations)

    target_net = create_net(env.num_of_features, num_of_actions)
    # deep_q = FixedTarget(deep_q, target_net)

    deep_q.train()

    evaluation.plot_actions_during_time(data['Price'], statistics.actions_for_last_iteration, model_name)
    evaluation.evaluate(statistics, model_name)

    agent_state = env.agent_state
    stats =  '''Budget: {},
                Num of stocks: {},
                Reward: {}, 
                Reward for last 5 iterations: {},
                Num of stocks bought: {}, 
                Num of stocks sold: {}, 
                Actions: {}, 
          '''.format(agent_state.budget,
                                    agent_state.num_of_stocks,
                                    statistics.rewards_history[-1],
                                    sum(statistics.rewards_history[-5:]),
                                    agent_state.num_of_stocks_bought,
                                    agent_state.num_of_stocks_sold,
                                      statistics.all_actions[-1],
                                    )

    print stats
    # with open("Output.txt", "w") as text_file:
    #     text_file.write("Purchase Amount: %s" % TotalAmount)


    nn.save_model(paths.get_models_path()+model_name)


def import_test_data(db_worker):
    start_date = dt.strptime("2018-11-26 15:00:00", '%Y-%m-%d %H:%M:%S')

    end_date = dt.strptime("2018-11-26 23:00:00", '%Y-%m-%d %H:%M:%S')
    data = db_worker.get_trades_for_period('BTC-EUR', start_date, end_date)
    # data = data.resample('30s').mean()
    # data = db_worker.get_trades('BTC-EUR', 50)
    return data


def validate_and_test(db_worker):
    test = build_test_validation_pipeline(db_worker)
    data = import_test_data(db_worker)

    confs_parameters= [([20, 20] [ActivationFunction.Relu, ActivationFunction.Relu]),
                       ([20, 20], [ActivationFunction.Tanh, ActivationFunction.Tanh]),
                       ([10, 10], [ActivationFunction.Relu, ActivationFunction.Relu]),
                       ([10, 10], [ActivationFunction.Tanh, ActivationFunction.Tanh])]

    confs = [Conf(*conf_parameters) for conf_parameters in confs_parameters]

    model = test.train_validate_test(data, confs)
    model.save_model(paths.get_models_path()+"model_novo5")
    DataPreprocessor.get_instance().save_scalars(paths.get_scalars_path())

def test(db_worker, model):
    test = build_test_validation_pipeline(db_worker)
    data = import_test_data(db_worker)
    # confs = [Conf([20, 20], [ActivationFunction.Relu, ActivationFunction.Relu])]
    # model = test.train_test(data)
    # model.save_model(paths.get_models_path()+"model_novo5")

    # DataPreprocessor.get_instance().save_scalars(paths.get_scalars_path())


    # test.test_existing_model("model_novo14/", data)
    if model != None:
        DataPreprocessor.get_instance().load_scalers(paths.get_scalars_path())
        test.test_existing_model(model_name, data)

def build_test_validation_pipeline(db_worker):
    reward = Reward()

    env_builder = EnvironmentBuilder(reward.get_reward)

    init_num_of_stocks, init_budget = 20, 40000

    dataset_splitter = DataSetSplitter()
    evaluator = Evaluator()
    test = TestAndValidation(env_builder, init_num_of_stocks, init_budget, dataset_splitter, evaluator)
    return test

def trade(db_worker):
    trader = Trader(db_worker, "test", False)
    num_of_stocks= 10
    budget = 20000

    agent_state = AgentState(num_of_stocks, budget)
    env = build_trading_env(agent_state, 'BTC-EUR')
    # trader.trade(agent_state, "BTC-EUR", action_performer, "model_novo14/")
    trader.trade(env, model_name)

def build_trading_env(agent_state, ticker):
    data_preprocessor = DataPreprocessor.get_instance()
    data_preprocessor.load_scalers(paths.get_scalars_path())
    realtime_data_getter = FakeRealTimeTradingDataGetter(db_worker, data_preprocessor, '2018-11-24 23:00:00')

    reward = Reward()
    env_builder = EnvironmentBuilder(reward.get_reward)
    action_performer = NullActionPerformer()

    env = env_builder.build_trading_environment(realtime_data_getter, ticker, action_performer, agent_state, data_preprocessor)

    return env

def run_local_agents(db_worker):
    local_agents = LocalAgents(db_worker)
    local_agents.run()

config = config_getter.get_config('config')
db_address = config.get('DB','db_address')
db_name = config.get('DB','db_name')
db_worker = DBWorker(db_address, db_name)
model_name = "model_updated-12-12-12-february10-tanh/"
# model_name = "50iters-14-14-10/"
# model_name = 'model_updated-14-10-10-new/'
# model_name = 'model_updated-12-12-12/'
# model_name = 'NOVOModel 14-10-10/'
# model_name = 'NOVO 14-10-10/'
# updated-14-14-12-february10/'
# model_name = 'model_updated-18-14-12-february10/'
# train(db_worker)
# test(db_worker, model_name)
# trade(db_worker)
run_local_agents(db_worker)


# if __name__ == "__main__":
#     # test()
# f = []
# for i in range(300):
#     p= random.uniform(0,1)
#     e = math.exp(-0.00008* i)
#
#     if p<e:
#         r+=1
#     f.append()
#
