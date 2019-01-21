import cPickle
import random
from threading import Timer

import datetime
import numpy as np
import time

from market_analysis.deep_q_learning.neural_net.neural_net import NeuralNet
from market_analysis.deep_q_learning.reinforcement.action import Action
from market_analysis.deep_q_learning.environment.environment_builder import EnvironmentBuilder
from market_analysis.deep_q_learning.environment.real_time_env.fake_real_time_trading_data_getter import FakeRealTimeTradingDataGetter
from market_analysis.deep_q_learning.preprocessing.data_preprocessor import DataPreprocessor
from market_analysis.deep_q_learning.reinforcement.reward import Reward
import market_analysis.deep_q_learning.paths
class Trader:

    def __init__(self, db_worker, name, random):
        self.name = name
        self.random = random
        self.db_worker = db_worker

    def trade(self, agent_state, ticker, action_performer, model):

        self.data_preprocessor = DataPreprocessor.get_instance()
        self.data_preprocessor.load_scalers(market_analysis.deep_q_learning.paths.get_scalars_path())
        realtime_data_getter = FakeRealTimeTradingDataGetter(self.db_worker, self.data_preprocessor, '2018-11-24 23:00:00')

        reward = Reward(self.data_preprocessor)
        env_builder = EnvironmentBuilder(reward)

        env = env_builder.build_realtime_environment(realtime_data_getter, ticker, action_performer, agent_state, self.data_preprocessor)

        # path = "neural_net/model/"
        nn = None
        if not self.random:
            nn = self.restore_model(model)

        t = Timer(0, self.perform_realtime_actions, args = (env, nn))
        self.start_time = time.time()
        t.start()
        return

    def restore_model(self, model):
        model_path = market_analysis.deep_q_learning.paths.get_models_path() + model
        model_parameters = self.load_model_parameters(model_path + "parameters")
        nn = NeuralNet(model_parameters.input_size, model_parameters.output_size,
                           [model_parameters.num_hidden_nodes1, model_parameters.num_hidden_nodes2, model_parameters.num_hidden_nodes1],
                           [model_parameters.activation_function1, model_parameters.activation_function2, model_parameters.activation_function2])

        # nn = NeuralNetwork(*model_parameters.__dict__.values)
        nn = nn.restore_model(model_path)
        return nn

    def perform_realtime_actions(self, env, nn):
        state = env.curr_state

        if self.random == False:
            q_values = nn.predict(state)
            action = np.argmax(q_values)
        else:
            action = random.randint(0, 2)

        # if (action == Action.Buy and env.agent_state.budget > self.data_preprocessor.inverse_transform_price(state[0]))\
        #     or (action == Action.Sell and env.agent_state.num_of_stocks>0) or action == Action.DoNothing:
        env.step(action)
        print 'Price: {}'.format(self.data_preprocessor.inverse_transform_price(state[0]))
        elapsed_time = (time.time() - self.start_time)
        elapsed_time_string = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        print 'Agent {} has budget {} and number of stocks {}, Current time: {}, Time elapsed: {}'.format(self.name, env.agent_state.budget,
                                                                                         env.agent_state.num_of_stocks,
                                                                                         datetime.datetime.now(),
                                                                                         elapsed_time_string)
        # return

        t = Timer(6, self.perform_realtime_actions, args =[env, nn])
        t.start()

    def perform_limited_realtime_actions(self, env, nn):
        state = env.curr_state
        q_values = nn.predict(state)
        action = np.argmax(q_values)
        state = env.agent_state

        if not(self.buy_with_no_money(env, action) or self.sell_with_no_stocks(env, action)):
            env.step(action)
            agent_state = env.agent_state
            print 'Agent {} has budget {} and number of stocks {}'.format(self.name, agent_state.budget, agent_state.num_of_stocks)
        return


        t = Timer(60, self.perform_realtime_actions(env, nn))
        t.start()

    def buy_with_no_money(self, env, action):
        state = env.agent_state
        return action == Action.Buy and state.budget < env.curr_state[0]

    def sell_with_no_stocks(self, env, action):
        state = env.agent_state
        return action == Action.Sell and state.stocks < 0

    def load_model_parameters(self, file_name):
        with open(file_name, 'rb') as file:
            return cPickle.load(file)