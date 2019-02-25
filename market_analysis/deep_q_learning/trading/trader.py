import random
from threading import Timer

import datetime
import random
import time
from threading import Timer

import numpy as np

import market_analysis.deep_q_learning.paths
from market_analysis.deep_q_learning.environment.environment_builder import EnvironmentBuilder
from market_analysis.deep_q_learning.environment.data_getter.fake_real_time_trading_data_getter import \
    FakeRealTimeTradingDataGetter
from market_analysis.deep_q_learning.neural_net.model_persister import ModelPersister
from market_analysis.deep_q_learning.preprocessing.data_preprocessor import DataPreprocessor
from market_analysis.deep_q_learning.reinforcement.action import Action
from market_analysis.deep_q_learning.reinforcement.reward import Reward


class Trader:

    def __init__(self, db_worker, name, random):
        self.name = name
        self.random = random
        self.db_worker = db_worker
        self.trading_frequency = 6
        self.actions = 0
        self.negative_states = 0

    def trade(self, env, model):
        nn = None if self.random else ModelPersister.restore_model(model)

        self.start_time = time.time()
        self.perform_realtime_actions(env, nn)

    def perform_realtime_actions(self, env, nn, limited = False):
        state = env.curr_state

        if self.random == False:
            q_values = nn.predict(state)
            action = np.argmax(q_values)
        else:
            action = random.randint(0, Action.num_of_actions-1)

        # if (action == Action.Buy and env.agent_state.budget > self.data_preprocessor.inverse_transform_price(state[0]))\
        #     or (action == Action.Sell and env.agent_state.num_of_stocks>0) or action == Action.DoNothing:
        # print 'Price: {}'.format(self.data_preprocessor.inverse_transform_price(state[0]))


        if env.agent_state.budget<0 or env.agent_state.num_of_stocks<0:
            self.negative_states+=1

        if not limited or (self.buy_with_no_money(env, action) or self.sell_with_no_stocks(env, action)):
            self.make_action(env, action)

        t = Timer(self.trading_frequency, self.perform_realtime_actions, args =[env, nn])
        t.start()

    def make_action(self, env, action):
        env.step(action)
        agent_state = env.agent_state
        self.actions+=1
        self.print_state(agent_state)

    def print_state(self, agent_state):
        elapsed_time = (time.time() - self.start_time)
        elapsed_time_string = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        print 'Agent {} has budget {} and number of stocks {}, ' \
              'Current time: {}, Time elapsed: {}, Action: {},' \
              'Negative states:{}'.format(self.name, agent_state.budget,
                                                          agent_state.num_of_stocks,datetime.datetime.now(),
                                                          elapsed_time_string, self.actions, self.negative_states)
    def buy_with_no_money(self, env, action):
        state = env.agent_state
        # return action == Action.Buy and state.budget < env.curr_state[0]
        return action == Action.Buy and state.budget <= 0

    def sell_with_no_stocks(self, env, action):
        state = env.agent_state
        return action == Action.Sell and state.stocks <= 0

