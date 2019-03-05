from collections import namedtuple

import numpy as np

from market_analysis.deep_q_learning.preprocessing.data_preprocessor import DataPreprocessor
from market_analysis.deep_q_learning.reinforcement.action import Action

State = namedtuple('State', ['data', 'stocks','profit', 'inv'])


# def clipper(rewardf):
#     def clip(min, max):
#         result = rewardf()
#         return np.clip(result, min, max)
#     return clip

class Reward:

    def clipper(self, rewardf):
        def clip(min, max):
            result = rewardf()
            return np.clip(result, min, max)
        return clip

    def get_reward(self, state, action, new_state):
        new_state_tuple = State(*new_state)
        state_tuple = State(*state)

        self.preprocessor = DataPreprocessor.get_instance()

        p0 = self.preprocessor.inverse_transform_budget(state_tuple.profit)
        n0 = self.preprocessor.inverse_transform_stocks(state_tuple.stocks)

        inv = self.preprocessor.inverse_transform_price(state_tuple.inv)
        r = 0

        if action == Action.Sell:
            if inv == 0:
                r = 0
            else:
                r = max(state_tuple.data - state_tuple.inv, 0)

        if p0<=0 and action == Action.Buy:
            r -= 0.2
        elif n0<=0 and action == Action.Sell:
            r-= 0.2
        return r

    def get_fee_punishment(self, profit, action, fee = 0.3):
        if action == Action.Sell or action == Action.Buy:
            return -abs(profit)*fee
        return 0

    def get_reward_for_being_alive(self):
        return 5

    def get_cummulative_return_reward(self, state):
        return state.cummulative_return
