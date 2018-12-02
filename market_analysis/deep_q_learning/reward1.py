from collections import namedtuple

from market_analysis.deep_q_learning.action import Action
import math
import numpy as np
State = namedtuple('State', ['s_return1', 's_return_2', 's_return_3', 's_return_4', 's_return_5', 'bollinger_band_diff', 'sharpe_ratio', 'daily_return','stocks', 'budget'])

class Reward:

    def __init__(self, min_value=-1, max_value=1):
        self.min_value = min_value
        self.max_value = max_value


    def get_reward(self, state, action, new_state):
        new_state_tuple = State(*new_state)
        state_tuple = State(*state)
        return self.reward_for_profit(state_tuple, action, new_state_tuple) + \
               self.get_reward_for_budget(new_state_tuple) + \
               self.get_reward_for_number_of_stocks(new_state_tuple) + \
               self.get_sharpe_ratio_reward(state_tuple) + \









    def reward_for_profit(self, state, action, new_state):
        # if action == Action.Sell:
        #     return (state.bollinger_band_diff+state.daily_return)*(new_state.stocks)

        # elif action == Action.Buy:
        #     return -(state.bollinger_band_diff+state.daily_return)*new_state.stocks


        # print "BB " + str(state.bollinger_band_diff)
        # print "DR " + str(state.daily_return)


        print (-state.bollinger_band_diff-state.s_return+state.sharpe_ratio)*new_state.stocks
        return (-state.bollinger_band_diff-state.s_return+state.sharpe_ratio)*new_state.stocks



    def get_reward_for_budget(self, state, new_state):
        if state.budget <=0:
            return -10

        return new_state.budget - state.budget

    def get_reward_for_number_of_stocks(self, state):
        if state.stocks <=0:
            return -10
        return state.stocks

    def get_sharpe_ratio_reward(self, state):
        # return self.get_sharpe_ratio()
        return state.sharpe_ratio

    def get_cummulative_return_reward(self, state):
        return state.cummulative_return

    def get_reward_for_being_alive(self):
        return 5

    def get_sparse_actions_reward(self, state, action, new_state):
        return self.reward_for_profit(state, action, new_state)

    def get_frequent_actions_reward(self, state, action, new_state):
        return self.reward_for_profit(state, action, new_state)+ \
               self.get_cummulative_return_reward(state)+ \
               self.get_sharpe_ratio_reward(state)

    def get_reward_for_best_actions(self, state, action, next_state):
        if action == Action.Buy and next_state.s_return < state.s_return:
            return -2

        elif action == Action.Sell and next_state.s_return > state.s_return:
            return -2


    def clip_reward(self, reward):
        clipped = np.clip(reward, self.min_value, self.max_value)
        return clipped

    def get_reward_simple(self, state, new_state):
        return new_state.budget + new_state.stocks*state.price

    def get_simple_reward1(self, state, new_state):
        return new_state.profit + new_state.stocks*(-state.bollinger_band_diff-state.s_return+state.sharpe_ratio)