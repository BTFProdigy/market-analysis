from collections import namedtuple

from market_analysis.deep_q_learning.action import Action
import math
import numpy as np
State = namedtuple('State', ['data', 'stocks', 'profit'])

class Reward:

    def __init__(self, preprocessor, min_value=-1, max_value=1 ):
        self.preprocessor = preprocessor
        self.min_value = min_value
        self.max_value = max_value

    def get_reward(self, state, action, new_state):
        new_state_tuple = State(*new_state)
        state_tuple = State(*state)
        # r = new_state_tuple.profit+new_state_tuple.stocks*new_state_tuple.data- \
        #     state_tuple.profit-state_tuple.stocks*state_tuple.s_return
        # r = self.reward_for_profit(state_tuple, action, new_state_tuple)

        p1 = self.preprocessor.inverse_transform_budget(new_state_tuple.profit)
        p0 = self.preprocessor.inverse_transform_budget(state_tuple.profit)
        #
        n1 = self.preprocessor.inverse_transform_stocks(new_state_tuple.stocks)
        n0 = self.preprocessor.inverse_transform_stocks(state_tuple.stocks)
        #
        d1 = self.preprocessor.inverse_transform_price(new_state_tuple.data)
        d0 = self.preprocessor.inverse_transform_price(state_tuple.data)

        # inv = self.preprocessor.inverse_transform_price(state_tuple.inv)
        # r = state_tuple.data

        # r = max(p1+d1*n1-p0-n0*d0, 0)
        #

        # if p0<=d0 and action == Action.Buy:
        #     r = -15
        # elif n0<=1 and action == Action.Sell:
        #     r = -15

        # if action == Action.Sell:
        #     if state_tuple.inv == 0:
        #         r = 0
        #     else:
        #         r = max(state_tuple.data - state_tuple.inv, 0)

        r = max(p1 + n1*d1- p0 - n0*d0, 0)

        # if p0<=d0 and action == Action.Buy:
        #     r = -10
        # elif n0<=1 and action == Action.Sell:
        #     r = -10
        if p1<=0 or n1<=0:
            r = -10

        # if new_state_tuple.profit <= 0:
            # r= -abs(r)*2

        return r

    def reward_if_state_good_enough(self, state):
        return

    def reward_for_profit(self, state, action, new_state):
        new_state_features = new_state.s_return
        state_features = state.s_return
        # gledamo razliku 2 stanja
        p1 = new_state.profit-state.profit
        p2 = new_state.stocks*new_state_features-state.stocks*state_features
        reward = new_state_features*new_state.stocks + new_state.profit - (state_features*state.stocks+state.profit)

        # self.get_fee_punishment(state_features, action)

        # od pocetka gledamo

        # reward = state_features*new_state.stocks+new_state.profit-state.profit +\
        #        self.get_fee_punishment(state_features, action)

        # return self.cl=

        return reward


    def get_fee_punishment(self, profit, action, fee = 0.3):
        if action == Action.Sell or action == Action.Buy:
            # price = abs(new_state.budget - state.budget)

            return -abs(profit)*fee
        return 0





    def get_reward_for_being_alive(self):
        return 5



    def clip_reward(self, reward):
        clipped = np.clip(reward, self.min_value, self.max_value)
        return clipped

    def get_reward_for_budget(self, state):
        if state.budget <=0:
            return -10
        return state.s_return

    def get_reward_for_number_of_stocks(self, state, new_state):
        if state.stocks <=0:
            return -(new_state.profit-state.profit)
        return 0

    def get_cummulative_return_reward(self, state):
        return state.cummulative_return

    def get_sparse_actions_reward(self, state, action, new_state):
        return self.reward_for_profit(state, action, new_state)

    def get_frequent_actions_reward(self, state, action, new_state):
        return self.reward_for_profit(state, action, new_state)+\
               self.get_cummulative_return_reward(state)+\
                self.get_sharpe_ratio_reward(state)


    def get_reward_for_best_actions(self, state, action, next_state):
        if action == Action.Buy and next_state.s_return < state.s_return:
            return -2

        elif action == Action.Sell and next_state.s_return > state.s_return:
            return -2