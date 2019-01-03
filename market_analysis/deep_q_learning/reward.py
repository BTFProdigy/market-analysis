from collections import namedtuple

from market_analysis.deep_q_learning.action import Action
import math
import numpy as np
State = namedtuple('State', ['data', 'stocks','profit', 'inv'])
# State = namedtuple('State', ['data', 'stocks','inv'])

class Reward:

    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    # za cenu dodati min
    def get_reward(self, state, action, new_state):
        new_state_tuple = State(*new_state)
        state_tuple = State(*state)
        p1 = self.preprocessor.inverse_transform_budget(new_state_tuple.profit)
        p0 = self.preprocessor.inverse_transform_budget(state_tuple.profit)
        #
        n1 = self.preprocessor.inverse_transform_stocks(new_state_tuple.stocks)
        n0 = self.preprocessor.inverse_transform_stocks(state_tuple.stocks)
        # # # #
        d1 = self.preprocessor.inverse_transform_price(new_state_tuple.data)
        d0 = self.preprocessor.inverse_transform_price(state_tuple.data)

        inv = self.preprocessor.inverse_transform_price(state_tuple.inv)
        # inv = self.agent_state.get_inventory()
        # r = 0
        r = 0
        # if action == Action.DoNothing:
        #     r = 0.15
        #
        # buying_frequence = 0.8
        #
        # if action == Action.Buy:
        #     r = -0.01
        #



        if action == Action.Sell:
            if inv == 0:
                r = 0
            else:

                r = max(state_tuple.data - state_tuple.inv, 0)
        #
        #



        # if n1<0 or p1<0:
        #     r-=0.5

        #         r = d0-inv
        # if action == Action.DoNothing:
        #     if inv == 0:
        #         r = 1
        #     elif d0<inv:
        #         r = 1
        # r = max(p1-p0 +n1*d1 - n0*d0, 0)
        if p0<=0 and action == Action.Buy:
            r -= 0.2
            # r = 0
        elif n0<=0 and action == Action.Sell:
            # r = 0
            r-= 0.2
        # if p1<=0. or n1 <=0.:
        #     r -=0.5

        # if p1<=5*d0 or n1 <= 5.:
        #     r = 0

        # if p1<5*d0 or n1<5:
        #     k = -abs(int(d0/p1))
        #     l = -abs(5-max(n1, 0))
        #     r = min(k, l)

        # r = max(p1-p0 +n1*d1 - n0*d0, 0)
        # r = p1-p0 +n1*d1 - n0*d0

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