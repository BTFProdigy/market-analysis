
import numpy as np

from market_analysis.deep_q_learning.action import Action


class Environment:

    def __init__(self, original_close, data, reward, num_of_stocks, budget):
        self.initial_num_of_stocks, self.initial_budget, self.original_close, self.data = \
            num_of_stocks, budget, original_close, data

        self.reward = reward
        self.num_of_features = None

        self.reset()

    def get_num_of_states_per_episode(self):
        return self.data.shape[0]

    def reset(self):
        self.num_of_stocks = self.initial_num_of_stocks
        self.budget = self.initial_budget

        self.data_index = 1

        self.curr_state = self.create_state(self.original_close[0], self.data.iloc[0],
                                            self.initial_num_of_stocks, self.initial_budget)

    def create_state(self, original_close, data, num_of_stocks, budget):
        values = np.append(data.values, [num_of_stocks, budget])
        values = np.insert(values, 0, original_close)
        self.num_of_features = len(values)
        return values

    def step(self, action):
        reward = self.reward.get_reward(self.curr_state,action)
        new_state, done = self.get_new_state(self.curr_state, action)

        self.data_index+=1
        self.curr_state = new_state

        return new_state, reward, done

    def get_new_state(self, state, action):
        if self.data_index == self.data.shape[0]:
            return None, True
        if action == Action.DoNothing:
            # do nothing
            return self.create_state(self.original_close.iloc[self.data_index],
                                     self.data.iloc[self.data_index], state[3], state[4]), False
        elif action == Action.Sell:
            # sell
            self.budget+= state[0]
            self.num_of_stocks-=1
        else:
        #     buy
            self.budget-= state[0]
            self.num_of_stocks+=1

        return self.create_state(self.original_close.iloc[self.data_index],self.data.iloc[self.data_index], self.num_of_stocks, self.budget), False