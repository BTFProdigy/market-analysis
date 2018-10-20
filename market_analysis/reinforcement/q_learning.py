import random

import pandas as pd


class QLearning:

    def __init__(self, owning_the_stock, budget, number_of_stocks_owning):
        self.alpha = 0.5
        self.discount_factor = 0.4
        self.chance_for_random_action = 0.7
        self.epsilon_decrease = 1e-3

        self.number_of_random_actions =0
        self.num_of_stocks_owning = number_of_stocks_owning

    def init(self, state_space, reward, q_table):
        self.state_space = state_space
        self.reward = reward
        self.q_table = q_table

    def start_learning(self, num_of_iterations, training_data):
        converged = False

        iterations_counter = 0
        self.actions = pd.DataFrame(columns = ["Action"], index = training_data.index)
        self.state_space.diskretize_features(training_data)
    
        while not converged and iterations_counter<num_of_iterations:
            i=0
            for index, instance in training_data.iterrows():
                if i < training_data.shape[0]-1:
                    state = self.state_space.get_state(instance, self.owning_the_stock, self.budget)
                    action = self.choose_action(state, i)
                    self.actions.loc[index] = action

                    new_state = self.state_space.get_state(training_data.iloc[i+1], self.owning_the_stock, self.budget)
                    i+=1
                    reward = self.reward.get_reward(state, action)
                    self.update_q(state, action, reward, new_state)
                  
            iterations_counter+=1
            self.alpha-=0.02

            print self.q_table.q_table
            print self.number_of_random_actions

    def update_q(self, state, action, reward, next_state):
        old_value = self.q_table.get_value(state, action)
        new_value = (1-self.alpha) * old_value + self.alpha * self.get_reward(reward, next_state)

        self.q_table.update_q(state, action, new_value)

    def get_reward(self, reward, next_state):
        return reward + self.get_discounted_reward(next_state)

    def get_discounted_reward(self, state):
        return self.discount_factor* self.q_table.get_max_value_for_state(state)

    def choose_if_random_action(self):
        return random.uniform(0,1) < self.chance_for_random_action

    def choose_action(self, state, instance_index):
        if self.choose_if_random_action() and instance_index<100:
            self.chance_for_random_action-=self.epsilon_decrease
            self.number_of_random_actions+=1

            return self.q_table.get_random_action(state)
        else:
            return self.q_table.get_best_action_for_state(state)