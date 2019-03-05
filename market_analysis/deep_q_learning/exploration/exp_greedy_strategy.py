import math

from market_analysis.deep_q_learning.exploration.linear_greedy_strategy import LinearGreedyStrategy


class ExpGreedyStrategy(LinearGreedyStrategy):


    def __init__(self, num_of_actions, num_of_iterations, num_of_states_per_episode):
        super(ExpGreedyStrategy, self).__init__(num_of_actions, num_of_iterations, num_of_states_per_episode)
        self.exp_decay_epsilon = self.get_exp_decay_factor()

    def get_epsilon(self):
        self.exp_decay_epsilon = self.get_exp_decay_factor()
        return self.init_epsilon * math.exp(-0.00008* self.steps)

        # return self.init_epsilon * math.exp(-self.exp_decay_epsilon * self.steps)

    def get_exp_decay_factor(self):

        step_size_per_iteration = self.num_of_states_per_episode
        total_num_of_steps = self.num_of_iterations*step_size_per_iteration

        self.steps+=1
        return math.log(self.min_epsilon)/-(total_num_of_steps*0.8)