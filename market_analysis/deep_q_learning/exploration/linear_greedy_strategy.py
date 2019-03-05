from strategy import Strategy


class LinearGreedyStrategy(Strategy):

    def __init__(self, num_of_actions, num_of_iterations, num_of_states_per_episode):
        self.num_of_actions = num_of_actions
        self.num_of_iterations = num_of_iterations
        self.num_of_states_per_episode = num_of_states_per_episode

        self.min_epsilon = 0.0001
        self.steps = -1
        self.init_epsilon = 1
        self.epsilon = self.init_epsilon

    def get_epsilon_linear(self):
        step_size_per_iteration = self.num_of_states_per_episode
        total_num_of_steps = self.num_of_iterations*step_size_per_iteration
        self.steps+=1
        return self.init_epsilon-(1./total_num_of_steps)*self.steps

    def get_epsilon(self):
        self.epsilon*=0.9995
        self.steps+=1
        return self.epsilon