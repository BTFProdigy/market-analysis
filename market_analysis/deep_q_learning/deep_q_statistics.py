
class DeepQStatistics:

    def __init__(self, num_of_states_per_episode):
        self.budget_history= []
        self.budget_for_last_iteration= []
        self.stocks_for_last_iteration= []
        self.num_of_stocks_history = []

        self.rewards_history = []
        self.avg_rewards_history = []
        self.epsilon_history = []

        self.actions_for_last_iteration = []
        self.random_actions = []

        self.rewards_for_last_iteration= []

        self.num_of_states_per_episode = num_of_states_per_episode

    def add_reward(self, reward):
        self.rewards_history.append(reward)

    def add_reward_avg(self, reward):
        self.avg_rewards_history.append(reward)

    def add_epsilon(self, epsilon):
        self.epsilon_history.append(epsilon)

    def add_budget(self, budget):
        self.budget_history.append(budget)

    def add_budget_for_last_iteration(self, budget):
        self.budget_for_last_iteration.append(budget)

    def add_stocks_for_last_iteration(self, stocks):
        self.stocks_for_last_iteration.append(stocks)

    def add_stocks(self, num_of_stocks):
        self.num_of_stocks_history.append(num_of_stocks)

    def add_random_actions(self, num_of_random_actions):
        self.random_actions.append(float(num_of_random_actions)/self.num_of_states_per_episode)


    def print_final_state(self):
        print '''Budget: {},
                Num of stocks: {},
                Reward: {}'''.format(self.budget_history[-1],
                                    self.num_of_stocks_history[-1],
                                    self.rewards_history[-1])
