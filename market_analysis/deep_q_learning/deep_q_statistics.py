
class DeepQStatistics:

    def __init__(self):
        self.budget_history= []
        self.num_of_stocks_history = []

        self.rewards_history = []
        self.epsilon_history = []
        self.actions = []
        self.random_actions = []

    def add_reward(self, reward):
        self.rewards_history.append(reward)

    def add_epsilon(self, epsilon):
        self.epsilon_history.append(epsilon)

    def add_budget(self, budget):
        self.budget_history.append(budget)

    def add_stocks(self, num_of_stocks):
        self.num_of_stocks_history.append(num_of_stocks)

    def add_random_actions(self, num_of_random_actions):
        self.random_actions.append(num_of_random_actions)

    def print_final_state(self):
        print '''Budget: {},
                Num of stocks: {},
                Reward: {}'''.format(self.budget_history[-1],
                                    self.num_of_stocks_history[-1],
                                    self.rewards_history[-1])
