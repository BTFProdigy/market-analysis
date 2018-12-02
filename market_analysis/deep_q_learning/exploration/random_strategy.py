from strategy import Strategy


class RandomStrategy(Strategy):

    def __init__(self, num_of_actions):
        self.num_of_actions = num_of_actions

    def get_epsilon(self, state):
        return 1
