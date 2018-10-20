import matplotlib.pyplot as plt
import pandas as pd

class Evaluation:
    def __init__(self):
        self.total_rewards = 0
        self.total_penalties = 0
        self.profit = 0

    def add_reward(self, reward):
        self.total_rewards+=reward

    def add_penalty(self, penalty):
        self.total_penalties+= penalty

    def plot_actions_during_time(self, price, actions):
        price.plot()

        data_and_actions = pd.concat([price, actions], axis = 1)
        data_and_actions.dropna(inplace=True)

        actions_buy = data_and_actions[actions.Action ==0]
        actions_sell = data_and_actions[actions.Action==1]

        plt.plot(actions_buy['Close'], "ro")
        plt.plot(actions_sell['Close'], "go")
        plt.show()

    def print_evaluation(self):
        print "Rewards: {}, Penalties: {}, Profit:{} ".format(self.total_penalties, self.total_rewards, self.profit)