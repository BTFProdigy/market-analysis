import matplotlib.pyplot as plt
import pandas as pd

from market_analysis.deep_q_learning.action import Action


class Evaluation:

    def plot_actions_during_time(self, price, actions):

        index = price.index[-len(actions):]
        actions = pd.DataFrame(data = actions, index = index, columns = ['Action'])
        data_and_actions = pd.concat([price, actions], axis = 1)
        data_and_actions.dropna(inplace=True)

        actions_buy = data_and_actions[actions.Action ==Action.Buy]
        actions_sell = data_and_actions[actions.Action==Action.Sell]

        price.plot()
        plt.plot(actions_buy['Close'], "ro")
        plt.plot(actions_sell['Close'], "go")
        plt.show()

    def evaluate(self, deep_q_statistics):
        self.plot_budget(deep_q_statistics.budget_history)
        self.plot_stocks(deep_q_statistics.num_of_stocks_history)
        self.plot_epsilon(deep_q_statistics.epsilon_history)
        self.plot_rewards(deep_q_statistics.rewards_history)
        self.plot_random_actions(deep_q_statistics.random_actions)

    def plot_rewards(self, rewards):
        self.plot_statistics(rewards, "Reward")

    def plot_stocks(self, stocks):
        self.plot_statistics(stocks, "Num of owned stocks")

    def plot_budget(self, budget):
        self.plot_statistics(budget, "Budget")

    def plot_epsilon(self, epsilon):
        self.plot_statistics(epsilon, "Epsilon")

    def plot_random_actions(self, random_actions):
        self.plot_statistics(random_actions, "No of random actions")

    def plot_statistics(self, list, ylabel, xlabel = "Iteration"):
        plt.plot(list)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(ylabel)
        plt.show()


    def print_final_state(self, statistics):
        print '''Budget: {},
                Num of stocks: {},
                Reward: {}'''.format(statistics.budget_history[-1],
                                     statistics.num_of_stocks_history[-1],
                                     statistics.rewards_history[-1])