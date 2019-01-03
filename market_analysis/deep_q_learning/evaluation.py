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

        ax = price.plot()
        ax.grid(color = 'gray', linestyle = '-', linewidth = 0.25, alpha = 0.5)
        plt.plot(actions_buy['Price'], "ro", label = 'Buy')
        plt.plot(actions_sell['Price'], "go", label = 'Sell')
        plt.title('Actions')
        plt.legend()
        plt.show()

    def evaluate(self, deep_q_statistics):
        self.plot_budget(deep_q_statistics.budget_history)
        self.plot_stocks(deep_q_statistics.num_of_stocks_history)
        self.plot_epsilon(deep_q_statistics.epsilon_history)
        self.plot_rewards(deep_q_statistics.rewards_history)
        self.plot_rewards_avg(deep_q_statistics.avg_rewards_history)

        self.plot_random_actions(deep_q_statistics.random_actions)
        self.plot_budget_for_last_iteration(deep_q_statistics.budget_for_last_iteration)
        self.plot_stocks_for_last_iteration(deep_q_statistics.stocks_for_last_iteration)

        self.scatter_plot_rewards(deep_q_statistics.actions_for_last_iteration, deep_q_statistics.rewards_for_last_iteration)
        # self.show_rewards_for_buy_and_sell_in_last_iteration(deep_q_statistics.actions_for_last_iteration, deep_q_statistics.rewards_for_last_iteration)
        self.scatter_plot_rewards1(deep_q_statistics.actions_for_last_iteration, deep_q_statistics.rewards_for_last_iteration)
        self.plot_rewards_for_last_iteration(deep_q_statistics.rewards_for_last_iteration)

    def plot_rewards(self, rewards):
        self.plot_statistics(rewards, "Reward")

    def plot_rewards_avg(self, rewards_avg):
        self.plot_statistics(rewards_avg, " Average Reward")

    def plot_stocks(self, stocks):
        self.plot_statistics(stocks, "Number of owned stocks")

    def plot_stocks_for_last_iteration(self, stocks):
        self.plot_statistics(stocks, "Stocks for last iteration")

    def plot_budget(self, budget):
        self.plot_statistics(budget, "Budget")

    def plot_budget_for_last_iteration(self, budget):
        self.plot_statistics(budget, "Budget for last iteration")

    def plot_epsilon(self, epsilon):
        self.plot_statistics(epsilon, "Epsilon")

    def plot_random_actions(self, random_actions):
        self.plot_statistics(random_actions, "Number of random actions")

    def plot_statistics(self, list, ylabel, xlabel = "Iteration"):
        x = range(list.__len__())
        plt.plot(x, list)

        plt.grid(color = 'gray', linestyle = '-', linewidth = 0.25, alpha = 0.5)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(ylabel)

        plt.show()

    def show_rewards_for_buy_and_sell_in_last_iteration(self, actions, rewards):

        for index, action in enumerate(actions):
            print 'Action {}, Reward {}'.format(action, rewards[index])
            plt.text(len%500, index, str(rewards[index]), fontsize=8)
        plt.show()

    def scatter_plot_rewards(self, actions, rewards):

        colors = map(lambda x: int(x), actions)
        instances = range(len(actions))
        plt.scatter(instances, rewards, c=colors, alpha=0.3,
                    cmap='viridis')
        plt.grid(color = 'gray', linestyle = '-', linewidth = 0.25, alpha = 0.5)

        plt.colorbar()  # show color scale
        plt.title('Rewards for specific actions in the last iteration')
        plt.xlabel('Instance')
        plt.ylabel('Reward')
        plt.show()

    def scatter_plot_rewards1(self, actions, rewards):

        plt.scatter(map(lambda x: int(x), actions), rewards, alpha=0.3,
                    cmap='viridis')
        plt.grid(color = 'gray', linestyle = '-', linewidth = 0.25, alpha = 0.5)

        plt.title('Rewards for specific actions in the last iteration')
        plt.xlabel('Action')
        plt.ylabel('Reward')

        plt.show()

    def plot_rewards_for_last_iteration(self, rewards):
        self.plot_statistics(rewards, "Reward", "Instance")

    def print_final_state(self, statistics):
        print '''Budget: {},
                Num of stocks: {},
                Reward: {}'''.format(statistics.budget_history[-1],
                                     statistics.num_of_stocks_history[-1],
                                     statistics.rewards_history[-1])